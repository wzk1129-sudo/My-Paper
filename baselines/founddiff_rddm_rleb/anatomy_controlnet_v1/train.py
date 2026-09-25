"""Serial C0/C1/C2 Fast64 screen; training runs have independent output directories."""
import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path
import torch
from context_2p5d.data import ContextPairs, ContextDataGuard
from mayo2016_smallset_v1.runtime import ROOT, write_json, sha256, rng_state
from .evaluate import validate
from .model import parameter_counts
from .protocol import HERE, STEPS, ARMS, DIRECT_CK, DIRECT_SHA, experiment_config, frozen_config
from .runtime import load_direct, construct, TrainState, rng_hash

def csv_write(path,rows):
    if not rows:return
    with Path(path).open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)

def check_access(a,training=False):
    assert a["L506_opens"]==a["external_memory_opens"]==0
    assert a["neighbor_target_opens"]==a["blocked_attempts"]==0
    assert a["validation_center_indices_opened"]==list(map(int,(ROOT/"splits/L333_fast64.txt").read_text().splitlines())) if not training else True

def smoke(arm,device):
    assert arm in ARMS[1:]
    out=HERE/"smoke_verified"/arm
    out.mkdir(parents=True,exist_ok=False)
    cfg=experiment_config(arm)
    guard=ContextDataGuard(cfg,"FAST",allow_training=True)
    try:
        with guard:
            state=TrainState(arm,device,guard)
            initial=[float(z.weight.detach().norm()) for z in state.model.model.unet0.zero_convs]
            assert initial==[0.]*4 and state.init_audit["step_zero_equivalent"]
            rows=[]
            for _ in range(20):
                r=state.update();rows.append(r)
                assert math.isfinite(r["train_loss"]) and math.isfinite(r["gradient_norm_pre_clip"])
            w=[float(z.weight.detach().norm()) for z in state.model.model.unet0.zero_convs]
            assert all(v>0 for v in w)
            assert state.model.model.unet0.main_state_hash()==state.main_hash
            assert state.ema.model.model.unet0.main_state_hash()==state.main_hash
            assert state.ema.updates==20
            if arm=="C2_EDGE":
                a=state.model.model.unet0.structure.p256[0].weight
                assert a.grad is not None and bool(torch.isfinite(a.grad).all())
        access=guard.report()
        assert access["L506_opens"]==access["external_memory_opens"]==access["neighbor_target_opens"]==access["blocked_attempts"]==0
        result={"status":"SMOKE_PASS","arm":arm,"steps":20,
                "initialization":state.init_audit,"parameter_audit":parameter_counts(state.model),
                "initial_zero_norms":initial,"final_zero_norms":w,
                "loss_first":rows[0]["train_loss"],"loss_last":rows[-1]["train_loss"],
                "gradient_norm_max":max(r["gradient_norm_pre_clip"] for r in rows),
                "peak_vram_bytes":state.peak_vram,
                "seconds_per_step":state.train_seconds/20,
                "main_hash_unchanged":True,"ema_updates":20,
                "data_access":access}
        write_json(out/"SMOKE_RESULT.json",result)
        print("SMOKE_PASS",arm,json.dumps(result),flush=True)
    except BaseException as exc:
        write_json(out/"SMOKE_RESULT.json",{"status":"FAILED","arm":arm,"error":repr(exc),
                                            "data_access":guard.report()})
        raise

def control(device):
    cfg=experiment_config("C0_DIRECT")
    out=HERE/"runs/C0_DIRECT"
    out.mkdir(parents=True,exist_ok=False)
    guard=ContextDataGuard(cfg,"FAST",allow_training=False)
    try:
        model=load_direct(device)
        data=ContextPairs(frozen_config(),"validation","FAST");data.bind_guard(guard)
        with guard:
            metrics,rows,_=validate(model,data,cfg,torch.device(device))
        access=guard.report()
        check_access(access)
        assert abs(metrics["psnr"]-40.76762730415604)<1e-4,metrics["psnr"]
        csv_write(out/"per_slice_step_0.csv",rows)
        write_json(out/"CONFIG.json",cfg)
        write_json(out/"DATA_ACCESS_AUDIT.json",access)
        write_json(out/"final_summary.json",{"arm":"C0_DIRECT","status":"COMPLETE","step":0,
            "metrics":metrics,"trainable_parameters":0,"checkpoint_sha256":DIRECT_SHA,
            "data_access":access})
        write_json(out/"progress.json",{"status":"COMPLETE","step":0})
        print("CONTROL_COMPLETE",json.dumps(metrics),flush=True)
    except BaseException as exc:
        write_json(out/"progress.json",{"status":"FAILED","error":repr(exc)})
        raise

def train(arm,device):
    assert arm in ARMS[1:]
    smoke_path=HERE/"smoke_verified"/arm/"SMOKE_RESULT.json"
    assert json.loads(smoke_path.read_text())["status"]=="SMOKE_PASS"
    out=HERE/"runs"/arm
    out.mkdir(parents=True,exist_ok=False)
    (out/"checkpoints").mkdir()
    cfg=experiment_config(arm)
    guard=ContextDataGuard(cfg,"FAST",allow_training=True)
    state=None
    history=[];activity=[]
    try:
        with guard:
            state=TrainState(arm,device,guard)
            assert state.init_audit==json.loads(smoke_path.read_text())["initialization"]
            write_json(out/"CONFIG.json",cfg)
            write_json(out/"INITIALIZATION_AUDIT.json",state.init_audit)
            write_json(out/"PARAMETER_AUDIT.json",parameter_counts(state.model))
            write_json(out/"TRAIN_RNG_AUDIT.json",{"initial_rng_sha256":state.initial_rng_sha256,
                "canonical_restored":True,"smoke_rng_sha256":
                json.loads(smoke_path.read_text())["initialization"]["canonical_training_rng_sha256"]})
            def evaluate(step):
                before=rng_hash(rng_state())
                metrics,rows,act=validate(state.ema.model,state.validation,cfg,state.device,activity=True)
                assert rng_hash(rng_state())==before
                assert metrics["slices"]==64 and metrics["weights"]=="ema"
                csv_write(out/("per_slice_step_%d.csv"%step),rows)
                row={"arm":arm,"step":step,**metrics,
                    "train_loss":None if state.last is None else state.last["train_loss"],
                    "gradient_norm_pre_clip":None if state.last is None else state.last["gradient_norm_pre_clip"],
                    "learning_rate":state.optimizer.param_groups[0]["lr"],
                    "peak_vram_bytes":state.peak_vram,
                    "seconds_per_step":None if step==0 else state.train_seconds/step,
                    "train_stream_sha256":state.stream_digest.hex(),
                    "pre_step_rng_sha256":None if state.last is None else state.last["pre_step_rng_sha256"]}
                history.append(row);activity.append({"arm":arm,"step":step,
                    **{k:json.dumps(v) for k,v in act.items()}})
                csv_write(out/"validation_history.csv",history)
                csv_write(out/"control_activity.csv",activity)
                print("VALIDATION",arm,step,json.dumps(row),flush=True)
            evaluate(0)
            with (out/"train.jsonl").open("x",buffering=1) as f:
                while state.step<6000:
                    row=state.update()
                    f.write(json.dumps(row,allow_nan=False)+"\n")
                    if state.step%100==0:
                        write_json(out/"progress.json",{"status":"RUNNING","step":state.step,"stop":6000})
                        print("TRAIN",arm,state.step,row["train_loss"],flush=True)
                    if state.step in STEPS[1:]:
                        evaluate(state.step)
                        path=out/"checkpoints"/("step_%d.pt"%state.step)
                        state.checkpoint(path)
                        write_json(out/"CHECKPOINT_MANIFEST.json",
                            {"arm":arm,"matched":[{"step":s,"sha256":sha256(out/"checkpoints"/("step_%d.pt"%s))}
                                for s in STEPS[1:] if s<=state.step]})
        access=guard.report()
        assert access["L506_opens"]==access["external_memory_opens"]==access["neighbor_target_opens"]==access["blocked_attempts"]==0
        assert access["validation_center_indices_opened"]==list(map(int,(ROOT/"splits/L333_fast64.txt").read_text().splitlines()))
        assert state.step==6000 and state.ema.updates==6000
        assert state.model.model.unet0.main_state_hash()==state.main_hash
        assert state.ema.model.model.unet0.main_state_hash()==state.main_hash
        assert state.nonfinite_count==0
        write_json(out/"DATA_ACCESS_AUDIT.json",access)
        write_json(out/"TRAIN_STREAM_AUDIT.json",{"sha256":state.stream_digest.hex(),
            "samples":state.step*2,"initial_rng_sha256":state.initial_rng_sha256})
        summary={"arm":arm,"status":"COMPLETE","optimizer_steps":state.step,
            "history":history,"parameter_audit":parameter_counts(state.model),
            "peak_vram_bytes":state.peak_vram,"training_seconds":state.train_seconds,
            "seconds_per_step":state.train_seconds/state.step,"nonfinite_count":state.nonfinite_count,
            "main_hash_before":state.main_hash,
            "main_hash_after":state.model.model.unet0.main_state_hash(),
            "ema_main_hash_after":state.ema.model.model.unet0.main_state_hash(),
            "train_stream_sha256":state.stream_digest.hex(),
            "data_access":access}
        write_json(out/"final_summary.json",summary)
        write_json(out/"progress.json",{"status":"COMPLETE","step":6000})
        print("ARM_COMPLETE",arm,flush=True)
    except BaseException as exc:
        write_json(out/"DATA_ACCESS_AUDIT.json",guard.report())
        write_json(out/"progress.json",{"status":"FAILED","step":0 if state is None else state.step,
                                        "error":repr(exc)})
        raise

def main():
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("smoke","control","train"))
    p.add_argument("--arm",choices=ARMS)
    p.add_argument("--device",default="cuda:0")
    a=p.parse_args()
    if a.mode=="control":control(a.device)
    elif a.mode=="smoke":smoke(a.arm,a.device)
    else:train(a.arm,a.device)
if __name__=="__main__":main()
