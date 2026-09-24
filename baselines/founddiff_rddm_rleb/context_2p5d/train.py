"""2-step/5-step smoke, independent 12k runs, and gated Full-L333."""
import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path
import torch
from fast_dev.protocol import FAST_INDICES
from mayo2016_smallset_v1.runtime import ROOT, atomic_save, rng_state, sha256, write_json
from .data import ContextPairs, ContextDataGuard
from .evaluate import validate
from .model import ARMS, construct, parameter_usage
from .protocol import HERE, STEPS, BASE_FULL, config, references, source_manifest
from .runtime import ScreenState, rng_hash


def csv_write(path,rows):
    if not rows:return
    with Path(path).open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)


def audit_access(access,full=False,smoke=False):
    assert access["L506_opens"]==access["external_memory_opens"]==0
    assert access["neighbor_target_opens"]==access["blocked_attempts"]==0
    assert access["data_directory_enumeration"] is False
    expected=list(range(610)) if full else list(FAST_INDICES)
    if smoke:assert access["validation_center_indices_opened"]==[]
    else:assert access["validation_center_indices_opened"]==expected
    assert set(access["training_patients_opened"])<=set(("L067","L143"))


def smoke(arm,device):
    cfg,base=config()
    out=HERE/"smoke_verified"/arm
    out.mkdir(parents=True,exist_ok=False)
    guard=ContextDataGuard(base,"FAST",allow_training=True)
    state=None
    try:
        with guard:
            state=ScreenState(arm,device,guard)
            assert state.init_audit["step_zero_equivalent"]
            expected_steps=2 if arm==ARMS[0] else 5
            seen=[]
            def output_hook(module,inputs,output):
                assert output.shape==(inputs[0].shape[0],1,*inputs[0].shape[-2:])
                assert bool(torch.isfinite(output).all())
                seen.append(tuple(output.shape))
            hook=state.model.model.unet0.register_forward_hook(output_hook)
            initial_hash=state.initial_rng_sha256
            rows=[state.update() for _ in range(expected_steps)]
            hook.remove()
            assert len(seen)==expected_steps and all(s==(2,1,512,512) for s in seen)
            assert all(math.isfinite(r["train_loss"]) and math.isfinite(r["gradient_norm_pre_clip"]) for r in rows)
            gradient=state.gradient_audit()
            if arm==ARMS[0]:
                assert gradient["neighbor_stem_gradient_abs"]>0
                assert gradient["parameter_usage"]["prev_stem_weight_norm"]>0
                assert gradient["parameter_usage"]["next_stem_weight_norm"]>0
            else:
                assert state.init_audit["initial_projection_norms"]==[0.,0.,0.]
                assert all(x>0 for x in gradient["first_step_projection_gradient_abs"])
                assert gradient["later_context_encoder_gradient_abs"]>0
                assert all(gradient["projection_optimizer_states_present"])
                assert gradient["context_encoder_optimizer_state_present"]
                assert all(x>0 for x in gradient["parameter_usage"]["projection_weight_norms"])
        access=guard.report();audit_access(access,smoke=True)
        result={
          "arm":arm,"status":"SMOKE_PASS","optimizer_steps":expected_steps,
          "losses":[r["train_loss"] for r in rows],
          "gradient_norms":[r["gradient_norm_pre_clip"] for r in rows],
          "output_shapes":[list(s) for s in seen],
          "initialization_audit":state.init_audit,"gradient_audit":gradient,
          "initial_training_rng_sha256":initial_hash,
          "train_stream_sha256":state.stream_digest.hex(),
          "training_state_discarded":True,"data_access":access,
          "source_manifest":state.manifest,
        }
        write_json(out/"SMOKE_RESULT.json",result)
        print("SMOKE_PASS",arm,json.dumps({"steps":expected_steps,"losses":result["losses"],
              "gradients":result["gradient_norms"]}),flush=True)
    except BaseException as exc:
        write_json(out/"SMOKE_RESULT.json",{
          "arm":arm,"status":"FAILED","error":repr(exc),"data_access":guard.report()})
        raise


def full_validate(arm,checkpoint,base,device,out):
    out.mkdir(parents=True,exist_ok=False)
    guard=ContextDataGuard(base,"FULL",allow_training=False)
    with guard:
        dataset=ContextPairs(base,"validation","FULL")
        dataset.bind_guard(guard)
        model,audit=construct(arm,device)
        ck=torch.load(str(checkpoint),map_location="cpu")
        assert ck["arm"]==arm and ck["optimizer_step"]==12000
        assert ck["source_manifest"]==source_manifest()
        model.load_state_dict(ck["ema_state_dict"],strict=True)
        summary=validate(model,dataset,base,torch.device(device),out/"full_l333_rows.json")
    access=guard.report();audit_access(access,full=True)
    observed=json.loads((out/"full_l333_rows.json").read_text())
    with BASE_FULL.open(newline="") as f:control=list(csv.DictReader(f))
    assert len(observed)==len(control)==610
    paired=[]
    for a,b in zip(observed,control):
        assert a["patient_id"]==b["patient_id"]=="L333"
        assert a["slice_index"]==int(b["slice_index"])
        row={"patient_id":"L333","slice_index":a["slice_index"]}
        for metric in ("psnr","ssim","rmse_hu","mae_hu"):
            row["candidate_"+metric]=a[metric]
            row["control_"+metric]=float(b["control_"+metric])
            row["delta_"+metric]=a[metric]-float(b["control_"+metric])
        paired.append(row)
    csv_write(out/"full_l333_paired.csv",paired)
    result={
      "candidate":summary,
      "control_mean":{m:sum(float(r["control_"+m]) for r in control)/610
                      for m in ("psnr","ssim","rmse_hu","mae_hu")},
      "delta_mean":{m:sum(r["delta_"+m] for r in paired)/610
                    for m in ("psnr","ssim","rmse_hu","mae_hu")},
      "comparison":"same L333 stack and archived 12k CONTROL_RDDM per-slice metrics; not patient-level significance",
      "control_source_sha256":sha256(BASE_FULL),
      "data_access":access,
    }
    write_json(out/"full_l333_summary.json",result)
    return result


def audited_smoke(arm):
    path=(HERE/"smoke_verified"/arm if arm==ARMS[0] else HERE/"smoke_pass"/arm)/"SMOKE_RESULT.json"
    result=json.loads(path.read_text())
    assert result["status"]=="SMOKE_PASS"
    current=source_manifest()
    changed=sorted(k for k in current if result["source_manifest"].get(k)!=current[k])
    assert set(changed)<={"context_2p5d/data.py","context_2p5d/train.py","context_2p5d/queue.py"}
    assert result["source_manifest"]["context_2p5d/model.py"]==current["context_2p5d/model.py"]
    assert result["source_manifest"]["src/rddm_rleb.py"]==current["src/rddm_rleb.py"]
    return result,changed

def train(arm,device,resume=None):
    cfg,base=config();ref=references()
    smoke_result,smoke_source_changes=audited_smoke(arm)
    out=HERE/"runs"/arm
    if resume:assert out.is_dir()
    else:
        out.mkdir(parents=True,exist_ok=False)
        (out/"checkpoints").mkdir()
    state=None;history=[];guard=ContextDataGuard(base,"FAST",allow_training=True)
    try:
        with guard:
            state=ScreenState(arm,device,guard)
            assert state.init_audit==smoke_result["initialization_audit"]
            assert state.initial_rng_sha256==smoke_result["initial_training_rng_sha256"]
            if resume:
                state.resume(resume)
                if (out/"validation_history.csv").is_file():
                    with (out/"validation_history.csv").open(newline="") as f:
                        history=list(csv.DictReader(f))
            else:
                write_json(out/"CONFIG.json",{**cfg,"arm":arm,
                  "only_architecture_change":"4-channel stem; zero neighbor channels" if arm==ARMS[0]
                  else "lightweight neighbor encoder; zero 256/128/64 projections"})
                write_json(out/"INITIALIZATION_AUDIT.json",state.init_audit)
                write_json(out/"TRAIN_RNG_AUDIT.json",{
                  "initial_rng_sha256":state.initial_rng_sha256,
                  "canonical_restored":True,
                  "smoke_initial_rng_sha256":smoke_result["initial_training_rng_sha256"]})
                write_json(out/"provenance.json",{
                  "git_parent":subprocess.check_output(["git","rev-parse","HEAD"],cwd=str(ROOT),text=True).strip(),
                  "source_manifest":state.manifest,"python":sys.version,
                  "torch":torch.__version__,"cuda":torch.version.cuda,
                  "gpu":torch.cuda.get_device_name(state.device) if state.device.type=="cuda" else None,
                  "seed":cfg["seed"],"fast_indices":list(FAST_INDICES),
                  "adjacency_audit_sha256":sha256(HERE/"ADJACENCY_AUDIT.json"),
                  "smoke_source_changes":smoke_source_changes,
                  "smoke_model_source_unchanged":True})
            assert state.step<12000
            manifest={"arm":arm,"matched":[],"latest":None}
            if resume and (out/"checkpoint_manifest.json").is_file():
                manifest=json.loads((out/"checkpoint_manifest.json").read_text())
            mode="a" if resume else "x"
            status="RUNNING"
            with (out/"train.jsonl").open(mode,buffering=1) as log:
                while state.step<12000:
                    row=state.update()
                    log.write(json.dumps(row,allow_nan=False)+"\n")
                    if state.step%100==0:
                        write_json(out/"progress.json",{
                          "status":"RUNNING","step":state.step,"stop":12000})
                        print("TRAIN",arm,state.step,row["train_loss"],flush=True)
                    if state.step in STEPS:
                        before=rng_hash(rng_state())
                        metrics=validate(state.ema.model,state.validation_data,base,state.device,
                            out/("validation_step_%d.json"%state.step))
                        assert rng_hash(rng_state())==before
                        assert metrics["slices"]==64 and metrics["weights"]=="ema"
                        assert state.stream_digest.hex()==ref[state.step]["train_stream_sha256"]
                        delta=metrics["psnr"]-ref[state.step]["psnr"]
                        record={"step":state.step,**metrics,"baseline_psnr":ref[state.step]["psnr"],
                          "delta_psnr":delta,"train_loss":row["train_loss"],
                          "gradient_norm_pre_clip":row["gradient_norm_pre_clip"],
                          "learning_rate":row["learning_rate"],
                          "parameter_count":state.init_audit["parameter_count"],
                          "peak_vram_bytes":state.peak_vram,
                          "seconds_per_step":state.train_seconds/state.step,
                          "train_stream_sha256":state.stream_digest.hex()}
                        assert all(math.isfinite(record[k]) for k in
                          ("psnr","ssim","rmse_hu","mae_hu","validation_loss","delta_psnr"))
                        history.append(record);csv_write(out/"validation_history.csv",history)
                        print("VALIDATION",arm,json.dumps(record),flush=True)
                    if state.step%1000==0:
                        latest=out/"checkpoints/latest.pt"
                        atomic_save(latest,state.payload())
                        manifest["latest"]={"step":state.step,"sha256":sha256(latest)}
                        if state.step in STEPS:
                            matched=out/"checkpoints"/("step_%d.pt"%state.step)
                            assert not matched.exists()
                            os.link(latest,matched)
                            manifest["matched"].append({"step":state.step,"sha256":sha256(matched)})
                        write_json(out/"checkpoint_manifest.json",manifest)
                    if state.step==4000 and history[-1]["delta_psnr"]<=-0.5:
                        status="EARLY_STOP_PERFORMANCE";break
                else:status="COMPLETE"
        access=guard.report();audit_access(access)
        assert [int(r["step"]) for r in history]==[s for s in STEPS if s<=state.step]
        flags={"promotion":"NO_PROMOTION","strong_fast_signal":False}
        if status=="COMPLETE":
            deltas={int(r["step"]):float(r["delta_psnr"]) for r in history}
            if all(deltas[s]>0 for s in (8000,10000,12000)) and deltas[12000]>=0.05:
                flags["promotion"]="PROMOTE_FULL_L333"
                flags["strong_fast_signal"]=deltas[12000]>=0.10
        full=None
        if flags["promotion"]=="PROMOTE_FULL_L333":
            full=full_validate(arm,out/"checkpoints/step_12000.pt",base,device,
                               HERE/"full_l333"/arm)
        usage=parameter_usage(state.model,arm)
        write_json(out/"parameter_usage_summary.json",usage)
        gradient=state.gradient_audit()
        if arm==ARMS[0]:
            assert gradient["neighbor_stem_gradient_abs"]>0
            assert usage["prev_stem_weight_norm"]>0 and usage["next_stem_weight_norm"]>0
        else:
            assert gradient["later_context_encoder_gradient_abs"]>0
            assert all(x>0 for x in usage["projection_weight_norms"])
        summary={
          "arm":arm,"status":status,"optimizer_steps":state.step,"history":history,
          "flags":flags,"full_l333":full,
          "parameter_count":state.init_audit["parameter_count"],
          "peak_vram_bytes":state.peak_vram,"training_seconds":state.train_seconds,
          "seconds_per_step":state.train_seconds/state.step,
          "gradient_min":state.gradient_min,"gradient_max":state.gradient_max,
          "nonfinite_count":state.nonfinite_count,
          "train_stream_sha256":state.stream_digest.hex(),
          "initial_rng_sha256":state.initial_rng_sha256,
          "gradient_audit":gradient,"parameter_usage":usage,
          "data_access":access,"source_manifest":state.manifest,
        }
        write_json(out/"final_summary.json",summary)
        write_json(out/"DATA_ACCESS_AUDIT.json",access)
        write_json(out/"TRAIN_STREAM_AUDIT.json",{
          "stream_sha256":state.stream_digest.hex(),"samples":state.step*2,
          "matched_control_sha256":ref[state.step]["train_stream_sha256"]})
        write_json(out/"TRAIN_RNG_AUDIT.json",{
          "initial_rng_sha256":state.initial_rng_sha256,
          "final_rng_sha256":rng_hash(rng_state()),"validation_rng_preserved":True})
        write_json(out/"progress.json",{"status":status,"step":state.step,"stop":12000})
        light=HERE/arm;light.mkdir(exist_ok=True)
        for name in ("CONFIG.json","INITIALIZATION_AUDIT.json","TRAIN_RNG_AUDIT.json",
          "TRAIN_STREAM_AUDIT.json","DATA_ACCESS_AUDIT.json","validation_history.csv",
          "final_summary.json","provenance.json","parameter_usage_summary.json"):
            src=out/name
            if src.is_file():(light/name).write_bytes(src.read_bytes())
        print("ARM_END",arm,status,flags,flush=True)
    except BaseException as exc:
        if out.is_dir():
            write_json(out/"DATA_ACCESS_AUDIT.json",guard.report())
            write_json(out/"progress.json",{
              "status":"FAILED","step":state.step if state else 0,"error":repr(exc)})
        raise


def main():
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("smoke","train"))
    p.add_argument("--arm",choices=ARMS,required=True)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--resume")
    a=p.parse_args()
    if a.mode=="smoke":smoke(a.arm,a.device)
    else:train(a.arm,a.device,a.resume)

if __name__=="__main__":main()
