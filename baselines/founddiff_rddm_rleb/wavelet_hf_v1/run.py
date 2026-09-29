"""Isolated wavelet-HF supervision on the frozen 2P5D_DIRECT_CONCAT RDDM."""
import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import context_2p5d.evaluate as evaluation
import context_2p5d.model as direct_model
import context_2p5d.runtime as direct_runtime
from context_2p5d.data import ContextDataGuard, ContextPairs
from context_2p5d.model import ARMS, construct, state_hash
from context_2p5d.protocol import config as direct_config
from fast_dev.protocol import FAST_INDICES
from mayo2016_smallset_v1.runtime import write_json
from wavelet_hf_v1.haar import Haar, BANDS, unit_test

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/wavelet_hf_v1"
REFERENCE = ROOT / "experiments/rddm_2p5d_screen_v1/2P5D_DIRECT_CONCAT"
BASE_HISTORY = REFERENCE / "validation_history.csv"
BASE_INIT = REFERENCE / "INITIALIZATION_AUDIT.json"
MATCHED_FULL_PSNR = 40.777830
SOURCE_WORKTREE = Path("/home/xujian/projects/CoreDiff/experiments/rddm_2p5d_screen_v1_20260924")
BASE_CKPT = SOURCE_WORKTREE / "baselines/founddiff_rddm_rleb/experiments/rddm_2p5d_screen_v1/runs/2P5D_DIRECT_CONCAT/checkpoints/step_12000.pt"
NODES = (2000,4000,6000,8000,10000,12000)
METRICS = ("psnr","ssim","rmse_hu","mae_hu")
LOSS_KEYS = ("loss_res","loss_lh","loss_hl","loss_hh","loss_hf","weighted_hf_loss","loss_total")

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1048576),b""):
            h.update(chunk)
    return h.hexdigest()

def jwrite(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    write_json(path,data)

def csv_write(path,rows):
    assert rows
    path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,"w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def reference_history():
    rows=list(csv.DictReader(BASE_HISTORY.open()))
    result={int(row["step"]):float(row["psnr"]) for row in rows}
    assert set(result)==set(NODES)
    assert abs(result[8000]-40.65797712281996)<1e-9
    return result

class LossComputer:
    def __init__(self,haar,lambda_hf):
        self.haar=haar
        self.lambda_hf=float(lambda_hf)
        self.last=None
        self.eval_records=[]

    def _compute(self,diffusion,target,center,prev,next_):
        assert diffusion.objective=="pred_res" and diffusion.model.num_unet==1
        residual_gt=2*(center-target)
        captured=[]
        def hook(_module,_inputs,output):
            assert len(output)==1
            captured.append(output[0])
        handle=diffusion.model.register_forward_hook(hook)
        try:
            loss_res=direct_model.training_loss(diffusion,target,center,prev,next_)
        finally:
            handle.remove()
        assert len(captured)==1
        predicted=captured[0]
        assert predicted.shape==residual_gt.shape
        direct_mse=F.mse_loss(predicted,residual_gt)
        assert torch.allclose(loss_res,direct_mse,atol=1e-6,rtol=1e-5),(
            loss_res.item(),direct_mse.item())
        errors=self.haar.errors(predicted,residual_gt)
        weighted=self.lambda_hf*errors["HF"]
        total=loss_res if self.lambda_hf==0 else loss_res+weighted
        components={
            "loss_res":float(loss_res.detach()),
            "loss_lh":float(errors["LH"].detach()),
            "loss_hl":float(errors["HL"].detach()),
            "loss_hh":float(errors["HH"].detach()),
            "loss_hf":float(errors["HF"].detach()),
            "weighted_hf_loss":float(weighted.detach()),
            "loss_total":float(total.detach()),
            "LL_MSE":float(errors["LL"].detach()),
        }
        assert all(math.isfinite(value) for value in components.values())
        return loss_res,total,components

    def train_loss(self,diffusion,target,center,prev,next_):
        _,total,self.last=self._compute(diffusion,target,center,prev,next_)
        return total

    def eval_loss(self,diffusion,target,center,prev,next_):
        residual,_,components=self._compute(diffusion,target,center,prev,next_)
        self.eval_records.append(components)
        return residual

def assert_init(state):
    prior=json.loads(BASE_INIT.read_text())
    assert state.init_audit==prior,("initialization differs from matched Direct Concat",state.init_audit)
    count=sum(p.numel() for p in state.model.parameters())
    assert count==13137217 and state.init_audit["parameter_count"]==count
    assert state.model.objective=="pred_res"
    assert state.model.num_timesteps==1000
    assert state.model.sampling_timesteps==2 and state.model.sum_scale==0.01
    assert state.model.model.unet0.init_conv.in_channels==4
    assert len(list(Haar(state.device).kernel.shape))==4
    return {"status":"PASS","matched_direct_initialization_audit":True,
            "canonical_state_sha256":prior["canonical_state_sha256"],
            "direct_initial_model_sha256":state_hash(state.model.state_dict()),
            "parameter_count":count,"network_topology_unchanged":True,
            "sampler_unchanged":True,"inference_path_unchanged":True,
            "wavelet_model_parameters":0}

def check_access(guard,training):
    access=guard.report()
    assert access["L506_opens"]==access["external_memory_opens"]==0
    assert access["neighbor_target_opens"]==access["blocked_attempts"]==0
    assert access["data_directory_enumeration"] is False
    if training:
        assert set(access["training_patients_opened"])=={"L067","L143"}
    else:
        assert access["training_patients_opened"]==[]
    return access

def prepare():
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG")==":4096:8"
    assert not (OUT/"CALIBRATION.json").exists()
    cfg,base=direct_config()
    source_cfg=json.loads((REFERENCE/"CONFIG.json").read_text())
    for key in ("seed","data","model","diffusion","training","ema","augmentation","evaluation","fast_dev"):
        assert source_cfg[key]==cfg[key]
    assert source_cfg["arm"]==ARMS[0]
    OUT.mkdir(parents=True,exist_ok=True)
    unit=unit_test()
    jwrite(OUT/"WAVELET_UNIT_TEST.json",unit)
    guard=ContextDataGuard(cfg,"FAST",allow_training=True)
    records=[]
    old=direct_runtime.training_loss
    with guard:
        state=direct_runtime.ScreenState(ARMS[0],"cuda:0",guard)
        init=assert_init(state)
        computer=LossComputer(Haar(state.device),0.0)
        direct_runtime.training_loss=computer.train_loss
        try:
            for _ in range(100):
                result=state.update()
                row={"step":state.step,**computer.last,
                     "gradient_norm_pre_clip":result["gradient_norm_pre_clip"],
                     "seconds_per_step":result["seconds_per_step"]}
                assert abs(row["loss_total"]-row["loss_res"])<1e-12
                records.append(row)
        finally:
            direct_runtime.training_loss=old
        access=check_access(guard,True)
    assert state.step==state.ema.updates==100 and len(records)==100
    assert state.nonfinite_count==0
    tail=records[50:]
    med={k:float(statistics.median(r[k] for r in tail))
         for k in ("loss_res","loss_lh","loss_hl","loss_hh","loss_hf")}
    lam=0.20*med["loss_res"]/(med["loss_hf"]+1e-12)
    status="PASS" if 1e-3<=lam<=10 and math.isfinite(lam) else "CALIBRATION_ABNORMAL"
    summary={
        "status":status,"optimizer_steps":100,"lambda_during_backward":0.0,
        "fresh_formal_initialization_required":True,
        "median_Lres":med["loss_res"],"median_LLH":med["loss_lh"],
        "median_LHL":med["loss_hl"],"median_LHH":med["loss_hh"],
        "median_LHF":med["loss_hf"],"lambda_hf":lam,
        "expected_weighted_ratio":lam*med["loss_hf"]/med["loss_res"],
        "gradient_norm_min":state.gradient_min,"gradient_norm_max":state.gradient_max,
        "peak_vram_bytes":state.peak_vram,
        "seconds_per_step":state.train_seconds/100,
        "nonfinite_count":state.nonfinite_count,
        "initialization_audit":init,"initial_rng_sha256":state.initial_rng_sha256,
        "data_access":access,
    }
    jwrite(OUT/"CALIBRATION.json",summary)
    jwrite(OUT/"calibration_100/summary.json",summary)
    csv_write(OUT/"calibration_100/train_100.csv",records)
    jwrite(OUT/"INITIALIZATION_AUDIT.json",init)
    new=json.loads((REFERENCE/"CONFIG.json").read_text())
    new["experiment_name"]="RDDM_2P5D_Wavelet_HF_V1"
    new["description"]="Direct Concat RDDM with fixed Haar high-frequency residual auxiliary loss"
    new["wavelet_hf"]={
        "baseline_arm":ARMS[0],"transform":"1-level orthonormal Haar",
        "kernel_size":2,"stride":2,"padding":0,"trainable_parameters":0,
        "loss":"L_res+lambda_hf*(MSE_LH+MSE_HL+MSE_HH)/3",
        "lambda_hf":lam,"calibration_optimizer_steps":100,
        "calibration_tail_steps":[51,100],"target_weighted_ratio":0.20,
        "initial_formal_steps":8000,"promotion_max_steps":12000,
        "validation_steps":list(NODES),"matched_baseline_history_sha256":sha(BASE_HISTORY),
        "matched_initialization_audit_sha256":sha(BASE_INIT),
    }
    jwrite(OUT/"CONFIG.json",new)
    print("CALIBRATION",status,"lambda_hf",lam,"median_res",med["loss_res"],
          "median_hf",med["loss_hf"],flush=True)
    if status!="PASS":
        raise RuntimeError("Abnormal lambda_hf; formal training not started")
    return summary

def validate_with_frequency(model,dataset,cfg,device,computer,output):
    old=evaluation.training_loss
    computer.eval_records=[]
    evaluation.training_loss=computer.eval_loss
    try:
        result=evaluation.validate(model,dataset,cfg,device,output=output)
    finally:
        evaluation.training_loss=old
    assert len(computer.eval_records)==len(dataset)
    values={k:float(np.mean([r[k] for r in computer.eval_records]))
            for k in ("LL_MSE","loss_lh","loss_hl","loss_hh","loss_hf")}
    return result,{
        "LL_MSE":values["LL_MSE"],"LH_MSE":values["loss_lh"],
        "HL_MSE":values["loss_hl"],"HH_MSE":values["loss_hh"],
        "mean_HF_MSE":values["loss_hf"],"slices":len(dataset),
    }

def full_validation(state,computer,cfg):
    out=OUT/"full_l333"
    assert not out.exists()
    out.mkdir()
    ckpt=OUT/"fast8k/checkpoints/step_12000.pt"
    payload=torch.load(ckpt,map_location="cpu")
    assert payload["optimizer_step"]==payload["ema_updates"]==12000
    candidate,_=construct(ARMS[0],"cuda:0",cfg["seed"])
    candidate.load_state_dict(payload["ema_state_dict"],strict=True)
    candidate.eval().requires_grad_(False)
    assert BASE_CKPT.is_file()
    control_payload=torch.load(BASE_CKPT,map_location="cpu")
    assert control_payload["optimizer_step"]==12000
    control,_=construct(ARMS[0],"cuda:0",cfg["seed"])
    control.load_state_dict(control_payload["ema_state_dict"],strict=True)
    control.eval().requires_grad_(False)
    guard=ContextDataGuard(cfg,"FULL",allow_training=False)
    with guard:
        dataset=ContextPairs(cfg,"validation","FULL")
        dataset.bind_guard(guard)
        assert len(dataset)==610
        candidate_metrics,candidate_freq=validate_with_frequency(
            candidate,dataset,cfg,state.device,computer,out/"candidate_per_slice.json")
        control_metrics,control_freq=validate_with_frequency(
            control,dataset,cfg,state.device,computer,out/"control_per_slice.json")
        access=check_access(guard,False)
    assert abs(control_metrics["psnr"]-MATCHED_FULL_PSNR)<0.01
    candidate_rows=json.loads((out/"candidate_per_slice.json").read_text())
    control_rows=json.loads((out/"control_per_slice.json").read_text())
    assert len(candidate_rows)==len(control_rows)==610
    assert all(a["slice_index"]==b["slice_index"] for a,b in zip(candidate_rows,control_rows))
    paired=[{"slice_index":a["slice_index"],
             **{f"delta_{k}":a[k]-b[k] for k in METRICS}}
            for a,b in zip(candidate_rows,control_rows)]
    csv_write(out/"paired_metrics.csv",paired)
    result={
        "candidate":candidate_metrics,"matched_direct_control":control_metrics,
        "mean_paired_delta":{k:float(np.mean([r[f"delta_{k}"] for r in paired])) for k in METRICS},
        "positive_psnr_slice_fraction":float(np.mean([r["delta_psnr"]>0 for r in paired])),
        "candidate_frequency":candidate_freq,"control_frequency":control_freq,
        "frequency_delta":{k:candidate_freq[k]-control_freq[k]
                           for k in ("LL_MSE","LH_MSE","HL_MSE","HH_MSE","mean_HF_MSE")},
        "patient":"L333","slices":610,"weights":"ema","candidate_checkpoint_sha256":sha(ckpt),
        "control_checkpoint_sha256":sha(BASE_CKPT),"data_access":access,
    }
    jwrite(out/"FULL_L333_SUMMARY.json",result)
    return result

def report(summary):
    nodes=summary["validation_history"]
    lines=["# Wavelet HF supervision on 2P5D Direct RDDM","",
           "Only the training loss adds fixed orthonormal Haar LH/HL/HH residual supervision. Model, sampler, evaluator, patients, seeds, optimizer and 2.5D inputs are unchanged.","",
           f"Calibration lambda_HF: {summary['lambda_hf']:.8g}; status PASS; formal training started from a fresh matched initialization.","",
           "| Step | PSNR | SSIM | RMSE HU | MAE HU | Delta PSNR vs Direct |",
           "|---:|---:|---:|---:|---:|---:|"]
    for row in nodes:
        lines.append(f"| {row['step']} | {row['psnr']:.6f} | {row['ssim']:.6f} | {row['rmse_hu']:.4f} | {row['mae_hu']:.4f} | {row['delta_PSNR_vs_2P5D_DIRECT']:+.6f} |")
    lines += ["",f"Gate: {summary['gate_decision']}.",f"Completed optimizer steps: {summary['optimizer_steps']}.",
              f"Parameter count: {summary['parameter_count']}; peak allocated VRAM: {summary['peak_vram_bytes']} bytes.",
              f"Mean train seconds/step: {summary['seconds_per_step']:.4f}.",
              "Validation uses EMA, fixed L333-fast64 and the original metric implementation."]
    if summary.get("full_l333"):
        full=summary["full_l333"]
        lines += ["",f"Full L333 (610) candidate PSNR: {full['candidate']['psnr']:.6f}; matched Direct: {full['matched_direct_control']['psnr']:.6f}.",
                  f"Paired mean delta PSNR: {full['mean_paired_delta']['psnr']:+.6f}; positive slice fraction: {full['positive_psnr_slice_fraction']:.4f}."]
    else:
        lines += ["","Full L333 not run because the predefined promotion gate did not pass."]
    lines += ["","L506, external-memory and neighbor NDCT access: 0. No other architecture or loss was added.",""]
    (OUT/"REPORT.md").write_text("\n".join(lines))

def train():
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG")==":4096:8"
    calibration=json.loads((OUT/"CALIBRATION.json").read_text())
    unit=json.loads((OUT/"WAVELET_UNIT_TEST.json").read_text())
    assert calibration["status"]==unit["status"]=="PASS"
    lam=float(calibration["lambda_hf"])
    cfg,base=direct_config()
    archived=json.loads((OUT/"CONFIG.json").read_text())
    assert archived["wavelet_hf"]["lambda_hf"]==lam
    for key in ("seed","data","model","diffusion","training","ema","augmentation","evaluation","fast_dev"):
        assert archived[key]==json.loads((REFERENCE/"CONFIG.json").read_text())[key]
    fast=OUT/"fast8k"
    assert not fast.exists()
    (fast/"checkpoints").mkdir(parents=True)
    (OUT/"diagnostics").mkdir(exist_ok=True)
    history=[];frequency=[];manifest=[];records=[]
    baseline=reference_history()
    guard=ContextDataGuard(cfg,"FAST",allow_training=True)
    old=direct_runtime.training_loss
    with guard:
        state=direct_runtime.ScreenState(ARMS[0],"cuda:0",guard)
        init=assert_init(state)
        assert init["direct_initial_model_sha256"]==calibration["initialization_audit"]["direct_initial_model_sha256"]
        assert state.initial_rng_sha256==calibration["initial_rng_sha256"]
        computer=LossComputer(Haar(state.device),lam)
        direct_runtime.training_loss=computer.train_loss
        try:
            for step in range(1,12001):
                result=state.update()
                assert result["step"]==step and state.ema.updates==step
                row={k:result[k] for k in ("step","train_loss","gradient_norm_pre_clip","learning_rate","seconds_per_step","peak_vram_bytes")}
                row.update(computer.last)
                assert abs(row["train_loss"]-row["loss_total"])<1e-7
                records.append(row)
                if step%100==0:
                    jwrite(fast/"progress.json",{
                        "optimizer_step":step,"max_possible_steps":12000,
                        "ema_updates":state.ema.updates,"last":row,
                        "peak_vram_bytes":state.peak_vram,"elapsed_train_seconds":state.train_seconds})
                if step in NODES:
                    metrics,freq=validate_with_frequency(
                        state.ema.model,state.validation_data,state.cfg,state.device,
                        computer,OUT/"diagnostics"/f"step_{step}_fast64_rows.json")
                    assert metrics["slices"]==64 and metrics["patient"]=="L333"
                    assert metrics["weights"]=="ema" and metrics["validation_mode"]=="FAST"
                    val={"step":step,**metrics,"baseline_psnr":baseline[step],
                         "delta_PSNR_vs_2P5D_DIRECT":metrics["psnr"]-baseline[step],
                         **{k:row[k] for k in ("train_loss","loss_res","loss_lh","loss_hl","loss_hh","loss_hf","weighted_hf_loss","loss_total","gradient_norm_pre_clip","learning_rate","peak_vram_bytes","seconds_per_step")}}
                    history.append(val)
                    frequency.append({"step":step,**freq})
                    csv_write(OUT/"validation_history.csv",history)
                    csv_write(OUT/"frequency_metrics.csv",frequency)
                    checkpoint=fast/"checkpoints"/f"step_{step}.pt"
                    assert not checkpoint.exists()
                    payload=state.payload()
                    payload["wavelet_hf"]={"lambda_hf":lam,"haar_source_sha256":sha(ROOT/"wavelet_hf_v1/haar.py"),
                                           "runner_source_sha256":sha(Path(__file__))}
                    torch.save(payload,checkpoint)
                    manifest.append({"step":step,"checkpoint":str(checkpoint),
                                     "sha256":sha(checkpoint),"optimizer_step":state.step,
                                     "ema_updates":state.ema.updates})
                    jwrite(fast/"checkpoint_manifest.json",manifest)
                    print("VALIDATION",step,metrics["psnr"],"delta",val["delta_PSNR_vs_2P5D_DIRECT"],
                          "HF_MSE",freq["mean_HF_MSE"],flush=True)
                if step==8000:
                    six=next(r for r in history if r["step"]==6000)["delta_PSNR_vs_2P5D_DIRECT"]
                    eight=history[-1]["delta_PSNR_vs_2P5D_DIRECT"]
                    if six<=0 and eight<=0:
                        gate="STOP_BOTH_NONPOSITIVE"
                    elif six>0 and eight>=0.02:
                        gate="PROMOTE_TO_12K"
                    else:
                        gate="NO_PROMOTION"
                    jwrite(OUT/"diagnostics/promotion_gate_8k.json",
                           {"delta_6k":six,"delta_8k":eight,"decision":gate})
                    if gate!="PROMOTE_TO_12K":
                        break
        finally:
            direct_runtime.training_loss=old
        access=check_access(guard,True)
    assert state.nonfinite_count==0 and state.step in (8000,12000)
    csv_write(fast/"training_summary_at_validation.csv",
              [{"step":r["step"],**{k:r[k] for k in LOSS_KEYS}}
               for r in records if r["step"] in NODES])
    with open(fast/"train.jsonl","w") as f:
        for row in records:f.write(json.dumps(row)+"\n")
    jwrite(OUT/"DATA_ACCESS_AUDIT.json",access)
    gate=json.loads((OUT/"diagnostics/promotion_gate_8k.json").read_text())["decision"]
    full=None
    if state.step==12000:
        assert gate=="PROMOTE_TO_12K"
        delta12=history[-1]["delta_PSNR_vs_2P5D_DIRECT"]
        if delta12>=0.03:
            full=full_validation(state,computer,cfg)
            gate="PROMOTE_FULL_L333"
        else:
            gate="STOP_12K_DELTA_BELOW_0.03"
    summary={
        "status":"COMPLETE","optimizer_steps":state.step,"ema_updates":state.ema.updates,
        "gate_decision":gate,"lambda_hf":lam,"parameter_count":init["parameter_count"],
        "initialization_audit":init,"calibration_initialization_equal":True,
        "validation_history":history,"frequency_metrics":frequency,
        "full_l333":full,"peak_vram_bytes":state.peak_vram,
        "seconds_per_step":state.train_seconds/state.step,
        "gradient_norm_min":state.gradient_min,"gradient_norm_max":state.gradient_max,
        "nonfinite_count":state.nonfinite_count,"data_access":access,
        "train_stream_sha256":state.stream_digest.hex(),
        "source_sha256":{"formal_rddm":sha(ROOT/"src/rddm_rleb.py"),
                         "haar":sha(ROOT/"wavelet_hf_v1/haar.py"),
                         "runner":sha(Path(__file__))},
        "checkpoint_manifest":manifest,
        "environment":{"python":__import__("sys").version.split()[0],
                       "torch":torch.__version__,"cuda":torch.version.cuda,
                       "gpu":torch.cuda.get_device_name(0)},
    }
    if state.step==12000:
        matched=json.loads((REFERENCE/"TRAIN_STREAM_AUDIT.json").read_text())
        assert summary["train_stream_sha256"]==matched["stream_sha256"]
        summary["matched_direct_train_stream_equal"]=True
    jwrite(OUT/"final_summary.json",summary)
    report(summary)
    print("WAVELET_EXPERIMENT_COMPLETE",state.step,gate,flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--phase",choices=("prepare","train"),required=True)
    args=parser.parse_args()
    if args.phase=="prepare":prepare()
    else:train()

if __name__=="__main__":
    main()

