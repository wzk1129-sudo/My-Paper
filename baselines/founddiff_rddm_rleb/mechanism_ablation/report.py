"""Final paired L333 analysis; within-stack block intervals are not patient-level inference."""
import csv
import json
import math
import os
import platform
import subprocess
from pathlib import Path
import numpy as np
import torch
from fast_dev.protocol import FAST_INDICES
from mayo2016_smallset_v1.runtime import ROOT, sha256, write_json
from .protocol import ARMS, HERE, STEPS, EXPECTED_SOURCE, experiment_config
from .evaluate import full_evaluate
from .train import source_manifest

METRICS = ("psnr","ssim","rmse_hu","mae_hu")

def csv_rows(path, rows):
    assert rows
    with path.open("x", newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        f.flush();os.fsync(f.fileno())

def bootstrap(values):
    values=np.asarray(values,dtype=np.float64)
    assert values.shape==(610,)
    rng=np.random.RandomState(20260924)
    means=np.empty(1000,dtype=np.float64)
    for i in range(1000):
        starts=rng.randint(0,601,size=61)
        idx=np.concatenate([np.arange(s,s+10) for s in starts])[:610]
        means[i]=values[idx].mean()
    return [float(np.percentile(means,2.5)),float(np.percentile(means,97.5))]

def build_report(device="cuda:0"):
    cfg,_=experiment_config()
    assert sha256(ROOT/"src/rddm_rleb.py")==EXPECTED_SOURCE
    summaries={}
    for arm in ARMS:
        folder=HERE/"runs"/arm
        s=json.loads((folder/"final_summary.json").read_text())
        assert s["status"]=="COMPLETE" and s["optimizer_steps"]==12000
        assert [r["step"] for r in s["history"]]==list(STEPS)
        assert s["nonfinite_count"]==0
        assert s["source_manifest"]==source_manifest(),"Source changed during experiment"
        summaries[arm]=s
    assert summaries[ARMS[0]]["initial_hash"]==summaries[ARMS[1]]["initial_hash"]
    assert summaries[ARMS[0]]["train_stream_sha256"]==summaries[ARMS[1]]["train_stream_sha256"]
    stream={"match":True,"control_sha256":summaries[ARMS[0]]["train_stream_sha256"],
            "regression_sha256":summaries[ARMS[1]]["train_stream_sha256"],
            "samples_per_arm":24000,"batch_stream_seed":20260829,
            "fields":["patient_id","slice_index","horizontal_flip","rotation_degrees"]}
    write_json(HERE/"TRAIN_STREAM_AUDIT.json",stream)
    control=summaries[ARMS[0]]["history"]
    regression=summaries[ARMS[1]]["history"]
    curves=[]
    for c,r in zip(control,regression):
        assert c["step"]==r["step"]
        assert c["train_stream_sha256"]==r["train_stream_sha256"]
        record={"step":c["step"]}
        for key in METRICS:
            record["control_"+key]=c[key]
            record["regression_"+key]=r[key]
            record["delta_"+key]=r[key]-c[key]
        for prefix, row in (("control",c),("regression",r)):
            for key in ("validation_loss","train_loss","gradient_norm_pre_clip",
                        "learning_rate","peak_vram_bytes","seconds_per_step","validation_seconds"):
                record[prefix+"_"+key]=row[key]
        curves.append(record)
    csv_rows(HERE/"CONTROL_HISTORY.csv",control)
    csv_rows(HERE/"REGRESSION_HISTORY.csv",regression)
    csv_rows(HERE/"MATCHED_CURVES.csv",curves)
    full=full_evaluate(device)
    a=full[ARMS[0]]["rows"]
    b=full[ARMS[1]]["rows"]
    assert len(a)==len(b)==610
    paired=[]
    for c,r in zip(a,b):
        assert c["patient_id"]==r["patient_id"]=="L333" and c["slice_index"]==r["slice_index"]
        item={"patient_id":"L333","slice_index":c["slice_index"]}
        for key in METRICS:
            item["control_"+key]=c[key]
            item["regression_"+key]=r[key]
            item["delta_"+key]=r[key]-c[key]
        paired.append(item)
    assert [r["slice_index"] for r in paired]==list(range(610))
    csv_rows(HERE/"FULL_L333_PER_SLICE.csv",paired)
    paired_stats={}
    for key in METRICS:
        values=np.array([r["delta_"+key] for r in paired])
        paired_stats[key]={"mean":float(values.mean()),"median":float(np.median(values)),
                           "iqr":[float(np.percentile(values,25)),float(np.percentile(values,75))],
                           "win_fraction":float(np.mean(values>0 if key in ("psnr","ssim") else values<0)),
                           "within_L333_block_bootstrap_95_interval":bootstrap(values)}
    result={"patient":"L333","slices":610,"weights":"ema","checkpoint_step":12000,
            "control":full[ARMS[0]]["summary"],"regression":full[ARMS[1]]["summary"],
            "paired_regression_minus_control":paired_stats,
            "bootstrap":{"block_length_slices":10,"replicates":1000,"seed":20260924,
                         "interpretation":"within one patient stack only; not patient-population significance"}}
    write_json(HERE/"FULL_L333_SUMMARY.json",result)
    delta=result["paired_regression_minus_control"]["psnr"]["mean"]
    last=[r["delta_psnr"] for r in curves[-2:]]
    if abs(delta)<=.10 and all(abs(v)<=.10 for v in last):
        flag="DYNAMIC_DIFFUSION_VALUE_SMALL_OR_TIED"
    elif delta<=-.20 and all(v<0 for v in last):
        flag="DIFFUSION_TRAINING_VALUE_SIGNAL"
    elif delta>=.10 and all(v>0 for v in last):
        flag="DIRECT_REGRESSION_ADVANTAGE_SIGNAL"
    else:
        flag="INCONCLUSIVE_OR_MODEST_EFFECT"
    flags={"flag":flag,"engineering_thresholds_not_statistical_significance":True,
           "full_delta_psnr":delta,"fast_10k_12k_delta_psnr":last}
    write_json(HERE/"ABLATION_FLAGS.json",flags)
    access={arm:{"training":summaries[arm]["data_access"],"full":full[arm]["access"]} for arm in ARMS}
    assert all(part["test_opens"]==part["external_memory_opens"]==part["blocked_attempts"]==0
               for arm in ARMS for part in access[arm].values())
    write_json(HERE/"DATA_ACCESS_AUDIT.json",access)
    provenance={"parent_git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                "git_branch":subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip(),
                "source_sha256":EXPECTED_SOURCE,"source_manifest":source_manifest(),
                "experiment_config_sha256":sha256(HERE/"EXPERIMENT_CONFIG.json"),
                "fast_protocol_sha256":sha256(ROOT/"configs/fast_dev_protocol.json"),
                "fast_indices_sha256":sha256(ROOT/"splits/L333_fast64.txt"),
                "seed":20260829,"python":platform.python_version(),
                "pytorch":torch.__version__,"cuda":torch.version.cuda,
                "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "training_order":list(ARMS)}
    write_json(HERE/"RUN_PROVENANCE.json",provenance)
    manifest={arm:json.loads((HERE/"runs"/arm/"checkpoint_manifest.json").read_text()) for arm in ARMS}
    assert all([r["step"] for r in manifest[arm]["matched"]]==list(STEPS) for arm in ARMS)
    write_json(HERE/"CHECKPOINT_MANIFEST.json",manifest)
    lines=["# RDDM vs matched LDCT-only residual regression V1","",
           "Both arms use the same 13,130,945-parameter UnetRes initialization, 24,000 paired augmented training samples, Adam/EMA settings, and frozen Mayo Fast protocol.",
           "Control retains the unchanged RDDM residual diffusion. Regression uses [LDCT, LDCT], fixed zero time, and direct residual MSE. No pretrained checkpoint was used.","",
           "## Fast64 EMA checkpoints","",
           "| Step | Control PSNR | Regression PSNR | Delta dB | Control SSIM | Regression SSIM | Control RMSE HU | Regression RMSE HU | Control MAE HU | Regression MAE HU |",
           "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in curves:
        lines.append("| %d | %.5f | %.5f | %+.5f | %.6f | %.6f | %.4f | %.4f | %.4f | %.4f |"%
                     (r["step"],r["control_psnr"],r["regression_psnr"],r["delta_psnr"],
                      r["control_ssim"],r["regression_ssim"],r["control_rmse_hu"],
                      r["regression_rmse_hu"],r["control_mae_hu"],r["regression_mae_hu"]))
    lines += ["","## Full L333 (610 slices, step 12000 EMA)",""]
    for arm,label in ((ARMS[0],"RDDM"),(ARMS[1],"Direct regression")):
        s=full[arm]["summary"]; t=summaries[arm]
        lines.append("%s: PSNR %.5f dB, SSIM %.6f, RMSE %.4f HU, MAE %.4f HU; validation %.2f s; training %.1f s (%.4f s/step), peak %.2f GiB."%
                     (label,s["psnr"],s["ssim"],s["rmse_hu"],s["mae_hu"],s["validation_seconds"],
                      t["training_seconds"],t["seconds_per_step"],t["peak_vram_bytes"]/2**30))
    ps=paired_stats["psnr"]
    lines += ["","Paired regression minus control PSNR: mean %+.5f dB, median %+.5f dB, IQR [%+.5f,%+.5f], win fraction %.3f, within-L333 block-bootstrap 95%% interval [%+.5f,%+.5f]."%
              (ps["mean"],ps["median"],*ps["iqr"],ps["win_fraction"],*ps["within_L333_block_bootstrap_95_interval"]),
              "This interval describes stability within one L333 CT stack; it is not patient-population significance.",
              "Engineering flag: **%s**."%flag,
              "Condition Reliance Probe asks what an already trained model uses at forward time. This training ablation asks whether dynamic diffusion training improves final restoration under this matched backbone/data/optimizer.",
              "Conclusions are limited to the FoundDiff-derived RDDM, this Mayo Fast protocol, and this residual objective.",
              "L506 and external memory were never opened; no formal baseline source or checkpoints were changed.",""]
    (HERE/"RDDM_VS_REGRESSION_REPORT.md").write_text("\n".join(lines))
    write_json(HERE/"FINAL_STATUS.json",{"status":"COMPLETE","flag":flag,"train_stream_equal":True,
                                         "l506_opens":0,"external_memory_opens":0,"next_experiment_started":False})
    print(json.dumps({"status":"COMPLETE","flag":flag,"delta_psnr":delta}),flush=True)

if __name__=="__main__":
    build_report()
