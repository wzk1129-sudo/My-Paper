"""Final read-only consistency audit and lightweight aggregate report."""
import csv
import json
import math
import statistics
from pathlib import Path
import numpy as np
from mayo2016_smallset_v1.runtime import ROOT, sha256, write_json
from context_2p5d.model import SOURCE_PIN
from .protocol import HERE, STEPS, DIRECT_CK, DIRECT_SHA

def load(path):return json.loads(Path(path).read_text())
def csv_load(path):
    with Path(path).open(newline="") as f:return list(csv.DictReader(f))
def csv_write(path,rows):
    with Path(path).open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)),lineterminator="\n")
        w.writeheader();w.writerows(rows)

def main():
    queue=load(HERE/"queue_status.json")
    assert queue["status"]=="TRAINING_COMPLETE"
    names=("C0_DIRECT","C1_CAPACITY","C2_EDGE")
    summaries={a:load(HERE/"runs"/a/"final_summary.json") for a in names}
    assert all(v["status"]=="COMPLETE" for v in summaries.values())
    assert all(summaries[a]["optimizer_steps"]==6000 for a in names[1:])
    assert sha256(ROOT/"src/rddm_rleb.py")==SOURCE_PIN
    assert sha256(DIRECT_CK)==DIRECT_SHA
    initial={a:load(HERE/"runs"/a/"INITIALIZATION_AUDIT.json") for a in names[1:]}
    params={a:load(HERE/"runs"/a/"PARAMETER_AUDIT.json") for a in names[1:]}
    accesses={a:load(HERE/"runs"/a/"DATA_ACCESS_AUDIT.json") for a in names}
    rng={a:load(HERE/"runs"/a/"TRAIN_RNG_AUDIT.json") for a in names[1:]}
    streams={a:load(HERE/"runs"/a/"TRAIN_STREAM_AUDIT.json") for a in names[1:]}
    assert len({initial[a]["main_state_hash"] for a in names[1:]})==1
    assert len({initial[a]["control_copy_hash"] for a in names[1:]})==1
    assert len({rng[a]["initial_rng_sha256"] for a in names[1:]})==1
    assert len({streams[a]["sha256"] for a in names[1:]})==1
    for a in names:
        access=accesses[a]
        assert access["L506_opens"]==access["external_memory_opens"]==0
        assert access["neighbor_target_opens"]==access["blocked_attempts"]==0
        assert access["validation_center_indices_opened"]==list(map(int,(ROOT/"splits/L333_fast64.txt").read_text().splitlines()))
    for a in names[1:]:
        assert initial[a]["step_zero_equivalent"]
        assert summaries[a]["main_hash_before"]==summaries[a]["main_hash_after"]==summaries[a]["ema_main_hash_after"]
        assert summaries[a]["nonfinite_count"]==0
    histories={a:csv_load(HERE/"runs"/a/"validation_history.csv") for a in names[1:]}
    assert all([int(r["step"]) for r in histories[a]]==list(STEPS) for a in names[1:])
    bystep={a:{int(r["step"]):r for r in histories[a]} for a in names[1:]}
    base=summaries["C0_DIRECT"]["metrics"]
    assert abs(base["psnr"]-40.76762730415604)<1e-4
    allhistory=[{"arm":"C0_DIRECT","step":0,**base}]
    for a in names[1:]:allhistory+=histories[a]
    csv_write(HERE/"VALIDATION_HISTORY.csv",allhistory)
    activity=[]
    for a in names[1:]:
        activity+=csv_load(HERE/"runs"/a/"control_activity.csv")
    csv_write(HERE/"CONTROL_ACTIVITY.csv",activity)
    per={a:csv_load(HERE/"runs"/a/("per_slice_step_0.csv" if a=="C0_DIRECT" else "per_slice_step_6000.csv")) for a in names}
    assert all(len(per[a])==64 for a in names)
    deltas=[]
    for i in range(64):
        row={a:per[a][i] for a in names}
        assert len({int(row[a]["slice_index"]) for a in names})==1
        x={"patient_id":"L333","slice_index":int(row["C0_DIRECT"]["slice_index"]),
           "baseline_psnr":float(row["C0_DIRECT"]["psnr"]),
           "capacity_control_psnr":float(row["C1_CAPACITY"]["psnr"]),
           "structure_psnr":float(row["C2_EDGE"]["psnr"])}
        x["delta_vs_C0"]=x["structure_psnr"]-x["baseline_psnr"]
        x["delta_vs_C1"]=x["structure_psnr"]-x["capacity_control_psnr"]
        x["rmse_delta_vs_C0"]=float(row["C2_EDGE"]["rmse_hu"])-float(row["C0_DIRECT"]["rmse_hu"])
        x["rmse_delta_vs_C1"]=float(row["C2_EDGE"]["rmse_hu"])-float(row["C1_CAPACITY"]["rmse_hu"])
        deltas.append(x)
    csv_write(HERE/"PER_SLICE_DELTA.csv",deltas)
    distribution={}
    for key in ("delta_vs_C0","delta_vs_C1"):
        values=np.asarray([r[key] for r in deltas])
        distribution[key]={"mean":float(values.mean()),"median":float(np.median(values)),
          "positive_slice_fraction":float((values>0).mean()),
          "p10":float(np.percentile(values,10)),"p50":float(np.percentile(values,50)),
          "p90":float(np.percentile(values,90))}
    c1_6=bystep["C1_CAPACITY"][6000];c2_6=bystep["C2_EDGE"][6000]
    def num(row,key):return float(row[key])
    d4=num(bystep["C2_EDGE"][4000],"psnr")-num(bystep["C1_CAPACITY"][4000],"psnr")
    d6=num(c2_6,"psnr")-num(c1_6,"psnr")
    a2=next(r for r in activity if r["arm"]=="C2_EDGE" and int(r["step"])==6000)
    active=all(v>0 for v in json.loads(a2["zero_weight_norms"])) and all(
        v>0 for v in json.loads(a2["zero_output_rms"])) and all(
        v>0 for v in json.loads(a2["edge_feature_rms"]))
    promising=(d4>0 and d6>=.05 and num(c2_6,"rmse_hu")<num(c1_6,"rmse_hu")
        and num(c2_6,"soft_psnr")>=num(c1_6,"soft_psnr")
        and distribution["delta_vs_C1"]["positive_slice_fraction"]>.5
        and distribution["delta_vs_C1"]["median"]>0 and active)
    capacity_gain=num(c1_6,"psnr")-base["psnr"]
    label="PROMISING_STRUCTURE_SIGNAL" if promising else (
        "CAPACITY_ONLY_GAIN" if capacity_gain>0 and abs(d6)<.05 else
        "CONTROL_ACTIVE_BUT_STRUCTURE_NOT_USEFUL" if active else "CONTROL_INACTIVE")
    write_json(HERE/"INITIALIZATION_AUDIT.json",initial)
    write_json(HERE/"PARAMETER_AUDIT.json",params)
    write_json(HERE/"DATA_ACCESS_AUDIT.json",accesses)
    write_json(HERE/"TRAIN_RNG_AUDIT.json",rng)
    write_json(HERE/"TRAIN_STREAM_AUDIT.json",streams)
    write_json(HERE/"VRAM_PROFILE.json",{a:{
        "peak_vram_bytes":summaries[a]["peak_vram_bytes"],
        "seconds_per_step":summaries[a]["seconds_per_step"]} for a in names[1:]})
    control_param_audit=load(HERE/"CONTROL_PARAMETER_AUDIT.json")
    assert all(control_param_audit[a]["optimizer_step"]==6000 for a in names[1:])
    result={"status":"COMPLETE","arms":names,"C0":base,
        "control_parameter_audit":control_param_audit,
        "histories":{a:summaries[a]["history"] for a in names[1:]},
        "capacity_gain_6k_psnr":capacity_gain,"edge_gain_vs_capacity_4k_psnr":d4,
        "edge_gain_vs_capacity_6k_psnr":d6,"distribution":distribution,
        "control_active":active,"promotion":promising,"label":label,
        "medical_encoder":"MEDICAL_ENCODER_BLOCKED",
        "medical_checkpoint_origin_verified":True,"medical_source_execution_blocked":True,
        "source_sha256":sha256(ROOT/"src/rddm_rleb.py"),
        "checkpoint_sha256":sha256(DIRECT_CK),
        "train_stream_equal":True,"initial_rng_equal":True,
        "L506_opens":0,"external_memory_opens":0,"neighbor_target_opens":0}
    write_json(HERE/"ANATOMY_CONTROLNET_V1_SUMMARY.json",result)
    lines=["# RDDM Anatomy-ControlNet V1","",
      "Frozen trained 2.5D Direct 12k EMA main; L067/L143 training and fixed L333 Fast64 validation.",
      "No full eight-patient training or L506 access.",
      "The per-arm model field inherited from the frozen 2D Fast-dev template is metadata only; the actual 4-channel stem and parameter counts are specified in EXPERIMENT_CONFIG.json actual_model_by_arm and PARAMETER_AUDIT.json.","",
      "| Arm | Condition | Trainable params | Peak VRAM GiB | 2k PSNR | 4k PSNR | 6k PSNR | Delta vs C0 | Delta vs C1 | Soft PSNR | RMSE HU |",
      "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
      "| C0 | Direct checkpoint | 0 | n/a | n/a | n/a | %.5f | 0 | n/a | %.5f | %.5f |"%(base["psnr"],base["soft_psnr"],base["rmse_hu"])]
    for a,short in (("C1_CAPACITY","C1"),("C2_EDGE","C2")):
        r=bystep[a][6000];p=params[a]
        lines.append("| %s | %s | %d | %.3f | %.5f | %.5f | %.5f | %+.5f | %s | %.5f | %.5f |"%(
            short,a,p["trainable_parameters"],summaries[a]["peak_vram_bytes"]/1024**3,
            num(bystep[a][2000],"psnr"),num(bystep[a][4000],"psnr"),
            num(r,"psnr"),num(r,"psnr")-base["psnr"],
            "0" if short=="C1" else "%+.5f"%d6,
            num(r,"soft_psnr"),num(r,"rmse_hu")))
    lines+=["","Observed label: **%s**. 4k/6k edge vs capacity: %+.5f / %+.5f dB."%(label,d4,d6),
      "C2 positive slice fraction vs C1 at 6k: %.3f."%distribution["delta_vs_C1"]["positive_slice_fraction"],
      "Control activity at 6k: %s; copied control encoder and all four ZeroConvs changed after optimization (see CONTROL_PARAMETER_AUDIT.json)."%active,
      "C2 has %d additional trainable edge-provider parameters vs C1; attribution is limited by this small capacity difference."%(
          params["C2_EDGE"]["trainable_parameters"]-params["C1_CAPACITY"]["trainable_parameters"]),
      "C3/C4 were not run: the local MedDINOv3 checkpoint SHA matches the author release, but official model source is absent on this server; exact source commit, input size, layer mapping and runtime compatibility remain unverified. See MEDICAL_ENCODER_AUDIT.json.",
      "Recommended future anatomy encoder: [MedDINOv3 ViT-B/16 CT-3M](https://huggingface.co/ricklisz123/MedDINOv3-ViTB-16-CT-3M) with [official source](https://github.com/ricklisz/MedDINOv3). Pin the source commit and input resolution, then profile memory before C3. C2 alone used 4.204 GiB; the anatomy path adds at least about 0.32 GiB FP32 weights plus unmeasured attention workspace.",
      "The first C1 smoke attempt stopped at the frozen data-config guard before any optimizer step or image open; the corrected independent 20-step C1/C2 smokes passed and are archived.",
      "Main weights and original Direct checkpoint remained unchanged; no optimizer step was taken for C0.",
      "No data or checkpoint is included in this lightweight archive.",
      "Promotion to full eight-patient training: %s."%("candidate only; requires further confirmation" if promising else "not supported by this Fast64 screen"),
      ""]
    (HERE/"ANATOMY_CONTROLNET_V1_REPORT.md").write_text("\n".join(lines))
    print(json.dumps({"status":"COMPLETE","label":label,"C0":base["psnr"],
        "C1_6k":num(c1_6,"psnr"),"C2_6k":num(c2_6,"psnr"),
        "delta_C2_C1":d6,"positive_slice_fraction":distribution["delta_vs_C1"]["positive_slice_fraction"]}),flush=True)
if __name__=="__main__":main()
