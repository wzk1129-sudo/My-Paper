"""Forward-only condition reliance probe for formal RDDM 60k EMA on L333."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy
import skimage
import torch

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
CORE = Path("/home/xujian/projects/CoreDiff")
RUN = CORE/"experiments/founddiff_table_v_20260921/rddm_rleb/mayo2016_smallset_v1/runs/FoundDiff_RDDM_RLEB_Mayo2016_smallset_v1_60k_20260921"
OUT = BASE/"experiments/diagnostics/rddm_condition_reliance_v1"
BOTTLE = BASE/"experiments/diagnostics/rddm_60k_bottleneck_v1"
CKPT = RUN/"checkpoints/step_60000.pt"
SOURCE = BASE/"src/rddm_rleb.py"
CONFIG = HERE/"PROBE_CONFIG.json"
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(HERE))
from mayo2016_smallset_v1.data import CenterPairs
from mayo2016_smallset_v1.frozen_protocol import slice_metrics
from mayo2016_smallset_v1.runtime import check_config, seed_everything
from src.rddm_rleb import build_baseline
from forward import q_state, raw_pred_res

METRICS = ("psnr","ssim","rmse_hu","mae_hu")
FIELDS = ("residual_mse","residual_nmse",*METRICS)


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def state_hash(model):
    h=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def write_json(path,value):
    path=Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")


def write_csv(path,rows):
    path=Path(path)
    if path.exists():
        raise FileExistsError(path)
    assert rows
    with path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def summary(values,worse_threshold=0.0):
    a=np.asarray(values,dtype=float)
    if not len(a) or not np.isfinite(a).all():
        raise ValueError("Nonfinite or empty summary")
    lo,hi=np.percentile(a,[25,75])
    return {"n":int(len(a)),"mean":float(a.mean()),"median":float(np.median(a)),
            "iqr":float(hi-lo),"q25":float(lo),"q75":float(hi),
            "p05":float(np.percentile(a,5)),"p95":float(np.percentile(a,95)),
            "fraction_of_samples_worse":float(np.mean(a>worse_threshold))}


def load_reference():
    rows={}
    with (BOTTLE/"TIMESTEP_PROBE.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            t=int(row["timestep"])
            if t not in (100,499,999):
                continue
            key=(int(row["slice_index"]),t,int(row["noise_replicate"]))
            assert key not in rows
            rows[key]=row
    assert len(rows)==128*3*3
    return rows


def measured(model,xt,condition,feed_t,x_input_original,x_res_original,target,cfg):
    raw=raw_pred_res(model,xt,condition,feed_t)
    assert raw.shape==x_res_original.shape and bool(torch.isfinite(raw).all())
    diff=raw-x_res_original
    mse=float(diff.square().mean())
    nmse=float(diff.square().sum()/(x_res_original.square().sum()+1e-12))
    pred_x0=torch.clamp(x_input_original-raw,-1,1)
    m=slice_metrics((pred_x0+1)/2,target,cfg)
    result={"residual_mse":mse,"residual_nmse":nmse,
            **{k:float(m[k]) for k in METRICS}}
    assert all(math.isfinite(v) for v in result.values())
    return result


def paired(variant,control):
    return {"control_psnr":control["psnr"],"variant_psnr":variant["psnr"],
            "psnr_drop":control["psnr"]-variant["psnr"],
            "control_nmse":control["residual_nmse"],
            "variant_nmse":variant["residual_nmse"],
            "nmse_ratio":variant["residual_nmse"]/(control["residual_nmse"]+1e-12),
            "ssim_drop":control["ssim"]-variant["ssim"],
            "rmse_hu_increase":variant["rmse_hu"]-control["rmse_hu"],
            "mae_hu_increase":variant["mae_hu"]-control["mae_hu"]}


def pair_summary(rows):
    return {"psnr_drop":summary([r["psnr_drop"] for r in rows]),
            "nmse_ratio":summary([r["nmse_ratio"] for r in rows],1.0),
            "ssim_drop":summary([r["ssim_drop"] for r in rows]),
            "rmse_hu_increase":summary([r["rmse_hu_increase"] for r in rows]),
            "mae_hu_increase":summary([r["mae_hu_increase"] for r in rows])}


def main(preflight=False):
    started=time.time()
    cfg_probe=json.loads(CONFIG.read_text())
    assert cfg_probe["patient"]=="L333" and cfg_probe["true_timesteps"]==[100,499,999]
    assert cfg_probe["feed_timesteps"]==[100,499,999]
    assert cfg_probe["noise_replicates"]==[0,1,2]
    assert cfg_probe["donor_position_offset"]==37 and cfg_probe["optimizer_steps"]==0
    cfg=json.loads((RUN/"CONFIG.json").read_text())
    check_config(cfg)
    assert cfg["data"]["validation_patients"]==["L333"]
    assert sha256(SOURCE)==cfg_probe["formal_source_sha256"]
    assert sha256(CKPT)==cfg_probe["formal_checkpoint_sha256"]
    src_hash=sha256(SOURCE)
    ck_hash=sha256(CKPT)
    formal_cfg_hash=sha256(RUN/"CONFIG.json")
    bottleneck_files={str(p.relative_to(BASE)):sha256(p) for p in sorted(BOTTLE.iterdir()) if p.is_file()}
    assert len(bottleneck_files)>=14
    assert json.loads((BOTTLE/"CONTROL.json").read_text())["pass"]
    audit_old=json.loads((BOTTLE/"DATA_ACCESS_AUDIT.json").read_text())
    assert audit_old["L506_opens"]==audit_old["external_memory_opens"]==audit_old["optimizer_steps"]==0
    reference=load_reference()
    indices=[round(float(v)) for v in np.linspace(0,609,128)]
    assert len(indices)==len(set(indices))==128
    assert set(k[0] for k in reference)==set(indices)
    donors=[indices[(position+37)%128] for position in range(128)]
    assert len(set(donors))==128 and all(a!=b for a,b in zip(indices,donors))
    torch.set_num_threads(cfg["training"]["cpu_threads"])
    seed_everything(cfg["seed"])
    assert torch.cuda.is_available()
    device=torch.device("cuda:0")
    ck=torch.load(str(CKPT),map_location="cpu")
    assert ck["optimizer_step"]==ck["ema_updates"]==60000
    assert ck["config"]==cfg and ck["source_hashes"]["src/rddm_rleb.py"]==src_hash
    model=build_baseline().eval().requires_grad_(False)
    model.load_state_dict(ck["ema_state_dict"],strict=True)
    del ck
    model=model.to(device)
    original_state=state_hash(model)
    data=CenterPairs(cfg,"validation")
    assert len(data)==610 and all(r[0]=="L333" for r in data.records)
    if preflight:
        index=indices[0]
        ld,gt,meta=data.load(index)
        assert meta["patient_id"]=="L333" and meta["slice_index"]==index
        xi=torch.from_numpy(ld).unsqueeze(0).to(device)*2-1
        xs=torch.from_numpy(gt).unsqueeze(0).to(device)*2-1
        gt_t=torch.from_numpy(gt).unsqueeze(0).to(device)
        with torch.no_grad():
            xt,xres=q_state(model,xs,xi,100,20260829+index)
            result=measured(model,xt,xi,100,xi,xres,gt_t,cfg)
        prior=reference[(index,100,0)]
        differences={k:result[k]-float(prior[k]) for k in FIELDS}
        assert all(abs(differences[k])<=cfg_probe["control_tolerances"][k] for k in FIELDS)
        assert state_hash(model)==original_state
        print("PREFLIGHT_CONTROL_PASS",json.dumps(differences),flush=True)
        return
    assert OUT.exists() and not any(OUT.iterdir()),"Refuse to overwrite probe output"
    (OUT/"PROBE_INDICES.txt").write_text("\n".join(map(str,indices))+"\n")
    pairs=[]
    for index in indices:
        ld,gt,meta=data.load(index)
        assert meta["patient_id"]=="L333" and meta["slice_index"]==index
        ld_t=torch.from_numpy(ld).unsqueeze(0).to(device)
        gt_t=torch.from_numpy(gt).unsqueeze(0).to(device)
        assert ld_t.shape==gt_t.shape==(1,1,512,512)
        pairs.append((ld_t,gt_t,ld_t*2-1,gt_t*2-1))
    print("CONTROL_START 1152 direct forwards",flush=True)
    control_rows=[]
    control={}
    state_digests={}
    max_deltas={k:0.0 for k in FIELDS}
    with torch.no_grad(),torch.random.fork_rng(devices=[0]):
        for position,index in enumerate(indices):
            ld_t,gt_t,xi,xs=pairs[position]
            donor=donors[position]
            for true_t in (100,499,999):
                for replicate in (0,1,2):
                    seed=20260829+index+replicate*1000000
                    xt,xres=q_state(model,xs,xi,true_t,seed)
                    result=measured(model,xt,xi,true_t,xi,xres,gt_t,cfg)
                    key=(index,true_t,replicate)
                    prior=reference[key]
                    differences={k:abs(result[k]-float(prior[k])) for k in FIELDS}
                    for k,v in differences.items():
                        max_deltas[k]=max(max_deltas[k],v)
                    control[key]=result
                    state_digests[key]=tensor_hash(xt)
                    control_rows.append({"slice_index":index,"donor_slice_index":donor,
                                         "true_t":true_t,"feed_t":true_t,
                                         "replicate":replicate,"condition_type":"CONTROL",**result})
            if (position+1)%16==0:
                print(f"CONTROL_PROGRESS {position+1}/128",flush=True)
    control_pass=(len(control_rows)==1152 and
                  all(max_deltas[k]<=cfg_probe["control_tolerances"][k] for k in FIELDS))
    control_summary={"pass":control_pass,"rows":len(control_rows),
                     "reference":"Bottleneck Diagnostic V1 TIMESTEP_PROBE.csv",
                     "reference_sha256":sha256(BOTTLE/"TIMESTEP_PROBE.csv"),
                     "max_abs_difference":max_deltas,
                     "tolerance":cfg_probe["control_tolerances"],
                     "weights":"ema_state_dict","patient":"L333",
                     "indices":indices,"timesteps":[100,499,999],"replicates":[0,1,2]}
    write_json(OUT/"CONTROL_SUMMARY.json",control_summary)
    if not control_pass:
        print("CONTROL_FAILED stop before ablations",json.dumps(max_deltas),flush=True)
        return 2
    assert state_hash(model)==original_state
    print("CONTROL_PASS",json.dumps(max_deltas),flush=True)
    long_rows=list(control_rows)
    time_pairs=[]
    xt_pairs=[]
    ldct_pairs=[]
    print("ABLATIONS_START 4608 direct forwards",flush=True)
    with torch.no_grad(),torch.random.fork_rng(devices=[0]):
        for position,index in enumerate(indices):
            ld_t,gt_t,xi,xs=pairs[position]
            donor_position=(position+37)%128
            donor_index=indices[donor_position]
            donor_xi=pairs[donor_position][2]
            donor_xs=pairs[donor_position][3]
            assert donor_index!=index
            for true_t in (100,499,999):
                for replicate in (0,1,2):
                    seed=20260829+index+replicate*1000000
                    xt,xres=q_state(model,xs,xi,true_t,seed)
                    key=(index,true_t,replicate)
                    assert tensor_hash(xt)==state_digests[key]
                    base=control[key]
                    for feed_t in (100,499,999):
                        if feed_t==true_t:
                            continue
                        result=measured(model,xt,xi,feed_t,xi,xres,gt_t,cfg)
                        long_rows.append({"slice_index":index,"donor_slice_index":donor_index,
                                          "true_t":true_t,"feed_t":feed_t,
                                          "replicate":replicate,"condition_type":"WRONG_TIME",**result})
                        time_pairs.append({"slice_index":index,"donor_slice_index":donor_index,
                                           "true_t":true_t,"feed_t":feed_t,"replicate":replicate,
                                           **paired(result,base)})
                    donor_seed=20260829+donor_index+replicate*1000000
                    donor_xt,_=q_state(model,donor_xs,donor_xi,true_t,donor_seed)
                    xt_result=measured(model,donor_xt,xi,true_t,xi,xres,gt_t,cfg)
                    long_rows.append({"slice_index":index,"donor_slice_index":donor_index,
                                      "true_t":true_t,"feed_t":true_t,"replicate":replicate,
                                      "condition_type":"SHUFFLE_XT",**xt_result})
                    xt_pairs.append({"slice_index":index,"donor_slice_index":donor_index,
                                     "true_t":true_t,"feed_t":true_t,"replicate":replicate,
                                     **paired(xt_result,base)})
                    ldct_result=measured(model,xt,donor_xi,true_t,xi,xres,gt_t,cfg)
                    long_rows.append({"slice_index":index,"donor_slice_index":donor_index,
                                      "true_t":true_t,"feed_t":true_t,"replicate":replicate,
                                      "condition_type":"SHUFFLE_LDCT",**ldct_result})
                    ldct_pairs.append({"slice_index":index,"donor_slice_index":donor_index,
                                       "true_t":true_t,"feed_t":true_t,"replicate":replicate,
                                       **paired(ldct_result,base)})
                    assert tensor_hash(xt)==state_digests[key]
            if (position+1)%16==0:
                print(f"ABLATION_PROGRESS {position+1}/128",flush=True)
    assert len(long_rows)==5760 and len(time_pairs)==2304
    assert len(xt_pairs)==len(ldct_pairs)==1152
    assert all(math.isfinite(float(r[k])) for r in long_rows for k in FIELDS)
    write_csv(OUT/"RELIANCE_PROBE_LONG.csv",long_rows)
    write_csv(OUT/"XT_RELIANCE.csv",xt_pairs)
    write_csv(OUT/"LDCT_RELIANCE.csv",ldct_pairs)
    matrix=[]
    for true_t in (100,499,999):
        c=[r for r in control_rows if r["true_t"]==true_t]
        for feed_t in (100,499,999):
            if feed_t==true_t:
                matrix.append({"true_t":true_t,"feed_t":feed_t,"condition_type":"CONTROL",
                               "n":len(c),"control_psnr_mean":float(np.mean([r["psnr"] for r in c])),
                               "variant_psnr_mean":float(np.mean([r["psnr"] for r in c])),
                               "variant_psnr_median":float(np.median([r["psnr"] for r in c])),
                               "psnr_drop_mean":0.0,"psnr_drop_median":0.0,
                               "nmse_ratio_mean":1.0,"nmse_ratio_median":1.0,
                               "fraction_psnr_worse":0.0})
            else:
                group=[r for r in time_pairs if r["true_t"]==true_t and r["feed_t"]==feed_t]
                assert len(group)==384
                matrix.append({"true_t":true_t,"feed_t":feed_t,"condition_type":"WRONG_TIME",
                               "n":len(group),"control_psnr_mean":float(np.mean([r["control_psnr"] for r in group])),
                               "variant_psnr_mean":float(np.mean([r["variant_psnr"] for r in group])),
                               "variant_psnr_median":float(np.median([r["variant_psnr"] for r in group])),
                               "psnr_drop_mean":float(np.mean([r["psnr_drop"] for r in group])),
                               "psnr_drop_median":float(np.median([r["psnr_drop"] for r in group])),
                               "nmse_ratio_mean":float(np.mean([r["nmse_ratio"] for r in group])),
                               "nmse_ratio_median":float(np.median([r["nmse_ratio"] for r in group])),
                               "fraction_psnr_worse":float(np.mean([r["psnr_drop"]>0 for r in group]))})
    write_csv(OUT/"TIME_RELIANCE_MATRIX.csv",matrix)
    time_summary={"off_diagonal":pair_summary(time_pairs),
                  "per_true_t":{str(t):pair_summary([r for r in time_pairs if r["true_t"]==t])
                                for t in (100,499,999)},
                  "matrix":matrix,
                  "paired_key":["slice_index","true_t","replicate"],
                  "x_t_and_ldct_unchanged_in_wrong_time":True}
    xt_summary={"overall":pair_summary(xt_pairs),
                "per_true_t":{str(t):pair_summary([r for r in xt_pairs if r["true_t"]==t])
                              for t in (100,499,999)},
                "donor_mapping":"(probe_position + 37) % 128",
                "donor_q_state":"same true_t and replicate, donor's own fixed seed"}
    ldct_summary={"overall":pair_summary(ldct_pairs),
                  "per_true_t":{str(t):pair_summary([r for r in ldct_pairs if r["true_t"]==t])
                                for t in (100,499,999)},
                  "reconstruction_uses_original_ldct":True,
                  "donor_mapping":"(probe_position + 37) % 128"}
    write_json(OUT/"TIME_RELIANCE_SUMMARY.json",time_summary)
    write_json(OUT/"XT_RELIANCE_SUMMARY.json",xt_summary)
    write_json(OUT/"LDCT_RELIANCE_SUMMARY.json",ldct_summary)
    td=time_summary["off_diagonal"]["psnr_drop"]["median"]
    tr=time_summary["off_diagonal"]["nmse_ratio"]["median"]
    xd=xt_summary["overall"]["psnr_drop"]["median"]
    xr=xt_summary["overall"]["nmse_ratio"]["median"]
    ld=ldct_summary["overall"]["psnr_drop"]["median"]
    lr=ldct_summary["overall"]["nmse_ratio"]["median"]
    ratio=ld/max(xd,.01)
    flags=[]
    if td<=.05 and tr<=1.05:flags.append("TIME_CONDITION_WEAK_SIGNAL")
    if xd<=.10 and xr<=1.10:flags.append("XT_SAMPLE_SPECIFIC_WEAK_SIGNAL")
    if ld>=1.0 and ld>=5*max(xd,.05):flags.append("LDCT_CONDITION_DOMINANT_SIGNAL")
    if xd>=.50 or xr>=1.25:flags.append("DIFFUSION_STATE_MATTERS_SIGNAL")
    assert not ("XT_SAMPLE_SPECIFIC_WEAK_SIGNAL" in flags and
                "DIFFUSION_STATE_MATTERS_SIGNAL" in flags),"Conflicting x_t flags"
    if all(x in flags for x in ("TIME_CONDITION_WEAK_SIGNAL",
                               "XT_SAMPLE_SPECIFIC_WEAK_SIGNAL",
                               "LDCT_CONDITION_DOMINANT_SIGNAL")):
        regression="supported"
    elif "DIFFUSION_STATE_MATTERS_SIGNAL" in flags:
        regression="not supported"
    else:
        regression="inconclusive"
    flag_result={"flags":flags,"engineering_heuristics_only":True,
                 "condition_to_xt_drop_ratio":ratio,
                 "median_psnr_drop_time":td,"median_nmse_ratio_time":tr,
                 "median_psnr_drop_xt":xd,"median_nmse_ratio_xt":xr,
                 "median_psnr_drop_ldct":ld,"median_nmse_ratio_ldct":lr,
                 "ldct_conditioned_residual_regression":regression}
    write_json(OUT/"RELIANCE_FLAGS.json",flag_result)
    lines=["# RDDM Condition Reliance Probe V1","",
           "Forward-only, EMA step_60000.pt, L333 fixed 128 indices; no DDIM loop or optimizer update.",
           "Control path matches all 1152 archived Bottleneck V1 direct-forward rows within configured tolerances.",
           "Reconstruction always uses the original LDCT slice; donor data only replaces one model-input channel.","",
           "## 3×3 time matrix","",
           "| true t | feed t | N | median PSNR drop (dB) | median NMSE ratio | fraction PSNR worse |",
           "|---:|---:|---:|---:|---:|---:|"]
    for r in matrix:
        lines.append(f"| {r['true_t']} | {r['feed_t']} | {r['n']} | "
                     f"{r['psnr_drop_median']:.6f} | {r['nmse_ratio_median']:.6f} | "
                     f"{r['fraction_psnr_worse']:.3f} |")
    lines += ["","## Paired dependence","",
              "| Intervention | N | Median PSNR drop | Mean | IQR | P05–P95 | Fraction worse | Median NMSE ratio |",
              "|---|---:|---:|---:|---:|---|---:|---:|"]
    for label,group in (("Wrong time (all off-diagonal)",time_summary["off_diagonal"]),
                        ("Shuffle x_t",xt_summary["overall"]),
                        ("Shuffle LDCT condition",ldct_summary["overall"])):
        a,b=group["psnr_drop"],group["nmse_ratio"]
        lines.append(f"| {label} | {a['n']} | {a['median']:.6f} | {a['mean']:.6f} | "
                     f"{a['iqr']:.6f} | {a['p05']:.6f} to {a['p95']:.6f} | "
                     f"{a['fraction_of_samples_worse']:.3f} | {b['median']:.6f} |")
    lines += ["",f"condition_to_xt_drop_ratio = {ratio:.6f}.","",
              "## Questions answered","",
              f"1. Wrong time embedding: median paired PSNR drop {td:.6f} dB; median NMSE ratio {tr:.6f}.",
              "2. True-t sensitivity: "+", ".join(
                  f"t={t}: {time_summary['per_true_t'][str(t)]['psnr_drop']['median']:.6f} dB"
                  for t in (100,499,999))+". Compare all six off-diagonal cells above.",
              f"3. Shuffled legal donor x_t: median PSNR drop {xd:.6f} dB, NMSE ratio {xr:.6f}.",
              f"4. Shuffled donor LDCT condition: median PSNR drop {ld:.6f} dB, NMSE ratio {lr:.6f}.",
              "5. Relative reliance: "+("LDCT condition has the larger median paired PSNR effect."
                  if ld>xd else "Sample-specific x_t has the larger median paired PSNR effect."),
              "6. Time embedding contribution: "+("small by preregistered engineering thresholds."
                  if "TIME_CONDITION_WEAK_SIGNAL" in flags else "not small by preregistered engineering thresholds."),
              f"7. LDCT-conditioned residual-regression hypothesis: **{regression}** under this probe's "
              "pre-registered heuristic; this is not a mathematical proof.","",
              f"Engineering flags: {flags if flags else ['none']}.",
              "All comparisons pair the same slice, true_t and replicate. The 128 slices come from one L333 patient; "
              "no patient-population statistical significance is claimed.",
              "No training, model parameter modification, new sampler optimization, L506 access, external memory access, "
              "or Git push of this branch.",""]
    (OUT/"CONDITION_RELIANCE_REPORT.md").write_text("\n".join(lines))
    assert state_hash(model)==original_state
    assert sha256(SOURCE)==src_hash==cfg_probe["formal_source_sha256"]
    assert sha256(CKPT)==ck_hash==cfg_probe["formal_checkpoint_sha256"]
    assert sha256(RUN/"CONFIG.json")==formal_cfg_hash
    assert {str(p.relative_to(BASE)):sha256(p) for p in sorted(BOTTLE.iterdir()) if p.is_file()}==bottleneck_files
    access={"validation_patient":"L333","L333_indices_opened":indices,
            "L333_unique_pair_loads":128,"L333_pair_files_loaded":256,
            "donor_sources":"fixed L333 probe indices only",
            "training_patients_opened":[],"L506_opens":0,"external_memory_opens":0,
            "blocked_attempts":0,"directory_enumeration_of_dataset":False,
            "optimizer_updates":0,"sampling_loop_calls":0,
            "model_parameters_unchanged":True,"model_state_sha256_before_after":original_state,
            "formal_source_sha256_before_after":src_hash,
            "formal_checkpoint_sha256_before_after":ck_hash,
            "formal_config_sha256_before_after":formal_cfg_hash,
            "bottleneck_files_sha256_before_after":bottleneck_files}
    write_json(OUT/"DATA_ACCESS_AUDIT.json",access)
    provenance={"git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=BASE,text=True).strip(),
                "git_branch":subprocess.check_output(["git","branch","--show-current"],cwd=BASE,text=True).strip(),
                "formal_source_sha256":src_hash,"formal_checkpoint_sha256":ck_hash,
                "formal_config_sha256":formal_cfg_hash,
                "bottleneck_reference_sha256":sha256(BOTTLE/"TIMESTEP_PROBE.csv"),
                "probe_source_sha256":{p.name:sha256(p) for p in sorted(HERE.glob("*.py"))},
                "probe_config_sha256":sha256(CONFIG),
                "python":sys.version,"torch":torch.__version__,"cuda":torch.version.cuda,
                "numpy":np.__version__,"scipy":scipy.__version__,"skimage":skimage.__version__,
                "gpu":torch.cuda.get_device_name(0),
                "seed_base":20260829,"replicate_stride":1000000,
                "started_unix":started,"ended_unix":time.time(),
                "seconds":time.time()-started,
                "weights":"ema_state_dict","optimizer_updates":0}
    write_json(OUT/"RUN_PROVENANCE.json",provenance)
    print("RELIANCE_COMPLETE",json.dumps({"flags":flags,"ratio":ratio,
                                      "seconds":provenance["seconds"]}),flush=True)
    return 0


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--preflight",action="store_true")
    args=parser.parse_args()
    with torch.no_grad():
        sys.exit(main(args.preflight) or 0)
