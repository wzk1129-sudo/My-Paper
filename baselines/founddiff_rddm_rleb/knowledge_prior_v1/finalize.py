"""Metrics-only final audit; never loads image arrays or checkpoint tensors."""
import csv,hashlib,json,math,statistics
from pathlib import Path
from context_2p5d.protocol import config as context_config
from fast_dev.protocol import FAST_INDICES
from mayo2016_smallset_v1.runtime import ROOT,sha256,write_json
from .engine import HERE,ARMS,NODES,read_json,csv_write,paired
from .stream import digest

def require(condition,message):
    if not condition:raise AssertionError(message)
def finite_rows(path):
    with open(path,newline="") as f: rows=list(csv.DictReader(f))
    require(bool(rows),str(path)+" empty")
    for row in rows:
        for key,value in row.items():
            if value not in ("",None):
                try: number=float(value)
                except ValueError:continue
                require(math.isfinite(number),str(path)+" "+key+" nonfinite")
    return rows
def access_check(name,access,expected_validation=True):
    for key in ("L506_opens","external_memory_opens","neighbor_target_opens","blocked_attempts"):
        require(access[key]==0,name+" "+key)
    require(access["training_patients_opened"]==["L067","L143"],name+" train patients")
    if expected_validation:require(access["validation_center_indices_opened"]==list(FAST_INDICES),name+" validation indices")
def per_slice(arm,summary):
    rows=[]
    for node in summary["history"]:
        step=node["step"]
        ours=read_json(HERE/arm/f"fast64_step_{step}.json")
        ctrl=read_json(HERE/ARMS[0]/f"fast64_step_{step}.json")
        require(len(ours)==len(ctrl)==64,arm+" per slice count")
        for x,y in zip(ours,ctrl):
            require(x["slice_index"]==y["slice_index"] and x["patient_id"]==y["patient_id"]=="L333",arm+" slice match")
            row={"step":step,"slice_index":x["slice_index"]}
            for key in ("psnr","ssim","rmse_hu","mae_hu"):
                row["method_"+key]=x[key]
                row["control_"+key]=y[key]
                row["delta_"+key]=x[key]-y[key]
            rows.append(row)
    path=HERE/("A_VS_CONTROL_PER_SLICE.csv" if arm==ARMS[1] else "B_VS_CONTROL_PER_SLICE.csv")
    require(not path.exists(),str(path)+" exists")
    csv_write(path,rows)
    return len(rows)
def main():
    cfg,base=context_config()
    original=read_json(HERE/"QUEUE_STATUS.json")
    retry=read_json(HERE/"QUEUE_STATUS_RETRY.json")
    require(original["status"]=="FAILED" and original["phase"]==ARMS[1],"original failure provenance")
    require("unsupported ScalarType BFloat16" in original["error"],"failure cause")
    require(retry["status"]==retry["phase"]=="COMPLETE","retry status")
    require((HERE/"QUEUE.exit").read_text().strip()=="1" and (HERE/"QUEUE_RETRY.exit").read_text().strip()=="0","exit codes")
    require(read_json(HERE/"STEP0_PARITY.json")["pass"],"step0 parity")
    require(read_json(HERE/"SMOKE_AUDIT.json")["status"]=="PASS","smoke")
    require(read_json(HERE/"DATA_SMOKE_AUDIT.json")["status"]=="PASS","data smoke")
    require(read_json(HERE/"A_INIT_RETRY_AUDIT.json")["status"]=="PASS","A init retry")
    manifest=read_json(HERE/"TRAIN_STREAM_MANIFEST.json")
    stream_sha=(HERE/"TRAIN_STREAM_SHA256.txt").read_text().strip()
    require(len(manifest)==6000 and digest(manifest)==stream_sha,"manifest SHA")
    require(sha256(ROOT/"src/rddm_rleb.py")=="e8ed93223fbdf4a0e2521f73c7a4f213561f6a191a1a7b2a5f95934b85ebf574","formal source changed")
    require(sha256(ROOT/"splits/L333_fast64.txt")=="0d3170fd2462c7b19d1cc0f187b12e03bac2c9b15babca7c1d881568adde6895","Fast64 split changed")
    env=read_json(HERE/"ENVIRONMENT_AUDIT.json")
    require(env["direct_checkpoint_sha256"]=="151b7e58d2ca0d936be0842434e3d17809b325f144d0d48e47c8565287bcdcdf","Direct SHA")
    require(env["meddino_checkpoint_sha256"]=="492b2b96c00e9d3b93ba3b07a8d387aa2196b5e04698378fcc23049b447966d2","Med SHA")
    require(env["meddino_official_commit"]=="a2460c2547899283446e520b84523454f7e25608","Med commit")
    summaries={arm:read_json(HERE/arm/"final_summary.json") for arm in ARMS}
    require([r["step"] for r in summaries[ARMS[0]]["history"]]==list(NODES),"control nodes")
    require([r["step"] for r in summaries[ARMS[1]]["history"]]==list(NODES),"A nodes")
    require([r["step"] for r in summaries[ARMS[2]]["history"]]==[0,1000,2000,4000],"B stop nodes")
    for arm,summary in summaries.items():
        require(summary["status"]=="COMPLETE" and summary["nonfinite_count"]==0,arm+" incomplete")
        require(summary["manifest_sha256"]==stream_sha,arm+" stream identity")
        access=read_json(HERE/arm/"DATA_ACCESS_AUDIT.json")
        access_check(arm,access)
        require(summary["access"]==access,arm+" access mismatch")
        init=read_json(HERE/arm/"INITIALIZATION_AUDIT.json")
        require(init["starting_ema_equal_model"] and init["fresh_adam_state_count"]==0,arm+" init")
        require(init["starting_direct_ema_sha256"]==env["direct_checkpoint_sha256"],arm+" checkpoint")
        for row in summary["history"]:
            require(row["weights"]=="ema" and row["slices"]==64,arm+" validation protocol")
            require(row["train_stream_sha256"]==digest(manifest[:row["step"]]),arm+" prefix hash")
            for key in ("psnr","ssim","rmse_hu","mae_hu","validation_loss"):
                require(math.isfinite(row[key]),arm+" "+key)
        finite_rows(HERE/arm/"validation_history.csv")
        finite_rows(HERE/arm/"train_history.csv")
    hashes=[read_json(HERE/arm/"INITIALIZATION_AUDIT.json")["starting_main_hash"] for arm in ARMS]
    require(len(set(hashes))==1,"different starting RDDM")
    require(read_json(HERE/ARMS[1]/"INITIALIZATION_AUDIT.json")["meddino_frozen"],"MedDINO not frozen")
    teacher=read_json(HERE/"B_TEACHER_PRETRAIN/TEACHER_AUDIT.json")
    require(teacher["status"]=="PASS" and teacher["steps"]==8000 and teacher["nonfinite_count"]==0,"teacher")
    ta=teacher["access"]
    require(ta["donor_patients_opened"]==["L096","L109","L192","L286","L291","L310"],"donor split")
    require(ta["donor_ldct_opens"]==ta["L333_opens"]==ta["L506_opens"]==ta["blocked_attempts"]==0,"donor access")
    finite_rows(HERE/"B_TEACHER_PRETRAIN/TEACHER_TRAIN_HISTORY.csv")
    calibration=read_json(HERE/"diagnostics/B_LAMBDA_CALIBRATION.json")
    require(calibration["batches"]==100 and calibration["optimizer_steps"]==0,"B calibration")
    require(1e-4<=calibration["lambda_prior"]<=10,"lambda range")
    ca=calibration["access"]
    require(ca["L506_opens"]==ca["external_memory_opens"]==ca["neighbor_target_opens"]==ca["blocked_attempts"]==0,"calibration access")
    final=read_json(HERE/"FINAL_SUMMARY.json")
    require(final["status"]=="COMPLETE" and final["stream_sha256"]==stream_sha,"final")
    require(not final["gates"]["A"]["promote"] and not final["gates"]["B"]["promote"],"unexpected promotion")
    require(final["gates"]["A"]["reason"]=="FAST_ONLY" and final["gates"]["B"]["reason"]=="STOP_AT_4K","gate reason")
    require(not (HERE/"full_l333").exists(),"unauthorized Full L333")
    comparison=finite_rows(HERE/"FINAL_COMPARISON.csv")
    require(len(comparison)==14,"comparison row count")
    counts={arm:per_slice(arm,summaries[arm]) for arm in ARMS[1:]}
    require(counts[ARMS[1]]==320 and counts[ARMS[2]]==256,"per slice count")
    btrain=finite_rows(HERE/ARMS[2]/"train_history.csv")
    prior=[float(r["loss_prior"]) for r in btrain]
    early=float(statistics.median(prior[:10]));late=float(statistics.median(prior[-10:]))
    audit={"status":"PASS","original_failure_preserved":True,"retry_complete":True,
       "control_steps":6000,"A_steps":6000,"teacher_steps":8000,"B_steps":4000,
       "matched_starting_main_hash":hashes[0],"manifest_sha256":stream_sha,
       "formal_source_sha256":sha256(ROOT/"src/rddm_rleb.py"),"fast64_split_sha256":sha256(ROOT/"splits/L333_fast64.txt"),
       "MedDINO_frozen":True,"teacher_NDCT_only_donors":list(ta["donor_patients_opened"]),
       "A_full_promotion":False,"B_full_promotion":False,"Full_L333_run":False,
       "L506_opens":0,"unauthorized_external_memory_opens":0,"neighbor_target_opens":0,"blocked_attempts":0,
       "nonfinite_counts":{"control":0,"A":0,"teacher":0,"B":0},
       "B_prior_loss_median_first10_logged":early,"B_prior_loss_median_last10_logged":late,
       "per_slice_row_counts":counts}
    path=HERE/"FINAL_AUDIT.json";require(not path.exists(),"FINAL_AUDIT already exists")
    write_json(path,audit)
    report=HERE/"REPORT.md"
    marker="## Post-run integrity audit"
    text=report.read_text()
    require(marker not in text,"report already finalized")
    a=summaries[ARMS[1]]["history"][-1];c=summaries[ARMS[0]]["history"][-1]
    b=summaries[ARMS[2]]["history"][-1]
    add=f"""
{marker}

PASS. First queue failed before A optimizer step in a BF16-only state-hash audit; the dtype-safe retry preserved Control 6k and completed the serial phases. The three RDDM arms share the same Direct 12k EMA main state and manifest SHA256 `{stream_sha}`.

Control continuation: {c['psnr']-summaries[ARMS[0]]['history'][0]['psnr']:+.6f} dB. A at 6k: ΔPSNR {a['delta_vs_control']:+.6f} dB, positive slice fraction {a['positive_slice_fraction']:.4f}; below the +0.03 dB Full-L333 gate. B stopped at 4k: ΔPSNR {b['delta_vs_control']:+.6f} dB. B prior loss median across first/last ten logged points: {early:.6f} → {late:.6f}; this is a training diagnostic, not a validation gain. Teacher trained 8k on disjoint donor NDCT only. No arm promoted to Full-L333. L506, unauthorized external-memory, neighbor-target and blocked opens were all zero. No A+B experiment was run.
"""
    report.write_text(text+add)
    print(json.dumps({"status":"PASS","control_psnr":c["psnr"],"A_delta":a["delta_vs_control"],"B_delta_4k":b["delta_vs_control"],"per_slice":counts},separators=(",",":")))
if __name__=="__main__":main()
