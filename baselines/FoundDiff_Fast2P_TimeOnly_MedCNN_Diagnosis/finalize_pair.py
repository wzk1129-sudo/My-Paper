"""Validate each completed run, and compare both matched Fast-Dev runs."""
import sys,json,csv,hashlib,math
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parent
NAMES=["Fast2P_TimeOnly_Control_8k","Fast2P_TimeOnly_MedCNN_8k"]
KEYS=("psnr","ssim","rmse_hu","mae_hu","soft_psnr","soft_ssim","soft_rmse_hu")
def readcsv(p):
    with p.open() as f:return list(csv.DictReader(f))
def audit(name):
    o=ROOT/"experiments"/name;done=json.loads((o/"COMPLETE.json").read_text());cfg=json.loads((o/"config.json").read_text())
    assert done["step"]==8000 and [r["step"] for r in done["history"]]==[2000,4000,6000,8000]
    for n,h in done["source_hashes"].items():assert hashlib.sha256((o/n).read_bytes()).hexdigest()==h
    valid=list(map(int,(o/"L333_fast64.txt").read_text().splitlines()))
    for row in done["history"]:
        st=row["step"];rr=readcsv(o/f"validation_step_{st}.csv")
        assert len(rr)==64 and [int(x["slice_index"]) for x in rr]==valid and all(x["patient_id"]=="L333" for x in rr)
        for k in KEYS:
            vals=[float(x[k]) for x in rr];assert all(math.isfinite(x) for x in vals) and abs(np.mean(vals)-row[k])<1e-10
        ck=torch.load(o/f"checkpoints/step_{st}.pt",map_location="cpu",weights_only=False)
        assert ck["step"]==ck["ema_updates"]==st and ck["config"]==cfg and ck["source_hashes"]==done["source_hashes"]
        assert all(torch.isfinite(v).all() for v in ck["ema_state_dict"].values());del ck
    rows=[json.loads(x) for x in (o/"train.jsonl").read_text().splitlines()]
    assert [x["step"] for x in rows]==list(range(1,8001))
    assert all(math.isfinite(x["loss"]) and len(x["samples"])==4 and all(y["patient_id"] in ["L067","L143"] for y in x["samples"]) for x in rows)
    access=json.loads((o/"DATA_ACCESS.json").read_text());assert access["L506_opens"]==0
    # Import report by path to keep A/B separate.
    import importlib.util
    spec=importlib.util.spec_from_file_location("report_"+name,o/"reporting.py");mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    pre=json.loads((ROOT/"PAIRED_PREFLIGHT.json").read_text());variant="A" if name==NAMES[0] else "B"
    mod.write_report(done["history"],pre["parameters"][variant]["total"],done["peak_allocated"],done["peak_reserved"])
    (o/"FINAL_AUDIT.json").write_text(json.dumps({"status":"PASS","steps":8000,"validations":4,"L506_opens":0,"source_hashes_unchanged":True},indent=2))
    return done,rows
def compare():
    da,ra=audit(NAMES[0]);db,rb=audit(NAMES[1])
    assert all(a["samples"]==b["samples"] and a["diffusion_rng_states"]==b["diffusion_rng_states"] for a,b in zip(ra,rb))
    a=ROOT/"experiments"/NAMES[0];b=ROOT/"experiments"/NAMES[1]
    ca=torch.load(a/"checkpoints/step_0.pt",map_location="cpu",weights_only=False);cb=torch.load(b/"checkpoints/step_0.pt",map_location="cpu",weights_only=False)
    assert all(torch.equal(v,cb["model_state_dict"][k]) for k,v in ca["model_state_dict"].items());del ca,cb
    rows=[]
    for xa,xb in zip(da["history"],db["history"]):
        rows.append({"step":xa["step"],**{"control_"+k:xa[k] for k in KEYS},**{"medcnn_"+k:xb[k] for k in KEYS},**{"delta_"+k:xb[k]-xa[k] for k in KEYS}})
    with (ROOT/"paired_comparison.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    sa=readcsv(a/"validation_step_8000.csv");sb=readcsv(b/"validation_step_8000.csv")
    delta=[float(y["psnr"])-float(x["psnr"]) for x,y in zip(sa,sb)]
    counts={"improved":sum(x>0 for x in delta),"declined":sum(x<0 for x in delta),"equal":sum(x==0 for x in delta)}
    text="# Paired Fast-Dev: TimeOnly control vs MedDINO+CNN\n\nBoth8000 steps/fourEMAvalidations completed and audited. Common backbone step0 exact equality; all8000 sample sequences and16000 CUDA RNG state pairs identical. TrainingL067/L143; validation originalL333Fast64;noL506.\n\n"
    text+="|Step|Control PSNR|MedCNN PSNR|Delta PSNR|Control SSIM|MedCNN SSIM|Delta SSIM|\n|---:|---:|---:|---:|---:|---:|---:|\n"
    for x in rows:text+="|"+"|".join(str(x[k]) if k=="step" else f"{x[k]:.8f}" for k in ["step","control_psnr","medcnn_psnr","delta_psnr","control_ssim","medcnn_ssim","delta_ssim"])+"|\n"
    text+="\n|Step|Control RMSE|MedCNN RMSE|Control MAE|MedCNN MAE|Control softPSNR|MedCNN softPSNR|Control softSSIM|MedCNN softSSIM|Control softRMSE|MedCNN softRMSE|\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    for x in rows:text+="|"+"|".join(str(x[k]) if k=="step" else f"{x[k]:.8f}" for k in ["step","control_rmse_hu","medcnn_rmse_hu","control_mae_hu","medcnn_mae_hu","control_soft_psnr","medcnn_soft_psnr","control_soft_ssim","medcnn_soft_ssim","control_soft_rmse_hu","medcnn_soft_rmse_hu"])+"|\n"
    text+=f"\nPrimary endpoint8k deltaPSNR={rows[-1]['delta_psnr']:+.8f}dB. Per-slice8k PSNR counts: {counts}. This is one seed and one validation patient's64 correlated slices; no significance claim is made.\n"
    pre=json.loads((ROOT/"PAIRED_PREFLIGHT.json").read_text())
    for label,done,tr,o in [("A",da,ra,a),("B",db,rb,b)]:
        text+=f"\n{label}: parameters {pre['parameters'][label]}, peak allocated {done['peak_allocated']/2**30:.3f}GiB, mean train seconds/step {np.mean([x['seconds'] for x in tr]):.6f}, elapsed {done['seconds']/3600:.3f}h; bestEMA {json.loads((o/'BEST_CHECKPOINT.json').read_text())}.\n"
    text+=f"\nMedDINO verified load: {pre['meddino']}. Architecture and full reproduction protocol in each REPORT.md. No automatic extension or third experiment.\n"
    (ROOT/"REPORT.md").write_text(text)
    (ROOT/"FINAL_AUDIT.json").write_text(json.dumps({"status":"PASS","both_steps":8000,"validation_count":8,"common_step0_equal":True,"all_sample_and_rng_sequences_equal":True,"slice_psnr_counts":counts,"delta_psnr_8k":rows[-1]["delta_psnr"]},indent=2))
    print("PAIRED_FINAL_AUDIT_PASS",flush=True)
if __name__=="__main__":
    if len(sys.argv)>1:audit(sys.argv[1])
    else:compare()
