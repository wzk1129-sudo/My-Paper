"""Audit/report full sixteen-thousand-step run; no invented >8k baseline."""
from pathlib import Path
import sys,json,csv,hashlib,shutil,tarfile
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
sys.path.insert(0,str(BASE))
import torch,numpy as np
from train import csv_write,write_json
from mayo_data import VALID
def read(p):
    with p.open() as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    torch.set_num_threads(2)
    done=json.loads((HERE/"COMPLETE.json").read_text())
    parent=json.loads((BASE/"COMPLETE.json").read_text())
    assert done["step"]==done["ema_updates"]==16000 and len(done["history"])==8
    assert json.loads((HERE/"RESUME_VERIFICATION.json").read_text())["status"]=="PASS"
    assert json.loads((BASE/"FINAL_AUDIT.json").read_text())["status"]=="PASS"
    for n,h in done["source_hashes"].items():assert sha(BASE/n)==h,n
    base={int(r["step"]):r for r in read(BASE/"baseline_validation_summary.csv")}
    keys=("psnr","ssim","rmse_hu","mae_hu","soft_psnr","soft_ssim","soft_rmse_hu")
    comparison=[];checks=[]
    for step in range(2000,16001,2000):
        folder=BASE if step<=8000 else HERE
        rows=read(folder/f"validation_step_{step}.csv")
        assert len(rows)==64 and [int(r["slice_index"]) for r in rows]==list(VALID)
        means={k:float(np.mean([float(r[k]) for r in rows])) for k in keys}
        expected=next(r for r in done["history"] if r["step"]==step)
        assert all(abs(means[k]-expected[k])<1e-10 for k in keys)
        ck=torch.load(folder/"checkpoints"/f"step_{step}.pt",map_location="cpu",weights_only=False)
        assert ck["step"]==ck["ema_updates"]==step
        assert ck["source_hashes"]==(parent if step<=8000 else done)["source_hashes"]
        assert all(int(x["step"])==step for x in ck["optimizer"]["state"].values())
        for key in ("model_state_dict","ema_state_dict"):
            assert all(torch.isfinite(x).all() for x in ck[key].values() if x.is_floating_point())
        del ck
        checks.append({"step":step,"checkpoint_csv_optimizer_ema":"PASS"})
        comparison.append({"step":step,"baseline_same_step_psnr":float(base[step]["psnr"]) if step in base else None,
          **means,"delta_same_step_psnr":means["psnr"]-float(base[step]["psnr"]) if step in base else None,
          "delta_vs_baseline_8k_reference":means["psnr"]-float(base[8000]["psnr"])})
    csv_write(HERE/"comparison.csv",comparison)
    audit={"status":"PASS","step":16000,"checks":checks,"resume_all_states_verified":True,"source_hashes_unchanged":True}
    write_json(HERE/"FINAL_AUDIT.json",audit)
    best=max(comparison,key=lambda r:r["psnr"]);last=comparison[-1]
    bestdir=BASE if best["step"]<=8000 else HERE
    text="# FoundDiff-MSFS-LACB-MDC: 16k extension\n\nCompleted 16000 optimizer steps and eight EMA L333-Fast64 validations. User extended the same training run during its initial stage. The first8k was preserved; continuation restored model, EMA, Adam state, random states and data-stream state at8k. No optimizer reset or weight reinitialization for continuation.\n\n"
    text+="|Step|Baseline same-step PSNR|Candidate PSNR|Matched delta|SSIM|RMSE HU|MAE HU|\n|---:|---:|---:|---:|---:|---:|---:|\n"
    for r in comparison:
        bp="N/A" if r["baseline_same_step_psnr"] is None else f"{r['baseline_same_step_psnr']:.8f}"
        delta="N/A" if r["delta_same_step_psnr"] is None else f"{r['delta_same_step_psnr']:+.8f}"
        text+=f"|{r['step']}|{bp}|{r['psnr']:.8f}|{delta}|{r['ssim']:.8f}|{r['rmse_hu']:.8f}|{r['mae_hu']:.8f}|\n"
    text+="\nNo TimeDACB10k/12k/14k/16k control was requested or trained. Those matched baseline cells remain N/A. Comparing these longer-trained checkpoints against baseline8k is only a training-budget-mismatched reference, not evidence of isolated module benefit.\n\n"
    text+="|Step|Soft PSNR|Soft SSIM|Soft RMSE HU|\n|---:|---:|---:|---:|\n"
    for r in comparison:text+=f"|{r['step']}|{r['soft_psnr']:.8f}|{r['soft_ssim']:.8f}|{r['soft_rmse_hu']:.8f}|\n"
    text+=f"\n16k PSNR={last['psnr']:.8f}; delta vs baseline8k reference={last['delta_vs_baseline_8k_reference']:+.8f} dB (different training steps). Best EMA step{best['step']}: {best['psnr']:.8f} dB. Best local checkpoint: {bestdir}/checkpoints/step_{best['step']}.pt, key ema_state_dict.\n"
    text+=f"\nParameters total={done['parameters_total']:,}, trainable={done['parameters_trainable']:,}. Peak allocated={done['peak_allocated']/2**30:.3f} GiB, reserved={done['peak_reserved']/2**30:.3f} GiB. Total training/validation runtime={done['seconds']/3600:.3f}h; extension={done['extension_seconds']/3600:.3f}h. Architecture, batch2/accum2, FP32, learning rate2e-4, EMA.995, residualMSE, normalization and metrics unchanged.\n"
    text+="\nAll nine LACBs, six MDCs, four FAMs and frozen real MedDINO were verified in the original preflight. See REPORT_8k.md for architecture, source references and original8k comparison. One seed and one validation patient's correlated slices remain insufficient for significance/generalization claims. Inspect late validation trends and per-slice/soft-window errors before another change; no further experiments were run.\n"
    text+="\nReproduce initial8k with root preflight.py/run.sh, then extension_16k/run_wait.sh resumes that checkpoint and completes16k. Keep original8k config.json; extension_16k/config.json documents the authorized16k limit. All data/weight paths require the existing external assets. No dataset, pretrained weights, checkpoints or images are published.\n"
    (HERE/"FINAL_REPORT.md").write_text(text)
    export=HERE/"github_export";shutil.copytree(BASE/"github_export",export)
    shutil.copyfile(BASE/"FINAL_REPORT.md",export/"REPORT_8k.md")
    for name in ("COMPLETE.json","FINAL_AUDIT.json","FINAL_REPORT.md","comparison.csv","validation_summary.csv"):
        shutil.copyfile(HERE/name,export/name)
    ext=export/"extension_16k";ext.mkdir()
    for name in ("train_continue.py","finalize_extended.py","config.json","run_wait.sh","EXTENSION_PLAN.json","RUN_MANIFEST.json","RESUME_VERIFICATION.json"):
        shutil.copyfile(HERE/name,ext/name)
    for step in (10000,12000,14000,16000):
        rows=read(HERE/f"validation_step_{step}.csv")
        csv_write(export/f"validation_step_{step}.csv",[{"slice_ordinal":i+1,**{k:float(r[k]) for k in keys}} for i,r in enumerate(rows)])
    (export/"README.md").write_text("# FoundDiff-MSFS-LACB-MDC, 16k\n\n"+f"Completed16k with eight EMA validations. Best PSNR {best['psnr']:.6f} at step{best['step']}. See FINAL_REPORT.md; comparisons above8k lack a same-budget baseline. REPORT_8k.md preserves the matched8k comparison.\n")
    manifest={str(p.relative_to(export)):sha(p) for p in export.rglob("*") if p.is_file() and p.name!="EXPORT_MANIFEST.json"}
    write_json(export/"EXPORT_MANIFEST.json",manifest)
    with tarfile.open(HERE/"github_export.tar.gz","x:gz") as tf:
        for p in sorted(export.rglob("*")):
            if p.is_file():assert p.stat().st_size<1024*1024;tf.add(p,arcname=str(p.relative_to(export)),recursive=False)
    print("FINAL_16K_AUDIT_PASS_EXPORT_READY",flush=True)
if __name__=="__main__":main()
