"""Finalize numeric audit, reports and an explicit text-only GitHub export."""
from pathlib import Path
import json,csv,hashlib,tarfile,shutil
import torch,numpy as np
from model import HERE
from train import write_json,csv_write
from mayo_data import VALID
def read(path):
    with path.open() as f:return list(csv.DictReader(f))
def main():
    torch.set_num_threads(2)
    done=json.loads((HERE/"COMPLETE.json").read_text())
    assert done["step"]==done["ema_updates"]==8000 and len(done["history"])==4
    for n,digest in done["source_hashes"].items():
        assert hashlib.sha256((HERE/n).read_bytes()).hexdigest()==digest,n
    base={int(r["step"]):r for r in read(HERE/"baseline_validation_summary.csv")}
    keys=("psnr","ssim","rmse_hu","mae_hu","soft_psnr","soft_ssim","soft_rmse_hu")
    comparison=[];checks=[]
    for step in (2000,4000,6000,8000):
        rows=read(HERE/f"validation_step_{step}.csv")
        assert len(rows)==64 and [int(r["slice_index"]) for r in rows]==list(VALID)
        means={k:float(np.mean([float(r[k]) for r in rows])) for k in keys}
        result=next(r for r in done["history"] if r["step"]==step)
        for k in keys:assert abs(means[k]-result[k])<1e-10
        ck=torch.load(HERE/"checkpoints"/f"step_{step}.pt",map_location="cpu",weights_only=False)
        assert ck["step"]==ck["ema_updates"]==step and ck["source_hashes"]==done["source_hashes"]
        for name in ("model_state_dict","ema_state_dict"):
            assert all(torch.isfinite(v).all() for v in ck[name].values() if v.is_floating_point())
        assert all(int(v["step"])==step for v in ck["optimizer"]["state"].values())
        checks.append({"step":step,"checkpoint_ema_optimizer":"PASS","fixed64_csv_means":"PASS"});del ck
        comparison.append({"step":step,"baseline_psnr":float(base[step]["psnr"]),**means,
            **{"delta_"+k:means[k]-float(base[step][k]) for k in keys}})
    csv_write(HERE/"comparison.csv",comparison)
    best=max(comparison,key=lambda r:r["psnr"]);last=comparison[-1]
    baseline_root=HERE.parents[1]
    originals=json.loads((HERE/"BASELINE_SOURCE_HASHES.json").read_text())
    external_available=all((baseline_root/n).is_file() for n in originals)
    if external_available:
        assert all(hashlib.sha256((baseline_root/n).read_bytes()).hexdigest()==s for n,s in originals.items())
    for n in ("src/DADiff.py","frozen_metrics.py"):
        assert hashlib.sha256((HERE/n).read_bytes()).hexdigest()==originals[n]
    pre=json.loads((HERE/"PREFLIGHT.json").read_text());assert pre["status"]=="PASS"
    audit={"status":"PASS","checks":checks,"baseline_files_unchanged":True if external_available else "external baseline unavailable; local frozen copies verified",
           "source_hashes_unchanged":True,"no_diffusion_checkpoint_loaded":True}
    write_json(HERE/"FINAL_AUDIT.json",audit)
    text="# FoundDiff-MSFS-LACB-MDC\n\nCompleted 8000 optimizer steps and all four EMA L333-Fast64 validations from random seed10 initialization. No old diffusion weights loaded. MedDINO frozen pretrained weights only.\n\n"
    text+="|Step|Baseline PSNR|Candidate PSNR|Delta PSNR|SSIM|RMSE HU|MAE HU|\n|---:|---:|---:|---:|---:|---:|---:|\n"
    for r in comparison:
        text+="|"+"|".join(str(r[k]) if k=="step" else f"{r[k]:.8f}" for k in ("step","baseline_psnr","psnr","delta_psnr","ssim","rmse_hu","mae_hu"))+"|\n"
    text+="\n|Step|Soft PSNR|Soft SSIM|Soft RMSE HU|\n|---:|---:|---:|---:|\n"
    for r in comparison:text+=f"|{r['step']}|{r['soft_psnr']:.8f}|{r['soft_ssim']:.8f}|{r['soft_rmse_hu']:.8f}|\n"
    text+=f"\nCandidate8k versus TimeDACB8k: PSNR {last['delta_psnr']:+.8f} dB; SSIM {last['delta_ssim']:+.8f}; RMSE {last['delta_rmse_hu']:+.8f} HU; MAE {last['delta_mae_hu']:+.8f} HU. Exceeds TimeDACB@8k PSNR: {last['psnr']>40.846045846787675}. Best candidate EMA step={best['step']}, PSNR={best['psnr']:.8f}.\n"
    text+=f"\nTotal parameters {done['parameters_total']:,}; trainable {done['parameters_trainable']:,}; frozen {done['parameters_total']-done['parameters_trainable']:,}. Peak allocated {done['peak_allocated']/2**30:.3f} GiB, peak reserved {done['peak_reserved']/2**30:.3f} GiB. Training+validation runtime {done['seconds']/3600:.3f} hours. Runtime does not include preflight/final audit.\n"
    text+="\n## Implementation\nmodel.py implements MSFS, ScaleFusion, FAM, LACB, LinearAttention, MDC and ConditionUnet.forward. src/DADiff.py is byte-identical to the baseline; all original RDDM math, 10 single-convolution ResnetBlocks, down/up/skip/head modules retained. Input [I_t,previous,center,next]; output single-channel center residual. Both prior branches see center LDCT only; MedDINO uses model center*1000 HU, then (HU-65.1084213256836)/178.01663208007812 repeated3.\n\n"
    text+="MSFS: frozen MedDINO zero-based blocks2/5/8/11, four patch grids768@32; CNN center-only 3x3/GN8/SiLU stages64@512,64@256,128@128,256@64. Each CNN pooled32 queries corresponding Med layer with projected KV, SDPA4heads x32dim. Attention128@32 upsampled, concatenated with native CNN and fused by3x3/GN8/SiLU. Four FAM3x3 concatenate same-scale64/64/128/256 U-Net and structure, before encoder ResBlock.\n\n"
    text+="Global condition M11 GAP768 -> Linear256; concatenate original time_mlp256 -> Linear512/256 -> SiLU -> Linear256. One joint vector feeds all nine independent LACBs.\n\n"
    text+="LACB: CPE1 depthwise3x3 -> non-affine LN eps1e-6 + condition shift/scale -> ACL in_proj/dwc/SiLU/ELU+1 linear attention with LePE -> multiply act_proj/SiLU branch -> out_proj/residual -> CPE2/residual -> LN/FFN(C,2C,C GELU)/residual = H. Output x+gate*(H-x); exactly one shift/scale/gate projection, zero initialized. Four LA heads at each scale; ACL norm2 remains default LayerNorm. No additional gates, stochastic depth, Mamba or SS2D. LA token matrix is never NxN; KTV normalization matches pinned ACL code.\n\n"
    text+="MDC: parallel depthwise3x3 dilation2 padding2 and5x5 dilation2 padding4, each GN8+SiLU, concatenate ->1x1 -> input residual. Six locations E0/E1/E2 and D1/D2/D3. Encoder FAM -> original ResBlock -> LACB -> optional MDC -> identical tensor to skip and downsample. Mid original ResBlock -> LACB. Decoder concat skip -> original ResBlock -> LACB -> optional MDC -> upsample. E3/MID/D0 are64x64; E3 ends same-resolution3x3, not an extra downsample. Decoder channel sequence512/256/128/64; original stem skip/final ResBlock/1x1 head unchanged.\n\n"
    text+="## Protocol and validation\nconfig.json inherits the original eight-patient training and fixed Fast64 order; seed10, batch2 x accumulation2, 512, FP32/TF32off, Adam2e-4 betas(.9,.99) wd0, clip1, EMA.995 every step, residual MSE only, diffusion1000/sample2/sumscale.01, no augmentation or L506. frozen_metrics.py unchanged. Per-slice means: main[-1000,1000], PSNRrange2000, SSIMwin7; soft[-160,240]. Only new trainable modules use activation checkpointing; resolution/batch/precision unchanged.\n\n"
    text+=f"Preflight status PASS; ACL LA forward max difference={pre['acl_linear_forward_max_diff']}, input gradients equal={pre['acl_linear_backward_equal']}. Nine LACBs, six MDCs, four FAMs and full dataflow checked. Time and global conditions influence updated outputs; MSFS/LACB/MDC finite gradients and optimizer updates checked. Frozen MedDINO SHA and eval/no-grad verified. Realbatch2 training and two-step sampling passed. No hidden auxiliary losses or extra experiments.\n\n"
    text+="## Interpretation and limitations\nThis compares the complete candidate architecture with the baseline at matched step counts; it cannot identify an individual module's contribution. A single seed and one validation patient's correlated64 slices do not establish statistical significance or generalization. Assess both main and soft windows and the saved per-slice errors before selecting a new architecture. Next useful direction: inspect where errors concentrate and whether high-resolution structure fusion preserves fine detail; no further variant or search was run.\n\n"
    text+=f"## Artifacts and reproduction\nBest local EMA checkpoint: {HERE}/checkpoints/step_{best['step']}.pt, key ema_state_dict. Frozen MedDINO weights are not duplicated in checkpoints and must match med_loader.py SHA.\n"
    text+="Run in a fresh copy of this directory: CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n flexict python preflight.py; then bash run.sh. Existing train logs/checkpoints are protected. Configure external data/model paths in config.json and med_loader.py for another machine. Exact launch/environment/source records are in RUN_MANIFEST.json and INITIAL_STATE.json.\n\n"
    text+="ACL reference: https://github.com/ClimBin/ACLNet/blob/905f6d04530d8b6ca4b5a962ef59fd5f3ea8c6b1/models/layers.py. Baseline: https://github.com/wzk1129-sudo/My-Paper/tree/adc8f138e1e02a1ac567177d6508436c2b5c6f36/baselines/FoundDiff_Lite_TimeDACB.\n"
    (HERE/"FINAL_REPORT.md").write_text(text)
    # Publish only explicit small text artifacts; never enumerate data/checkpoints/images.
    out=HERE/"github_export";out.mkdir()
    allow=["model.py","train.py","preflight.py","finalize.py","mayo_data.py","frozen_metrics.py","med_loader.py",
           "src/DADiff.py","src/__init__.py","config.json","L333_fast64.txt","baseline_validation_summary.csv",
           "validation_summary.csv","comparison.csv","PREFLIGHT.json","FINAL_AUDIT.json","FINAL_REPORT.md",
           "COMPLETE.json","INITIAL_STATE.json","RUN_MANIFEST.json","run.sh","reference/provenance.json","reference/ACL_layers.py","BASELINE_SOURCE_HASHES.json"]
    for name in allow:
        p=HERE/name;assert p.is_file() and p.stat().st_size<1024*1024
        dest=out/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    for step in (2000,4000,6000,8000):
        rows=read(HERE/f"validation_step_{step}.csv")
        clean=[{"slice_ordinal":i+1,**{k:float(row[k]) for k in keys}} for i,row in enumerate(rows)]
        csv_write(out/f"validation_step_{step}.csv",clean)
    # Remove access counters/patient IDs from public preflight; protocol stays reproducible.
    public_pre=json.loads((out/"PREFLIGHT.json").read_text());public_pre.pop("data_access",None)
    write_json(out/"PREFLIGHT.json",public_pre)
    (out/"README.md").write_text("# FoundDiff-MSFS-LACB-MDC\n\n"+f"Completed8k, best EMA PSNR {best['psnr']:.6f} dB at step{best['step']}; final delta versus TimeDACB {last['delta_psnr']:+.6f} dB. See FINAL_REPORT.md for architecture, full metrics and limitations.\n\n"+ "No CT data, model weights, checkpoints, validation images, intermediate tensors or large training logs are included. Per-slice CSVs use anonymous ordinals in the fixed validation order. MedDINO must be supplied separately with the recorded SHA.\n")
    manifest={str(p.relative_to(out)):sha_file(p) for p in out.rglob("*") if p.is_file()}
    write_json(out/"EXPORT_MANIFEST.json",manifest)
    with tarfile.open(HERE/"github_export.tar.gz","x:gz") as tf:
        for p in sorted(out.rglob("*")):
            if p.is_file():tf.add(p,arcname=str(p.relative_to(out)),recursive=False)
    print("FINAL_AUDIT_PASS_EXPORT_READY",flush=True)
def sha_file(p):return hashlib.sha256(p.read_bytes()).hexdigest()
if __name__=="__main__":main()
