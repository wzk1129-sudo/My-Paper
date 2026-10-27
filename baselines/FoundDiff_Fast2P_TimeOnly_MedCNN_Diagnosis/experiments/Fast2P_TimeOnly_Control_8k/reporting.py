"""Matched two-patient comparison; values only from actual validation outputs."""
import csv,json
from pathlib import Path
OUT=Path(__file__).resolve().parent
KEYS=("psnr","ssim","rmse_hu","mae_hu","soft_psnr","soft_ssim","soft_rmse_hu")
def write_report(history,params,peak,reserved):
    cfg=json.loads((OUT/"config.json").read_text());root=OUT.parents[1]
    pre=json.loads((root/"PAIRED_PREFLIGHT.json").read_text());variant="A" if cfg["variant"]=="control" else "B"
    best=max(history,key=lambda r:r["psnr"])
    (OUT/"BEST_CHECKPOINT.json").write_text(json.dumps({"step":best["step"],"psnr":best["psnr"],"path":str(OUT/f"checkpoints/step_{best['step']}.pt"),"weights_key":"ema_state_dict"},indent=2))
    text=f"# {cfg['experiment_name']}\n\nCompleted steps: {history[-1]['step']}/8000. Matched Fast-Dev: L067(560),L143(585); L333 original Fast64 validation. L506 excluded.\n\n"
    text+="|Step|PSNR|SSIM|RMSE HU|MAE HU|Soft PSNR|Soft SSIM|Soft RMSE HU|\n|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    for row in history:text+="|"+"|".join(str(row[k]) if k=="step" else f"{row[k]:.8f}" for k in ("step",*KEYS))+"|\n"
    p=pre["parameters"][variant]
    text+=f"\nTotal parameters: {p['total']:,}; trainable: {p['trainable']:,}. Peak allocated {peak/2**30:.3f} GiB; reserved {reserved/2**30:.3f} GiB. Best EMA: checkpoints/step_{best['step']}.pt (ema_state_dict).\n"
    if (OUT/"COMPLETE.json").exists():
        done=json.loads((OUT/"COMPLETE.json").read_text())
        rows=[json.loads(x) for x in (OUT/"train.jsonl").read_text().splitlines()]
        text+=f"Training+validation runtime: {done['seconds']:.2f}s. Mean optimizer-step time: {sum(x['seconds'] for x in rows)/len(rows):.6f}s.\n"
    text+="\n## Shared model and training\nExact archived FoundDiff Lite TimeOnlyDACB backbone, commit adc8f138e1e02a1ac567177d6508436c2b5c6f36. Four encoders/middle/four decoders, ten original single-convolution RLEBs, nine unchanged TimeOnlyDACBs, all skips and RDDM mathematics retained. 2.5D [I_t,previous,center,next], center residual1 channel. Encoder TimeDACB->RLEB->skip->down; mid RLEB->TimeDACB; decoder concat->RLEB->TimeDACB->up. Widths E64/64/128/256,MID512,D512/256/128/64; resolutions512/256/128/64,64,64/128/256/512.\n\n"
    text+="Stored-1024 HU,clip[-1000,1000],model[-1,1],patient-local neighbors with center replication; seed10,batch2accum2,Adam2e-4 betas(.9,.99),wd0,clip1,no scheduler,FP32/TF32off,EMA.995 eachstep,1000 diffusionsteps,2 samplingsteps,sumscale.01,pred_resMSE,no augmentation. Metrics copied byte-for-byte,PSNRrange2000,SSIMwin7,fixed validation seed20260829+index; softwindow[-160,240]. A fresh random initialization; B loads A STEP0 shared state only. Both reset diffusion RNG toseed10 after model construction; sample sequence and per-microbatch CUDA RNG hashes logged and compared after completion. No trained A weights transferred.\n\n"
    if variant=="B":
        text+="## Structural conditioning\nOnly frozen MedDINO receives center LDCT; 2.5D CNN receives previous/center/next. Official MedDINOv3 CT-3M ViT-B/16 weights loaded STRICTLY, no random substitute. Official CT demo preprocessing: HUclip[-1000,1000], mean65.1084213256836/std178.01663208007812,repeat3channels,keep512; RoPE usesFP32 to honor requested precision. Layers[2,5,8,11] zero-based normalized patch tokens only,4x[B,768,32,32]. Four1x1 projections768->32,concat128,1x1/GN/SiLU. Frozen model always eval/no_grad.\n\n"
        text+="Residual CNN channels32@512,32@256,64@128,128@64,each stage Conv3x3/GN/SiLU+two two-convolution residual blocks; stride2 between scales. Local CA at64/128 uses CNNQ and MedK/V,4heads*32,each query maps by pixel center to32grid and attends only valid3x3tokens; invalid neighbors masked before softmax. A projected to CNNchannels; F=C+sigmoid(Conv1x1([C,A]))*A. F128 upsampled bilinear->1x1 to32 then separate3x3 gate/delta fusion withC256; repeat to512. No high-resolution global attention.\n\n"
        text+="Nine independent adapters: E0/F512,E1/F256,D2/F256,D3/F512 use X+sigmoid(Linear(SiLU(t)))*sigmoid(Conv3x3([meanX,meanF]))*Conv1x1(F),last delta projection zero. E2/F128,E3/F64,MID/F64,D0/F64,D1/F128 use X+time_gate*(gamma*GN(X)+beta),gamma/beta output projection zero. Other gates initialized normally. Inject before encoder TimeDACB; after midRLEB before midTimeDACB; after decoderRLEB before TimeDACB. New CNN/fusion/adapters use non-reentrant activation checkpointing; no backbone changes. No SkipGate,LA,TransposedAttention,Mamba,DA-CLIP,new losses or other modules.\n\n"
        text+=f"MedDINO load confirmation: {json.dumps(pre['meddino'])}\n"
    text+="\nPreflight: both backbones and time blocks preserved, all requested feature shapes, nine adapters called, step0 outputs identical, actual sampled timestep/noise identical, frozen weights unchanged, gradients and updates after zero adapters open, batch2FP32,two-step sampling. See PAIRED_PREFLIGHT.json. Final paired report is generated only after A8k then B8k and validates all samples/RNG hashes and checkpoints.\n"
    (OUT/"REPORT.md").write_text(text)
