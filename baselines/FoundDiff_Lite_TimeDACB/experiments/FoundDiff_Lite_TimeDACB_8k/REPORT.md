# FoundDiff-Lite-TimeDACB-2p5D
Official commit: c85a3a0e87ae2e4587650f0fb7efc15f3311d8e2.
Completed optimizer steps: 8000. Fresh random initialization, seed=10.
Trainable parameters: 13,614,209. Peak allocated: 4.930 GiB; peak reserved: 6.314 GiB.

## Architecture
Input [I_t, previous LDCT, center LDCT, next LDCT], 4 channels; stem 7x7 conv to 64.
Encoder (TimeOnlyDACB -> original single-convolution ResnetBlock -> downsample):
E0 64@512 ->64@256; E1 64@256 ->128@128; E2 128@128 ->256@64;
E3 256@64 ->512@64 (official final 3x3 convolution, no spatial downsampling).
Bottleneck: original ResnetBlock 512@64 -> TimeOnlyDACB 512@64.
Decoder (concat skip -> original ResnetBlock -> TimeOnlyDACB -> upsample):
D0 (512+256)->512@64 ->256@128;
D1 (256+128)->256@128 ->128@256;
D2 (128+64)->128@256 ->64@512;
D3 (64+64)->64@512 ->64@512 (official final 3x3 convolution).
Stem skip concat -> original ResnetBlock 128->64 -> original 1x1 head 64->1.
Ten original ResnetBlocks; nine TimeOnlyDACBs. Commented second conv/time modulation remain commented.
time_mlp: official sinusoid 64 -> Linear 256 -> GELU -> Linear 256.

## Removed and retained
Removed CLIP/DA-CLIP loading, dose/anatomy embeddings, prompt/text/prompt MLP,
CSSM/SS2D/Mamba and all attention modules. Removed unused LPIPS instantiation.
src/DADiff.py: TimeOnlyDACB implements channel-last affine-free LayerNorm eps=1e-6:
shift, scale, gate = Linear(SiLU(time_mlp(t))); x_out=x+gate*(LN(x)*(1+scale)+shift).
Its final Linear is zero initialized. No convolution, attention or external conditions within DACB.
ResidualDiffusion.condition=True, channels=out_dim=1. Residual is center LDCT minus center NDCT.
Official q_sample, linear-beta convert_to_ddim=True schedule, residual MSE and two-step sampling retained.

## EMA validation (arithmetic mean of 64 per-slice metrics)
|step|PSNR dB|SSIM|RMSE HU|MAE HU|soft PSNR|soft SSIM|soft RMSE HU|
|---:|---:|---:|---:|---:|---:|---:|---:|
|2000|40.26604638|0.94336170|19.49894484|13.45456354|28.08629335|0.83785298|15.88561910|
|4000|40.69278263|0.94716584|18.56903827|12.65551351|28.41417944|0.84135980|15.30653989|
|6000|40.79711561|0.94798205|18.34886742|12.50682195|28.49170267|0.84225572|15.17367980|
|8000|40.84604585|0.94831217|18.24744758|12.43825823|28.52752354|0.84232740|15.11270288|

## Protocol and reference comparisons
Main window [-1000,1000] HU, PSNR range 2000, SSIM win_size=7. Auxiliary window [-160,240] HU,
range 400. frozen_metrics.py is byte-identical to historical metric implementation.
Historical Fast64 slice list and order and per-slice validation seed=20260829+index retained.
Eight train patients: L067 L096 L109 L143 L192 L286 L291 L310. No L506. No augmentation.
Stored pixel offset -1024 HU; explicit requested training clip [-1000,1000], then model [-1,1].
Historical referenced model used [-1024,3072] normalization and only L067/L143 training, so not a matched experiment.
MLSF+Pyramid@8k supplied reference: 40.956836033 dB. Current step 8000 delta: -0.11079019 dB.
Models, losses, initialization, train patients, normalization, effective batch and seed differ; no isolated attribution.
Historical simplified FoundDiff_RDDM_RLEB Control (2D, no DACB) recorded:
2k 38.4787, 4k 40.1454, 6k 40.3872, 8k 40.4846 dB (rounded report values).
Its 8k SSIM=.9449, RMSE=19.0149 HU, MAE=12.4597 HU.
Exact metrics/source are preserved in historical_reference/. These are contextual references, not matched ablations.
Historical training used two patients, different normalization, seed, effective batch, and paired flip/rotation;
the current baseline also preserves upstream FoundDiff schedule/sampler rather than historical CT alterations.
Adam lr=2e-4, betas=(.9,.99), weight_decay=0 (official optimizer); batch=2, accumulation=2;
EMA=.995 every optimizer step, initialized from step0 (historical matched EMA convention).
amp=False; no scheduler; clip grad norm=1. Peak memory includes training/validation.

## Reproduction
Best checkpoint: /home/xujian/projects/FoundDiff_Lite_TimeDACB/experiments/FoundDiff_Lite_TimeDACB_8k/checkpoints/step_8000.pt (ema_state_dict).
Step0 with model, optimizer, RNG and stream state: checkpoints/step_0.pt.
Full command: cd /home/xujian/projects/FoundDiff_Lite_TimeDACB && CUDA_VISIBLE_DEVICES=0 /home/xujian/miniconda3/bin/conda run --no-capture-output -n flexict python -u train.py
Requires preflight.py PASS. Existing outputs are protected; resume explicitly with --resume PATH.
Config: config.json; training: train.jsonl; launch: RUN_MANIFEST.json; per-slice: validation_step_*.csv.

## Final artifact verification
FINAL_AUDIT.json: PASS. Exactly 8000 sequential optimizer-step log entries; all logged losses and gradient norms finite.
All four checkpoints loaded on CPU: optimizer step and EMA update count equal 2000/4000/6000/8000;
all model and EMA tensors finite. All four 64-slice CSVs preserve the fixed L333 order; seven metric means verified.
Training source hashes match all checkpoints and COMPLETE.json. L506 opens=0 in loader audit.
TimeOnlyDACB source: src/DADiff.py:258. Full run including validation: 3853.657 seconds (64.23 minutes).
Verified facts above are artifact-based. No claim of causal superiority over historical protocols is supported.
No additional ablation or inference run was performed during this final audit.
