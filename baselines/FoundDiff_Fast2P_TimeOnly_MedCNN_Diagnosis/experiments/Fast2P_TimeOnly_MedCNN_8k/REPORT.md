# Fast2P_TimeOnly_MedCNN_8k

Completed steps: 8000/8000. Matched Fast-Dev: L067(560),L143(585); L333 original Fast64 validation. L506 excluded.

|Step|PSNR|SSIM|RMSE HU|MAE HU|Soft PSNR|Soft SSIM|Soft RMSE HU|
|---:|---:|---:|---:|---:|---:|---:|---:|
|2000|40.48729257|0.94580665|19.01317868|13.08108638|28.27628464|0.84032897|15.54945415|
|4000|40.68986172|0.94745620|18.57752419|12.71979173|28.42346953|0.84199599|15.29317293|
|6000|40.66971248|0.94721243|18.62098691|12.72989781|28.38502790|0.84165479|15.36405317|
|8000|40.36719784|0.94443200|19.28930130|13.03688560|28.05437322|0.83642567|15.97349683|

Total parameters: 101,426,285; trainable: 15,784,301. Peak allocated 6.253 GiB; reserved 7.701 GiB. Best EMA: checkpoints/step_4000.pt (ema_state_dict).
Training+validation runtime: 6174.95s. Mean optimizer-step time: 0.765045s.

## Shared model and training
Exact archived FoundDiff Lite TimeOnlyDACB backbone, commit adc8f138e1e02a1ac567177d6508436c2b5c6f36. Four encoders/middle/four decoders, ten original single-convolution RLEBs, nine unchanged TimeOnlyDACBs, all skips and RDDM mathematics retained. 2.5D [I_t,previous,center,next], center residual1 channel. Encoder TimeDACB->RLEB->skip->down; mid RLEB->TimeDACB; decoder concat->RLEB->TimeDACB->up. Widths E64/64/128/256,MID512,D512/256/128/64; resolutions512/256/128/64,64,64/128/256/512.

Stored-1024 HU,clip[-1000,1000],model[-1,1],patient-local neighbors with center replication; seed10,batch2accum2,Adam2e-4 betas(.9,.99),wd0,clip1,no scheduler,FP32/TF32off,EMA.995 eachstep,1000 diffusionsteps,2 samplingsteps,sumscale.01,pred_resMSE,no augmentation. Metrics copied byte-for-byte,PSNRrange2000,SSIMwin7,fixed validation seed20260829+index; softwindow[-160,240]. A fresh random initialization; B loads A STEP0 shared state only. Both reset diffusion RNG toseed10 after model construction; sample sequence and per-microbatch CUDA RNG hashes logged and compared after completion. No trained A weights transferred.

## Structural conditioning
Only frozen MedDINO receives center LDCT; 2.5D CNN receives previous/center/next. Official MedDINOv3 CT-3M ViT-B/16 weights loaded STRICTLY, no random substitute. Official CT demo preprocessing: HUclip[-1000,1000], mean65.1084213256836/std178.01663208007812,repeat3channels,keep512; RoPE usesFP32 to honor requested precision. Layers[2,5,8,11] zero-based normalized patch tokens only,4x[B,768,32,32]. Four1x1 projections768->32,concat128,1x1/GN/SiLU. Frozen model always eval/no_grad.

Residual CNN channels32@512,32@256,64@128,128@64,each stage Conv3x3/GN/SiLU+two two-convolution residual blocks; stride2 between scales. Local CA at64/128 uses CNNQ and MedK/V,4heads*32,each query maps by pixel center to32grid and attends only valid3x3tokens; invalid neighbors masked before softmax. A projected to CNNchannels; F=C+sigmoid(Conv1x1([C,A]))*A. F128 upsampled bilinear->1x1 to32 then separate3x3 gate/delta fusion withC256; repeat to512. No high-resolution global attention.

Nine independent adapters: E0/F512,E1/F256,D2/F256,D3/F512 use X+sigmoid(Linear(SiLU(t)))*sigmoid(Conv3x3([meanX,meanF]))*Conv1x1(F),last delta projection zero. E2/F128,E3/F64,MID/F64,D0/F64,D1/F128 use X+time_gate*(gamma*GN(X)+beta),gamma/beta output projection zero. Other gates initialized normally. Inject before encoder TimeDACB; after midRLEB before midTimeDACB; after decoderRLEB before TimeDACB. New CNN/fusion/adapters use non-reentrant activation checkpointing; no backbone changes. No SkipGate,LA,TransposedAttention,Mamba,DA-CLIP,new losses or other modules.

MedDINO load confirmation: {"path": "/home/xujian/research/FoundDiff_TimeOnly_PairedFastDev/pretrained/MedDINOv3_ViTB16_CT3M_model.pth", "sha256": "492b2b96c00e9d3b93ba3b07a8d387aa2196b5e04698378fcc23049b447966d2", "strict": true, "missing": [], "unexpected": [], "layers_zero_based": [2, 5, 8, 11]}

Preflight: both backbones and time blocks preserved, all requested feature shapes, nine adapters called, step0 outputs identical, actual sampled timestep/noise identical, frozen weights unchanged, gradients and updates after zero adapters open, batch2FP32,two-step sampling. See PAIRED_PREFLIGHT.json. Final paired report is generated only after A8k then B8k and validates all samples/RNG hashes and checkpoints.
