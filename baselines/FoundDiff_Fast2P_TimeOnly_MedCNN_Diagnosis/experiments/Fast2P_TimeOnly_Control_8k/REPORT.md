# Fast2P_TimeOnly_Control_8k

Completed steps: 8000/8000. Matched Fast-Dev: L067(560),L143(585); L333 original Fast64 validation. L506 excluded.

|Step|PSNR|SSIM|RMSE HU|MAE HU|Soft PSNR|Soft SSIM|Soft RMSE HU|
|---:|---:|---:|---:|---:|---:|---:|---:|
|2000|40.32677099|0.94437664|19.35555065|13.22731061|28.13003133|0.84307195|15.80266136|
|4000|40.55813769|0.94657186|18.85361121|12.80113469|28.30807933|0.84337915|15.49085961|
|6000|40.29571945|0.94438764|19.43505744|13.16568336|28.05084498|0.83896450|15.96121514|
|8000|40.09270496|0.94232764|19.89754465|13.34778744|27.82716724|0.83494646|16.38211235|

Total parameters: 13,614,209; trainable: 13,614,209. Peak allocated 4.930 GiB; reserved 6.314 GiB. Best EMA: checkpoints/step_4000.pt (ema_state_dict).
Training+validation runtime: 3555.29s. Mean optimizer-step time: 0.440347s.

## Shared model and training
Exact archived FoundDiff Lite TimeOnlyDACB backbone, commit adc8f138e1e02a1ac567177d6508436c2b5c6f36. Four encoders/middle/four decoders, ten original single-convolution RLEBs, nine unchanged TimeOnlyDACBs, all skips and RDDM mathematics retained. 2.5D [I_t,previous,center,next], center residual1 channel. Encoder TimeDACB->RLEB->skip->down; mid RLEB->TimeDACB; decoder concat->RLEB->TimeDACB->up. Widths E64/64/128/256,MID512,D512/256/128/64; resolutions512/256/128/64,64,64/128/256/512.

Stored-1024 HU,clip[-1000,1000],model[-1,1],patient-local neighbors with center replication; seed10,batch2accum2,Adam2e-4 betas(.9,.99),wd0,clip1,no scheduler,FP32/TF32off,EMA.995 eachstep,1000 diffusionsteps,2 samplingsteps,sumscale.01,pred_resMSE,no augmentation. Metrics copied byte-for-byte,PSNRrange2000,SSIMwin7,fixed validation seed20260829+index; softwindow[-160,240]. A fresh random initialization; B loads A STEP0 shared state only. Both reset diffusion RNG toseed10 after model construction; sample sequence and per-microbatch CUDA RNG hashes logged and compared after completion. No trained A weights transferred.


Preflight: both backbones and time blocks preserved, all requested feature shapes, nine adapters called, step0 outputs identical, actual sampled timestep/noise identical, frozen weights unchanged, gradients and updates after zero adapters open, batch2FP32,two-step sampling. See PAIRED_PREFLIGHT.json. Final paired report is generated only after A8k then B8k and validates all samples/RNG hashes and checkpoints.
