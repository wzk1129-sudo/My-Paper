# MedCNN eight-patient synchronized augmentation 8k

Completed 8000/8000 optimizer steps, 4/4 EMA validations.

|Step|New 8patient+aug PSNR|Old 2patient noaug PSNR|Delta PSNR|SSIM|RMSE HU|MAE HU|Soft PSNR|Soft SSIM|Soft RMSE HU|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|2000|40.33329178|40.48729257|-0.15400078|0.94411911|19.34488571|13.09750715|28.14080013|0.83597422|15.78959010|
|4000|40.60578952|40.68986172|-0.08407220|0.94622679|18.75278923|12.65056993|28.33294488|0.83881089|15.44865978|
|6000|40.69343255|40.66971248|0.02372007|0.94693750|18.56732539|12.49613369|28.39569976|0.83924182|15.34140369|
|8000|40.74921300|40.36719784|0.38201516|0.94739372|18.44966536|12.54776634|28.44099103|0.84011320|15.26158063|

Parameters {'total': 101426285, 'trainable': 15784301}; peak allocated 6.253 GiB; reserved 7.701 GiB. Best EMA /home/xujian/research/My-Paper/baselines/FoundDiff_MedCNN_8patient_Aug_8k/experiments/MedCNN_8patient_Aug_8k/checkpoints/step_8000.pt, key ema_state_dict.

Training plus validation elapsed 6423.18 seconds.

Train L067/L096/L109/L143/L192/L286/L291/L310,4800 centers; original L333-Fast64 order; no L506. Seed10,batch2,accum2,Adam2e-4 betas(.9,.99),wd0,FP32,clip1,no scheduler,EMA.995 every optimizer step,original residualMSE,RDDM1000diffusion/2sampling,sumscale.01. Stored-1024 HU,clip[-1000,1000],dataset[0,1],model[-1,1]. Main PSNRrange2000,SSIMwin7;soft[-160,240].

Model and computational train.py copied byte-for-byte from the actual archived MedCNN run. Common backbone loaded only from original A untrained step0; no trained A/B weights used. New modules freshly initialized with original seed and verified equal to original B step0. MedDINO official weight SHA256 492b2b96c00e9d3b93ba3b07a8d387aa2196b5e04698378fcc23049b447966d2 strictly loaded, frozen/eval. Original CNN,local cross-attention,Fusion,nine independent spatial/FiLM/time gates,TimeOnlyDACBs and RDDM unchanged. See src/DADiff.py and src/medcnn.py.

Augmentation only in training Stream.next: independent private RandomState10 chooses horizontal flip probability.5 and uniform rotation0/90/180/270. Same transform applied jointly to all3 LDCT channels and center NDCT; contiguous index permutations, no interpolation,crop,resize,filtering. Validation uses unaugmented Pairs.load. Aug RNG saved/restored in checkpoints; independent of shuffle and diffusion RNG. Actual transforms logged per sample.

Historical comparison changes BOTH training patients and augmentation. It cannot isolate the augmentation effect. No extra ablation or architecture change was run. Preflight real batch2accum2,all8 transformations,synchronization,resume RNG,initialization,gradients,frozen MedDINO and2-step sampling passed.
