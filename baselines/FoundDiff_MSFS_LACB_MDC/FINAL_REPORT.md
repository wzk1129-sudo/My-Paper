# FoundDiff-MSFS-LACB-MDC: 16k extension

Completed 16000 optimizer steps and eight EMA L333-Fast64 validations. User extended the same training run during its initial stage. The first8k was preserved; continuation restored model, EMA, Adam state, random states and data-stream state at8k. No optimizer reset or weight reinitialization for continuation.

|Step|Baseline same-step PSNR|Candidate PSNR|Matched delta|SSIM|RMSE HU|MAE HU|
|---:|---:|---:|---:|---:|---:|---:|
|2000|40.26604638|39.86870050|-0.39734588|0.93901455|20.41044092|13.77905004|
|4000|40.69278263|40.74317238|+0.05038975|0.94763056|18.46375516|12.53946275|
|6000|40.79711561|40.85308429|+0.05596868|0.94839645|18.23207669|12.33213024|
|8000|40.84604585|40.89735173|+0.05130588|0.94873618|18.14114612|12.29543646|
|10000|N/A|40.90778112|N/A|0.94882810|18.12038961|12.26295061|
|12000|N/A|40.89968606|N/A|0.94873889|18.13793325|12.28060393|
|14000|N/A|40.86583175|N/A|0.94845916|18.20980849|12.28202398|
|16000|N/A|40.82448189|N/A|0.94812728|18.29894455|12.32368205|

No TimeDACB10k/12k/14k/16k control was requested or trained. Those matched baseline cells remain N/A. Comparing these longer-trained checkpoints against baseline8k is only a training-budget-mismatched reference, not evidence of isolated module benefit.

|Step|Soft PSNR|Soft SSIM|Soft RMSE HU|
|---:|---:|---:|---:|
|2000|27.72870879|0.83483356|16.55546969|
|4000|28.44453110|0.84200331|15.25659444|
|6000|28.52690681|0.84288617|15.11464556|
|8000|28.55993071|0.84313972|15.05932867|
|10000|28.56587607|0.84315371|15.05055033|
|12000|28.55497773|0.84289701|15.07016622|
|14000|28.51931940|0.84232114|15.13382552|
|16000|28.47759482|0.84178597|15.20923905|

16k PSNR=40.82448189; delta vs baseline8k reference=-0.02156396 dB (different training steps). Best EMA step10000: 40.90778112 dB. Best local checkpoint: /home/xujian/projects/FoundDiff_Lite_TimeDACB/experiments/FoundDiff_MSFS_LACB_MDC_8k/extension_16k/checkpoints/step_10000.pt, key ema_state_dict.

Parameters total=110,587,073, trainable=24,945,089. Peak allocated=8.007 GiB, reserved=10.963 GiB. Total training/validation runtime=9.013h; extension=4.484h. Architecture, batch2/accum2, FP32, learning rate2e-4, EMA.995, residualMSE, normalization and metrics unchanged.

All nine LACBs, six MDCs, four FAMs and frozen real MedDINO were verified in the original preflight. See REPORT_8k.md for architecture, source references and original8k comparison. One seed and one validation patient's correlated slices remain insufficient for significance/generalization claims. Inspect late validation trends and per-slice/soft-window errors before another change; no further experiments were run.

Reproduce initial8k with root preflight.py/run.sh, then extension_16k/run_wait.sh resumes that checkpoint and completes16k. Keep original8k config.json; extension_16k/config.json documents the authorized16k limit. All data/weight paths require the existing external assets. No dataset, pretrained weights, checkpoints or images are published.
