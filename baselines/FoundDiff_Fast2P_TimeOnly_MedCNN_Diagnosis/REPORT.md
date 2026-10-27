# Paired Fast-Dev: TimeOnly control vs MedDINO+CNN

Both8000 steps/fourEMAvalidations completed and audited. Common backbone step0 exact equality; all8000 sample sequences and16000 CUDA RNG state pairs identical. TrainingL067/L143; validation originalL333Fast64;noL506.

|Step|Control PSNR|MedCNN PSNR|Delta PSNR|Control SSIM|MedCNN SSIM|Delta SSIM|
|---:|---:|---:|---:|---:|---:|---:|
|2000|40.32677099|40.48729257|0.16052157|0.94437664|0.94580665|0.00143001|
|4000|40.55813769|40.68986172|0.13172403|0.94657186|0.94745620|0.00088434|
|6000|40.29571945|40.66971248|0.37399303|0.94438764|0.94721243|0.00282479|
|8000|40.09270496|40.36719784|0.27449288|0.94232764|0.94443200|0.00210436|

|Step|Control RMSE|MedCNN RMSE|Control MAE|MedCNN MAE|Control softPSNR|MedCNN softPSNR|Control softSSIM|MedCNN softSSIM|Control softRMSE|MedCNN softRMSE|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|2000|19.35555065|19.01317868|13.22731061|13.08108638|28.13003133|28.27628464|0.84307195|0.84032897|15.80266136|15.54945415|
|4000|18.85361121|18.57752419|12.80113469|12.71979173|28.30807933|28.42346953|0.84337915|0.84199599|15.49085961|15.29317293|
|6000|19.43505744|18.62098691|13.16568336|12.72989781|28.05084498|28.38502790|0.83896450|0.84165479|15.96121514|15.36405317|
|8000|19.89754465|19.28930130|13.34778744|13.03688560|27.82716724|28.05437322|0.83494646|0.83642567|16.38211235|15.97349683|

Primary endpoint8k deltaPSNR=+0.27449288dB. Per-slice8k PSNR counts: {'improved': 64, 'declined': 0, 'equal': 0}. This is one seed and one validation patient's64 correlated slices; no significance claim is made.

A: parameters {'total': 13614209, 'trainable': 13614209}, peak allocated 4.930GiB, mean train seconds/step 0.440347, elapsed 0.988h; bestEMA {'step': 4000, 'psnr': 40.55813768571127, 'path': '/home/xujian/research/FoundDiff_TimeOnly_PairedFastDev/experiments/Fast2P_TimeOnly_Control_8k/checkpoints/step_4000.pt', 'weights_key': 'ema_state_dict'}.

B: parameters {'total': 101426285, 'trainable': 15784301}, peak allocated 6.253GiB, mean train seconds/step 0.765045, elapsed 1.715h; bestEMA {'step': 4000, 'psnr': 40.68986171941624, 'path': '/home/xujian/research/FoundDiff_TimeOnly_PairedFastDev/experiments/Fast2P_TimeOnly_MedCNN_8k/checkpoints/step_4000.pt', 'weights_key': 'ema_state_dict'}.

MedDINO verified load: {'path': '/home/xujian/research/FoundDiff_TimeOnly_PairedFastDev/pretrained/MedDINOv3_ViTB16_CT3M_model.pth', 'sha256': '492b2b96c00e9d3b93ba3b07a8d387aa2196b5e04698378fcc23049b447966d2', 'strict': True, 'missing': [], 'unexpected': [], 'layers_zero_based': [2, 5, 8, 11]}. Architecture and full reproduction protocol in each REPORT.md. No automatic extension or third experiment.
