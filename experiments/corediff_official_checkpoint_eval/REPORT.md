# Official CoreDiff Mayo 2016 25% dose / 150k EMA: L506 dual-protocol evaluation

Status: COMPLETE. Read-only inference; no training, OSL, parameter selection, or L333 evaluation.

## Results

Arithmetic means of per-slice scores over 524 valid centers:

| Protocol | PSNR (dB) | SSIM | RMSE (HU) | MAE (HU) |
|---|---:|---:|---:|---:|
| OFFICIAL | 41.60937266 | 0.95650445 | 16.91536899 | Not requested |
| UNIFIED | 41.60937146 | 0.95580814 | 16.91536520 | 11.23689846 |

## Checkpoint and source verification

Existing checkpoint: /home/xujian/projects/CoreDiff/output/corediff_dose25_mayo_2016/save_models/ema_model-150000
Bytes: 19337897
SHA256: 16e09d7b13a01e360e064194c3c5c334b558c4b12ce260d4c39f5ff4ad865b39
Public official release SHA256 matches exactly. The two local reproduce copies have a different SHA256 and were rejected.
Official repository: https://github.com/qgao21/CoreDiff
Official source commit: 9bc31dc1d6184a880bd6334db34c466569a3642a
Actual project commit: b754bcbb384ba087a15db4f4f4f9c54fbb4bfc6d
Eight relevant source files match the official commit byte for byte (SOURCE_AUDIT.json).
Official load_network and strict Diffusion.load_state_dict loaded 84 state entries; 4,825,953 trainable-architecture parameters, all finite.
File and loaded model state hashes are unchanged after evaluation.

## Data and inference

Original Mayo 2016 stored-pixel DICOM preprocessing; dataset=mayo_2016, dose=25, test_id=9, patient=L506.
Both raw series contain 526 slices. Official dataset excludes first/last as centers; first/last remain context neighbors.
The evaluated centers are L506_25_001_img.npy through L506_25_524_img.npy.
Exact DICOM stored-pixel versus existing NPY equality was checked at first, middle, last for both series, without rewriting data.
Normalization uses the unmodified CTDataset.normalize_: subtract 1024, clip [-1024,3072], map to [0,1].
Batch=1, context=True (prev/center/next), T=10, sampling_routine=ddim (official default), n_iter=150000, start_adjust_iter=1.
only_adjust_two_step=True is the official training flag; it is not an argument to sample and does not change inference.
The existing official Network and Diffusion.sample are called directly. No optimizer is instantiated.
Each center is sampled once: 524 sample calls, 5240 denoiser calls.
The identical normalized prediction tensor is used by both protocols and hash-checked before and after scoring.
Predictions are retained in memory only; no CT image/prediction array export. Both CSVs carry matching per-slice prediction hashes.
Direct dataset iteration replaces worker-based loading only; slice ordering and returned arrays are unchanged.

## Evaluation protocols

OFFICIAL: use the exact TrainTask.transfer_calculate_window and utils.measure.compute_measure on CUDA.
Normalized images become HU, clipped [-1000,1000], then mapped [0,255].
Per slice data_range=target.max()-target.min() in this mapped space.
Official SSIM uses an 11x11 Gaussian (sigma 1.5), official convolution/padding and dynamic range.
Official RMSE converts mapped values back to HU inside compute_RMSE, and is reported in HU.

UNIFIED: reuse the identical prediction; convert to HU and clip [-1000,1000].
PSNR=10*log10(2000^2/MSE_HU); RMSE=sqrt(MSE_HU); MAE=mean(abs(error)).
SSIM is skimage.structural_similarity with data_range=2000, win_size=7,
gaussian_weights=False, use_sample_covariance=True, K1=0.01, K2=0.03.
HU conversion/clipping is float32 like the official model; unified accumulation is float64.
These are two scoring protocols applied to the same outputs; SSIM differences do not represent a model change.
No parameters, data_range, windows, or slice counts were tuned to match paper numbers.

## Execution and audit

Command:
~~~bash
CUDA_VISIBLE_DEVICES=0 /home/xujian/miniconda3/envs/corediff/bin/python -u /home/xujian/projects/CoreDiff/evaluate_official_checkpoint_dual_protocol.py --checkpoint /home/xujian/projects/CoreDiff/output/corediff_dose25_mayo_2016/save_models/ema_model-150000 --output /home/xujian/projects/CoreDiff/experiments/corediff_official_checkpoint_eval
~~~
tmux: corediff_official_dual_eval_20261002
Conda: corediff; GPU: NVIDIA GeForce RTX 3090; PyTorch: 1.10.1+cu111; CUDA runtime: 11.1.
Started: 2026-10-02T15:01:24+0800; finished: 2026-10-02T15:04:25+0800.
Inference/scoring elapsed: 149.34 seconds.
Peak allocated GPU memory: 0.430 GiB.
Peak reserved GPU memory: 0.508 GiB.
Detailed environment, command/options, source, checkpoint, data boundary and access/state audits are in the neighboring JSON files.
No original source, checkpoint, or medical data was modified. No other-patient file reads occurred in this evaluation.
No claim is made about independent performance on L333.

## Final independent file checks

Both CSVs contain exactly 524 unique centers, in the same order and with identical prediction SHA256 values. Arithmetic means were independently recomputed from each CSV and match SUMMARY.json. Process exit code is 0.

All target slices span the complete clipped evaluation window. The actual official float32 mapped dynamic range is 255.00001525878906 on every slice, retained exactly without rounding or replacing it. Thus PSNR and RMSE agree to numerical precision; SSIM differs because the two requested implementations/windows differ. No inference was repeated for this check.

Git tracked/staged diffs remain empty. Only the new evaluator and this experiment output directory were added; pre-existing untracked work was preserved. No Git commit or push was performed for this evaluation. See FINAL_VALIDATION.json and ARTIFACT_MANIFEST.json.
