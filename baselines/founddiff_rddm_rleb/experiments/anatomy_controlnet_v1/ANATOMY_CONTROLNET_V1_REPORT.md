# RDDM Anatomy-ControlNet V1

Frozen trained 2.5D Direct 12k EMA main; L067/L143 training and fixed L333 Fast64 validation.
No full eight-patient training or L506 access.
The per-arm model field inherited from the frozen 2D Fast-dev template is metadata only; the actual 4-channel stem and parameter counts are specified in EXPERIMENT_CONFIG.json actual_model_by_arm and PARAMETER_AUDIT.json.

| Arm | Condition | Trainable params | Peak VRAM GiB | 2k PSNR | 4k PSNR | 6k PSNR | Delta vs C0 | Delta vs C1 | Soft PSNR | RMSE HU |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | Direct checkpoint | 0 | n/a | n/a | n/a | 40.76763 | 0 | n/a | 28.45386 | 18.40853 |
| C1 | C1_CAPACITY | 6088384 | 4.136 | 40.77147 | 40.77505 | 40.77864 | +0.01101 | 0 | 28.46530 | 18.38618 |
| C2 | C2_EDGE | 6190880 | 4.204 | 40.77153 | 40.77506 | 40.77889 | +0.01126 | +0.00024 | 28.46543 | 18.38566 |

Observed label: **CAPACITY_ONLY_GAIN**. 4k/6k edge vs capacity: +0.00001 / +0.00024 dB.
C2 positive slice fraction vs C1 at 6k: 0.766.
Control activity at 6k: True; copied control encoder and all four ZeroConvs changed after optimization (see CONTROL_PARAMETER_AUDIT.json).
C2 has 102496 additional trainable edge-provider parameters vs C1; attribution is limited by this small capacity difference.
C3/C4 were not run: the local MedDINOv3 checkpoint SHA matches the author release, but official model source is absent on this server; exact source commit, input size, layer mapping and runtime compatibility remain unverified. See MEDICAL_ENCODER_AUDIT.json.
Recommended future anatomy encoder: [MedDINOv3 ViT-B/16 CT-3M](https://huggingface.co/ricklisz123/MedDINOv3-ViTB-16-CT-3M) with [official source](https://github.com/ricklisz/MedDINOv3). Pin the source commit and input resolution, then profile memory before C3. C2 alone used 4.204 GiB; the anatomy path adds at least about 0.32 GiB FP32 weights plus unmeasured attention workspace.
The first C1 smoke attempt stopped at the frozen data-config guard before any optimizer step or image open; the corrected independent 20-step C1/C2 smokes passed and are archived.
Main weights and original Direct checkpoint remained unchanged; no optimizer step was taken for C0.
No data or checkpoint is included in this lightweight archive.
Promotion to full eight-patient training: not supported by this Fast64 screen.
