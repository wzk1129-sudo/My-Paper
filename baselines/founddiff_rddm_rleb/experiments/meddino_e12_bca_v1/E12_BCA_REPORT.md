# E12-BCA Fast6k / Full-L333

E12 topology: frozen MedDINO 2/5/8/11, SALF, hierarchical reassembly.
The actual E12 source has no Edge branch; no new extractor was added.
Q comes from RDDM encoder; K/V come from E12 structure at 128/64.
RDDM main was trainable here, unlike frozen D1/E12-HAF; comparison is therefore a joint architecture/training-parameter change.

## Full L333 (610 slices, EMA)

| Arm | PSNR | SSIM | RMSE HU | MAE HU |
|---|---:|---:|---:|---:|
| E12-BCA | 40.856054 | 0.948571 | 18.225127 | 12.011960 |
| D1 | 40.821165 | 0.948359 | 18.296843 | 12.064741 |
| E12-HAF | 40.819139 | 0.948336 | 18.300924 | 12.061774 |

## Paired comparisons

- BCA - D1: delta PSNR +0.034890 dB; paired slice bootstrap 95% CI [+0.033680, +0.036089] dB; positive slice fraction 0.998.
- BCA - E12_HAF: delta PSNR +0.036915 dB; paired slice bootstrap 95% CI [+0.035709, +0.038126] dB; positive slice fraction 1.000.

alpha128=0.202174, alpha64=0.161799.
Train step time 0.360s; ratios vs D1 1.24x, vs E12-HAF 0.99x.
Full inference 115.7s; peak VRAM 0.81 GiB.
No training during Full-L333; L506 and external memory untouched.
