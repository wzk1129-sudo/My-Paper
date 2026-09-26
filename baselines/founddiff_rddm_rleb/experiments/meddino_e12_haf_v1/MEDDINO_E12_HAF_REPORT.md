# E12-HAF: SALF + hierarchical reassembly + hybrid anatomy fusion

Independent 6k run from canonical Direct base. E1 standalone 6k metric was 40.810964 dB, but its checkpoint was unavailable and was never used as E12 initialization.

| Arm | 2k | 4k | 6k | Delta 6k vs D1 | SSIM | RMSE HU | Soft PSNR | Params | Peak VRAM GiB | sec/step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D1_MULTI_512 | 40.797164 | 40.806988 | 40.813156 | +0.000000 | 0.948272 | 18.3154 | 28.500742 | 6776736 | 4.878 | 0.290 |
| E12_HAF | 40.796695 | 40.806565 | 40.811034 | -0.002122 | 0.948244 | 18.3196 | 28.499855 | 7385900 | 5.318 | 0.362 |

| Full L333 | PSNR | SSIM | RMSE HU | MAE HU | Soft PSNR | Inference s |
|---|---:|---:|---:|---:|---:|---:|
| D1_MULTI_512 | 40.821165 | 0.948359 | 18.2968 | 12.0647 | 28.505206 | 100.4 |
| E12_HAF | 40.819139 | 0.948336 | 18.3009 | 12.0618 | 28.504572 | 122.8 |

E12-D1 paired PSNR: mean -0.002025 dB, median -0.002268, positive slices 0.261, 95% slice-bootstrap CI [-0.002374, -0.001677].
Z thirds mean deltas: {'first': {'start': 0, 'end': 203, 'n': 204, 'mean_delta_psnr': -0.0005930225125230618}, 'middle': {'start': 204, 'end': 406, 'n': 203, 'mean_delta_psnr': -0.0022929627138477205}, 'last': {'start': 407, 'end': 609, 'n': 203, 'mean_delta_psnr': -0.0031970761540188177}}.

Hierarchy residuals nonzero: True. Attention and local conditioning residuals nonzero: True.
Training time ratio E12/D1: 1.25x; Full inference time ratio: 1.22x.
Attention entropy was not computed to avoid allocating dense attention maps.
L333 is development validation; slice-bootstrap CI is not cross-patient generalization.
No L506, external memory, neighbor NDCT, MedDINO fine-tuning, E3, 8-patient training, or Git push.
