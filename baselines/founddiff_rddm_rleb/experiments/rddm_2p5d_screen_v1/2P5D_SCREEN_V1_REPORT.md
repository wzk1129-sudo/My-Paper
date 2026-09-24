# RDDM 2.5D Context Fast Screen V1

Two independent additions to the frozen RDDM+RLEB Fast protocol. L067/L143 optimization; fixed L333 Fast64 EMA.

| Arm | Step | PSNR | Delta PSNR | SSIM | RMSE HU | MAE HU | Params | Peak VRAM GiB | s/step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2P5D_DIRECT_CONCAT | 2000 | 37.974800 | -0.503901 | 0.918596 | 25.3495 | 17.1814 | 13137217 | 4.172 | 0.3460 |
| 2P5D_DIRECT_CONCAT | 4000 | 40.204364 | +0.058978 | 0.942562 | 19.6254 | 12.8991 | 13137217 | 4.172 | 0.3357 |
| 2P5D_DIRECT_CONCAT | 6000 | 40.529617 | +0.142415 | 0.945690 | 18.9102 | 12.4073 | 13137217 | 4.172 | 0.3321 |
| 2P5D_DIRECT_CONCAT | 8000 | 40.657977 | +0.173409 | 0.946719 | 18.6386 | 12.2788 | 13137217 | 4.172 | 0.3302 |
| 2P5D_DIRECT_CONCAT | 10000 | 40.724556 | +0.202506 | 0.947337 | 18.4986 | 12.1914 | 13137217 | 4.172 | 0.3297 |
| 2P5D_DIRECT_CONCAT | 12000 | 40.767627 | +0.191121 | 0.947724 | 18.4085 | 12.1283 | 13137217 | 4.172 | 0.3295 |
| 2P5D_CONTEXT_BRANCH | 2000 | 38.756806 | +0.278106 | 0.929049 | 23.1787 | 15.5994 | 13276609 | 4.353 | 0.3404 |
| 2P5D_CONTEXT_BRANCH | 4000 | 40.182035 | +0.036650 | 0.942575 | 19.6747 | 12.9033 | 13276609 | 4.353 | 0.3404 |
| 2P5D_CONTEXT_BRANCH | 6000 | 40.519932 | +0.132730 | 0.945549 | 18.9329 | 12.4507 | 13276609 | 4.353 | 0.3403 |
| 2P5D_CONTEXT_BRANCH | 8000 | 40.641795 | +0.157227 | 0.946563 | 18.6734 | 12.2976 | 13276609 | 4.353 | 0.3404 |
| 2P5D_CONTEXT_BRANCH | 10000 | 40.704160 | +0.182109 | 0.947150 | 18.5414 | 12.2210 | 13276609 | 4.353 | 0.3409 |
| 2P5D_CONTEXT_BRANCH | 12000 | 40.749180 | +0.172674 | 0.947559 | 18.4485 | 12.1766 | 13276609 | 4.353 | 0.3413 |

## 2P5D_DIRECT_CONCAT

Status: COMPLETE; steps: 12000; promotion: PROMOTE_FULL_L333.
Training seconds: 3953.7; nonfinite: 0.
Neighbor parameter usage: {"prev_stem_weight_norm": 0.6067071557044983, "next_stem_weight_norm": 0.6093493103981018}.
Full L333 paired delta: {"psnr": 0.19781377387796795, "ssim": 0.0020388250628746284, "rmse_hu": -0.42498018864637505, "mae_hu": -0.22989085932246975}

## 2P5D_CONTEXT_BRANCH

Status: COMPLETE; steps: 12000; promotion: PROMOTE_FULL_L333.
Training seconds: 4095.2; nonfinite: 0.
Neighbor parameter usage: {"projection_weight_norms": [1.9084898233413696, 0.5545899271965027, 0.4460459053516388]}.
Full L333 paired delta: {"psnr": 0.17911392812286825, "ssim": 0.0018731839423823215, "rmse_hu": -0.38457257804186, "mae_hu": -0.1805328306604604}

L506, external-memory, and neighbor-target opens: 0, 0, 0.
Engineering screening only; no patient-level statistical significance claim.
