# V1@8k 与 MLSF@8k：Full-L333 匹配评价

使用同一 L333 全部 610 张切片、正式 8k EMA 和原验证函数。逐切片患者、中心/相邻索引及采样 seed 均逐项一致；以下差值为 MLSF − V1。

| 指标 | V1@8k | MLSF@8k | 差值 |
|---|---:|---:|---:|
| PSNR (dB) | 40.915844079 | 40.946909358 | +0.031065279 |
| SSIM | 0.948919638 | 0.949096333 | +0.000176695 |
| RMSE (HU) | 18.101864711 | 18.037720601 | -0.064144111 |
| MAE (HU) | 11.975785166 | 11.949061839 | -0.026723327 |

## 逐切片 PSNR

改善 607/610 张（99.51%）；配对差值均值 +0.031065279 dB，中位数 +0.029790030 dB。
差值分布：min -0.017930214，P5 +0.012505070，P25 +0.021895652，P75 +0.039472288，P95 +0.052858778，max +0.077595918 dB。

## 与 Fast64 对照

Fast64 的 PSNR 差值为 +0.031444759 dB；Full-L333 为 +0.031065279 dB。方向一致；两者差 -0.000379480 dB。Fast64 是固定 64 张子集，Full-L333 是完整 610 张，不能把两者当作独立患者重复。

## 权重、协议与运行时间

- V1：正式 MedDINO 条件模型，8k EMA：`/home/xujian/projects/CoreDiff_official_2p5d_meddino_condition_v1/experiments/rddm_official_2p5d_meddino_condition_v1/runs/OFFICIAL_RDDM_2P5D_MEDDINO_CONDITION_retry1/checkpoints/step_8000.pt`；已存 SHA256 `1ea09e9d0a4f2d56a636a62dc4a02afe80ff6342f506628077dd0003af748ce1`。
- MLSF：8k EMA：`/home/xujian/projects/CoreDiff_official_2p5d_meddino_condition_v1/experiments/rddm_mlsf_v1/runs/MLSF_RDDM_8K_20261009/checkpoints/step_8000.pt`；已存 SHA256 `3861477ec1edb3867c19e06e17ab34ae56918a1acf690dbbf916f2de41500bab`。
- V1 仅调用原 `postprocess.py` 的 `eval_checkpoint('OFFICIAL_RDDM_2P5D_MEDDINO_CONDITION', 'FULL', 'FULL_L333_V1_8K_MLSF_MATCH')`；MLSF 直接复用已完成的 `FULL_L333_RESULT.json` 和逐切片评价。
- 两者沿用原始预处理、[-1000,1000] HU 评价窗、PSNR data_range=2000 HU、SSIM/RMSE/MAE 函数、sampling_timesteps=2 和每张切片固定随机种子。
- V1 Full 验证 389.215 s；MLSF Full 验证 383.886 s。

## 数据访问审计

- V1：L333 中心 NDCT 610 次，中心 LDCT 612 次，邻近 LDCT 1218 次。
- MLSF：L333 中心 NDCT 610 次，中心 LDCT 612 次，邻近 LDCT 1218 次。
- 两边的 L506、外部库、训练目标、邻近 NDCT 及阻断访问计数均为 0。

本比较仅覆盖 L333 一个验证患者；逐切片分布不能当作跨患者独立性证据。
