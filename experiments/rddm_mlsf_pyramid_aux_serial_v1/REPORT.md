# MLSF-RDDM：Pyramid + SWT / MS-SSIM 8k 对照实验

> **记录性质**：2026-10-20 将远程实验结束摘要归档到 GitHub。本文各项实验完成情况、数值和远程路径**来自实验汇报**，并非本次 GitHub 提交时独立重新运行的结果。原始训练日志、完整远程 `REPORT.md`、checkpoint 和逐节点详细指标**未同步至本分支**。

## 1. 实验概览

- **任务**：低剂量 CT（LDCT）恢复；以原 MLSF-RDDM 为基础，对比 Pyramid、Pyramid+SWT 和 Pyramid+MS-SSIM。
- **验证**：L333-Fast64，EMA；下表为 **8,000 optimizer steps** 最终结果。
- **完成状态**：A（+SWT）和 B（+MS-SSIM）均完成 8k 步及各自全部四次验证。
- **训练流**：A、B 均报告与匹配训练流一致。
- **MedDINO**：冻结，未更新。
- **数据访问**：L506 访问次数为 0。
- **训练结束状态**：队列已退出，GPU 已释放。
- **成果保存情况**：A、B 每组四个 checkpoint 和独立报告均已保存在远程实验目录；本分支暂仅存结果摘要和 CSV。

## 2. 8k EMA L333-Fast64 结果

| 模型 | PSNR (dB) | SSIM | RMSE (HU) | MAE (HU) | 相对仅 Pyramid 的 PSNR (dB) |
|---|---:|---:|---:|---:|---:|
| 原 MLSF | 40.939682 | 0.949028 | 18.054190 | 11.959569 | -0.017154 |
| MLSF + Pyramid | 40.956836 | 0.949182 | 18.018937 | 11.925059 | 0.000000 |
| A：MLSF + Pyramid + SWT | 40.956532 | 0.949187 | 18.019565 | 11.925177 | **-0.000304** |
| B：MLSF + Pyramid + MS-SSIM | **40.960498** | **0.949264** | **18.011197** | **11.921201** | **+0.003662** |

独立的机器可读结果在 [RESULTS_8K_EMA_FAST64.csv](./RESULTS_8K_EMA_FAST64.csv)。

## 3. 配对比较与结论

- **Pyramid 相对原 MLSF**：PSNR **+0.017154 dB**。
- **A（额外 SWT）相对仅 Pyramid**：PSNR **-0.000304 dB**；因此这次实验中 SWT **没有额外 PSNR 收益**。展示到 6 位小数的 SSIM 为 0.949187 vs. 0.949182。
- **B（额外 MS-SSIM）相对仅 Pyramid**：PSNR **+0.003662 dB**；SSIM 由 0.949182 提高到 0.949264。远程摘要给出的未完全展示精度的 SSIM 增量为 **+0.00008228**（根据表中 6 位小数相减得到 +0.000082）。
- **B 相对原 MLSF**：PSNR **+0.020816 dB**。

**谨慎结论**：MS-SSIM 在本次 **L333-Fast64** 评价中带来很小的正向收益，SWT 没有带来额外 PSNR 收益。这些数值**不代表** Full-L333 或独立测试集 L506 的结果，也不能单凭该差值声称具有统计显著性。

## 4. 原始资料与完整性说明

- 远程汇总报告（**未通过本次 GitHub 写入流程读取或上传**）：
  `/home/xujian/projects/CoreDiff_official_2p5d_meddino_condition_v1/experiments/rddm_mlsf_pyramid_aux_serial_v1/REPORT.md`
- 原 MLSF 实现及基线记录：本分支继承自 `experiment/rddm-mlsf-v1`，目录为 `experiments/rddm_mlsf_v1/`。
- 本提交**不修改**原 MLSF 源码、原 checkpoint、原实验结果或其他分支。
- Pyramid/SWT/MS-SSIM 的本次实际训练源码、全部 checkpoint 与训练日志没有包含在当前新建的结果文件中；要做可复现性审计，应后续从远程同步相应代码、配置、文件哈希与运行产物。

记录日期：2026-10-20。
