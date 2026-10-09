# MLSF-RDDM V1：正式 MedDINO V1 的匹配 8k 消融

本目录保存实际使用的 MLSF 源码、配置、标量审计和评价结果。正式 MedDINO V1 的源码在同一分支的 [V1 实验目录](../rddm_official_2p5d_meddino_condition_v1/)；本分支从该 V1 分支建立。V1 源码的 Git 原始字节哈希与本次 SOURCE_PROVENANCE.json 一致。

## 方法

固定 Official RDDM 残差扩散、2.5D LDCT 输入、冻结的 MedDINOv3 L2/L5/L8/L11 Adapter/Fusion、9 个原生 Attention 与 9 个 V1 FiLM 位点，仅加入：

1. local_detail_encoder.py 从相邻和中心 LDCT 构建 256/128 分辨率局部特征。
2. local_cross_attention.py 在每个 128×128 查询位置对齐读取 MedDINO L8/L11 的 3×3 token 邻域，即每个查询 9 个位置；没有全局 attention 矩阵。
3. mlsf_model.py 在 E1/E2/D1/D2 四个 FiLM 位点加入零初始化的 gamma/beta 修正；step 0 与正式 V1 输出完全相同。原 9 个 Attention 和其他 5 个 FiLM 位点保留。

新增可训练参数 336,032 个。没有额外损失、额外训练组、外部检索库或邻近切片 NDCT 条件。

## 匹配训练与验收

从同一原始 Official 2D@8k EMA 父权重初始化；训练 L067/L143，seed 20260829、batch 2、accumulation 1、Adam lr 2e-4 / betas (0.9,0.99) / weight decay 1e-9、EMA 0.995、无 AMP、grad clip 1。目标 pred_res，T=1000，采样步数 2；在 L333-fast64 的 0/1k/2k/4k/6k/8k 节点评价 EMA。

MODEL_AUDIT.json 核对 498 个共有状态张量、9 个原生 Attention 和 step-0 恒等性；READINESS.json 记录真实 batch=2 梯度与显存验收；V1_STREAM_REFERENCE.json、STEP0_AUDIT.json 和最终状态记录训练流一致性。配置、源码哈希与命令见 BATCH_CONFIG.json、RUN_MANIFEST.json、SOURCE_PROVENANCE.json。

| L333-fast64 EMA | V1 PSNR dB | MLSF PSNR dB | 差值 dB |
|---:|---:|---:|---:|
| 1k | 40.678683 | 40.818798 | +0.140115 |
| 2k | 40.775686 | 40.863531 | +0.087845 |
| 4k | 40.862841 | 40.900376 | +0.037535 |
| 6k | 40.892207 | 40.924381 | +0.032174 |
| 8k | 40.908237 | 40.939682 | +0.031445 |

完整 PSNR/SSIM/RMSE/MAE 轨迹见 VALIDATION_COMPARISON.csv；训练关键日志见 logs/train.log；Fast64 逐切片配对差值见 PER_SLICE_DELTA.csv。逐步训练标量大表保留在远程原实验目录，未纳入此轻量发布包。

## Full-L333 匹配评价

8k Fast64 达到预设门槛后运行一次 MLSF 全量评价。随后只调用原 V1 postprocess.py 的 eval_checkpoint("OFFICIAL_RDDM_2P5D_MEDDINO_CONDITION", "FULL", "FULL_L333_V1_8K_MLSF_MATCH") 补充正式 V1@8k EMA 的 610 张评价；没有重训或重跑 MLSF。两边逐张的患者、切片索引及采样 seed 严格一致。

| Full-L333, 610 张 | V1@8k | MLSF@8k | MLSF − V1 |
|---|---:|---:|---:|
| PSNR dB | 40.915844 | 40.946909 | **+0.031065** |
| SSIM | 0.948920 | 0.949096 | +0.000177 |
| RMSE HU | 18.101865 | 18.037721 | −0.064144 |
| MAE HU | 11.975785 | 11.949062 | −0.026723 |

607/610 张切片 PSNR 提升，中位配对差值 +0.029790 dB。Fast64 的 +0.031445 dB 与 Full 的 +0.031065 dB 方向一致。精确数值和方法见 FULL_L333_MATCHED_COMPARISON.csv / .md。两套原始标量结果分别在相邻 V1 目录的 FULL_L333_V1_8K_MLSF_MATCH.json 和本目录的 FULL_L333_RESULT.json；两套逐切片评价指标文件也保留，不含像素数组。

## 阅读顺序与边界

先读本文件、REPORT.md、FULL_L333_MATCHED_COMPARISON.md；再看 mlsf_model.py、local_detail_encoder.py、local_cross_attention.py、train_mlsf.py、evaluate_mlsf.py、postprocess_mlsf.py。原 V1 的 model.py、med_loader.py、context_data.py、evaluate.py 在相邻 V1 目录。write_full_l333_match.py 是只读配对汇总脚本。

V1 8k EMA checkpoint SHA256 为 1ea09e9d0a4f2d56a636a62dc4a02afe80ff6342f506628077dd0003af748ce1；MLSF 8k EMA 为 3861477ec1edb3867c19e06e17ab34ae56918a1acf690dbbf916f2de41500bab。权重、患者数据、影像、医学数组、特征缓存及预测张量未发布，复现 GPU 推理需另行提供私有依赖。DATA_ACCESS_AUDIT.json 中 L506、外部库、邻近 NDCT 访问数均为 0。

结果仅来自 L333 一个验证患者；逐切片比例不构成跨患者显著性证据。
