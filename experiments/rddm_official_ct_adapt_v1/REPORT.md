# Official RDDM restoration backbone → Mayo CT：matched Fast-Dev

本目录包含能够复核代码与结果的完整实验版本。只替换 U-Net backbone；训练、扩散、采样、CT 预处理和评价沿用 matched baseline。官方来源为 [nachifur/RDDM](https://github.com/nachifur/RDDM) commit `524caea34cedb3733aa5bf6c27dc1cd6e0469216` 的 `experiments/2_Image_Restoration_deraing_raindrop_noise1/src/residual_denoising_diffusion_pytorch.py`。官方 clone 未修改。

## 代码入口与结构差异（A–C）

- `src/rddm_rleb.py`：既有 simplified RDDM/RLEB Control 的固定副本，13,130,945 参数。
- `src/official_rddm_ct.py`：官方 restoration U-Net 源码加相同 CT diffusion wrapper 的 Candidate，36,248,513 参数（+23,117,568；2.7605 倍）。`topology_audit.json` 证明 Candidate U-Net 与官方 AST 一致，Candidate 的 `UnetRes` 和 `ResidualDiffusion` 与 Control 的 AST 一致；`SOURCE_PROVENANCE.json` 给出源码哈希。
- `matched_train.py`：两臂共用的数据顺序、训练、EMA 与记录入口；`readiness.py`：B=2 全分辨率结构和梯度审计；`run_queue.sh`：严格各 8k 的顺序运行命令；`collect_results.py`：核验日志并产生结果表。

| 位置 | Simplified Control | Official Candidate |
|---|---|---|
| Encoder 每级 ×4 | 1 ResnetBlock、1 skip、无 Attention | 2 ResnetBlock、2 skip、LinearAttention |
| Middle | 1 ResnetBlock | ResnetBlock → full Attention → ResnetBlock |
| Decoder 每级 ×4 | 1 ResnetBlock、1 skip、无 Attention | 2 ResnetBlock、2 skip、LinearAttention |
| 每个 ResnetBlock 内 | 1 个卷积 Block | 官方 2 个卷积 Block |
| 末端 | stem skip → final ResnetBlock → Conv1×1 | 同一末端拓扑，恢复官方内部双卷积 Block |

Control 有 10 个 ResnetBlock、0 个 Attention；Candidate 有 19 个 ResnetBlock、8 个 LinearAttention、1 个 middle full Attention。Downsample/Upsample、基本 Block 与时间位置编码的共享实现经 AST 比对一致。

## 固定协议

Train：L067、L143；Validation：固定 L333-fast64；不访问 L506 或 external memory。单通道配对 Mayo 2016 LDCT/NDCT，512×512，LDCT 为 condition。存储值偏移 −1024 HU，截断 [−1024,3072] HU 并线性归一化到 [0,1]；指标使用 HU window [−1000,1000]，PSNR range 2000 HU，同时给出 SSIM、RMSE HU、MAE HU。

两臂均为 `pred_res`、residual `LDCT−NDCT`、相同 `q_sample`、1000 timesteps、2-step sampler、`sum_scale=0.01`、Adam lr `2e-4` / betas `(0.9,0.99)` / wd `1e-9`、batch 2、EMA 0.995、AMP=False、grad clip 1.0、seed 20260829。没有双头、额外先验或新 loss。验证使用 EMA 权重。

## EMA Fast64 结果（D–E）

完整精度和四项 Candidate−Control delta 见 [fast_dev_metrics.csv](results/fast_dev_metrics.csv)。RMSE、MAE 单位为 HU；ΔPSNR = Candidate−Control。step 0 是未训练权重。

| Step | Control PSNR / SSIM / RMSE / MAE | Candidate PSNR / SSIM / RMSE / MAE | ΔPSNR |
|---:|---|---|---:|
| 0 | 13.0154 / 0.1083 / 447.0620 / 386.1935 | 8.5760 / 0.0213 / 745.2883 / 687.7655 | −4.4394 |
| 2k | 38.4787 / 0.9260 / 23.9292 / 15.7929 | 40.0288 / 0.9405 / 20.0238 / 13.0317 | +1.5501 |
| 4k | 40.1454 / 0.9416 / 19.7640 / 13.0245 | 40.4472 / 0.9445 / 19.0939 / 12.4888 | +0.3018 |
| 6k | 40.3872 / 0.9440 / 19.2251 / 12.5807 | 40.5665 / 0.9455 / 18.8399 / 12.3635 | +0.1793 |
| 8k | 40.4846 / 0.9449 / 19.0149 / 12.4597 | 40.6234 / 0.9461 / 18.7191 / 12.3019 | +0.1388 |

8k 的其余 delta 为 SSIM +0.001186、RMSE −0.295735 HU、MAE −0.157858 HU。Candidate 在固定 64/64 个 L333-fast64 slice 上四项指标均同向改善；逐 slice PSNR 增量为 +0.0739 至 +0.2899 dB，见 [paired_slice_8k_audit.json](results/paired_slice_8k_audit.json)。

## Readiness、Attention 与数据边界（F–G）

[readiness.json](readiness.json) 记录 4 级 encoder/decoder、28 个 ResnetBlock/Attention hook 在 B=2 forward 中各调用一次，输出形状正确，backward 有限；9 个 Attention 模块参数梯度有限且非零，timestep 与 LDCT condition sensitivity 非零。合成 readiness 显存峰值 18.53 GiB。真实训练中 Candidate 每步 attention gradient norm 均为有限正值（范围 1.71e−5 至 0.1796），Control 为 0；两臂各 81 次参数有限性审计全部通过。实际训练峰值为 Control 4.16 GiB、Candidate 18.83 GiB。每步 loss、attention gradient、step time、VRAM 等见 [training_diagnostics.csv](results/training_diagnostics.csv)。

首个真实 batch 的样本及 augmentation、timestep `[102,842]`、noise SHA-256 和训练前后 RNG SHA-256 在两臂完全一致，见 [matched_readiness_audit.json](matched_readiness_audit.json) 与 [audit_summary.json](results/audit_summary.json)。DataAccessGuard 实际文件打开记录只包含 L067、L143、L333-fast64；两臂均为 L506 0、external memory 0、其他 test 0、blocked attempts 0。未做 Full-L333。最终两个 step-8000 checkpoint 已成功加载，`optimizer_step=ema_updates=8000`。

## 结论（H）

**短线结果支持“过度简化 backbone 可能压低本次 RDDM-CT baseline”的判断**：官方 restoration 主干在 2k、4k、6k、8k 的四项匹配指标均优于 Control。但优势到 8k 缩至 +0.1388 dB，因此不能据此断言先前所有性能差距均由 backbone 造成，也不能拆分额外 ResBlock、内部第二卷积、Attention 或 skip 的独立贡献。只有单 seed、一个验证患者的 64 slice 与 8k 预算；64 个相关 slice 的同向结果不是独立患者层面的显著性证据。

## 运行与工件

环境版本见 [ENVIRONMENT.json](ENVIRONMENT.json)，原始启动信息见 [RUN_MANIFEST.json](RUN_MANIFEST.json)。在本目录使用 corediff 环境执行 `python readiness.py`，随后由远程 `tmux` 运行 `bash run_queue.sh`，完成后执行 `python collect_results.py --output results_recheck`。本轮真实训练命令、输出、日志和 GPU 编号均记录在启动信息中。checkpoint 和原始日志保存在 Git 忽略的 `runs/`、`logs/`；提交内容只包含代码、配置、审计、汇总指标和报告，不包含 CT 图像或 checkpoint。
