# CoreDiff 官方 150k EMA：L506 双评价协议

本分支保存官方 Mayo 2016 **25% dose / 150000-step EMA checkpoint** 的一次只读推理评估，以及用于复现和审计的轻量代码与记录。

## 结果

两套协议复用完全相同的预测；对每张有效 center slice 只调用一次官方 sampler。表中为 **524 张逐 slice 指标的算术平均**。

| 协议 | PSNR (dB) | SSIM | RMSE (HU) | MAE (HU) |
|---|---:|---:|---:|---:|
| OFFICIAL | 41.60937266 | 0.95650445 | 16.91536899 | 未要求 |
| UNIFIED | 41.60937146 | 0.95580814 | 16.91536520 | 11.23689846 |

- [完整报告](experiments/corediff_official_checkpoint_eval/REPORT.md)
- [OFFICIAL 逐 slice 指标](experiments/corediff_official_checkpoint_eval/OFFICIAL_METRICS.csv)
- [UNIFIED 逐 slice 指标](experiments/corediff_official_checkpoint_eval/UNIFIED_METRICS.csv)
- [结构化汇总](experiments/corediff_official_checkpoint_eval/SUMMARY.json)

## 数据与推理条件

- Mayo 2016 original DICOM preprocessing，25% dose，官方 test_id=9，仅 L506。
- LDCT/NDCT 各 526 张；官方 dataset 排除 000、525 作为中心，评价 001–524，共 524 张。边界两张仍用于邻片 context。
- context=True，prev/center/next 三张 LDCT，batch=1，T=10，sampling_routine=ddim（官方默认），n_iter=150000，start_adjust_iter=1。
- 官方 normalization：stored pixel → subtract 1024 → clip [-1024,3072] HU → map [0,1]。
- only_adjust_two_step=True 是官方训练 forward 的设置；官方 sample 没有此参数，评估保持 sample 原样。
- checkpoint 严格加载成功，84 个 state entries、4,825,953 个网络参数。
- checkpoint、加载后的模型状态和官方源码在推理前后哈希不变；训练、optimizer、backward、OSL 次数均为 0。未读取 L333 或其他患者文件。
- 不将 L333 作为这个 checkpoint 的独立测试集。

## 两套评价协议

**OFFICIAL**：直接调用原有 TrainTask.transfer_calculate_window 与 utils.measure.compute_measure。预测/target 从 [0,1] 还原 HU，clip [-1000,1000]，map [0,255]；每张 slice 的 data_range=target.max()-target.min()。SSIM 为官方 Gaussian 11×11、sigma=1.5、官方 padding。官方 compute_RMSE 内部还原 HU，结果单位为 HU。

**UNIFIED**：同一 prediction tensor 还原 HU 并 clip [-1000,1000]，不映射 [0,255]；固定 data_range=2000。PSNR=10*log10(2000²/MSE_HU)，RMSE=sqrt(MSE_HU)，MAE=mean(abs(error))。SSIM 使用 skimage.metrics.structural_similarity，win_size=7、gaussian_weights=False、use_sample_covariance=True、K1=0.01、K2=0.03。

两份 CSV 保存相同的逐 slice prediction SHA256；预测只保留在内存，没有导出影像数组。全部 target 覆盖完整窗口，官方 float32 实际 range=255.00001525878906，保持原值；因此 PSNR/RMSE 仅有浮点差异，SSIM 差异来自评价实现和窗口。没有调参或为匹配论文数字改动协议。

## 代码与来源

- [实际评估脚本](evaluate_official_checkpoint_dual_protocol.py)
- models/、utils/、main.py、test_mayo2016.sh、data_preprocess/prep_mayo2016.py 为实际运行工程中的轻量源码快照。
- [官方 CoreDiff](https://github.com/qgao21/CoreDiff)，核心 8 个源码文件与官方 commit **9bc31dc1d6184a880bd6334db34c466569a3642a** 逐字节一致。
- 实际评估工程 Git commit：**b754bcbb384ba087a15db4f4f4f9c54fbb4bfc6d**。本 GitHub 发布提交与评估时工程 commit 分别记录。
- [官方 README 快照](docs/OFFICIAL_COREDIFF_README.md)
- [源码审计](experiments/corediff_official_checkpoint_eval/SOURCE_AUDIT.json)

## Checkpoint

复用远程已有官方测试目录文件，未覆盖已有文件。先检索本机与远程候选，再验证官方公开 payload 的 SHA256；两份本地复现训练权重因 SHA 不同被排除。

官方发布文件：ema_model-150000，19,337,897 bytes。

SHA256：
~~~
16e09d7b13a01e360e064194c3c5c334b558c4b12ce260d4c39f5ff4ad865b39
~~~

[官方发布目录](https://drive.google.com/drive/folders/1rGb34H_6ktP79vMYYJOLSoCE3579TDZ5)

[Checkpoint 审计](experiments/corediff_official_checkpoint_eval/CHECKPOINT_AUDIT.json) 包含候选路径、大小、SHA、来源核验与严格 loader 检查。本分支不包含 checkpoint。

## 执行与复现

实际命令、tmux 会话、数据位置和环境版本在 RUN_CONFIG.json、LAUNCH_AUDIT.json、ENVIRONMENT.json 中完整记录。实际环境为 Python 3.8.13、PyTorch 1.10.1+cu111、CUDA runtime 11.1、RTX 3090；推理及评分耗时约 149.34 秒，进程退出码 0。

复现时需自行提供合法取得的原始 Mayo 2016 L506 DICOM、按官方 prep_mayo2016.py 生成的 stored-pixel NPY，以及上述 SHA256 的官方 EMA 权重。评估脚本直接使用当前工程 dataset 和 sampler；只选择 L506，不运行 main.py 的训练入口。

该脚本记录的是原始运行环境，NPY 使用仓库相对路径 data_preprocess/gen_data/mayo_2016_npy；DICOM spot-check 使用 /home/xujian/data/Mayo2016/L506/{quarter_1mm,full_1mm}。复现环境须提供相应数据路径。它在两个 series 的首、中、末各检查一次 DICOM stored-pixel 与现有 NPY 的逐值相等性，不重新写数据。

为保留这次输出，使用新的空输出目录，放入这次 PRECHECK_PROVENANCE.json 的副本，并更新副本的 command/tmux_session 为复现实际命令。随后从工程根目录执行：
~~~bash
CUDA_VISIBLE_DEVICES=0 python -u evaluate_official_checkpoint_dual_protocol.py \
  --checkpoint /path/to/verified/ema_model-150000 \
  --output /path/to/new/empty-output-directory
~~~
已有 SUMMARY.json 会使脚本拒绝覆盖。依赖版本以 ENVIRONMENT.json 为准；checkpoint SHA、核心源码 SHA、split/context 和模型结构不匹配时会停止。

## 审计文件索引

- [实际配置](experiments/corediff_official_checkpoint_eval/RUN_CONFIG.json)
- [环境](experiments/corediff_official_checkpoint_eval/ENVIRONMENT.json)
- [数据与边界审计](experiments/corediff_official_checkpoint_eval/DATA_PROTOCOL_AUDIT.json)
- [数据访问与只读审计](experiments/corediff_official_checkpoint_eval/READ_ONLY_AUDIT.json)
- [最终 CSV/均值/状态核验](experiments/corediff_official_checkpoint_eval/FINAL_VALIDATION.json)
- [推理前来源记录](experiments/corediff_official_checkpoint_eval/PRECHECK_PROVENANCE.json)
- [运行日志](experiments/corediff_official_checkpoint_eval/run.log)
- [原始运行产物清单](experiments/corediff_official_checkpoint_eval/ARTIFACT_MANIFEST.json)
- [本次发布文件清单](PUBLICATION_MANIFEST.json)

REPORT 中“未提交或推送”描述的是推理结束时的 Git 快照；本分支是随后由用户授权发布的结果包。原始结果与审计文件保持原样。没有上传 checkpoint、DICOM、医学 NPY、预测影像或其他大文件。
