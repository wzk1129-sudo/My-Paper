# Official RDDM + 2.5D + MedDINOv3 Anatomy Condition V1

两臂从此前 Official RDDM 2D 8k EMA 初始化，再分别进行 8k 个 fresh-Adam continuation steps。Official RDDM ResBlock、8 个 LinearAttention、middle Full Attention 和双 skip topology 全部保留；diffusion、pred_res MSE、sampler 与评价函数原样复用。

## 起点与实现

2D 8k 起点 EMA Fast64：PSNR 40.623374 dB；SSIM 0.946056；RMSE 18.719123 HU；MAE 12.301865 HU。
2.5D stem 通道为 [x_t, prev LDCT, center LDCT, next LDCT]；两邻居权重初始为零，旧 x_t/center 权重和 bias 完整复制。逐患者相邻 slice、边界复制及固定 Fast64 manifest 来自已验证 Direct 实验，context_data.py 字节一致。
MedDINO ViT-B/16 只接收 center LDCT（HU clip [-1000,1000]，mean 65.1084213256836，std 178.01663208007812，repeat 到三通道），官方 get_intermediate_layers(n=[2,5,8,11],norm=True,return_class_token=False,return_extra_tokens=False)。四个 768→64 独立 1×1 adapter 后 concat，256→128 1×1 + SiLU 得到共享 anatomy source。
九个独立 residual affine modulation：E0/E1/D2/D3 global；E2/E3/MID/D0/D1 spatial。GN groups=8、affine=False，最后 gamma/beta projection zero-init；没有 Cross-Attention、额外 gate、sigmoid/tanh 或新 loss。每处均在原生 Attention 前，原生 Attention 参数和代码保持原样。
实际 batch=2、accum=1，effective batch=2。Adam lr 2e-4，betas=(0.9,0.99)，wd=1e-9；EMA=0.995；clip=1；AMP=False；seed=20260829。训练 L067/L143；评价 L333 Fast64，HU window [-1000,1000]，PSNR data_range=2000 HU。
代码入口：model.py、med_loader.py、train.py、evaluate.py、readiness.py、postprocess.py；官方 diffusion 完整副本为 src/official_rddm_ct.py。源码、父 checkpoint 与 loader/配对实现身份见 SOURCE_PROVENANCE.json；运行环境和命令见 ENVIRONMENT.json、RUN_MANIFEST.json。

## Matched EMA Fast64

| Step | Control PSNR / SSIM / RMSE HU / MAE HU | Candidate PSNR / SSIM / RMSE HU / MAE HU | ΔPSNR dB |
|---:|---|---|---:|
| 0 | 40.623375 / 0.946056 / 18.719123 / 12.301865 | 40.623375 / 0.946056 / 18.719123 / 12.301865 | +0.000000 |
| 1000 | 40.678378 / 0.946600 / 18.601160 / 12.228678 | 40.678683 / 0.946614 / 18.600489 / 12.237826 | +0.000305 |
| 2000 | 40.768102 / 0.947583 / 18.410287 / 12.121031 | 40.775686 / 0.947631 / 18.394970 / 12.140861 | +0.007585 |
| 4000 | 40.853634 / 0.948401 / 18.231005 / 12.023502 | 40.862841 / 0.948459 / 18.212387 / 12.040945 | +0.009207 |
| 6000 | 40.872903 / 0.948539 / 18.190771 / 12.007369 | 40.892207 / 0.948718 / 18.151863 / 11.996179 | +0.019305 |
| 8000 | 40.889946 / 0.948667 / 18.156012 / 11.984496 | 40.908237 / 0.948847 / 18.119234 / 11.986737 | +0.018291 |

8k ΔPSNR=+0.018291 dB；PSNR 改善 64/64 slices（100.00%）。完整四项 delta 和逐 slice 结果见 VALIDATION_HISTORY.csv、PER_SLICE_DELTA.csv。

## Condition 实际强度（8k EMA，两次 sampler forward ×64 slices 的均值）

| Stage | delta RMS | base RMS | mean delta/base | gamma RMS | beta RMS |
|---|---:|---:|---:|---:|---:|
| E0 | 0.069468 | 0.872437 | 0.079626 | 0.0324945 | 0.0518283 |
| E1 | 0.126291 | 0.935212 | 0.135244 | 0.0584245 | 0.0884958 |
| E2 | 0.15944 | 0.961735 | 0.165806 | 0.0776232 | 0.118787 |
| E3 | 0.0122609 | 0.990804 | 0.0123734 | 0.00559631 | 0.00890443 |
| MID | 0.004804 | 0.688242 | 0.00697993 | 0.00429151 | 0.00253577 |
| D0 | 0.0124679 | 0.923087 | 0.0135087 | 0.00664094 | 0.00665393 |
| D1 | 0.119926 | 0.952944 | 0.125825 | 0.0663934 | 0.0716601 |
| D2 | 0.0739348 | 0.870272 | 0.0847274 | 0.0705565 | 0.0264533 |
| D3 | 0.0232259 | 0.993251 | 0.0233746 | 0.0274688 | 0.0139652 |

全部节点见 CONDITION_ACTIVITY.csv。Native Attention 每个训练 step 的聚合梯度有限且非零，详见 TRAINING_DIAGNOSTICS.csv；readiness 对九处实际 forward 调用以及 Attention 输入为 condition 输出进行了 hook/指针验证。
readiness：2D→2.5D max_abs_diff=0；Control→Candidate step0 max_abs_diff=0。九处 step0 delta RMS=0；首 backward 九个 projection 梯度有限非零；optimizer step 后第二 backward 四个层 adapter、fusion 和九个 stage adapter 梯度有限非零。MedDINO grad=None、requires_grad=False、eval，训练后权重哈希不变。
两臂全部 8000 step 的样本、augmentation、timestep 和 noise stream hash 一致；first-batch 完整证据见 DATA_ACCESS_AUDIT.json。readiness GPU 峰值 17.693 GiB，训练峰值 Control 16.296 GiB、Candidate 17.835 GiB。

## 数据边界与 promotion

L506 access=0；external memory access=0；neighbor NDCT access=0；blocked attempts=0。guard 按 patient/index/view 审核实际打开，监督仅 center NDCT。所有数据访问证据见 DATA_ACCESS_AUDIT.json。
Full-L333 gate（ΔPSNR≥0.03 dB 且改善≥60% slices）：未通过，未执行 Full-L333。

## 免费推理诊断

| Mode | PSNR dB |
|---|---:|
| NORMAL | 40.908237 |
| CONDITION_OFF | 40.762979 |
| SHUFFLED_MEDDINO | 40.902731 |
CONDITION_OFF 将九处 delta 强制为零；SHUFFLED_MEDDINO 在固定 Fast64 的相邻二元 batch 中交换 frozen anatomy maps，RDDM 输入和逐 slice sampling seed 不变。诊断差值用于观察实际依赖程度，不改变 promotion 结果。

## 结果判断

本次 matched Fast64 的 Candidate 获得正增益。结论仅针对单 seed、一个验证患者和这次 8k continuation。Condition 活动非零证明模块参与计算；是否改善重建由指标判断。若 NORMAL 与 OFF/SHUFFLED 差异很小，不能仅依据模块活动宣称 patient-specific anatomy 被有效利用。
checkpoint、CT 图像、DICOM、NPY 医疗数据、MedDINO checkpoint 和训练缓存均不进入 Git。checkpoint 保存在忽略的 runs/<arm>/checkpoints；原始日志在 logs/ 和 runs/<arm>/train.jsonl。

## 运行恢复审计

Control 已完成后，Candidate 在训练 step 0 前的 MedDINO state hash 审计中因 BFloat16→NumPy 不支持而退出。修复仅将哈希数据读取改为 contiguous uint8 byte view；train main AST、所有 model/readiness/diffusion 依赖不变，float32 shared-parameter hash 与已完成 Control 的初始化完全一致。HASH_FIX_AUDIT.json 记录 dtype 兼容、实际 MedDINO hash、RNG 不变及初始参数匹配证明。Candidate 在新的 retry1 目录从相同父 EMA 和 fresh Adam 开始；Control 结果与失败目录/log 均保留。RUN_PATHS.json 明确给出实际两臂路径。最终 matched stream 审计仍要求全部 8000 steps 一致。

## 后处理数值一致性恢复

两臂 8000 步训练和 EMA checkpoint 均已完成。首轮后处理硬编码 cpu_threads=4，与训练配置 cpu_threads=2 不一致，导致 CPU float32 metric mean 归约顺序产生 2.46e-8 dB 的平均 PSNR 差异，触发 1e-8 断言。POSTPROCESS_THREAD_AUDIT.json 使用同一不变预测图，分别精确复现两组逐 slice 指标，并确认重新加载的 frozen MedDINO hash 与训练初始化一致。postprocess.py 改为读取相同配置的 cpu_threads；保留原断言、评价函数、训练结果及 checkpoint。三组 Fast64 诊断在 2 threads 下重新执行；未重跑训练。旧日志和首次后处理输出保存在 logs/postprocess_retry1.log 与忽略目录 runs/postprocess_attempt1_metadata_20261002。

8k NORMAL − CONDITION_OFF ΔPSNR = 0.145259 dB；NORMAL − SHUFFLED_MEDDINO ΔPSNR = 0.005506 dB。64/64 slices 的正向增益仍只有 0.018291 dB 平均值，未达到预定 Full-L333 门槛。关闭条件后的下降说明已训练 Candidate 使用了该条件路径；shuffle 差异很小，不足以证明强 patient-specific anatomy 利用。
