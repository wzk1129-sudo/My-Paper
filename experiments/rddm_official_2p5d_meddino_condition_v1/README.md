# Official RDDM + 2.5D + MedDINOv3 Anatomy Condition V1

## 结果和审计入口

本目录保存本次实际运行代码、原始轻量运行记录及可以离线核验的结果。首先阅读 [REPORT.md](REPORT.md)、[EFFECTIVE_RUN_CONFIG.json](EFFECTIVE_RUN_CONFIG.json) 和 [AUDIT_GUIDE.md](AUDIT_GUIDE.md)。

两臂从相同 Official RDDM **2D 8k EMA** 初始化，分别完成 **额外 8000 个 fresh-Adam continuation steps**。Control 是 Official RDDM + 2.5D；Candidate 增加 frozen MedDINOv3 anatomy condition。以下为 EMA / 固定 L333-fast64：

| Step | Control PSNR | Candidate PSNR | Candidate − Control |
|---:|---:|---:|---:|
| 0 | 40.623375 | 40.623375 | 0 |
| 1000 | 40.678378 | 40.678683 | +0.000305 |
| 2000 | 40.768102 | 40.775686 | +0.007585 |
| 4000 | 40.853634 | 40.862841 | +0.009207 |
| 6000 | 40.872903 | 40.892207 | +0.019305 |
| 8000 | 40.889946 | 40.908237 | +0.018291 |

8k Candidate SSIM 0.948847、RMSE 18.119234 HU、MAE 11.986737 HU。64/64 slices PSNR 改善，但 +0.018291 dB 未达到预定 +0.03 dB promotion 门槛，未执行 Full-L333。NORMAL / CONDITION_OFF / SHUFFLED_MEDDINO PSNR 为 40.908237 / 40.762979 / 40.902731 dB。结论只覆盖单 seed、单验证患者和此次 8k continuation。

## 实际核心代码

| 内容 | 文件 |
|---|---|
| Official U-Net、residual diffusion、q_sample、两步 sampler | [src/official_rddm_ct.py](src/official_rddm_ct.py) |
| 4-channel stem、四层 MedDINO map fusion、九处 pre-attention residual affine modulation | [model.py](model.py) |
| frozen official ViT-B/16 loader 与 checkpoint pin | [med_loader.py](med_loader.py) |
| 实际 MedDINO Python dependency 源码只读快照 | [vendor/meddinov3/SOURCE_MANIFEST.json](vendor/meddinov3/SOURCE_MANIFEST.json) |
| matched 数据顺序、Adam、EMA、MSE、noise/timestep stream、记录与 checkpoint 接口 | [train.py](train.py) |
| 精确 Control 训练时版本与两臂所有运行源码 SHA 映射 | [audit_artifacts/TRAINING_SOURCE_MANIFEST.json](audit_artifacts/TRAINING_SOURCE_MANIFEST.json) |
| center-only NDCT、patient-local LDCT 邻居、同步增强、actual-open guard | [context_data.py](context_data.py)、[fast_dev/data.py](fast_dev/data.py) |
| 归一化、HU 指标与 SSIM | [mayo2016_smallset_v1/frozen_protocol.py](mayo2016_smallset_v1/frozen_protocol.py) |
| EMA Fast64、逐 slice noise seed、OFF/SHUFFLED 实现 | [evaluate.py](evaluate.py) |
| readiness / promotion / 结果汇总 | [readiness.py](readiness.py)、[postprocess.py](postprocess.py) |

## 配置解析与复现边界

configs/rddm_fast_dev.json 原样复用原 baseline 的数据、diffusion、optimizer、EMA 和指标合同，其中历史 experiment_name/model 描述字段不是此次实际 backbone。实际模型由 model.py:build_model() 构造；实际 parameter counts、4-channel 输入、condition、全部 validation nodes 见 EFFECTIVE_RUN_CONFIG.json 和两臂 INITIAL_STATE.json。训练采用 batch=2 / accum=1、seed=20260829、Adam lr=2e-4、betas=(0.9,0.99)、wd=1e-9、EMA=.995、clip=1、AMP=False。评价 CPU threads=2。

CT：stored value −1024 HU，clip [-1024,3072] 并归一化到 [0,1]；diffusion 内转换 [-1,1]。MedDINO 只使用 center LDCT，clip [-1000,1000] 后 mean=65.1084213256836 / std=178.01663208007812，repeat 3 channels。评价 HU window [-1000,1000]、PSNR range=2000 HU；SSIM 参数在冻结指标代码内。

训练只访问 L067/L143；validation NDCT 只访问固定 64 个中心 slice，相邻 LDCT 只用于 2.5D context。遗留合同中 external-memory patient 列表只用于禁止访问和校验；实际访问计数为 0。Full-L333、L506、neighbor NDCT 均未访问。

数据和三个必要私有 weight artifact（父 2D 8k EMA、MedDINO weights、最终 run checkpoints）不在 GitHub。可独立静态审计全部协议和结果；重新训练或重新 GPU 推理需要合法取得匹配数据及 pinned weights，并设置代码/配置内的路径绑定。保留原始路径和哈希便于判定实验身份。

## 离线复核

在本目录执行：

~~~bash
python3 verify_lightweight_artifacts.py
~~~

仅标准库；不访问医学数据、checkpoint、GPU、网络，不启动训练。核验文件哈希、实际源码快照、两臂全部 8000 stream hashes、六个 Fast64 节点、逐 slice 指标聚合、初始化、MedDINO 冻结与 checkpoint reload、三组诊断和数据边界。

历史真实启动命令见 RUN_MANIFEST.json、RECOVERY_RUN_MANIFEST.json、POSTPROCESS_RUN_MANIFEST.json；失败和恢复证据均保留。run_queue.py 是一次性执行入口，已有输出不能再次运行。当前已 COMPLETE，不应在历史目录重启。

checkpoint、DICOM、NPY、医学图像、feature map、缓存及原始大日志均不发布。TRAINING_DIAGNOSTICS.csv 仅为两臂每步标量与 stream hashes；audit_artifacts/ 中的 JSON/CSV 是原始轻量记录的字节一致副本。
