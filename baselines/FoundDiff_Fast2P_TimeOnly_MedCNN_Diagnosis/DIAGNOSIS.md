# FoundDiff TimeOnly 与 MedDINO+CNN：8k 代码与训练诊断

## 范围与结果

本报告仅依据真实运行源码、两份 8,000-step 日志、四次 EMA 逐片验证 CSV 和 step 0/2k/4k/6k/8k 检查点；本轮没有重新训练、推理或修改原实验。两组均使用 L067/L143 训练、L333 固定 Fast64 验证。最终审计记录：共同主干 step0 一致；全部 8,000 步样本顺序及 16,000 次微批次 CUDA 随机状态相同。实验只有一个种子和一位验证患者，64 张相邻切片不能视为 64 个独立患者。

| Step | A PSNR | B PSNR | B−A PSNR | A SSIM | B SSIM | A RMSE HU | B RMSE HU | A MAE HU | B MAE HU |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2k | 40.326771 | 40.487293 | +0.160522 | 0.944377 | 0.945807 | 19.355551 | 19.013179 | 13.227311 | 13.081086 |
| 4k | 40.558138 | 40.689862 | +0.131724 | 0.946572 | 0.947456 | 18.853611 | 18.577524 | 12.801135 | 12.719792 |
| 6k | 40.295719 | 40.669712 | +0.373993 | 0.944388 | 0.947212 | 19.435057 | 18.620987 | 13.165683 | 12.729898 |
| 8k | 40.092705 | 40.367198 | +0.274493 | 0.942328 | 0.944432 | 19.897545 | 19.289301 | 13.347787 | 13.036886 |

4k→8k 逐片 PSNR：A 下降 64/64，均值 −0.465433 dB，中位数 −0.460320，范围 −0.653149 至 −0.312374；B 下降 64/64，均值 −0.322664，中位数 −0.339986，范围 −0.449475 至 −0.081328。两组都呈系统性下降，不由少数异常切片单独造成。8k B 对 A 为 64 张提高、0 张下降；同一患者的切片相关，不能据此声称跨患者普遍成立。

## 确定存在的实现不一致

### 1. 训练扩散与采样起点的噪声尺度不一致

[A 的日程代码](experiments/Fast2P_TimeOnly_Control_8k/src/DADiff.py#L691-L715) 中 convert_to_ddim=True 使用线性 beta 构造 betas_cumsum，该分支没有使用 sum_scale 缩放训练日程。[q_sample](experiments/Fast2P_TimeOnly_Control_8k/src/DADiff.py#L1131-L1136) 的训练分布为 x_t=x_start+alpha_t*res+beta_t*epsilon。检查 8k checkpoint：alpha_999=0.9936471581，beta_999=0.9999797940，beta_499=0.9599024653。

但[两步采样初始化](experiments/Fast2P_TimeOnly_Control_8k/src/DADiff.py#L1035-L1044) 从 x_input+sqrt(sum_scale)*epsilon，即 x_input+0.1*epsilon 开始，并将其标记为 t=999。采样起点噪声标准差 0.1 与前向日程末端约 1.0 不一致，方差相差约 100 倍；严格比较时还有 alpha_999 未完全等于 1 的残差项。这是代码及 checkpoint 确认的分布错配，继承自指定的官方/Lite 实现，两组共同受影响。本轮没有执行修正前后的推理对照，因此无法断言它导致 4k 后回落，也不能保证修复必然提高 PSNR。

### 2. 首个确定性反向步未匹配前向边缘噪声系数

[采样代码](experiments/Fast2P_TimeOnly_Control_8k/src/DADiff.py#L1050-L1093) 固定使用 use_pred_noise；eta=0 时实际更新为 x_next=x_t-(alpha_t-alpha_next)*pred_res，没有使用已计算的 pred_noise 修正噪声系数。若要求固定同一 epsilon 的确定性边缘一致性，由 q_sample 得到目标更新 x_next=x_t-(alpha_t-alpha_next)*pred_res+(beta_next-beta_t)*pred_noise。t=999→499 时 beta_next-beta_t≈−0.0400773。现有分支不满足该边缘一致公式；其也继承官方实现，可能是作者选择的启发式，不能仅凭静态检查断言改动能改善效果或解释后期回落。

建议下一轮在独立副本中明确选择一套训练—采样一致的方案：例如保留训练 beta 日程，将采样初始噪声与 betas_cumsum[-1] 对齐并核对反向噪声系数；或设计末端噪声为 0.1 的完整训练日程，重算依赖系数。先以固定 checkpoint、相同切片和种子对照，报告协议变化。本轮未做此对照。

### 3. 日志缺失和过时状态工件

[训练日志](experiments/Fast2P_TimeOnly_Control_8k/train.py#L107-L117) 记录 loss、裁剪前全局梯度范数和随机状态，但未逐步记录 LR、各时间门控的激活/梯度或验证残差损失。源码设定 Adam LR=2e-4、无 scheduler，两组 0/2k/4k/6k/8k 优化器检查点均为 2e-4；归档曲线中的 LR 由源码与检查点重建，不是逐步实测日志。下一轮应记录各参数组实际 LR、分模块梯度和门控激活、时间步分桶残差损失以及在线/EMA 模型差异。

原实验根目录旧 STATUS.json 仍显示下载权重前的 BLOCKED；随后 SERIAL_COMPLETE.txt 和 FINAL_AUDIT.json 显示完成。它是过时状态文件，会误导查询，但不是模型计算错误。本次未修改原文件。

## 有证据支持的训练问题

最可能的后期回落机制是小数据重复训练带来的泛化差距，可能与恒定学习率和上述训练—采样错配共同作用。两名患者共 1,145 张；有效 batch=4，4k/8k 约相当于 14/28 次样本遍历，无增强。每 1,000 步均值如下：

| 步数区间 | A loss | B loss | A 全局梯度范数 | B 全局梯度范数 |
|---:|---:|---:|---:|---:|
| 3001–4000 | 0.000465438 | 0.000481539 | 0.021624 | 0.021568 |
| 4001–5000 | 0.000437233 | 0.000471529 | 0.016435 | 0.017411 |
| 5001–6000 | 0.000412081 | 0.000460675 | 0.013948 | 0.015815 |
| 6001–7000 | 0.000390800 | 0.000438381 | 0.013455 | 0.012952 |
| 7001–8000 | 0.000377554 | 0.000413092 | 0.012381 | 0.012210 |

训练 loss 持续下降、L333 PSNR 在 4k 后下降，符合泛化差距/过拟合的观察模式；但训练损失是随机 timestep 残差 MSE，验证指标是两步采样 PSNR，目标和输入分布不同。没有同分布验证损失、更多患者或多 seed，不能断定是纯粹过拟合。

两组 1001–8000 步没有全局梯度范数超过 1 的记录，也无非有限 loss/梯度；第一千步分别有 19、26 次超过阈值并被裁剪。后期梯度均值下降，没有全局梯度爆炸证据。固定 2e-4 LR 是可能因素，但没有衰减对照或逐步 Adam 更新量，无法确定因果。优先保留 4k EMA；下一轮可在独立配对设置下比较预先固定的 LR 衰减/早停方案并记录实际 LR。

## TimeOnlyDACB 与 MedCNN

[TimeOnlyDACB](experiments/Fast2P_TimeOnly_Control_8k/src/DADiff.py#L258-L270) 的九组调制线性层在 0/2k/4k/6k/8k 均有限。4k→8k 全部调制参数相对 4k 的整体 L2 变化为 A 0.472791、B 1.082616；它说明参数持续变化，不能单独证明不稳定。gate 参数最大绝对值由 A 0.067482 增至 0.106215、B 0.050346 增至 0.059744；参数值不等于门控激活，现有日志无法判断饱和或各时间步异常。训练日志为全局梯度，不能推出每个门控模块的后期梯度。

[MedDINO](experiments/Fast2P_TimeOnly_MedCNN_8k/src/medcnn.py#L16-L37) 严格加载官方预训练权重，冻结参数，重载 train() 强制 eval 并以 no_grad 提取特征。164 个 MedDINO 参数/缓冲区在 step0 与 step8k 全部逐元素相同；8k EMA 中也与在线模型相同。[CNN、局部 Cross-Attention 和 Adapter](experiments/Fast2P_TimeOnly_MedCNN_8k/src/medcnn.py#L39-L119) 与 [九处注入](experiments/Fast2P_TimeOnly_MedCNN_8k/src/DADiff.py#L447-L461) 实际参与前向；预检记录了形状、调用和梯度，step0→8k CNN、投影、各尺度融合与九个 Adapter 的参数组均有非零变化。计算图正常不等于每个模块均对提升有独立贡献。

## 目标、EMA、数据和评价

残差目标为中心 LDCT − NDCT；[q_sample 与 p_losses](experiments/Fast2P_TimeOnly_Control_8k/src/DADiff.py#L1131-L1213) 使用该残差和 L2。[forward](experiments/Fast2P_TimeOnly_Control_8k/src/DADiff.py#L1233-L1248) 均匀采样 0–999，输入原 alpha 时间编码；除上述采样问题外，未发现残差符号、时间传递或条件通道的确定错误。[EMA](experiments/Fast2P_TimeOnly_Control_8k/train.py#L40-L47) 每 optimizer step 更新可训练参数并复制缓冲区；检查点均满足 ema_updates=step，验证使用 EMA 和固定逐片种子。

[加载器](experiments/Fast2P_TimeOnly_Control_8k/mayo_data.py#L18-L40) 同患者取邻片并在边界复制；存储值减 1024 后裁剪至 [-1000,1000]。原版 [frozen_metrics.py](experiments/Fast2P_TimeOnly_Control_8k/frozen_metrics.py#L10-L53) 按片计算 PSNR（data_range=2000）、SSIM（win_size=7）后取平均。静态核对未发现尺度或聚合错误。DICOM 预处理时 FD/QD 几何逐片匹配，但 FrameOfReferenceUID 不同；旧 NPY 不可用，因此与历史图像的逐像素身份尚无法证实。

当前结果支持使用两组 4k EMA，也支持新结构在本次 Fast-Dev 的配对验证中优于对照；尚不能证明跨患者或多 seed 泛化。本轮未执行任何新实验。公开归档仅含源码、配置、聚合指标和脱敏的 100-step 曲线，不含 CT、逐片结果、权重、checkpoint 或大型原始日志。
