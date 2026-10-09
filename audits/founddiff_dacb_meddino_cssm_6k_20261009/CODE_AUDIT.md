# FoundDiff DACB + MedDINO CSSM 6k：只读源码与证据审计

审计日期：2026-10-09。范围为 CONTROL、MED_G、MED_S 三组已完成的 6k 实验。原实验目录不作修改。本轮没有训练、模型前向、L333 验证或医学数组读取。审计方法是阅读冻结源码/配置/日志，并对既有 checkpoint 作 CPU 标量与哈希检查。下述“已验证”仅指此范围，不能代替重新运行的动态验证。

## 结论

没有在所审静态执行路径中发现 CSSM 断开、Med-S 注入位置错误、父 RLEB/残差扩散路径被替换、MedDINO 用 NDCT 作为条件、或三组串行随机流不匹配的功能性错误。发现两项可确证的审计/元数据缺陷，以及一个预检承诺与实测不一致。多层零初始化造成第一步条件梯度为零，但后续梯度和投影参数更新均非零；它不是永久的死分支。MED_S 6k 的空间条件关闭仅使 PSNR 下降 0.000133555 dB，证明该条件对输出有极微小影响，不能证明它改善了跨组重建性能。6k MED_S 相对 CONTROL 为 -0.003769792 dB；本组结果支持 NO-GO。

## 1. 拓扑、官方模块与扩散路径

实际模型在 experiment/model_cssm.py:21-26 继承父 Unet，并为 E0/E1/E2/E3/MID/D0/D1/D2/D3 建立九个 Mamba_block。前向在 :82-91 延续父 init_conv、ResnetBlock skip、time_mlp 和最终预测头，E/MID/D 阶段按明确位置插入模块。官方 Mamba_block 由 vendor/official_blocks.py:46-80 移植，包含 SS2D、TransposedAttention 和 timestep adaLN-Zero；SS2D 的 scan 运算在 vendor/official_ss2d.py:741 仍调用原 forward_core。原父网络和 RDDM 损失/采样代码在 vendor/parent_rleb.py:230-261、:764-840、:884-920；训练调用 experiment/train_cssm_6k.py:129-140 的同一 p_losses。父 RLEB@60k 本身不含这些 DACB/CSSM，故本模型是从父 RLEB 扩展的筛选实现，不等于声称复现完整原始 FoundDiff 系统。DA-CLIP/cross-attention 未加入本次路径。

外部“残差门控”在实际代码里是官方 adaLN 的 gate_msa/gate_mlp，乘在两条 Mamba/Attention 残差更新上（vendor/official_blocks.py:65-80）；model_cssm.py:73-81 没有另设可学习输出门。branch_off 是诊断开关，正常训练为 False。若报告将其描述成另一个已训练门，属于不准确表述。

## 2. 初始化、零门控和梯度

模型构造在 model_cssm.py:93-109 以相同 seed 装载父 60k EMA，并逐项迁移父 Unet 权重，再装入同一 COMMON_INITIAL_STATE；原始公共参数哈希在三组均为 e303a436cd762d1ec29d48795ec8b2c757ae99ecf3ef15d1d81d511620d822ab。Control 的 c 为全零 [B,1,256]，不实例化 MedDINO（:52-56）。MED_G/S 的 global_projection 权重及 bias 为零（:33-34），MED_S 的三处 spatial_projections 权重及 bias 为零（:35-37）；官方每个 Mamba_block 的 adaLN 最后一层也全零（official_blocks.py:65-67）。因此初始前向等于公共父输出，但条件分支第一步梯度为零：preflight.py:97 已明确记录，TRAINING_GRADIENT_SUMMARY.json 也显示 global/spatial 第一步都是 0。这不满足用户原先“首次反向传播即非零”的字面预检要求，应记为预检契约偏差；不能将其改写成已通过第一步梯度检查。

后续没有永久梯度阻断：MED_G global_projection 在 5999/6000 步非零，MED_S global/spatial 各 5999/6000 步非零。MED_S 梯度范数中位数：main 1.050066e-3，DACB 4.945613e-6，global 4.223128e-8，spatial 9.879112e-8（audit/TRAINING_GRADIENT_SUMMARY.json）；组范数受参数量影响，不可直接作为功能贡献比例。6k global_projection 参数非零，MED_S 三处 spatial 投影亦非零（audit/PARAMETER_UPDATE_AND_GATE_PROXY_STATS.json）。adapter/fusion 起始私有权重没有完整 step0 快照，只能报告 1k→6k 更新，不能伪称 0→6k。adaLN gate 行权重/bias 的 RMS 仅为参数代理；实际随 timestep 的 gate_msa/gate_mlp 输出幅度没有保存，故未测得。

## 3. MedDINO 输入、冻结与缓存

model_cssm.py:38-51 的 train() 强制 med.eval().requires_grad_(False)，extract() 在 no_grad 下只接受中心 LDCT。输入由存储规范化值按 center*4096-1024 转回 HU，再 clip[-1000,1000]，以 V1 的 MEAN_HU=65.1084213256836、STD_HU=178.01663208007812 标准化成 3 通道；第 [2,5,8,11] 层提取 normalized 1024x768 tokens，恢复四层 32x32 特征。模型权重 SHA 为 492b2b96c00e9d3b93ba3b07a8d387aa2196b5e04698378fcc23049b447966d2，哈希和冻结状态在初始化/最终审计中一致。NDCT 在 protocol_cssm.py:44-55 只作中心监督目标，train_cssm_6k.py:129-140 不把它传给 set_context。

set_context 在每张图采样前执行一次，clear_context 在采样后清空（protocol_cssm.py:65-93；model_cssm.py:70-72）。已有 EMA 条件活动审计每变体为 64 次提取、128 次扩散 forward，Control 提取 0 次。训练和验证都使用同一 set_context/update_branch 路径；未见静态缓存错配。尚无逐 forward 缓存张量哈希或 hook 证据，不能声称动态层面完全排除了任何别名/缓存问题。

## 4. MED_G / MED_S 条件路由

model_cssm.py:57-69 的四层 adapter→fusion 产生 [B,128,32,32]。GAP→Linear(128,256) 构成 [B,1,256] 全局 c，update_branch 在九处均调用 Mamba_block(x,c,t)（:73-91）。SS2D 在 official_ss2d.py:715 通过 attn(c) 进入输出投影前的 y+local 分支（:752）；它不是 pre-scan 的全局 c 注入。MED_S 在 E2、MID、D1 以 bilinear resize 后用 1x1 投影到实时读取的 d_inner，模型断言尺寸（model_cssm.py:25,35-37,62-69）。SS2D 在深度卷积和激活之后、forward_core 扫描之前将空间张量加到 scan 输入（official_ss2d.py:722-741）；相同 shape 断言阻止错误广播。E2/MID/D1 为 128x128 / 64x64 / 128x128；d_inner 分别由官方权重确定为 256/1024/512，不硬编码。

MED_S 的 c RMS 均值 0.004578、fusion RMS 0.224902；空间条件 RMS E2 0.007929、MID 0.000729、D1 0.004985（audit/CONDITION_ACTIVITY_SUMMARY.csv）。总 DACB delta/base RMS 约 E2 0.0184、MID 0.0111、D1 0.0226，是全部 DACB 更新，不可当作 MedDINO 独立贡献。

## 5. 对照、训练流、optimizer 与评价

三个变体各自 fresh Adam（train_cssm_6k.py:66-69），仅收 requires_grad=True 参数；MedDINO 不入 optimizer。Control 1k→6k 从完整 model/EMA/Adam/torch/CUDA/NumPy RNG、独立 noise generator、Stream permutation/cursor/rng 恢复（:79-110），并核验首个续接 batch。MED_G/S 不加载前组训练权重。三组 6000 步数据/augmentation/timestep/noise 记录逐项相等，最终链哈希同为 3a1f2797bcd9aa788a78a228faea70db36c3ebbbdaa39166f98980ae73db213d（audit/TRAIN_STREAM_ALIGNMENT_SUMMARY.json）。EMA 2k/4k/6k 与同一 Fast64、同一种子评价；step0 四指标逐字一致。NaN/Inf/overflow/optimizer skip 日志总数为零。

数据 AccessGuard 只允许 L067/L143 和固定 L333-fast64 的 np.load，审计 L506/external opens 为 0（protocol_cssm.py:27-55）。该守卫覆盖此执行路径的 np.load，但不是操作系统级所有 I/O 拦截；只读源码检查也未发现其他数据读取路径。本轮审计本身没有打开 L333 或 L506 医学数组，未运行验证。

## 6. 条件关闭幅度与数值结论

6k L333-fast64：CONTROL 40.684519 dB / SSIM 0.946670206 / RMSE 18.591804 HU / MAE 12.157335 HU；MED_G 40.678648 / 0.946619591 / 18.604376 / 12.162352；MED_S 40.680749 / 0.946648663 / 18.599658 / 12.157785（experiment/VALIDATION_HISTORY.csv）。MED_G−CONTROL = -0.005871 dB；MED_S−MED_G = +0.002101 dB；MED_S−CONTROL = -0.003770 dB。2k MED_S−CONTROL +0.001236 dB，4k +0.003149 dB，6k 转负，缺乏稳定增益。

同一 MED_S@6k 模型 NORMAL−SPATIAL_OFF = +0.000133555 dB，64/64 片 PSNR 变化为正，区间 +0.000047552 至 +0.000221714 dB；NORMAL−MEDDINO_OFF = +0.000121665 dB。MED_G NORMAL−MEDDINO_OFF = -0.000017222 dB，仅 12/64 片正（experiment/CONDITION_OFF_DIAGNOSTICS.csv；audit/CONDITION_OFF_AGGREGATED_SLICE_SUMMARY.json）。这证实空间条件对数值预测有极微小作用；关闭空间条件仍保留全局条件，关闭 MedDINO 条件仍保留已训练的 DACB 和条件投影结构，均不能替代不同模型的 Control 对照。极小响应与弱梯度、层层门控相容，但“为何恰是 0.000134 dB”不能仅凭静态代码给出因果分解。

## 分类

### VERIFIED BUG / 确证缺陷

1. **续训 RNG 审计措辞过强**：train_cssm_6k.py:106-110 声称 default_RNG_exact=True，:165-168 的 checkpoint 却没有保存 Python random.getstate()，恢复也没有 random.setstate()。当前 self_condition=False，父代码中的 random.random() 路径（vendor/parent_rleb.py:904）不可达，因此没有证据表明此次数值流受影响；但“所有默认 RNG 精确恢复”这一审计命题不成立。
2. **旧元数据残留**：experiment/METRIC_CONFIG.json 的 training.max_optimizer_steps=8000、matched_checkpoint_steps 含 8000，而实际6k训练由 CONFIG_6K.json 与 train_cssm_6k.py:52-55、:123-125 控制；preflight.py:99 也保留 8k ETA 文案。运行步数、验证历史和 6k checkpoint 均证实实际是6k。该问题污染资料说明，不构成证实的训练超步错误。

### SUSPECTED / 待验证

1. 逐层 gate 与小幅条件梯度可能使 MedDINO 输入对 DACB 的实际作用很弱；现有参数 RMS 不能代替逐 timestep gate(t) 输出或反事实贡献归因。
2. 现有缓存证据支持每 slice 一次特征提取、两次采样 forward 复用，但没有逐 forward 特征哈希或 hook；因此不能动态穷尽排除缓存/别名问题。

### DESIGN LIMITATION / 设计与证据限制

1. 多个串联零初始化使**第一步** global/spatial 梯度为零，尽管第二步后可学习。这与最初硬性预检的字面目标不一致，须与永久死路区别。
2. MED_G 的 c 在 SS2D 扫描后汇合，MED_S 仅三处扫描前注入；这种条件接口可能不足以在残差已较小的任务上产生可测收益。单一 seed、单一 Fast64 验证患者和 6k 小幅差异限制泛化主张。
3. 条件关闭诊断测得功能活动，跨组同 step 指标没有改善；不能把“条件发生作用”写成“方法有效”。最终按原报告为 NO-GO，不据此自动启动后续实验。

## 后续证明所需工作

见 VERIFICATION_PLAN_NOT_EXECUTED.md。本轮未执行该方案。
