# FoundDiff DACB + MedDINO CSSM 6k 只读审计

本资料包对应 CONTROL、MED_G、MED_S 三组各 6000 optimizer steps 的既有实验。完整源码、有效配置、上游实现参考、初始化与恢复审计、训练日志摘要、条件关闭结果、原最终报告及本轮代码审计均在 ZIP 内。原远程实验文件没有修改。

- 完整资料包：[founddiff_dacb_meddino_cssm_6k_readonly_audit_20261009.zip](founddiff_dacb_meddino_cssm_6k_readonly_audit_20261009.zip)
- 可直接阅读的结论：[CODE_AUDIT.md](CODE_AUDIT.md)
- 校验摘要：[ZIP_VERIFICATION.json](ZIP_VERIFICATION.json)
- SHA256：5862f1bd4f35546f905e31e869018f50a20a6d0d895d94c405daf8b352b1e508

主要结论：源码静态路径未发现 CSSM 断开或条件注入位置错误；存在 Python random 状态未纳入续训 checkpoint 却宣称默认 RNG 完全精确的审计缺陷，以及旧 8k 元数据残留。条件分支在第 1 步因串联零初始化梯度为零，第 2 步后非零，但 6k MED_S 相对 CONTROL 的 PSNR 为 -0.003770 dB。MED_S 关闭空间条件仅下降 0.000134 dB；“发生作用”与“改善重建”需要区别。

本轮没有训练、推理、L333 验证或 L506 数据访问。ZIP 不含 checkpoint、模型权重、医学数组、医学影像、逐切片标识清单或大缓存。
