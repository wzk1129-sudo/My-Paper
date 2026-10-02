# 复现与审计索引

| 需要判断 | 主证据 | 交叉核验 |
|---|---|---|
| 实际配置与历史 donor metadata 区别 | EFFECTIVE_RUN_CONFIG.json | EXPERIMENT_CONFIG.json、BATCH_CONFIG.json、configs/rddm_fast_dev.json、model.py、train.py |
| 数据配对、患者、邻居与 HU normalization | context_data.py、fast_dev/data.py、mayo2016_smallset_v1/frozen_protocol.py | splits/L333_fast64.txt、inherited_audits/、DATA_ACCESS_AUDIT.json |
| Official topology、注入位置和实际 forward/backward | model.py、src/official_rddm_ct.py | READINESS.json、CONDITION_ACTIVITY.csv、TRAINING_DIAGNOSTICS.csv |
| MedDINO source/weights pin、frozen loader、input transform | med_loader.py、model.py、vendor/meddinov3/ | SOURCE_PROVENANCE.json、audit_artifacts/MEDDINO_FROZEN_AUDIT.json |
| fresh Adam、同一父 EMA 与零初始化 | model.py、train.py | audit_artifacts/INITIALIZATION_AUDIT.json、两臂 INITIAL_STATE.json、READINESS.json |
| 全部 step matched data/aug/timestep/noise | train.py、context_data.py | audit_artifacts/RANDOM_STREAM_AUDIT.json、TRAINING_DIAGNOSTICS.csv、DATA_ACCESS_AUDIT.json |
| 实际 Control 和 Candidate 版本 | audit_artifacts/TRAINING_SOURCE_MANIFEST.json | source_snapshots/、两臂 FINAL_STATE.json、各 RUN_MANIFEST.json |
| 原始六节点、64 slice 指标和采样 seed | audit_artifacts/<arm>/validation_step_*.json | VALIDATION_HISTORY.csv、PER_SLICE_DELTA.csv、validation_metrics.csv、evaluate.py |
| 正常、condition-off、shuffled-feature 三组诊断 | DIAGNOSTIC_*.json、DIAGNOSTIC_*_PER_SLICE.json | evaluate.py、FINAL_SUMMARY.json、REPORT.md |
| frozen MedDINO 未更新 | audit_artifacts/MEDDINO_FROZEN_AUDIT.json | INITIAL_STATE.json、FINAL_STATE.json、POSTPROCESS_THREAD_AUDIT.json、train.py 的 final state hash assertion |
| checkpoint reload 与训练验证完全一致 | audit_artifacts/CHECKPOINT_RELOAD_AUDIT.json | 原始 step8000 JSON、DIAGNOSTIC_NORMAL_PER_SLICE.json、POSTPROCESS_THREAD_AUDIT.json |
| 两次失败及恢复是否影响训练 | READINESS_FAILURE.json、QUEUE_STATUS_FAILED_20261002.json、HASH_FIX_AUDIT.json、POSTPROCESS_THREAD_AUDIT.json | RECOVERY_SOURCE_PROVENANCE.json、source_snapshots/、REPORT.md 的恢复记录 |
| 发布文件是否完整且没有医学数组或模型权重 | PUBLICATION_MANIFEST.json | verify_lightweight_artifacts.py、Git blob identities |

## 解读审计强度

本包包含真实记录和可独立重算的哈希/标量一致性检查。随机流哈希覆盖 all normalized augmented batch bytes + metadata + timestep + noise，原始图像/随机噪声张量不公开；只能重新核验两臂记录的 digest equality，不能从 digest 恢复医学数据。

MedDINO 初始 hash 与 frozen-state reload hash 均有记录。训练结束时 train.py 执行全 state hash equality assertion，通过后才写 meddino_weights_unchanged=true。FINAL_STATE 只记录 assertion 通过的布尔状态；不是新增一次独立重训练验证。

Frozen legacy configuration 还引用历史 source pins 和禁止的 memory patient labels，运行采用 FastPairs + ContextPairs 严格 allowed patient/index/view guard。监督只使用 center NDCT，neighbors 只读取 LDCT。审计中的患者代码是 Mayo 公开数据集编号，不包含 DICOM headers 或私人影像。

本轮从 prior official 2D 8k EMA 再训练 8k，不能把相对 parent 的增益单独归因于 2.5D（还同时增加了训练步数）。matched Control 与 Candidate 的差值才对应本轮 anatomy condition；单 seed / 单患者不支持跨患者显著性结论。关闭条件诊断使用已经共同训练的 Candidate，不等同于重新训练一个无条件 Control。

## SHA 版本约定

Remote CoreDiff commit 与 GitHub publication commit 属于不同 Git 历史，SHA 可以不同。PUBLICATION_MANIFEST 对每个新实验文件记录 SHA256、Git blob SHA1 和字节数，GitHub 发布后按 tree blob identity 核验逐文件内容相同。manifest 自身不包含自身 hash；其 Git blob 由发布 tree 单独核验。
