"""Read-only matched Full-L333 comparison; writes two summary files only."""
import csv
import json
from pathlib import Path

import numpy as np

project = Path("/home/xujian/projects/CoreDiff_official_2p5d_meddino_condition_v1")
v1 = project / "experiments/rddm_official_2p5d_meddino_condition_v1"
mlsf = project / "experiments/rddm_mlsf_v1"
v1_name = "FULL_L333_V1_8K_MLSF_MATCH"
v1_result = json.loads((v1 / (v1_name + ".json")).read_text())
v1_slices = json.loads((v1 / (v1_name + "_PER_SLICE.json")).read_text())
mlsf_result = json.loads((mlsf / "FULL_L333_RESULT.json").read_text())
mlsf_slices = json.loads((mlsf / "validation_step_8000.json").read_text())
gate = json.loads((mlsf / "FULL_L333_GATE.json").read_text())
paths = json.loads((v1 / "RUN_PATHS.json").read_text())
summary = json.loads((mlsf / "PAIRED_SUMMARY.json").read_text())

v1_metrics = v1_result["metrics"]
mlsf_metrics = mlsf_result["metrics"]
assert len(v1_slices) == len(mlsf_slices) == 610
for result, metrics in ((v1_result, v1_metrics), (mlsf_result, mlsf_metrics)):
    assert metrics["slices"] == 610
    assert metrics["patient"] == "L333"
    assert metrics["weights"] == "ema"
    assert metrics["validation_mode"] == "FULL"
assert mlsf_result["step"] == 8000 and mlsf_result["mode"] == "FULL"
identity = ("patient_id", "slice_index", "prev_index", "next_index", "sampling_seed")
for left, right in zip(v1_slices, mlsf_slices):
    assert tuple(left[k] for k in identity) == tuple(right[k] for k in identity)
assert len({(row["patient_id"], row["slice_index"]) for row in v1_slices}) == 610
assert [row["slice_index"] for row in v1_slices] == list(range(610))
for field in ("psnr", "ssim", "rmse_hu", "mae_hu"):
    assert abs(float(np.mean([r[field] for r in v1_slices])) - v1_metrics[field]) < 1e-8
    assert abs(float(np.mean([r[field] for r in mlsf_slices])) - mlsf_metrics[field]) < 1e-8

v1_access = v1_result["data_access"]
mlsf_access = mlsf_result["access"]
for audit in (v1_access, mlsf_access):
    for field in ("L506_opens", "external_memory_opens", "neighbor_target_opens",
                  "training_target_opens", "blocked_attempts"):
        assert audit[field] == 0, (field, audit[field])
assert gate["eligible"] and abs(gate["delta_psnr"] - 0.031444759188119065) < 1e-10
v1_checkpoint = Path(paths["OFFICIAL_RDDM_2P5D_MEDDINO_CONDITION"]) / "checkpoints/step_8000.pt"
mlsf_checkpoint = mlsf / "runs/MLSF_RDDM_8K_20261009/checkpoints/step_8000.pt"
assert v1_checkpoint.is_file() and mlsf_checkpoint.is_file()

fields = ("psnr", "ssim", "rmse_hu", "mae_hu")
labels = {"psnr": "PSNR (dB)", "ssim": "SSIM", "rmse_hu": "RMSE (HU)", "mae_hu": "MAE (HU)"}
rows = []
for field in fields:
    v = float(v1_metrics[field])
    m = float(mlsf_metrics[field])
    rows.append({"metric": labels[field], "v1_8k": v, "mlsf_8k": m,
                 "delta_mlsf_minus_v1": m - v, "slices": 610})
out_csv = mlsf / "FULL_L333_MATCHED_COMPARISON.csv"
out_md = mlsf / "FULL_L333_MATCHED_COMPARISON.md"
assert not out_csv.exists() and not out_md.exists()
with out_csv.open("x", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

psnr_delta = np.array([m["psnr"] - v["psnr"] for v, m in zip(v1_slices, mlsf_slices)],
                      dtype=np.float64)
quantiles = np.quantile(psnr_delta, [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1])
count = int(np.count_nonzero(psnr_delta > 0))
fast_delta = float(gate["delta_psnr"])
full_delta = float(rows[0]["delta_mlsf_minus_v1"])
agree = (fast_delta > 0 and full_delta > 0) or (fast_delta < 0 and full_delta < 0)
table = "\n".join(
    f"| {r['metric']} | {r['v1_8k']:.9f} | {r['mlsf_8k']:.9f} | "
    f"{r['delta_mlsf_minus_v1']:+.9f} |" for r in rows
)
lines = [
    "# V1@8k 与 MLSF@8k：Full-L333 匹配评价",
    "",
    "使用同一 L333 全部 610 张切片、正式 8k EMA 和原验证函数。"
    "逐切片患者、中心/相邻索引及采样 seed 均逐项一致；以下差值为 MLSF − V1。",
    "",
    "| 指标 | V1@8k | MLSF@8k | 差值 |",
    "|---|---:|---:|---:|",
    table,
    "",
    "## 逐切片 PSNR",
    "",
    f"改善 {count}/610 张（{count / 610:.2%}）；配对差值均值 {float(np.mean(psnr_delta)):+.9f} dB，"
    f"中位数 {float(quantiles[3]):+.9f} dB。",
    f"差值分布：min {quantiles[0]:+.9f}，P5 {quantiles[1]:+.9f}，"
    f"P25 {quantiles[2]:+.9f}，P75 {quantiles[4]:+.9f}，"
    f"P95 {quantiles[5]:+.9f}，max {quantiles[6]:+.9f} dB。",
    "",
    "## 与 Fast64 对照",
    "",
    f"Fast64 的 PSNR 差值为 {fast_delta:+.9f} dB；Full-L333 为 "
    f"{full_delta:+.9f} dB。方向{'一致' if agree else '不一致'}；"
    f"两者差 {full_delta - fast_delta:+.9f} dB。Fast64 是固定 64 张子集，"
    "Full-L333 是完整 610 张，不能把两者当作独立患者重复。",
    "",
    "## 权重、协议与运行时间",
    "",
    f"- V1：正式 MedDINO 条件模型，8k EMA：`{v1_checkpoint}`；"
    f"已存 SHA256 `{summary['v1_checkpoint_sha256']}`。",
    f"- MLSF：8k EMA：`{mlsf_checkpoint}`；"
    f"已存 SHA256 `{summary['mlsf_checkpoint_sha256']}`。",
    "- V1 仅调用原 `postprocess.py` 的 "
    "`eval_checkpoint('OFFICIAL_RDDM_2P5D_MEDDINO_CONDITION', 'FULL', "
    "'FULL_L333_V1_8K_MLSF_MATCH')`；MLSF 直接复用已完成的 "
    "`FULL_L333_RESULT.json` 和逐切片评价。",
    "- 两者沿用原始预处理、[-1000,1000] HU 评价窗、PSNR data_range=2000 HU、"
    "SSIM/RMSE/MAE 函数、sampling_timesteps=2 和每张切片固定随机种子。",
    f"- V1 Full 验证 {v1_metrics['validation_seconds']:.3f} s；"
    f"MLSF Full 验证 {mlsf_metrics['validation_seconds']:.3f} s。",
    "",
    "## 数据访问审计",
    "",
    f"- V1：L333 中心 NDCT {v1_access['center_target_opens']} 次，"
    f"中心 LDCT {v1_access['center_ldct_opens']} 次，"
    f"邻近 LDCT {v1_access['neighbor_ldct_opens']} 次。",
    f"- MLSF：L333 中心 NDCT {mlsf_access['center_target_opens']} 次，"
    f"中心 LDCT {mlsf_access['center_ldct_opens']} 次，"
    f"邻近 LDCT {mlsf_access['neighbor_ldct_opens']} 次。",
    "- 两边的 L506、外部库、训练目标、邻近 NDCT 及阻断访问计数均为 0。",
    "",
    "本比较仅覆盖 L333 一个验证患者；逐切片分布不能当作跨患者独立性证据。",
    "",
]
out_md.write_text("\n".join(lines))
print(json.dumps({"full_delta_psnr": full_delta, "improved_slices": count,
                  "improved_fraction": count / 610, "v1_seconds": v1_metrics["validation_seconds"],
                  "mlsf_seconds": mlsf_metrics["validation_seconds"],
                  "csv": str(out_csv), "report": str(out_md)}, ensure_ascii=False))
