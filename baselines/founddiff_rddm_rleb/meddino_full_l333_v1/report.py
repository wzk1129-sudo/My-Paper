"""Paired slice-level analysis for the isolated full-L333 inference outputs."""
import csv
import hashlib
import json
import math
import platform
import subprocess
from pathlib import Path

import numpy as np

from mayo2016_smallset_v1.runtime import ROOT, sha256
from .evaluate import OUT, NAMES, METRICS, checkpoint_audit, write_json, write_csv

PAIRS = (
    ("C3_D1", "D1_MULTI_512", "C3_FINAL"),
    ("D1_D3", "D3_MULTI_1024", "D1_MULTI_512"),
    ("C3_D3", "D3_MULTI_1024", "C3_FINAL"),
)
BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 20260926


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def values(rows, key):
    result = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    assert result.shape == (610,) and np.isfinite(result).all()
    return result


def statistics(delta):
    assert delta.shape == (610,) and np.isfinite(delta).all()
    return {
        "mean": float(delta.mean()), "median": float(np.median(delta)),
        "std": float(delta.std(ddof=1)),
        "positive_delta_fraction": float(np.mean(delta > 0)),
        **{f"p{p}": float(np.percentile(delta, p)) for p in (5, 10, 25, 50, 75, 90, 95)},
    }


def bootstrap_ci(delta, seed):
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, 610, size=(BOOTSTRAP_N, 610))
    means = delta[draws].mean(axis=1)
    return {
        "mean_delta": float(delta.mean()),
        "ci_95_low": float(np.percentile(means, 2.5)),
        "ci_95_high": float(np.percentile(means, 97.5)),
        "positive_slice_fraction": float(np.mean(delta > 0)),
        "n_bootstrap": BOOTSTRAP_N, "seed": seed,
        "bootstrap_unit": "L333 slice",
        "interpretation": "Within-patient development validation; not cross-patient generalization",
    }


def report():
    config = json.loads((OUT / "EXPERIMENT_CONFIG.json").read_text())
    repro = json.loads((OUT / "EVALUATOR_REPRO_AUDIT.json").read_text())
    access = json.loads((OUT / "DATA_ACCESS_AUDIT.json").read_text())
    runtime = json.loads((OUT / "RUNTIME_COMPARISON.json").read_text())
    recorded_checkpoints = json.loads((OUT / "CHECKPOINT_AUDIT.json").read_text())
    assert config["slices"] == 610 and repro["status"] == "PASS"
    assert set(runtime) == set(NAMES) and set(access["by_model"]) == set(NAMES)
    assert all(access[key] == 0 for key in
               ("optimizer_steps", "L506_opens", "external_memory_opens",
                "neighbor_target_opens", "blocked_attempts"))
    assert sha256(OUT / "FULL_L333_SLICE_LIST.txt") == config["slice_list_sha256"]
    actual_checkpoints = checkpoint_audit()
    assert actual_checkpoints == recorded_checkpoints
    rows = {}
    for name in NAMES:
        path = OUT / (name + "_FULL_L333.csv")
        rows[name] = read_csv(path)
        assert len(rows[name]) == 610
        assert [int(row["slice_index"]) for row in rows[name]] == list(range(610))
        assert all(row["slice_name"] == f"L333_25_{i:03d}_img.npy"
                   for i, row in enumerate(rows[name]))
        assert all(np.isfinite(values(rows[name], key)).all() for key in METRICS)
        opened = access["by_model"][name]
        assert opened["training_target_opens"] == opened["neighbor_target_opens"] == 0
        assert opened["validation_center_indices_opened"] == list(range(610))
        assert opened["center_target_opens"] == 610
        # At z=0 and z=609, a clamped neighbor opens the center LDCT again.
        assert opened["center_ldct_opens"] == 612
        assert opened["neighbor_ldct_opens"] == 1218
        assert opened["L506_opens"] == opened["external_memory_opens"] == opened["blocked_attempts"] == 0
    for name in NAMES:
        runtime[name]["mean_sampling_seconds_per_slice"] = runtime[name].pop(
            "mean_seconds_per_slice")
        runtime[name]["mean_seconds_per_slice"] = (
            runtime[name]["total_inference_wall_seconds"] / 610)
    runtime["D3_vs_D1"] = {
        "wall_time_ratio": runtime["D3_MULTI_1024"]["total_inference_wall_seconds"] /
                           runtime["D1_MULTI_512"]["total_inference_wall_seconds"],
        "vram_ratio": runtime["D3_MULTI_1024"]["peak_vram_bytes"] /
                      runtime["D1_MULTI_512"]["peak_vram_bytes"],
    }
    write_json(OUT / "RUNTIME_COMPARISON.json", runtime)
    paired = {}
    ci = {}
    for label, left, right in PAIRS:
        assert [row["slice_name"] for row in rows[left]] == [
            row["slice_name"] for row in rows[right]]
        out_rows = []
        for a, b in zip(rows[left], rows[right]):
            entry = {"slice_index": int(a["slice_index"]), "slice_name": a["slice_name"]}
            for key in METRICS:
                entry["delta_" + key] = float(a[key]) - float(b[key])
            out_rows.append(entry)
        write_csv(OUT / ("PAIRED_DELTA_" + label + ".csv"), out_rows)
        differences = {key: values(out_rows, "delta_" + key) for key in METRICS}
        paired[label] = {
            "left": left, "right": right,
            "psnr": statistics(differences["psnr"]),
            "rmse_hu": statistics(differences["rmse_hu"]),
            "soft_psnr": statistics(differences["soft_psnr"]),
        }
        ci[label] = bootstrap_ci(differences["psnr"], BOOTSTRAP_SEED)
    write_json(OUT / "PAIRED_STATS.json", paired)
    write_json(OUT / "BOOTSTRAP_CI.json", ci)
    thirds = np.array_split(np.arange(610), 3)
    position = {}
    for label in ("C3_D1", "D1_D3"):
        csv_rows = read_csv(OUT / ("PAIRED_DELTA_" + label + ".csv"))
        delta = values(csv_rows, "delta_psnr")
        position[label] = {
            title: {"start_index": int(indices[0]), "end_index": int(indices[-1]),
                    "n": len(indices), "mean_delta_psnr": float(delta[indices].mean())}
            for title, indices in zip(("first_third", "middle_third", "last_third"), thirds)
        }
    write_json(OUT / "SLICE_POSITION_TRENDS.json", position)
    summary = []
    for name in NAMES:
        row = {"model": name}
        row.update({key: float(values(rows[name], key).mean()) for key in METRICS})
        row["positive_slices_vs_reference"] = (
            "" if name == "C3_FINAL" else
            paired["C3_D1" if name == "D1_MULTI_512" else "D1_D3"]
                  ["psnr"]["positive_delta_fraction"])
        row["mean_seconds_per_slice"] = runtime[name]["mean_seconds_per_slice"]
        row["total_inference_wall_seconds"] = runtime[name]["total_inference_wall_seconds"]
        row["peak_vram_bytes"] = runtime[name]["peak_vram_bytes"]
        summary.append(row)
    write_csv(OUT / "FULL_L333_SUMMARY.csv", summary)
    judgments = {}
    for label in ("C3_D1", "D1_D3"):
        psnr = paired[label]["psnr"]
        trend = position[label]
        judgments[label] = {
            "all_three_thirds_positive": all(x["mean_delta_psnr"] > 0
                                              for x in trend.values()),
            "majority_slices_positive": psnr["positive_delta_fraction"] > 0.5,
            "bootstrap_ci_lower_positive": ci[label]["ci_95_low"] > 0,
            "stable_positive_on_L333": (
                psnr["positive_delta_fraction"] > 0.5 and
                ci[label]["ci_95_low"] > 0 and
                all(x["mean_delta_psnr"] > 0 for x in trend.values())),
        }
    audit = {
        "status": "PASS", "branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "parent_commit": config["parent_commit"], "optimizer_steps": 0,
        "weights_unchanged": True, "checkpoint_sha256": {
            k: v["sha256"] for k, v in actual_checkpoints.items()},
        "model_counts": {name: len(rows[name]) for name in NAMES},
        "ema_weights": True, "sampling_timesteps": 2, "sum_scale": 0.01,
        "patient": "L333", "slice_list_sha256": config["slice_list_sha256"],
        "reproduction_status": repro["status"],
        "max_fast64_reproduction_error": {
            name: repro["models"][name]["max_abs_metric_difference"] for name in NAMES},
        "full_l333_only": True, "L506_opens": 0, "external_memory_opens": 0,
        "neighbor_target_opens": 0, "blocked_attempts": 0,
        "data_access": access, "judgments": judgments,
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in (
            ROOT / "meddino_full_l333_v1/evaluate.py",
            ROOT / "meddino_full_l333_v1/report.py",
            ROOT / "meddino_full_l333_v1/run_full.sh")},
        "evaluator_sha256": config["frozen_evaluator_sha256"],
        "frozen_protocol_sha256": config["frozen_protocol_sha256"],
        "fast64_split_sha256": config["fast64_split_sha256"],
        "python": platform.python_version(),
    }
    write_json(OUT / "FINAL_AUDIT.json", audit)
    lines = [
        "# MedDINO Full-L333 Validation V1", "",
        "Inference-only L333 development validation: all 610 ordered slices, EMA step-6000 checkpoints, "
        "two-step RDDM sampler with sum_scale=0.01 and slice-wise seed 20260829+index.",
        "Wide-window [-1000,1000] HU is primary; [-160,240] HU is "
        "**SECONDARY_SOFT_TISSUE_METRIC** on the same predictions.", "",
        "| Model | PSNR | SSIM | RMSE HU | MAE HU | Soft PSNR | Soft SSIM | Soft RMSE HU | "
        "Positive slices vs reference | Mean wall sec/slice | Peak VRAM GiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary:
        frac = "NA" if r["positive_slices_vs_reference"] == "" else (
            f'{r["positive_slices_vs_reference"]:.3f}')
        lines.append(f'| {r["model"]} | {r["psnr"]:.6f} | {r["ssim"]:.6f} | '
                     f'{r["rmse_hu"]:.4f} | {r["mae_hu"]:.4f} | '
                     f'{r["soft_psnr"]:.6f} | {r["soft_ssim"]:.6f} | '
                     f'{r["soft_rmse_hu"]:.4f} | {frac} | '
                     f'{r["mean_seconds_per_slice"]:.3f} | '
                     f'{r["peak_vram_bytes"]/1024**3:.3f} |')
    lines += ["", "| Comparison | Mean Delta PSNR | Median | Slice-bootstrap 95% CI | "
              "Positive slices | Mean Delta Soft PSNR |",
              "|---|---:|---:|---:|---:|---:|"]
    for label, _, _ in PAIRS:
        p = paired[label]["psnr"]
        s = paired[label]["soft_psnr"]
        interval = ci[label]
        lines.append(f'| {label} | {p["mean"]:+.6f} | {p["median"]:+.6f} | '
                     f'[{interval["ci_95_low"]:+.6f}, {interval["ci_95_high"]:+.6f}] | '
                     f'{p["positive_delta_fraction"]:.3f} | {s["mean"]:+.6f} |')
    lines += [
        "", "## Paired interpretation", "",
        f'- D1 Multi-512 vs C3 Final: stable positive on this L333 volume = '
        f'{judgments["C3_D1"]["stable_positive_on_L333"]}. '
        f'All-three-thirds positive = {judgments["C3_D1"]["all_three_thirds_positive"]}.',
        f'- D3 Multi-1024 vs D1 Multi-512: stable positive on this L333 volume = '
        f'{judgments["D1_D3"]["stable_positive_on_L333"]}. '
        f'All-three-thirds positive = {judgments["D1_D3"]["all_three_thirds_positive"]}.',
        f'- D3/D1 inference wall-time ratio = {runtime["D3_vs_D1"]["wall_time_ratio"]:.2f}x; '
        f'peak VRAM ratio = {runtime["D3_vs_D1"]["vram_ratio"]:.2f}x. '
        "The observed PSNR gain and compute cost should both determine whether D3 remains a candidate.",
        "", "## Slice-position thirds", "",
    ]
    for label in ("C3_D1", "D1_D3"):
        for title, entry in position[label].items():
            lines.append(f'- {label} {title} (indices {entry["start_index"]}-{entry["end_index"]}, '
                         f'n={entry["n"]}): {entry["mean_delta_psnr"]:+.6f} dB.')
    lines += [
        "", "## Candidate priority", "",
        "C3 Final remains the archived final-layer comparator.",
        f"D1 Multi-512 is the primary candidate for any later formal 8-patient test: "
        f"its full-L333 paired gain over C3 is {paired['C3_D1']['psnr']['mean']:+.6f} dB "
        f"with {paired['C3_D1']['psnr']['positive_delta_fraction']:.3f} positive slices.",
        f"D3 Multi-1024 is a conditional compute-cost candidate, not the first formal-training "
        f"choice: it adds only {paired['D1_D3']['psnr']['mean']:+.6f} dB over D1 "
        f"at {runtime['D3_vs_D1']['wall_time_ratio']:.2f}x inference wall time.",
        "Peak VRAM here means peak CUDA allocated memory measured by PyTorch, not "
        "process-level nvidia-smi reservation.",
        "", "## Scope and safety", "",
        "The 10,000 paired-bootstrap draws resample L333 slices only. Their intervals are "
        "not cross-patient generalization intervals.",
        "No training, optimizer step, external memory, neighbor NDCT, or L506 access occurred. "
        "Model weights, checkpoints, formal source, and frozen evaluator were unchanged.",
        "This is development validation, not a test-set or final paper result. "
        "No formal 8-patient training was started.", "",
    ]
    (OUT / "MEDDINO_FULL_L333_REPORT.md").write_text("\n".join(lines))
    print("FULL_L333_REPORT_PASS", flush=True)


if __name__ == "__main__":
    report()
