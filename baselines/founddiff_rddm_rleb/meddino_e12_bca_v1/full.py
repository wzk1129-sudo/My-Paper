"""Frozen Full-L333 inference and paired D1/E12-HAF comparison."""
import csv
import json
import math

import numpy as np
import torch

from context_2p5d.model import state_hash
from mayo2016_smallset_v1.runtime import ROOT, sha256, write_json
from meddino_four_arm_v1.runtime import load_direct
from meddino_four_arm_v1.structure import MultiProvider
from meddino_full_l333_v1.evaluate import dataset as full_dataset, evaluate as frozen_evaluate
from meddino_salf_v1.train import write_csv
from .model import graft_bca
from .protocol import ARM, HERE, FORMAL_SOURCE_SHA, FAST64_SPLIT_SHA, config

D1_CSV = ROOT / "experiments/meddino_full_l333_validation_v1/D1_MULTI_512_FULL_L333.csv"
E12_CSV = ROOT / "experiments/meddino_e12_haf_v1/FULL_L333_PER_SLICE.csv"
METRICS = ("psnr", "ssim", "rmse_hu", "mae_hu")


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_ema(device):
    ck_path = HERE / "checkpoints/step_6000.pt"
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    assert ck["arm"] == ARM and ck["optimizer_step"] == ck["ema_updates"] == 6000
    model = load_direct("cpu")
    graft_bca(model.model.unet0, MultiProvider("D1_MULTI_512"))
    model.load_state_dict(ck["ema_state_dict"], strict=True)
    assert model.objective == "pred_res" and model.sampling_timesteps == 2
    assert model.sum_scale == .01
    return model.to(device).eval()


def metric_mean(rows):
    return {key: float(np.mean([float(row[key]) for row in rows])) for key in METRICS}


def paired(rows, reference, name):
    assert len(rows) == len(reference) == 610
    assert [int(r["slice_index"]) for r in rows] == list(range(610))
    assert [int(r["slice_index"]) for r in reference] == list(range(610))
    assert [r["slice_name"] for r in rows] == [r["slice_name"] for r in reference]
    per_slice = []
    for a, b in zip(rows, reference):
        per_slice.append({
            "slice_index": int(a["slice_index"]), "slice_name": a["slice_name"],
            **{"delta_" + key: float(a[key]) - float(b[key]) for key in METRICS},
        })
    delta = np.array([r["delta_psnr"] for r in per_slice], dtype=np.float64)
    draws = np.random.default_rng(20260926).integers(0, 610, size=(10000, 610))
    means = delta[draws].mean(axis=1)
    comparison = {
        "reference": name,
        "candidate_metrics": metric_mean(rows),
        "reference_metrics": metric_mean(reference),
        "aggregate_delta": {key: metric_mean(rows)[key] - metric_mean(reference)[key]
                            for key in METRICS},
        "paired_slice_mean_delta_psnr": float(delta.mean()),
        "paired_slice_bootstrap_95_ci_psnr": [float(v) for v in np.quantile(means, [.025, .975])],
        "positive_slice_fraction": float((delta > 0).mean()),
        "bootstrap_seed": 20260926, "bootstrap_draws": 10000,
        "inference_seconds_ratio": float(
            np.mean([float(r["inference_seconds"]) for r in rows]) /
            np.mean([float(r["inference_seconds"]) for r in reference])),
    }
    write_csv(HERE / ("PAIRED_VS_" + name + ".csv"), per_slice)
    return comparison


def main():
    assert not (HERE / "FULL_L333_PER_SLICE.csv").exists()
    cfg = config()
    training = json.loads((HERE / "FINAL_TRAIN_SUMMARY.json").read_text())
    assert training["status"] == "COMPLETE"
    assert training["optimizer_step"] == training["ema_updates"] == 6000
    assert training["nonfinite_count"] == 0
    assert all(v > 0 for v in training["grad_usage"].values())
    data_cfg, data = full_dataset()
    assert len(data) == 610
    device = torch.device("cuda:0")
    torch.set_num_threads(2)
    model = load_ema(device)
    rows, runtime, access = frozen_evaluate(
        ARM, model, data_cfg, data, range(610), device)
    assert len(rows) == 610
    assert access["validation_center_indices_opened"] == list(range(610))
    assert all(access[k] == 0 for k in (
        "L506_opens", "external_memory_opens", "neighbor_target_opens",
        "blocked_attempts", "training_target_opens"))
    write_csv(HERE / "FULL_L333_PER_SLICE.csv", rows)
    d1, e12 = read_csv(D1_CSV), read_csv(E12_CSV)
    comparisons = {
        "D1": paired(rows, d1, "D1"),
        "E12_HAF": paired(rows, e12, "E12_HAF"),
    }
    u = model.model.unet0
    attention = {
        "alpha128": float(u.bca128.alpha.detach()),
        "alpha64": float(u.bca64.alpha.detach()),
        "attention128": {k: float(v) for k, v in u.bca128.last_activity.items()},
        "attention64": {k: float(v) for k, v in u.bca64.last_activity.items()},
    }
    d1_training = json.loads((ROOT / "experiments/meddino_four_arm_v1/runs/D1_MULTI_512/final_summary.json").read_text())
    e12_training = json.loads((ROOT / "experiments/meddino_e12_haf_v1/training_summary.json").read_text())
    summary = {
        "status": "COMPLETE", "arm": ARM, "weights": "step6000 EMA",
        "slices": 610, "optimizer_steps_during_full": 0,
        "metrics": metric_mean(rows), "comparisons": comparisons,
        "runtime": runtime, "attention": attention,
        "train_seconds_per_step": training["seconds_per_step"],
        "train_step_time_ratio_vs_D1": training["seconds_per_step"] / d1_training["seconds_per_step"],
        "train_step_time_ratio_vs_E12_HAF": training["seconds_per_step"] / e12_training["seconds_per_step"],
        "access": access,
    }
    write_json(HERE / "FULL_L333_SUMMARY.json", summary)
    write_json(HERE / "FULL_L333_DATA_ACCESS.json", access)
    source_sha = sha256(ROOT / "src/rddm_rleb.py")
    split_sha = sha256(ROOT / "splits/L333_fast64.txt")
    eval_sha = sha256(ROOT / "meddino_full_l333_v1/evaluate.py")
    assert source_sha == FORMAL_SOURCE_SHA and split_sha == FAST64_SPLIT_SHA
    assert eval_sha == "94ddc8b2d9e9bae08457bb0f4e57c20f2d282eb80ef5b77c50304818f3f29366"
    audit = {
        "status": "PASS", "branch": "experiment/rddm-meddino-e12-bca-v1",
        "formal_source_sha256": source_sha, "fast64_split_sha256": split_sha,
        "full_evaluator_sha256": eval_sha,
        "checkpoint_sha256": sha256(HERE / "checkpoints/step_6000.pt"),
        "smoke_pass": True, "training_6000_complete": True,
        "full_l333_610_complete": True,
        "nonfinite_count": training["nonfinite_count"],
        "cross_attention_gradient_usage": training["grad_usage"],
        "conditioning_residual_nonzero": all(
            attention[key]["weighted_residual_rms"] > 0
            for key in ("attention128", "attention64")),
        "L506_opens": training["data_access"]["L506_opens"] + access["L506_opens"],
        "external_memory_opens": training["data_access"]["external_memory_opens"] + access["external_memory_opens"],
        "neighbor_target_opens": training["data_access"]["neighbor_target_opens"] + access["neighbor_target_opens"],
        "blocked_attempts": training["data_access"]["blocked_attempts"] + access["blocked_attempts"],
        "meddino_frozen": True,
        "comparison_caveat": "D1/E12-HAF froze RDDM main; BCA trained main per this experiment request, so deltas do not isolate injection mechanism alone.",
        "source_discrepancy": "Actual E12-HAF has no Sobel/Laplacian/Edge branch; none was added.",
    }
    assert audit["L506_opens"] == audit["external_memory_opens"] == 0
    assert audit["neighbor_target_opens"] == audit["blocked_attempts"] == 0
    assert audit["conditioning_residual_nonzero"]
    write_json(HERE / "FINAL_AUDIT.json", audit)
    lines = [
        "# E12-BCA Fast6k / Full-L333",
        "",
        "E12 topology: frozen MedDINO 2/5/8/11, SALF, hierarchical reassembly.",
        "The actual E12 source has no Edge branch; no new extractor was added.",
        "Q comes from RDDM encoder; K/V come from E12 structure at 128/64.",
        "RDDM main was trainable here, unlike frozen D1/E12-HAF; comparison is therefore a joint architecture/training-parameter change.",
        "",
        "## Full L333 (610 slices, EMA)",
        "",
        "| Arm | PSNR | SSIM | RMSE HU | MAE HU |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in (("E12-BCA", summary["metrics"]),
                         ("D1", comparisons["D1"]["reference_metrics"]),
                         ("E12-HAF", comparisons["E12_HAF"]["reference_metrics"])):
        lines.append(f'| {name} | {values["psnr"]:.6f} | {values["ssim"]:.6f} | {values["rmse_hu"]:.6f} | {values["mae_hu"]:.6f} |')
    lines += ["", "## Paired comparisons", ""]
    for name, values in comparisons.items():
        lo, hi = values["paired_slice_bootstrap_95_ci_psnr"]
        lines.append(f'- BCA - {name}: delta PSNR {values["aggregate_delta"]["psnr"]:+.6f} dB; paired slice bootstrap 95% CI [{lo:+.6f}, {hi:+.6f}] dB; positive slice fraction {values["positive_slice_fraction"]:.3f}.')
    lines += [
        "", f'alpha128={attention["alpha128"]:.6f}, alpha64={attention["alpha64"]:.6f}.',
        f'Train step time {summary["train_seconds_per_step"]:.3f}s; ratios vs D1 {summary["train_step_time_ratio_vs_D1"]:.2f}x, vs E12-HAF {summary["train_step_time_ratio_vs_E12_HAF"]:.2f}x.',
        f'Full inference {runtime["total_inference_wall_seconds"]:.1f}s; peak VRAM {runtime["peak_vram_bytes"] / 2**30:.2f} GiB.',
        "No training during Full-L333; L506 and external memory untouched.",
    ]
    (HERE / "E12_BCA_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": audit["status"], "psnr": summary["metrics"]["psnr"],
                      "delta_d1": comparisons["D1"]["aggregate_delta"]["psnr"],
                      "delta_e12": comparisons["E12_HAF"]["aggregate_delta"]["psnr"]}), flush=True)


if __name__ == "__main__":
    main()
