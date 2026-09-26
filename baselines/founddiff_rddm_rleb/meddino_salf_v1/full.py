"""E1 EMA full-L333 validation and paired D1 comparison; inference only."""
import csv
import gc
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from anatomy_controlnet_v1.model import parameter_counts
from context_2p5d.data import ContextDataGuard
from mayo2016_smallset_v1.runtime import ROOT, sha256, write_json
from src.rddm_rleb import normalize_to_neg_one_to_one
from meddino_four_arm_v1.model import FourArmControlledUnet
from meddino_four_arm_v1.runtime import load_direct
from meddino_full_l333_v1.evaluate import dataset as full_dataset, evaluate as frozen_full_evaluate
from .model import SALFProvider
from .protocol import ARM, HERE, D1_FULL_ROOT, config

KEYS = ("psnr", "ssim", "rmse_hu", "mae_hu", "soft_psnr",
        "soft_ssim", "soft_rmse_hu", "soft_mae_hu")


def read_csv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    assert rows
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_ema(device):
    manifest = json.loads((HERE / "CHECKPOINT_MANIFEST.json").read_text())
    path = HERE / "checkpoints/step_6000.pt"
    assert manifest["arm"] == ARM and manifest["step"] == 6000
    assert sha256(path) == manifest["sha256"]
    model = load_direct("cpu")
    main = model.model.unet0
    model.model.unet0 = FourArmControlledUnet(main, "D1_MULTI_512")
    u = model.model.unet0
    u.structure = SALFProvider(u.structure)
    u.arm = ARM
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert checkpoint["arm"] == ARM and checkpoint["optimizer_step"] == 6000
    assert checkpoint["ema_updates"] == 6000
    model.load_state_dict(checkpoint["ema_state_dict"], strict=True)
    del checkpoint, main
    model = model.to(device).eval().requires_grad_(False)
    assert model.sampling_timesteps == 2 and model.sum_scale == 0.01
    assert not any(p.requires_grad for p in model.parameters())
    return model


@torch.inference_mode()
def collect_full_gates(model, cfg, data, device):
    guard = ContextDataGuard(cfg, "FULL", allow_training=False)
    data.bind_guard(guard)
    p = model.model.unet0.structure
    p.capture_weights = True
    p.eval()
    rows = []
    started = time.perf_counter()
    try:
        with guard:
            for pos in range(610):
                _, center, _, _, meta = data.load(pos)
                assert meta["patient_id"] == "L333" and meta["slice_index"] == pos
                x = torch.from_numpy(center).unsqueeze(0).to(device)
                p(normalize_to_neg_one_to_one(x))
                record = {"slice_index": pos,
                          "slice_name": data.records[pos][2].name}
                for scale, tag in (("s256", "w256"), ("s128", "w128"), ("s64", "w64")):
                    weights = p.last_weights[scale][0].tolist()
                    assert len(weights) == 4 and abs(sum(weights) - 1) < 1e-6
                    for layer, value in zip((2, 5, 8, 11), weights):
                        record[f"{tag}_{layer}"] = value
                    record[f"{tag}_entropy"] = -sum(
                        w * math.log(max(w, 1e-12)) for w in weights)
                rows.append(record)
    finally:
        p.capture_weights = False
        p.last_weights = None
    access = guard.report()
    assert len(rows) == 610
    assert access["validation_center_indices_opened"] == list(range(610))
    assert all(access[k] == 0 for k in ("L506_opens", "external_memory_opens",
                                        "neighbor_target_opens", "blocked_attempts",
                                        "training_target_opens"))
    return rows, access, time.perf_counter() - started


def gate_stats(rows):
    result = {}
    for scale, tag in (("s256", "w256"), ("s128", "w128"), ("s64", "w64")):
        matrix = np.array([[float(r[f"{tag}_{l}"]) for l in (2, 5, 8, 11)]
                           for r in rows], dtype=np.float64)
        assert matrix.shape == (len(rows), 4)
        assert np.isfinite(matrix).all()
        assert np.allclose(matrix.sum(axis=1), 1, atol=1e-6)
        entropy = -(matrix * np.log(np.maximum(matrix, 1e-12))).sum(axis=1)
        result[scale] = {
            "layers": {
                str(l): {"mean": float(matrix[:, j].mean()),
                         "std": float(matrix[:, j].std(ddof=1)),
                         "p10": float(np.percentile(matrix[:, j], 10)),
                         "p50": float(np.percentile(matrix[:, j], 50)),
                         "p90": float(np.percentile(matrix[:, j], 90))}
                for j, l in enumerate((2, 5, 8, 11))},
            "mean_entropy": float(entropy.mean()),
            "fraction_max_weight_gt_0_95": float(np.mean(matrix.max(axis=1) > 0.95)),
            "collapse_diagnostic": bool(np.mean(matrix.max(axis=1) > 0.95) > 0.9),
        }
    return result


def full():
    training = json.loads((HERE / "training_summary.json").read_text())
    assert training["status"] == "COMPLETE"
    assert training["optimizer_steps"] == training["ema_updates"] == 6000
    assert training["nonfinite_count"] == 0
    assert training["main_hash_before"] == training["main_hash_after"] == training["ema_main_hash_after"]
    assert not (HERE / "FULL_L333_PER_SLICE.csv").exists()
    cfg, data = full_dataset()
    assert len(data) == 610
    device = torch.device("cuda:0")
    torch.set_num_threads(2)
    model = load_ema(device)
    rows, runtime, access = frozen_full_evaluate(
        ARM, model, cfg, data, range(610), device)
    assert len(rows) == 610
    write_csv(HERE / "FULL_L333_PER_SLICE.csv", rows)
    gates, gate_access, gate_seconds = collect_full_gates(model, cfg, data, device)
    write_csv(HERE / "FULL_L333_GATE_WEIGHTS.csv", gates)
    for audit in (access, gate_access):
        assert audit["validation_center_indices_opened"] == list(range(610))
        assert all(audit[k] == 0 for k in ("L506_opens", "external_memory_opens",
                                          "neighbor_target_opens", "blocked_attempts",
                                          "training_target_opens"))
    write_json(HERE / "FULL_L333_DATA_ACCESS.json", {
        "inference": access, "gate_only": gate_access, "optimizer_steps": 0,
        "L506_opens": 0, "external_memory_opens": 0,
        "neighbor_target_opens": 0, "blocked_attempts": 0})
    write_json(HERE / "FULL_L333_RUNTIME.json", {
        **runtime, "gate_diagnostic_seconds": gate_seconds,
        "full_evaluator_source_sha256": sha256(
            ROOT / "meddino_full_l333_v1/evaluate.py")})
    print("SALF_FULL_L333_COMPLETE", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def report():
    training = json.loads((HERE / "training_summary.json").read_text())
    full_runtime = json.loads((HERE / "FULL_L333_RUNTIME.json").read_text())
    train_access = json.loads((HERE / "DATA_ACCESS_AUDIT.json").read_text())
    full_access = json.loads((HERE / "FULL_L333_DATA_ACCESS.json").read_text())
    e1_fast = read_csv(HERE / "FAST64_PER_SLICE.csv")
    e1_full = read_csv(HERE / "FULL_L333_PER_SLICE.csv")
    d1_full = read_csv(D1_FULL_ROOT / "D1_MULTI_512_FULL_L333.csv")
    fast_gates = read_csv(HERE / "FAST64_GATE_WEIGHTS.csv")
    full_gates = read_csv(HERE / "FULL_L333_GATE_WEIGHTS.csv")
    assert len(e1_full) == len(d1_full) == len(full_gates) == 610
    assert len(e1_fast) == len(fast_gates) == 6 * 64
    assert [int(r["slice_index"]) for r in e1_full] == list(range(610))
    assert [r["slice_name"] for r in e1_full] == [r["slice_name"] for r in d1_full]
    delta_rows = []
    for e1, d1 in zip(e1_full, d1_full):
        row = {"slice_index": int(e1["slice_index"]), "slice_name": e1["slice_name"]}
        for key in KEYS:
            row["delta_" + key] = float(e1[key]) - float(d1[key])
        delta_rows.append(row)
    write_csv(HERE / "PAIRED_DELTA_D1_E1.csv", delta_rows)
    delta = np.array([r["delta_psnr"] for r in delta_rows])
    rng = np.random.default_rng(20260926)
    draws = rng.integers(0, 610, size=(10000, 610))
    means = delta[draws].mean(axis=1)
    bootstrap = {
        "mean_delta_psnr": float(delta.mean()),
        "median_delta_psnr": float(np.median(delta)),
        "positive_slice_fraction": float(np.mean(delta > 0)),
        "ci_95_low": float(np.percentile(means, 2.5)),
        "ci_95_high": float(np.percentile(means, 97.5)),
        "n_bootstrap": 10000, "seed": 20260926,
        "unit": "L333 slice, not cross-patient",
    }
    write_json(HERE / "BOOTSTRAP_CI.json", bootstrap)
    fast_last = [r for r in fast_gates if int(r["step"]) == 6000]
    assert len(fast_last) == 64
    thirds = np.array_split(np.arange(610), 3)
    regional = {}
    for label, indices in zip(("first_third", "middle_third", "last_third"), thirds):
        regional[label] = {
            "start_index": int(indices[0]), "end_index": int(indices[-1]),
            "n": len(indices),
            "mean_delta_psnr": float(delta[indices].mean()),
            "gate_weights": gate_stats([full_gates[i] for i in indices]),
        }
    gate_report = {
        "Fast64_6k": gate_stats(fast_last),
        "Full_L333": gate_stats(full_gates),
        "slice_position_thirds": regional,
    }
    write_json(HERE / "GATE_DIAGNOSTICS.json", gate_report)
    e1_mean = {key: float(np.mean([float(r[key]) for r in e1_full])) for key in KEYS}
    d1_mean = {key: float(np.mean([float(r[key]) for r in d1_full])) for key in KEYS}
    d1_train = json.loads((ROOT / "experiments/meddino_four_arm_v1/runs/D1_MULTI_512/final_summary.json").read_text())
    d1_full_runtime = json.loads((D1_FULL_ROOT / "RUNTIME_COMPARISON.json").read_text())["D1_MULTI_512"]
    runtime = {
        "E1_trainable_parameters": training["parameter_audit"]["trainable_parameters"],
        "D1_trainable_parameters": d1_train["parameter_audit"]["trainable_parameters"],
        "added_salf_parameters": 9228,
        "E1_train_peak_vram_bytes": training["peak_vram_bytes"],
        "D1_train_peak_vram_bytes": d1_train["peak_vram_bytes"],
        "E1_seconds_per_step": training["seconds_per_step"],
        "D1_seconds_per_step": d1_train["seconds_per_step"],
        "training_step_time_ratio": training["seconds_per_step"] / d1_train["seconds_per_step"],
        "total_training_seconds": training["total_training_seconds"],
        "full_inference_seconds": full_runtime["total_inference_wall_seconds"],
        "D1_full_inference_seconds": d1_full_runtime["total_inference_wall_seconds"],
        "full_inference_ratio": full_runtime["total_inference_wall_seconds"] /
                                d1_full_runtime["total_inference_wall_seconds"],
        "full_peak_cuda_allocated_bytes": full_runtime["peak_vram_bytes"],
    }
    write_json(HERE / "RUNTIME_AUDIT.json", runtime)
    assert all(train_access[k] == 0 for k in ("L506_opens", "external_memory_opens",
                                               "neighbor_target_opens", "blocked_attempts"))
    assert all(full_access[k] == 0 for k in ("L506_opens", "external_memory_opens",
                                             "neighbor_target_opens", "blocked_attempts",
                                             "optimizer_steps"))
    history = read_csv(HERE / "VALIDATION_HISTORY.csv")
    assert [int(r["step"]) for r in history] == [0, 500, 1000, 2000, 4000, 6000]
    for r in history:
        assert r["weights"] == "ema" and r["mode"] == "FAST" and int(r["slices"]) == 64
        assert math.isfinite(float(r["psnr"])) and math.isfinite(float(r["delta_psnr_vs_d1"]))
    gate_mean = gate_report["Full_L333"]
    sample_dependent = any(
        gate_mean[s]["layers"][str(l)]["std"] > 1e-4
        for s in ("s256", "s128", "s64") for l in (2, 5, 8, 11))
    scale_dependent = any(
        abs(gate_mean[a]["layers"][str(l)]["mean"] -
            gate_mean[b]["layers"][str(l)]["mean"]) > 1e-4
        for a, b in (("s256", "s128"), ("s128", "s64"))
        for l in (2, 5, 8, 11))
    audit = {
        "status": "PASS", "arm": ARM, "optimizer_steps_train": 6000,
        "optimizer_steps_full_inference": 0, "ema": True,
        "fast64_nodes": [0, 500, 1000, 2000, 4000, 6000],
        "full_L333_slices": 610, "L506_opens": 0,
        "external_memory_opens": 0, "neighbor_target_opens": 0,
        "blocked_attempts": 0, "nonfinite_count": training["nonfinite_count"],
        "main_unchanged": training["main_hash_before"] ==
                          training["main_hash_after"] ==
                          training["ema_main_hash_after"],
        "extra_salf_parameters": 9228,
        "step_zero_equal": json.loads((HERE / "INITIALIZATION_PARITY_AUDIT.json").read_text())
                           ["step_zero_equivalent"],
        "checkpoint_sha256": json.loads((HERE / "CHECKPOINT_MANIFEST.json").read_text())["sha256"],
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p)
                          for p in (ROOT / "meddino_salf_v1").glob("*.py")},
        "full_evaluator_sha256": full_runtime["full_evaluator_source_sha256"],
        "formal_source_sha256": sha256(ROOT / "src/rddm_rleb.py"),
        "fast64_split_sha256": sha256(ROOT / "splits/L333_fast64.txt"),
        "sample_dependent_weights": sample_dependent,
        "scale_dependent_weights": scale_dependent,
        "bootstrap": bootstrap,
    }
    assert audit["main_unchanged"] and audit["step_zero_equal"]
    assert audit["formal_source_sha256"] == "e8ed93223fbdf4a0e2521f73c7a4f213561f6a191a1a7b2a5f95934b85ebf574"
    write_json(HERE / "FINAL_AUDIT.json", audit)
    lines = [
        "# E1-SALF: Scale-Aware MedDINO Layer Fusion", "",
        "Development validation only: frozen MedDINO, D1 ControlNet/RDDM, 512 input; "
        "only 12 scale/layer Linear scorers were added.", "",
        "| Arm | 2k PSNR | 4k PSNR | 6k PSNR | 6k Delta vs D1 | "
        "Fast64 positive slices | Trainable parameters | Seconds/step |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, source in (("D1_MULTI_512", ROOT / "experiments/meddino_four_arm_v1/runs/D1_MULTI_512/validation_history.csv"),
                         (ARM, HERE / "VALIDATION_HISTORY.csv")):
        entries = {int(r["step"]): r for r in read_csv(source)}
        if name == ARM:
            e1_6k = [r for r in e1_fast if int(r["step"]) == 6000]
            d1_6k = read_csv(ROOT / "experiments/meddino_four_arm_v1/runs/D1_MULTI_512/per_slice_step_6000.csv")
            positive = float(np.mean([float(a["psnr"]) > float(b["psnr"])
                                      for a, b in zip(e1_6k, d1_6k)]))
            params, secs = runtime["E1_trainable_parameters"], runtime["E1_seconds_per_step"]
        else:
            positive = None
            params, secs = runtime["D1_trainable_parameters"], runtime["D1_seconds_per_step"]
        lines.append(
            f'| {name} | {float(entries[2000]["psnr"]):.6f} | '
            f'{float(entries[4000]["psnr"]):.6f} | {float(entries[6000]["psnr"]):.6f} | '
            f'{float(entries[6000]["psnr"])-float(read_csv(ROOT / "experiments/meddino_four_arm_v1/runs/D1_MULTI_512/validation_history.csv")[-1]["psnr"]):+.6f} | '
            f'{positive if positive is not None else "NA"} | {params} | {secs:.3f} |')
    lines += ["", "| Model | PSNR | SSIM | RMSE HU | MAE HU | Soft PSNR |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, m in (("D1_MULTI_512", d1_mean), (ARM, e1_mean)):
        lines.append(f'| {name} | {m["psnr"]:.6f} | {m["ssim"]:.6f} | '
                     f'{m["rmse_hu"]:.4f} | {m["mae_hu"]:.4f} | '
                     f'{m["soft_psnr"]:.6f} |')
    lines += [
        "", f'E1-D1 Full-L333 mean PSNR delta: {bootstrap["mean_delta_psnr"]:+.6f} dB; '
        f'median {bootstrap["median_delta_psnr"]:+.6f}; '
        f'positive slice fraction {bootstrap["positive_slice_fraction"]:.3f}; '
        f'slice-bootstrap 95% CI [{bootstrap["ci_95_low"]:+.6f}, '
        f'{bootstrap["ci_95_high"]:+.6f}].', "",
        "| Scale | F2 | F5 | F8 | F11 | Mean entropy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for scale in ("s256", "s128", "s64"):
        item = gate_mean[scale]
        vals = [item["layers"][str(l)]["mean"] for l in (2, 5, 8, 11)]
        lines.append(f'| {scale} | ' + " | ".join(f"{v:.4f}" for v in vals) +
                     f' | {item["mean_entropy"]:.4f} |')
    lines += ["", "## Interpretation", "",
              f'Full-L333 positive mean: {bootstrap["mean_delta_psnr"] > 0}; '
              f'positive fraction >0.5: {bootstrap["positive_slice_fraction"] > 0.5}; '
              f'CI lower bound >0: {bootstrap["ci_95_low"] > 0}.',
              f'Sample-dependent weights: {sample_dependent}; '
              f'scale-dependent weights: {scale_dependent}.',
              f'E1/D1 training step-time ratio: {runtime["training_step_time_ratio"]:.2f}x; '
              f'Full inference ratio: {runtime["full_inference_ratio"]:.2f}x.',
              "The three slice-index thirds are descriptive and have no anatomical labels.",
              "This is L333 development validation. Slice-level bootstrap intervals are not "
              "cross-patient generalization intervals.",
              "No L506, external memory, neighbor NDCT, MedDINO fine-tuning, "
              "E2, or formal 8-patient training was performed.",
              ""]
    (HERE / "MEDDINO_SALF_REPORT.md").write_text("\n".join(lines))
    print("SALF_FINAL_AUDIT_PASS", flush=True)


if __name__ == "__main__":
    full()
    report()
