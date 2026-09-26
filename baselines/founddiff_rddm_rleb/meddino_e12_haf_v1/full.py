"""Inference-only EMA Full-L333 and paired D1 report for E12-HAF."""
import csv
import gc
import json
import math
from pathlib import Path

import numpy as np
import torch

from anatomy_controlnet_v1.model import parameter_counts
from mayo2016_smallset_v1.runtime import ROOT, sha256, write_json
from meddino_four_arm_v1.model import FourArmControlledUnet
from meddino_four_arm_v1.runtime import load_direct
from meddino_full_l333_v1.evaluate import dataset as full_dataset, evaluate as frozen_full_evaluate
from meddino_salf_v1.full import collect_full_gates, gate_stats
from .model import graft_e12
from .protocol import ARM, HERE, D1_FULL_ROOT, D1_ROOT, STEPS, config

KEYS = ("psnr", "ssim", "rmse_hu", "mae_hu",
        "soft_psnr", "soft_ssim", "soft_rmse_hu", "soft_mae_hu")


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    assert rows and not Path(path).exists()
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_ema(device):
    audit = json.loads((HERE / "CHECKPOINT_AUDIT.json").read_text())
    path = HERE / "checkpoints/step_6000.pt"
    assert audit["arm"] == ARM and audit["step"] == 6000
    assert sha256(path) == audit["sha256"]
    model = load_direct("cpu")
    main = model.model.unet0
    model.model.unet0 = FourArmControlledUnet(main, "D1_MULTI_512")
    model = graft_e12(model)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert checkpoint["arm"] == ARM
    assert checkpoint["optimizer_step"] == checkpoint["ema_updates"] == 6000
    model.load_state_dict(checkpoint["ema_state_dict"], strict=True)
    del checkpoint, main
    model = model.to(device).eval().requires_grad_(False)
    assert model.sampling_timesteps == 2 and model.sum_scale == .01
    assert not any(p.requires_grad for p in model.parameters())
    return model


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
        assert all(audit[k] == 0 for k in (
            "L506_opens", "external_memory_opens", "neighbor_target_opens",
            "blocked_attempts", "training_target_opens"))
    write_json(HERE / "FULL_L333_DATA_ACCESS.json", {
        "inference": access, "gate_only": gate_access, "optimizer_steps": 0,
        "L506_opens": 0, "external_memory_opens": 0,
        "neighbor_target_opens": 0, "blocked_attempts": 0})
    write_json(HERE / "FULL_L333_RUNTIME.json", {
        **runtime, "gate_diagnostic_seconds": gate_seconds,
        "full_evaluator_source_sha256": sha256(ROOT / "meddino_full_l333_v1/evaluate.py")})
    print("E12_FULL_L333_COMPLETE", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def report():
    training = json.loads((HERE / "training_summary.json").read_text())
    runtime_full = json.loads((HERE / "FULL_L333_RUNTIME.json").read_text())
    train_access = json.loads((HERE / "DATA_ACCESS_AUDIT.json").read_text())
    full_access = json.loads((HERE / "FULL_L333_DATA_ACCESS.json").read_text())
    fast = read_csv(HERE / "FAST64_PER_SLICE.csv")
    e12 = read_csv(HERE / "FULL_L333_PER_SLICE.csv")
    d1 = read_csv(D1_FULL_ROOT / "D1_MULTI_512_FULL_L333.csv")
    fast_gates = read_csv(HERE / "FAST64_GATE_WEIGHTS.csv")
    full_gates = read_csv(HERE / "FULL_L333_GATE_WEIGHTS.csv")
    assert len(e12) == len(d1) == len(full_gates) == 610
    assert len(fast) == len(fast_gates) == 6 * 64
    assert [int(r["slice_index"]) for r in e12] == list(range(610))
    assert [r["slice_name"] for r in e12] == [r["slice_name"] for r in d1]
    paired = []
    for a, b in zip(e12, d1):
        row = {"slice_index": int(a["slice_index"]), "slice_name": a["slice_name"]}
        for key in KEYS:
            row["delta_" + key] = float(a[key]) - float(b[key])
        paired.append(row)
    write_csv(HERE / "PAIRED_DELTA_D1_E12.csv", paired)
    delta = np.array([r["delta_psnr"] for r in paired], dtype=np.float64)
    draws = np.random.default_rng(20260926).integers(0, 610, size=(10000, 610))
    means = delta[draws].mean(axis=1)
    thirds = {}
    for name, indices in zip(("first", "middle", "last"),
                             np.array_split(np.arange(610), 3)):
        thirds[name] = {"start": int(indices[0]), "end": int(indices[-1]),
                        "n": len(indices),
                        "mean_delta_psnr": float(delta[indices].mean())}
    bootstrap = {
        "mean_delta_psnr": float(delta.mean()),
        "median_delta_psnr": float(np.median(delta)),
        "positive_slice_fraction": float(np.mean(delta > 0)),
        "p10": float(np.percentile(delta, 10)),
        "p25": float(np.percentile(delta, 25)),
        "p50": float(np.percentile(delta, 50)),
        "p75": float(np.percentile(delta, 75)),
        "p90": float(np.percentile(delta, 90)),
        "ci_95_low": float(np.percentile(means, 2.5)),
        "ci_95_high": float(np.percentile(means, 97.5)),
        "n_bootstrap": 10000, "seed": 20260926,
        "unit": "L333 slice, not independent patient",
        "z_thirds": thirds,
    }
    write_json(HERE / "BOOTSTRAP_CI.json", bootstrap)
    final_gates = [r for r in fast_gates if int(r["step"]) == 6000]
    assert len(final_gates) == 64
    gates = {"Fast64_6k": gate_stats(final_gates),
             "Full_L333": gate_stats(full_gates)}
    write_json(HERE / "SALF_GATE_DIAGNOSTICS.json", gates)
    activity0 = json.loads((HERE / "ACTIVITY_STEP_0.json").read_text())
    activity6 = json.loads((HERE / "ACTIVITY_STEP_6000.json").read_text())
    hierarchy_keys = ("A64_rms", "H64_rms", "A128_rms",
                      "hier64to128_residual_rms", "H128_rms",
                      "A256_rms", "hier128to256_residual_rms", "H256_rms")
    activity0s, activity6s = activity0["summary"], activity6["summary"]
    hierarchy = {
        "step0": {k: activity0s[k] for k in hierarchy_keys},
        "step6000": {k: activity6s[k] for k in hierarchy_keys},
        "coarse_to_fine_nonzero": all(
            activity6s[k]["mean"] > 0 for k in
            ("hier64to128_residual_rms", "hier128to256_residual_rms")),
    }
    write_json(HERE / "HIERARCHICAL_REASSEMBLY_DIAGNOSTICS.json", hierarchy)
    attention = {
        "step0": {k: v for k, v in activity0s.items()
                  if "attention" in k or "conditioning_residual" in k},
        "step6000": {k: v for k, v in activity6s.items()
                     if "attention" in k or "conditioning_residual" in k},
        "entropy": "not computed; need_weights=False avoids dense maps",
    }
    for key in ("hybrid.s128_conditioning_residual_rms",
                "hybrid.s64_conditioning_residual_rms",
                "hybrid.s256_conditioning_residual_rms"):
        assert activity6s[key]["mean"] > 0, key
    write_json(HERE / "ATTENTION_DIAGNOSTICS.json", attention)
    e12_mean = {key: float(np.mean([float(r[key]) for r in e12])) for key in KEYS}
    d1_mean = {key: float(np.mean([float(r[key]) for r in d1])) for key in KEYS}
    d1_train = json.loads((D1_ROOT / "final_summary.json").read_text())
    d1_full_runtime = json.loads((D1_FULL_ROOT / "RUNTIME_COMPARISON.json").read_text())["D1_MULTI_512"]
    run = {
        "E12_trainable_parameters": training["parameter_audit"]["trainable_parameters"],
        "D1_trainable_parameters": d1_train["parameter_audit"]["trainable_parameters"],
        "added_trainable_parameters": training["parameter_audit"]["trainable_parameters"] -
                                      d1_train["parameter_audit"]["trainable_parameters"],
        "E12_train_peak_vram_bytes": training["peak_vram_bytes"],
        "D1_train_peak_vram_bytes": d1_train["peak_vram_bytes"],
        "E12_seconds_per_step": training["seconds_per_step"],
        "D1_seconds_per_step": d1_train["seconds_per_step"],
        "training_step_time_ratio": training["seconds_per_step"] / d1_train["seconds_per_step"],
        "total_training_seconds": training["total_training_seconds"],
        "E12_full_inference_seconds": runtime_full["total_inference_wall_seconds"],
        "D1_full_inference_seconds": d1_full_runtime["total_inference_wall_seconds"],
        "full_inference_ratio": runtime_full["total_inference_wall_seconds"] /
                                d1_full_runtime["total_inference_wall_seconds"],
        "full_peak_vram_bytes": runtime_full["peak_vram_bytes"],
        "attention_cost": "included in end-to-end time; per-module forward timing not recorded",
    }
    write_json(HERE / "RUNTIME_AUDIT.json", run)
    history = read_csv(HERE / "VALIDATION_HISTORY.csv")
    assert [int(r["step"]) for r in history] == list(STEPS)
    assert all(r["weights"] == "ema" and r["mode"] == "FAST" and int(r["slices"]) == 64
               for r in history)
    assert all(train_access[k] == 0 for k in (
        "L506_opens", "external_memory_opens", "neighbor_target_opens", "blocked_attempts"))
    assert all(full_access[k] == 0 for k in (
        "L506_opens", "external_memory_opens", "neighbor_target_opens",
        "blocked_attempts", "optimizer_steps"))
    init = json.loads((HERE / "INITIALIZATION_PARITY_AUDIT.json").read_text())
    checkpoint = json.loads((HERE / "CHECKPOINT_AUDIT.json").read_text())
    audit = {
        "status": "PASS", "arm": ARM,
        "optimizer_steps_train": 6000, "optimizer_steps_full_inference": 0,
        "ema": True, "fast64_nodes": list(STEPS), "full_L333_slices": 610,
        "L506_opens": 0, "external_memory_opens": 0,
        "neighbor_target_opens": 0, "blocked_attempts": 0,
        "nonfinite_count": training["nonfinite_count"],
        "main_unchanged": training["main_hash_before"] ==
                          training["main_hash_after"] == training["ema_main_hash_after"],
        "step_zero_equal": init["step_zero_equivalent"],
        "shared_state_bitwise_equal": init["shared_state_tensors_bitwise_identical"],
        "checkpoint_sha256": checkpoint["sha256"],
        "formal_source_sha256": sha256(ROOT / "src/rddm_rleb.py"),
        "fast64_split_sha256": sha256(ROOT / "splits/L333_fast64.txt"),
        "full_evaluator_sha256": runtime_full["full_evaluator_source_sha256"],
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p)
                          for p in (ROOT / "meddino_e12_haf_v1").glob("*.py")},
        "hierarchy_nonzero": hierarchy["coarse_to_fine_nonzero"],
        "conditioning_nonzero": True,
        "bootstrap": bootstrap,
    }
    assert audit["main_unchanged"] and audit["step_zero_equal"]
    assert audit["hierarchy_nonzero"]
    assert audit["formal_source_sha256"] == "e8ed93223fbdf4a0e2521f73c7a4f213561f6a191a1a7b2a5f95934b85ebf574"
    write_json(HERE / "FINAL_AUDIT.json", audit)
    lines = [
        "# E12-HAF: SALF + hierarchical reassembly + hybrid anatomy fusion", "",
        "Independent 6k run from canonical Direct base. E1 standalone 6k metric was 40.810964 dB, but its checkpoint was unavailable and was never used as E12 initialization.", "",
        "| Arm | 2k | 4k | 6k | Delta 6k vs D1 | SSIM | RMSE HU | Soft PSNR | Params | Peak VRAM GiB | sec/step |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    d1hist = {int(r["step"]): r for r in read_csv(D1_ROOT / "validation_history.csv")}
    e12hist = {int(r["step"]): r for r in history}
    for name, entries, params, vram, seconds in (
        ("D1_MULTI_512", d1hist, run["D1_trainable_parameters"],
         run["D1_train_peak_vram_bytes"], run["D1_seconds_per_step"]),
        (ARM, e12hist, run["E12_trainable_parameters"],
         run["E12_train_peak_vram_bytes"], run["E12_seconds_per_step"]),
    ):
        last = entries[6000]
        d = float(last["psnr"]) - float(d1hist[6000]["psnr"])
        lines.append(f'| {name} | {float(entries[2000]["psnr"]):.6f} | '
                     f'{float(entries[4000]["psnr"]):.6f} | {float(last["psnr"]):.6f} | '
                     f'{d:+.6f} | {float(last["ssim"]):.6f} | '
                     f'{float(last["rmse_hu"]):.4f} | {float(last["soft_psnr"]):.6f} | '
                     f'{params} | {vram / 2**30:.3f} | {seconds:.3f} |')
    lines += ["", "| Full L333 | PSNR | SSIM | RMSE HU | MAE HU | Soft PSNR | Inference s |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name, mean, seconds in (("D1_MULTI_512", d1_mean, run["D1_full_inference_seconds"]),
                                (ARM, e12_mean, run["E12_full_inference_seconds"])):
        lines.append(f'| {name} | {mean["psnr"]:.6f} | {mean["ssim"]:.6f} | '
                     f'{mean["rmse_hu"]:.4f} | {mean["mae_hu"]:.4f} | '
                     f'{mean["soft_psnr"]:.6f} | {seconds:.1f} |')
    lines += [
        "", f'E12-D1 paired PSNR: mean {bootstrap["mean_delta_psnr"]:+.6f} dB, '
        f'median {bootstrap["median_delta_psnr"]:+.6f}, '
        f'positive slices {bootstrap["positive_slice_fraction"]:.3f}, '
        f'95% slice-bootstrap CI [{bootstrap["ci_95_low"]:+.6f}, {bootstrap["ci_95_high"]:+.6f}].',
        f'Z thirds mean deltas: {thirds}.', "",
        f'Hierarchy residuals nonzero: {hierarchy["coarse_to_fine_nonzero"]}. '
        f'Attention and local conditioning residuals nonzero: True.',
        f'Training time ratio E12/D1: {run["training_step_time_ratio"]:.2f}x; '
        f'Full inference time ratio: {run["full_inference_ratio"]:.2f}x.',
        "Attention entropy was not computed to avoid allocating dense attention maps.",
        "L333 is development validation; slice-bootstrap CI is not cross-patient generalization.",
        "No L506, external memory, neighbor NDCT, MedDINO fine-tuning, E3, 8-patient training, or Git push.", "",
    ]
    (HERE / "MEDDINO_E12_HAF_REPORT.md").write_text("\n".join(lines))
    print("E12_FINAL_AUDIT_PASS", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(2)
    full()
    report()

