"""Final matched-step, per-slice, feature, runtime, and access audit."""
import csv
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from context_2p5d.data import ContextPairs, ContextDataGuard
from mayo2016_smallset_v1.runtime import ROOT, write_json, sha256
from .protocol import HERE, ARMS, STEPS, BASE_COMMIT, CONTROL_HASHES, archived_c3, config, frozen_config, verify
from .structure import MultiProvider, MEAN_HU, STD_HU, OFFICIAL_COMMIT, WEIGHT_SHA256, WEIGHT_PATH


def read_csv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_history(path):
    return {int(r["step"]): r for r in read_csv(path / "validation_history.csv")}


def load_slices(path, step):
    rows = read_csv(path / ("per_slice_step_%d.csv" % step))
    assert len(rows) == 64
    return rows


def slice_stats(left, right):
    assert [r["slice_index"] for r in left] == [r["slice_index"] for r in right]
    delta = np.array([float(a["psnr"]) - float(b["psnr"])
                      for a, b in zip(left, right)], dtype=np.float64)
    return {
        "mean": float(delta.mean()), "median": float(np.median(delta)),
        "positive_slice_fraction": float(np.mean(delta > 0)),
        **{f"p{p}": float(np.percentile(delta, p)) for p in (10, 25, 50, 75, 90)},
    }


def _med_input(center, input_size):
    hu = ((center.float() + 1.) * 2048. - 1024.).clamp(-1000., 1000.)
    value = ((hu - MEAN_HU) / STD_HU).repeat(1, 3, 1, 1)
    if input_size == 1024:
        value = F.interpolate(value, size=(1024, 1024), mode="bilinear",
                              align_corners=False, antialias=True)
    return value


def _provider_from_ema(arm, device):
    path = HERE / "runs" / arm / "checkpoints/step_6000.pt"
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    assert ck["arm"] == arm and ck["optimizer_step"] == 6000
    state = {k.removeprefix("model.unet0.structure."): v
             for k, v in ck["ema_state_dict"].items()
             if k.startswith("model.unet0.structure.")}
    provider = MultiProvider(arm)
    provider.load_state_dict(state, strict=True)
    provider = provider.to(device).eval()
    del ck, state
    return provider


def feature_diagnostics(available):
    device = torch.device("cuda:0")
    compare = ["D1_MULTI_512"]
    if "D3_MULTI_1024" in available:
        compare.append("D3_MULTI_1024")
    compare.append("D4_MULTI_FT_512")
    providers = {arm: _provider_from_ema(arm, device) for arm in compare}
    data_cfg = frozen_config()
    guard = ContextDataGuard(data_cfg, "FAST", allow_training=False)
    dataset = ContextPairs(data_cfg, "validation", "FAST")
    dataset.bind_guard(guard)
    sums = {}
    count = 0
    with guard, torch.no_grad():
        for pos in range(64):
            _, center, _, _, meta = dataset.load(pos)
            assert meta["patient_id"] == "L333"
            center = torch.from_numpy(center).unsqueeze(0).to(device) * 2. - 1.
            layers = {
                arm: provider.med.get_intermediate_layers(
                    _med_input(center, provider.input_size),
                    n=[2, 5, 8, 11], reshape=True, norm=True)
                for arm, provider in providers.items()
            }
            base = layers["D1_MULTI_512"]
            for arm in ("D3_MULTI_1024", "D4_MULTI_FT_512"):
                if arm not in layers:
                    continue
                for index, (a, b) in enumerate(zip(base, layers[arm])):
                    if arm == "D3_MULTI_1024":
                        b = F.interpolate(b, size=(32, 32), mode="bilinear",
                                          align_corners=False)
                    key = arm + "_vs_D1_F" + str((2, 5, 8, 11)[index])
                    row = sums.setdefault(key, {"cosine": 0., "l1": 0., "rms_difference": 0.})
                    row["cosine"] += float(F.cosine_similarity(
                        a.flatten(1), b.flatten(1), dim=1).mean())
                    row["l1"] += float((a - b).abs().mean())
                    row["rms_difference"] += float((a - b).square().mean().sqrt())
            count += 1
    access = guard.report()
    assert count == 64
    assert access["L506_opens"] == access["external_memory_opens"] == 0
    assert access["neighbor_target_opens"] == access["blocked_attempts"] == 0
    assert access["validation_center_indices_opened"] == list(
        map(int, (ROOT / "splits/L333_fast64.txt").read_text().splitlines()))
    for row in sums.values():
        for key in row:
            row[key] /= count
    return {"slices": 64, "indices": list(map(int, (ROOT / "splits/L333_fast64.txt").read_text().splitlines())),
            "cross_arm": sums, "data_access": access}


def build_report():
    verify()
    queue = json.loads((HERE / "queue_status.json").read_text())
    retry = json.loads((HERE / "D4_RETRY_STATUS.json").read_text())
    original_smoke = json.loads((HERE / "smoke/D4_MULTI_FT_512/SMOKE_RESULT.json").read_text())
    assert queue["status"] == "FAILED"
    assert original_smoke["status"] == "FAILED"
    assert original_smoke["error"] == "TypeError('Got unsupported ScalarType BFloat16')"
    assert retry["status"] == "FAILED" and retry["from_step"] == 2000 and retry["to_step"] == 6000
    assert "Use the frozen fast-dev config" in retry["error"]
    d4_completed = json.loads((HERE / "runs/D4_MULTI_FT_512/final_summary.json").read_text())
    assert d4_completed["status"] == "COMPLETE" and d4_completed["optimizer_steps"] == 6000
    queue["arms"]["D4_MULTI_FT_512"] = {"status": "COMPLETE", "optimizer_steps": 6000,
                                         "source": "D4_BF16_RETRY_POSTPROCESS_ONLY"}
    available = [arm for arm in ARMS if queue["arms"][arm]["status"] == "COMPLETE"]
    assert all(arm in available for arm in ("D1_MULTI_512", "D2_MULTI_ADD_512", "D4_MULTI_FT_512"))
    d3_oom = queue["arms"]["D3_MULTI_1024"]["status"] == "OOM"
    assert d3_oom or "D3_MULTI_1024" in available
    c3_path, c3 = archived_c3()
    histories = {arm: load_history(HERE / "runs" / arm) for arm in available}
    summaries = {arm: json.loads((HERE / "runs" / arm / "final_summary.json").read_text())
                 for arm in available}
    accesses = {arm: summary["data_access"] for arm, summary in summaries.items()}
    for arm in available:
        expected_steps = [2000, 4000, 6000] if arm == "D4_MULTI_FT_512" else list(STEPS)
        assert sorted(histories[arm]) == expected_steps
        summary = summaries[arm]
        assert summary["status"] == "COMPLETE" and summary["optimizer_steps"] == 6000
        assert summary["nonfinite_count"] == 0
        assert summary["main_hash_before"] == summary["main_hash_after"] == summary["ema_main_hash_after"]
        access = accesses[arm]
        assert access["training_patients_opened"] == ["L067", "L143"]
        assert access["validation_center_indices_opened"] == list(map(
            int, (ROOT / "splits/L333_fast64.txt").read_text().splitlines()))
        assert all(access[k] == 0 for k in ("L506_opens", "external_memory_opens",
                                           "neighbor_target_opens", "blocked_attempts"))
        for step, row in histories[arm].items():
            assert row["weights"] == "ema" and row["mode"] == "FAST" and int(row["slices"]) == 64
            assert all(math.isfinite(float(row[k])) for k in (
                "psnr", "ssim", "rmse_hu", "mae_hu", "soft_psnr",
                "soft_ssim", "soft_rmse_hu", "soft_mae_hu", "validation_loss"))
            assert len(load_slices(HERE / "runs" / arm, step)) == 64
    assert abs(float(histories["D4_MULTI_FT_512"][2000]["psnr"]) -
               float(histories["D1_MULTI_512"][2000]["psnr"])) <= 1.e-6
    comparison = []
    for step in STEPS:
        for arm in ARMS:
            if arm not in available:
                continue
            source = "D1_SHARED_PRE_FINETUNE" if arm == "D4_MULTI_FT_512" and step < 2000 else "OWN_RUN"
            row = histories["D1_MULTI_512"][step] if source.startswith("D1_SHARED") else histories[arm][step]
            c3_row = c3.get(step)
            d1_row = histories["D1_MULTI_512"][step]
            comparison.append({
                "arm": arm, "step": step, "source": source,
                "psnr": row["psnr"], "ssim": row["ssim"],
                "rmse_hu": row["rmse_hu"], "mae_hu": row["mae_hu"],
                "soft_psnr": row["soft_psnr"],
                "validation_loss": row["validation_loss"],
                "delta_psnr_vs_C3": None if c3_row is None else
                float(row["psnr"]) - float(c3_row["psnr"]),
                "delta_psnr_vs_D1": float(row["psnr"]) - float(d1_row["psnr"]),
            })
    write_csv(HERE / "FOUR_ARM_VALIDATION_HISTORY.csv", comparison)
    per_slice = []
    stats = {}
    for arm, reference, ref_path in (
        ("D1_MULTI_512", "C3_FINAL", c3_path),
        ("D2_MULTI_ADD_512", "D1_MULTI_512", HERE / "runs/D1_MULTI_512"),
        ("D3_MULTI_1024", "D1_MULTI_512", HERE / "runs/D1_MULTI_512"),
        ("D4_MULTI_FT_512", "D1_MULTI_512", HERE / "runs/D1_MULTI_512"),
    ):
        if arm not in available:
            continue
        left = load_slices(HERE / "runs" / arm, 6000)
        right = load_slices(ref_path, 6000)
        stats[arm + "_vs_" + reference] = slice_stats(left, right)
        for a, b in zip(left, right):
            per_slice.append({
                "arm": arm, "reference": reference, "patient_id": "L333",
                "slice_index": a["slice_index"],
                "psnr": a["psnr"], "reference_psnr": b["psnr"],
                "delta_psnr": float(a["psnr"]) - float(b["psnr"]),
            })
    write_csv(HERE / "SLICE_LEVEL_COMPARISON.csv", per_slice)
    runtime = {}
    final_table = []
    for arm in ARMS:
        if arm not in available:
            runtime[arm] = {"status": "OOM"}
            continue
        summary = summaries[arm]
        h = histories[arm]
        params = summary["parameter_audit"]
        runtime[arm] = {
            "total_parameters": params["total_parameters"],
            "trainable_parameters_at_end": params["trainable_parameters"],
            "peak_vram_bytes": summary["peak_vram_bytes"],
            "seconds_per_step": summary["seconds_per_step"],
            "training_seconds_including_D1_shared": summary["training_seconds_including_D1_shared"],
            "training_seconds_after_fork": summary["training_seconds_after_fork"],
            "validation_seconds": {str(step): float(row["validation_seconds"]) for step, row in h.items()},
        }
        final_table.append({
            "arm": arm,
            "psnr_500": float((histories["D1_MULTI_512"] if arm == "D4_MULTI_FT_512" else h)[500]["psnr"]),
            "psnr_1000": float((histories["D1_MULTI_512"] if arm == "D4_MULTI_FT_512" else h)[1000]["psnr"]),
            "psnr_2000": float(h[2000]["psnr"]),
            "psnr_4000": float(h[4000]["psnr"]),
            "psnr_6000": float(h[6000]["psnr"]),
            "ssim_6000": float(h[6000]["ssim"]),
            "rmse_hu_6000": float(h[6000]["rmse_hu"]),
            "mae_hu_6000": float(h[6000]["mae_hu"]),
            "soft_psnr_6000": float(h[6000]["soft_psnr"]),
            "soft_ssim_6000": float(h[6000]["soft_ssim"]),
            "delta_vs_C3_6000": float(h[6000]["psnr"]) - float(c3[6000]["psnr"]),
            "delta_vs_D1_6000": float(h[6000]["psnr"]) - float(histories["D1_MULTI_512"][6000]["psnr"]),
            "positive_slices_vs_D1": None if arm == "D1_MULTI_512" else
            stats[arm + "_vs_D1_MULTI_512"]["positive_slice_fraction"],
            "trainable_parameters": params["trainable_parameters"],
            "peak_vram_bytes": summary["peak_vram_bytes"],
            "seconds_per_step": summary["seconds_per_step"],
            "total_training_seconds": summary["training_seconds_including_D1_shared"],
        })
    write_csv(HERE / "FOUR_ARM_COMPARISON.csv", final_table)
    write_json(HERE / "RUNTIME_COMPARISON.json", runtime)
    feat = {
        "per_arm_6k_fast64_average": {
            arm: json.loads((HERE / "runs" / arm / "feature_activity.json").read_text())[-1]["features"]
            for arm in available
        },
        "cross_arm": feature_diagnostics(available),
    }
    write_json(HERE / "FEATURE_DIAGNOSTICS.json", feat)
    finetune = summaries["D4_MULTI_FT_512"]["finetune_audit"]
    assert finetune["blocks_0_7_initial_hash"] == finetune["blocks_0_7_final_hash"]
    assert finetune["blocks_8_11_initial_hash"] != finetune["blocks_8_11_final_hash"]
    write_json(HERE / "FINETUNE_AUDIT.json", finetune)
    write_json(HERE / "DATA_ACCESS_AUDIT.json", {
        "per_arm": accesses, "feature_diagnostics": feat["cross_arm"]["data_access"],
        "L506_opens": 0, "external_memory_opens": 0,
        "neighbor_target_opens": 0,
        "blocked_attempts": 0,
    })
    write_json(HERE / "FOUR_ARM_COMPARISON.json", {
        "status": "COMPLETE_WITH_D3_OOM" if d3_oom else "COMPLETE",
        "table": final_table, "slice_stats": stats,
        "archived_reference_C3": {str(k): c3[k] for k in (2000, 4000, 6000)},
    })
    source_hashes = {str(p.relative_to(ROOT)): sha256(p)
                     for p in sorted((ROOT / "meddino_four_arm_v1").glob("*.py"))}
    audit = {
        "status": "PASS_WITH_D3_OOM" if d3_oom else "PASS",
        "base_commit": BASE_COMMIT,
        "branch": "experiment/rddm-meddino-four-arm-v1",
        "source_sha256": source_hashes,
        "control_source_hashes": {k: sha256(ROOT / k) for k in CONTROL_HASHES},
        "official_meddino_commit": OFFICIAL_COMMIT,
        "official_meddino_weight_sha256": WEIGHT_SHA256,
        "completed_arms": available, "D3_oom": d3_oom,
        "L506_opens": 0, "external_memory_opens": 0,
        "neighbor_target_opens": 0, "blocked_attempts": 0,
        "full_L333_run": False, "formal_training": False,
        "python": sys.version.split()[0], "torch": torch.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "seed": 20260829,
        "previous_D4_failure": "instrumentation-only BF16 to NumPy conversion in audit state_hash",
        "BF16_conversion_fix": "detach().to(torch.float32).cpu().contiguous().numpy()",
        "scientific_training_protocol_changed": False,
        "original_failure_record_preserved": True,
        "D4_training_completed_before_report_failure": True,
        "prior_report_failure": retry["error"],
        "postprocess_recovery_only": True,
        "diagnostic_dataset_config": "frozen_config()",
    }
    assert audit["control_source_hashes"] == CONTROL_HASHES
    write_json(HERE / "FINAL_AUDIT.json", audit)
    lines = [
        "# MedDINOv3 four-arm RDDM ControlNet study", "",
        "Fast64 EMA evaluation only. The archived C3 final-layer arm was not retrained.",
        "All arms use the same trained Direct 2.5D RDDM; no performance-based early stopping.",
        "",
        "| Arm | 500 | 1k | 2k | 4k | 6k PSNR | 6k SSIM | 6k RMSE HU | 6k MAE HU | 6k soft PSNR | 6k soft SSIM | Δ vs C3 | Δ vs D1 | Positive slices vs D1 | Trainable params | Peak VRAM GiB | Sec/step |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in final_table:
        frac = "NA" if r["positive_slices_vs_D1"] is None else f"{r['positive_slices_vs_D1']:.3f}"
        lines.append(
            f"| {r['arm']} | {r['psnr_500']:.6f} | {r['psnr_1000']:.6f} | "
            f"{r['psnr_2000']:.6f} | {r['psnr_4000']:.6f} | {r['psnr_6000']:.6f} | "
            f"{r['ssim_6000']:.6f} | {r['rmse_hu_6000']:.4f} | {r['mae_hu_6000']:.4f} | "
            f"{r['soft_psnr_6000']:.6f} | {r['soft_ssim_6000']:.6f} | "
            f"{r['delta_vs_C3_6000']:+.6f} | {r['delta_vs_D1_6000']:+.6f} | "
            f"{frac} | {r['trainable_parameters']} | {r['peak_vram_bytes']/1024**3:.3f} | "
            f"{r['seconds_per_step']:.3f} |"
        )
    if d3_oom:
        lines.append("| D3_MULTI_1024 | OOM | OOM | OOM | OOM | OOM | NA | NA | NA | NA | NA | NA | NA | NA | NA |")
    lines += ["", "D4 500/1k values are the shared D1 pre-finetune history, not independent D4 training.",
              "C3 is ARCHIVED_REFERENCE; its 2k/4k/6k PSNR is " +
              ", ".join(f"{step}: {float(c3[step]['psnr']):.6f}" for step in (2000, 4000, 6000)) + ".",
              "", "## Slice-level 6k comparisons", ""]
    for name, s in stats.items():
        lines.append(f"- {name}: mean {s['mean']:+.6f}, median {s['median']:+.6f}, positive {s['positive_slice_fraction']:.3f}; p10/p25/p50/p75/p90 = " +
                     "/".join(f"{s[f'p{p}']:+.6f}" for p in (10,25,50,75,90)) + ".")
    d1_psnr = float(histories["D1_MULTI_512"][6000]["psnr"])
    d3_psnr = float(histories["D3_MULTI_1024"][6000]["psnr"])
    d4_psnr = float(histories["D4_MULTI_FT_512"][6000]["psnr"])
    d4_delta = d4_psnr - d1_psnr
    d3_delta = d3_psnr - d1_psnr
    d4_fraction = stats["D4_MULTI_FT_512_vs_D1_MULTI_512"]["positive_slice_fraction"]
    d3_fraction = stats["D3_MULTI_1024_vs_D1_MULTI_512"]["positive_slice_fraction"]
    d3_step_ratio = runtime["D3_MULTI_1024"]["seconds_per_step"] / runtime["D1_MULTI_512"]["seconds_per_step"]
    lines += [
        "", "## Four scientific questions", "",
        f"- Partial fine-tuning vs frozen at 6k: D4-D1 = {d4_delta:+.6f} dB; "
        f"4k delta = {float(histories['D4_MULTI_FT_512'][4000]['psnr'])-float(histories['D1_MULTI_512'][4000]['psnr']):+.6f} dB.",
        f"- Positive Fast64 slices for D4 vs D1: {d4_fraction:.3f}; "
        + ("a majority." if d4_fraction > .5 else "not a majority."),
        f"- 1024 resolution vs 512 frozen at 6k: D3-D1 = {d3_delta:+.6f} dB; "
        f"D4-D3 = {d4_psnr-d3_psnr:+.6f} dB.",
        f"- Next-step priority under Fast64: resolution has the stronger signal "
        f"(D3-D1 {d3_delta:+.6f} dB, {d3_fraction:.3f} positive slices) than partial fine-tuning "
        f"(D4-D1 {d4_delta:+.6f} dB, {d4_fraction:.3f} positive slices), "
        f"although D3 training costs {d3_step_ratio:.2f}x seconds/step. "
        "The absolute gain is small and needs full-L333 confirmation before any formal claim; "
        "no next experiment is started.",
        "", "## D4 matched metrics vs D1", "",
    ]
    for step in (4000, 6000):
        d4 = histories["D4_MULTI_FT_512"][step]
        d1 = histories["D1_MULTI_512"][step]
        lines.append(
            f"- {step}: PSNR {float(d4['psnr']):.6f} ({float(d4['psnr'])-float(d1['psnr']):+.6f} dB), "
            f"SSIM {float(d4['ssim']):.6f} ({float(d4['ssim'])-float(d1['ssim']):+.6f}), "
            f"RMSE {float(d4['rmse_hu']):.4f} ({float(d4['rmse_hu'])-float(d1['rmse_hu']):+.4f} HU), "
            f"MAE {float(d4['mae_hu']):.4f} ({float(d4['mae_hu'])-float(d1['mae_hu']):+.4f} HU), "
            f"soft PSNR {float(d4['soft_psnr']):.6f} ({float(d4['soft_psnr'])-float(d1['soft_psnr']):+.6f} dB), "
            f"soft SSIM {float(d4['soft_ssim']):.6f} ({float(d4['soft_ssim'])-float(d1['soft_ssim']):+.6f})."
        )
    lines += [
        "", "## Scope and interpretation", "",
        "The prior D4 attempt failed before any optimizer step because audit state_hash converted "
        "a BFloat16 tensor directly to NumPy. The retry casts audit tensors to float32 first. "
        "This instrumentation fix did not change model, checkpoint, loss, optimizer, AMP or data protocol.",
        "D4 training reached step 6000 before a report-only configuration error. Feature diagnostics "
        "now load L333-Fast64 through the canonical frozen_config(); no optimizer step was repeated.",
        "Matched Fast64 curves, feature activity, VRAM, and training time are archived in CSV/JSON.",
        "D4 blocks 0–7 hash unchanged; blocks 8–11 changed with finite nonzero gradients.",
        "No Full-L333, 8-patient formal run, L506, external memory, or neighbor NDCT access occurred.",
        "These observations apply to this implementation and Fast64 protocol; the report does not establish a final paper method.",
    ]
    (HERE / "MEDDINO_FOUR_ARM_REPORT.md").write_text("\n".join(lines) + "\n")
    print("FOUR_ARM_REPORT_COMPLETE", flush=True)


if __name__ == "__main__":
    build_report()
