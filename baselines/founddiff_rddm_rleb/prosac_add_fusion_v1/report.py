"""Matched-step Eq.(7) add-fusion report; reads archived controls only."""
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from mayo2016_smallset_v1.runtime import ROOT, sha256, write_json
from .protocol import (HERE, OLD, OLD_PROSAC, STEPS, CONTROL_HASHES,
                       DIRECT_CK, DIRECT_SHA, verify)
from .structure import OFFICIAL_ROOT, OFFICIAL_COMMIT, WEIGHT_PATH, WEIGHT_SHA256


def rows(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def csv_out(path, values):
    if values:
        with Path(path).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)


def history(path):
    return {int(r["step"]): r for r in rows(path / "validation_history.csv")}


def slices(path, step):
    data = rows(path / ("per_slice_step_%d.csv" % step))
    assert len(data) == 64
    return data


def stats(left, right):
    assert [r["slice_index"] for r in left] == [r["slice_index"] for r in right]
    d = np.asarray([float(a["psnr"]) - float(b["psnr"])
                    for a, b in zip(left, right)], dtype=np.float64)
    return {"mean": float(d.mean()), "median": float(np.median(d)),
            "positive_slice_fraction": float(np.mean(d > 0)),
            **{"p%d" % p: float(np.percentile(d, p))
               for p in (10, 25, 50, 75, 90)}}


def build_report():
    verify()
    arm = "C4_ADD"
    path = HERE / "runs" / arm
    summary = json.loads((path / "final_summary.json").read_text())
    decision = json.loads((HERE / "PROMOTION_DECISION.json").read_text())[arm]
    expected = 6000 if decision["continue_to_6k"] else 2000
    assert summary["optimizer_steps"] == expected
    assert summary["status"] == ("COMPLETE" if expected == 6000 else "FIRST_STAGE_COMPLETE")
    assert summary["nonfinite_count"] == 0
    assert summary["main_hash_before"] == summary["main_hash_after"] == summary["ema_main_hash_after"]
    init = json.loads((path / "INITIALIZATION_AUDIT.json").read_text())
    params = json.loads((path / "PARAMETER_AUDIT.json").read_text())
    access = json.loads((path / "DATA_ACCESS_AUDIT.json").read_text())
    assert init["step_zero_max_abs_difference"] <= 1.e-6
    assert init["meddino_frozen"] and init["meddino_eval"]
    assert access["training_patients_opened"] == ["L067", "L143"]
    assert access["validation_center_indices_opened"] == list(map(
        int, (ROOT / "splits/L333_fast64.txt").read_text().splitlines()))
    assert all(access[k] == 0 for k in ("L506_opens", "external_memory_opens",
                                       "neighbor_target_opens", "blocked_attempts"))
    references = {
        "C1_CAPACITY": OLD / "runs/C1_CAPACITY",
        "C3_ANATOMY": OLD_PROSAC / "runs/C3_ANATOMY",
        "C4_APGC": OLD_PROSAC / "runs/C4_APGC",
    }
    h = history(path)
    ref_h = {name: history(p) for name, p in references.items()}
    assert sorted(h) == list(STEPS if expected == 6000 else STEPS[:4])
    old_c4_history = history(OLD_PROSAC / "runs/C4_APGC")
    for step, row in h.items():
        assert row["weights"] == "ema" and row["mode"] == "FAST" and int(row["slices"]) == 64
        assert all(math.isfinite(float(row[k])) for k in (
            "psnr", "ssim", "rmse_hu", "mae_hu", "soft_psnr",
            "soft_ssim", "soft_rmse_hu", "soft_mae_hu", "validation_loss"))
        assert row["train_stream_sha256"] == old_c4_history[step]["train_stream_sha256"]
    old_config = json.loads((OLD_PROSAC / "EXPERIMENT_CONFIG.json").read_text())["C4_APGC"]
    new_config = json.loads((path / "CONFIG.json").read_text())
    a, b = dict(old_config), dict(new_config)
    for key in ("arm", "experiment_name", "description"):
        a.pop(key); b.pop(key)
    assert b.pop("edge_aggregation") == "Sobel plus Laplacian before Conv1x1_1toC"
    assert a == b, "C4_ADD config drift outside edge aggregation"
    old_params = json.loads((references["C4_APGC"] / "PARAMETER_AUDIT.json").read_text())
    assert params["trainable_parameters"] - old_params["trainable_parameters"] == -224
    comparison, per_slice, validation = {}, [], []
    for step, row in sorted(h.items()):
        current = slices(path, step)
        r = dict(row)
        for name, directory in references.items():
            baseline = ref_h[name][step]
            delta = float(row["psnr"]) - float(baseline["psnr"])
            reference = slices(directory, step)
            comparison[name + "_at_" + str(step)] = stats(current, reference)
            r["delta_psnr_vs_" + name] = delta
            for now, old in zip(current, reference):
                per_slice.append({
                    "reference": name, "step": step,
                    "patient_id": "L333", "slice_index": now["slice_index"],
                    "C4_ADD_psnr": now["psnr"], "reference_psnr": old["psnr"],
                    "delta_psnr": float(now["psnr"]) - float(old["psnr"]),
                    "C4_ADD_soft_psnr": now["soft_psnr"],
                    "reference_soft_psnr": old["soft_psnr"],
                })
        validation.append(r)
    csv_out(HERE / "VALIDATION_HISTORY.csv", validation)
    csv_out(HERE / "PER_SLICE_DELTA.csv", per_slice)
    activity = rows(path / "control_activity.csv")
    assert [int(r["step"]) for r in activity] == sorted(h)
    for row in activity:
        for k in ("sobel_rms", "laplacian_rms", "edge_sum_rms",
                  "edge_feature_rms", "structure_rms", "control_residual_rms",
                  "zero_output_rms", "zero_weight_norms"):
            assert k in row
    csv_out(HERE / "CONTROL_ACTIVITY.csv", activity)
    csv_out(HERE / "OLD_C4_CONTROL_ACTIVITY.csv",
            rows(references["C4_APGC"] / "control_activity.csv"))
    write_json(HERE / "EXPERIMENT_CONFIG.json", new_config)
    write_json(HERE / "INITIALIZATION_AUDIT.json", init)
    write_json(HERE / "PARAMETER_AUDIT.json", {
        "C4_ADD": params, "C4_APGC": old_params,
        "trainable_parameter_delta": -224})
    write_json(HERE / "DATA_ACCESS_AUDIT.json", access)
    write_json(HERE / "TRAIN_STREAM_AUDIT.json",
               json.loads((path / "TRAIN_STREAM_AUDIT.json").read_text()))
    write_json(HERE / "TRAIN_RNG_AUDIT.json",
               json.loads((path / "TRAIN_RNG_AUDIT.json").read_text()))
    sources = {k: sha256(ROOT / k) for k in CONTROL_HASHES}
    assert sources == CONTROL_HASHES
    assert sha256(DIRECT_CK) == DIRECT_SHA
    assert sha256(WEIGHT_PATH) == WEIGHT_SHA256
    assert subprocess.check_output(["git", "rev-parse", "HEAD"],
                                   cwd=OFFICIAL_ROOT, text=True).strip() == OFFICIAL_COMMIT
    audit = {
        "status": "PASS", "arm": arm, "optimizer_steps": expected,
        "smoke_steps": 20, "step_zero_max_abs_difference": init["step_zero_max_abs_difference"],
        "main_hash_unchanged": True, "meddino_frozen": True,
        "nonfinite_count": 0, "L506_opens": 0, "external_memory_opens": 0,
        "neighbor_target_opens": 0, "blocked_attempts": 0,
        "control_source_hashes": sources, "direct_checkpoint_sha256": DIRECT_SHA,
        "meddino_source_commit": OFFICIAL_COMMIT,
        "meddino_checkpoint_sha256": WEIGHT_SHA256,
        "fast64_slices": 64, "weights": "EMA", "full_L333_run": False,
    }
    write_json(HERE / "FINAL_AUDIT.json", audit)
    provenance = {
        "parent_commit": "6595ca325b204769400653836ac447d919aaa40e",
        "branch": "experiment/rddm-prosac-add-fusion-v1",
        "source_sha256": {
            str(p.relative_to(ROOT)): sha256(p)
            for p in sorted((ROOT / "prosac_add_fusion_v1").glob("*.py"))},
        "control_source_hashes": sources,
        "direct_checkpoint_sha256": DIRECT_SHA,
        "meddino_source_commit": OFFICIAL_COMMIT,
        "meddino_checkpoint_sha256": WEIGHT_SHA256,
        "python": sys.version, "torch": torch.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "seed": 20260829,
    }
    write_json(HERE / "PROVENANCE.json", provenance)
    print_rows = []
    for name, values in [("C1_CAPACITY", ref_h["C1_CAPACITY"]),
                         ("C3_ANATOMY", ref_h["C3_ANATOMY"]),
                         ("C4_APGC", ref_h["C4_APGC"]),
                         ("C4_ADD", h)]:
        step = 2000
        row = values[step]
        print_rows.append({
            "arm": name, "psnr_500": float(values[500]["psnr"]),
            "psnr_1000": float(values[1000]["psnr"]),
            "psnr_2000": float(row["psnr"]),
            "delta_vs_C3": float(row["psnr"]) - float(ref_h["C3_ANATOMY"][step]["psnr"]),
            "delta_vs_old_C4": float(row["psnr"]) - float(ref_h["C4_APGC"][step]["psnr"]),
            "positive_slices_vs_C3": (
                None if name != "C4_ADD" else
                comparison["C3_ANATOMY_at_2000"]["positive_slice_fraction"]),
            "rmse_hu": float(row["rmse_hu"]),
            "soft_psnr": float(row["soft_psnr"]),
        })
    lines = ["# ProSAC Eq.(7) add-fusion probe on RDDM ControlNet", "",
             "Only the edge aggregation changed: concat(Sobel,Laplacian) -> Conv1x1_2toC became (Sobel + Laplacian) -> Conv1x1_1toC. The [-1000,1000] HU window, frozen MedDINO, APGC multiplication and all training/evaluation conditions remained fixed.",
             "", "| Arm | 500 PSNR | 1k PSNR | 2k PSNR | Delta vs C3 | Delta vs old C4 | Positive slices vs C3 | RMSE HU | Soft PSNR |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in print_rows:
        fraction = "NA" if r["positive_slices_vs_C3"] is None else "%.3f" % r["positive_slices_vs_C3"]
        lines.append("| %s | %.8f | %.8f | %.8f | %+.8f | %+.8f | %s | %.6f | %.6f |" %
                     (r["arm"], r["psnr_500"], r["psnr_1000"], r["psnr_2000"],
                      r["delta_vs_C3"], r["delta_vs_old_C4"], fraction,
                      r["rmse_hu"], r["soft_psnr"]))
    lines += ["", "## Gate, slices, and structure activity", "",
              "- 2k promotion to 4k/6k: %s." % decision["continue_to_6k"],
              "- C4_ADD trainable parameters: %d; old C4: %d; difference: -224." %
              (params["trainable_parameters"], old_params["trainable_parameters"])]
    for name in ("C3_ANATOMY", "C4_APGC", "C1_CAPACITY"):
        s = comparison[name + "_at_2000"]
        lines.append("- 2k vs %s: mean %+.8f, median %+.8f, positive fraction %.3f, p10/p25/p50/p75/p90 = %+.6f/%+.6f/%+.6f/%+.6f/%+.6f." %
                     (name, s["mean"], s["median"], s["positive_slice_fraction"],
                      *(s["p%d" % p] for p in (10, 25, 50, 75, 90))))
    last_activity = activity[-1]
    old_activity = {int(r["step"]): r for r in rows(
        references["C4_APGC"] / "control_activity.csv")}[int(last_activity["step"])]
    lines += ["- Activity at last matched step is in CONTROL_ACTIVITY.csv and OLD_C4_CONTROL_ACTIVITY.csv. Sobel, Laplacian, and their sum RMS are recorded for C4_ADD; the archived old C4 did not record separate Sobel/Laplacian RMS.",
              "", "## Integrity and scope", "",
              "- Direct main, copied Control Encoder, ZeroConv taps, loss, sampler, evaluator, MedDINO source/weights, Fast64 split and seed were unchanged.",
              "- Main and MedDINO remained frozen; L506, external memory and neighbor NDCT opens were zero.",
              "- Conclusions are limited to this Fast64 implementation and matched checkpoints. No Full-L333, 8-patient, or test evaluation was run."]
    (HERE / "PROSAC_EQ7_ADD_REPORT.md").write_text("\n".join(lines) + "\n")
    write_json(HERE / "PROSAC_EQ7_ADD_SUMMARY.json", {
        "status": "COMPLETE", "gate": decision, "table": print_rows,
        "slice_comparisons": comparison, "history": validation,
        "parameters": {"C4_ADD": params, "C4_APGC": old_params},
        "peak_vram_bytes": summary["peak_vram_bytes"],
        "training_seconds": summary["training_seconds"],
        "data_access": access})
    print("ADD_FUSION_REPORT_COMPLETE", flush=True)


if __name__ == "__main__":
    build_report()
