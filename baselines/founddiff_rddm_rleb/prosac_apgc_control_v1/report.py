"""Lightweight, source-pinned Fast64 result aggregation."""
import csv
import json
import math
import statistics
from pathlib import Path

import numpy as np

from mayo2016_smallset_v1.runtime import ROOT, sha256, write_json
from .protocol import HERE, OLD, ARMS, CONTROL_HASHES, DIRECT_CK, DIRECT_SHA, verify


def rows(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def csv_out(path, items):
    if not items:
        return
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(items[0]))
        writer.writeheader()
        writer.writerows(items)


def reference(arm, step):
    path = OLD / "runs" / arm
    history = next(r for r in rows(path / "validation_history.csv")
                   if int(r["step"]) == step)
    slices = rows(path / ("per_slice_step_%d.csv" % step))
    return history, slices


def summarize_delta(left, right):
    assert [r["slice_index"] for r in left] == [r["slice_index"] for r in right]
    delta = np.array([float(a["psnr"]) - float(b["psnr"])
                      for a, b in zip(left, right)], dtype=np.float64)
    return {"mean": float(delta.mean()), "median": float(np.median(delta)),
            "positive_slice_fraction": float(np.mean(delta > 0)),
            "p10": float(np.percentile(delta, 10)),
            "p50": float(np.percentile(delta, 50)),
            "p90": float(np.percentile(delta, 90))}


def build_report():
    verify()
    decisions = json.loads((HERE / "PROMOTION_DECISION.json").read_text())
    histories = {}
    summaries = {}
    access = {}
    init = {}
    params = {}
    activity = []
    validation = []
    per_slice_delta = []
    comparisons = {}
    for arm in ARMS:
        path = HERE / "runs" / arm
        summary = json.loads((path / "final_summary.json").read_text())
        assert summary["status"] in ("FIRST_STAGE_COMPLETE", "COMPLETE")
        assert summary["optimizer_steps"] == (
            6000 if decisions[arm]["continue_to_6k"] else 2000)
        expected_steps = ([0, 500, 1000, 2000, 4000, 6000]
                          if summary["optimizer_steps"] == 6000 else
                          [0, 500, 1000, 2000])
        history = rows(path / "validation_history.csv")
        assert [int(r["step"]) for r in history] == expected_steps
        histories[arm] = history
        summaries[arm] = summary
        init[arm] = json.loads((path / "INITIALIZATION_AUDIT.json").read_text())
        params[arm] = json.loads((path / "PARAMETER_AUDIT.json").read_text())
        access[arm] = json.loads((path / "DATA_ACCESS_AUDIT.json").read_text())
        a = access[arm]
        assert a["L506_opens"] == a["external_memory_opens"] == 0
        assert a["neighbor_target_opens"] == a["blocked_attempts"] == 0
        assert a["training_patients_opened"] == ["L067", "L143"]
        assert a["validation_center_indices_opened"] == list(map(
            int, (ROOT / "splits/L333_fast64.txt").read_text().splitlines()))
        assert summary["nonfinite_count"] == 0
        assert summary["main_hash_before"] == summary["main_hash_after"]
        assert summary["main_hash_before"] == summary["ema_main_hash_after"]
        assert init[arm]["step_zero_max_abs_difference"] <= 1e-6
        assert init[arm]["meddino_frozen"] and init[arm]["meddino_eval"]
        assert int(history[0]["step"]) == 0
        assert all(math.isfinite(float(r[k])) for r in history
                   for k in ("psnr", "ssim", "rmse_hu", "mae_hu",
                             "soft_psnr", "soft_ssim", "soft_rmse_hu",
                             "soft_mae_hu", "validation_loss"))
        activity.extend(rows(path / "control_activity.csv"))
        for r in history:
            step = int(r["step"])
            c1, _ = reference("C1_CAPACITY", step)
            validation.append({
                **r,
                "delta_psnr_vs_C1": float(r["psnr"]) - float(c1["psnr"]),
                "delta_soft_psnr_vs_C1":
                    float(r["soft_psnr"]) - float(c1["soft_psnr"]),
            })
    assert init[ARMS[0]]["canonical_training_rng_sha256"] == \
           init[ARMS[1]]["canonical_training_rng_sha256"]
    for arm in ARMS:
        path = HERE / "runs" / arm
        h = {int(r["step"]): r for r in histories[arm]}
        for step in sorted(h):
            left = rows(path / ("per_slice_step_%d.csv" % step))
            for other in ("C1_CAPACITY", "C2_EDGE"):
                _, right = reference(other, step)
                name = arm + "_minus_" + other + "_at_" + str(step)
                comparisons[name] = summarize_delta(left, right)
                for a, b in zip(left, right):
                    per_slice_delta.append({
                        "arm": arm, "reference": other, "step": step,
                        "patient_id": "L333", "slice_index": a["slice_index"],
                        "delta_psnr": float(a["psnr"]) - float(b["psnr"]),
                        "delta_soft_psnr":
                            float(a["soft_psnr"]) - float(b["soft_psnr"]),
                    })
            if arm == "C4_APGC" and step in {
                int(r["step"]) for r in histories["C3_ANATOMY"]}:
                right = rows(HERE / "runs/C3_ANATOMY" /
                             ("per_slice_step_%d.csv" % step))
                name = "C4_APGC_minus_C3_ANATOMY_at_" + str(step)
                comparisons[name] = summarize_delta(left, right)
                for a, b in zip(left, right):
                    per_slice_delta.append({
                        "arm": arm, "reference": "C3_ANATOMY", "step": step,
                        "patient_id": "L333", "slice_index": a["slice_index"],
                        "delta_psnr": float(a["psnr"]) - float(b["psnr"]),
                        "delta_soft_psnr":
                            float(a["soft_psnr"]) - float(b["soft_psnr"]),
                    })
    csv_out(HERE / "VALIDATION_HISTORY.csv", validation)
    csv_out(HERE / "PER_SLICE_DELTA.csv", per_slice_delta)
    csv_out(HERE / "CONTROL_ACTIVITY.csv", activity)
    write_json(HERE / "INITIALIZATION_AUDIT.json", init)
    write_json(HERE / "PARAMETER_AUDIT.json", params)
    write_json(HERE / "DATA_ACCESS_AUDIT.json", access)
    source = {k: sha256(ROOT / k) for k in CONTROL_HASHES}
    assert source == CONTROL_HASHES
    assert sha256(DIRECT_CK) == DIRECT_SHA
    write_json(HERE / "FINAL_AUDIT.json", {
        "status": "PASS", "arms": {
            arm: {"optimizer_steps": summaries[arm]["optimizer_steps"],
                  "nonfinite_count": summaries[arm]["nonfinite_count"],
                  "L506_opens": access[arm]["L506_opens"],
                  "external_memory_opens": access[arm]["external_memory_opens"],
                  "neighbor_target_opens": access[arm]["neighbor_target_opens"],
                  "blocked_attempts": access[arm]["blocked_attempts"],
                  "main_hash_unchanged": True}
            for arm in ARMS},
        "control_hashes": source,
        "direct_checkpoint_sha256": DIRECT_SHA,
        "input_size": 512,
        "evaluation_weights": "ema",
        "validation_indices_sha256":
            CONTROL_HASHES["splits/L333_fast64.txt"],
    })
    report = [
        "# ProSAC-CT APGC structure probe on frozen RDDM ControlNet",
        "",
        "The official MedDINOv3 ViT-B/16 CT-3M teacher was frozen. This is an APGC transfer probe, not a ProSAC-CT reproduction.",
        "C3 uses its final normalized dense patch tokens; C4 adds explicit Sobel/Laplacian modulation.",
        "The original RDDM Direct main and Control Encoder topology, ZeroConv taps, loss, sampler, and Fast64 evaluator were unchanged.",
        "",
        "| Arm | step | PSNR | delta vs C1 | soft PSNR | RMSE HU | positive slices vs C1 | trainable params | peak GiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        summary = summaries[arm]
        for r in histories[arm]:
            step = int(r["step"])
            c1, _ = reference("C1_CAPACITY", step)
            frac = comparisons[arm + "_minus_C1_CAPACITY_at_" + str(step)]["positive_slice_fraction"]
            report.append(
                "| %s | %d | %.8f | %+.8f | %.8f | %.8f | %.3f | %d | %.3f |" %
                (arm, step, float(r["psnr"]),
                 float(r["psnr"]) - float(c1["psnr"]),
                 float(r["soft_psnr"]), float(r["rmse_hu"]), frac,
                 params[arm]["trainable_parameters"],
                 summary["peak_vram_bytes"] / 1024 ** 3))
    report += ["", "## Gate and interpretation", ""]
    for arm in ARMS:
        d = decisions[arm]
        report.append(
            "- %s: 2k delta vs C1 %+.8f dB; positive slice fraction %.3f; continue to 6k: %s." %
            (arm, d["delta_psnr_vs_C1"]["2000"],
             d["positive_slice_fraction_vs_C1"], d["continue_to_6k"]))
    c3 = float(histories["C3_ANATOMY"][-1]["psnr"])
    c4 = float(histories["C4_APGC"][-1]["psnr"])
    report += [
        "",
        "C3 anatomy and C4 APGC claims are limited to observed Fast64 deltas at matched steps. Small positive differences do not establish a robust benefit.",
        "C4 minus C3 at their final common step: see PER_SLICE_DELTA.csv and FINAL_AUDIT.json.",
        "",
        "## Provenance and limits",
        "",
        "- MedDINO input is 512x512 native Mayo resolution, chosen explicitly for this experiment. The official notebook example sets 2048; weights, HU clip/normalization, channel repeat, constructor, and final patch-token convention follow that source.",
        "- Official paper Eq.(7) writes Sobel + Laplacian. This probe instead uses concat(Sobel,Laplacian) followed by trainable Conv1x1 as specified by the experiment request: RECONSTRUCTION_CHOICE_NOT_EXPLICIT_IN_PAPER.",
        "- C0/C1/C2 controls were reused only after protocol, Fast64 split and evaluator source hashes matched. C0 was separately reproduced in the flexict runtime within 1e-7 dB.",
        "- PyTorch/skimage runtime versions differ from the archived C1/C2 run; this is a cross-runtime comparison. Source and evaluation formulas are unchanged, and the C0 numeric reproduction bounds the observed mismatch.",
        "- L506, external memory, and neighbor NDCT access were zero; no formal 8-patient training was run.",
    ]
    (HERE / "PROSAC_APGC_REPORT.md").write_text("\n".join(report) + "\n")
    write_json(HERE / "PROSAC_APGC_SUMMARY.json", {
        "status": "COMPLETE", "decisions": decisions,
        "comparisons": comparisons,
        "histories": histories,
        "parameters": params,
        "peak_vram_bytes": {arm: summaries[arm]["peak_vram_bytes"] for arm in ARMS},
        "training_seconds": {arm: summaries[arm]["training_seconds"] for arm in ARMS},
        "source_hashes": source,
        "L506_opens": 0, "external_memory_opens": 0,
        "neighbor_target_opens": 0,
    })
    print("REPORT_COMPLETE", flush=True)


if __name__ == "__main__":
    build_report()

