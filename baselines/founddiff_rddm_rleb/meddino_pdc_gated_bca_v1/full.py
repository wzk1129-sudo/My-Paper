"""Frozen Full-L333 EMA inference and paired E12-BCA slice comparison."""
import csv
import json
import math

import numpy as np
import torch

from mayo2016_smallset_v1.runtime import ROOT, sha256, write_json
from meddino_four_arm_v1.runtime import load_direct
from meddino_four_arm_v1.structure import MultiProvider
from meddino_full_l333_v1.evaluate import dataset as full_dataset, evaluate as frozen_evaluate
from meddino_salf_v1.train import write_csv
from .model import graft_pdc, numeric_activity
from .protocol import ARM, HERE, FORMAL_SOURCE_SHA, FAST64_SPLIT_SHA, EVALUATOR_SHA, config

BCA_CSV = (ROOT.parent.parent.parent / "rddm_prosac_add_fusion_v1_20260926"
           / "baselines/founddiff_rddm_rleb/experiments/meddino_e12_bca_v1/FULL_L333_PER_SLICE.csv")
METRICS = ("psnr", "ssim", "rmse_hu", "mae_hu")


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_ema(device):
    ck = torch.load(HERE / "checkpoints/step_6000.pt", map_location="cpu", weights_only=False)
    assert ck["arm"] == ARM and ck["optimizer_step"] == ck["ema_updates"] == 6000
    model = load_direct("cpu")
    graft_pdc(model.model.unet0, MultiProvider("D1_MULTI_512"))
    model.load_state_dict(ck["ema_state_dict"], strict=True)
    assert model.objective == "pred_res" and model.sampling_timesteps == 2
    assert model.sum_scale == .01
    return model.to(device).eval()


def means(rows):
    return {key: float(np.mean([float(row[key]) for row in rows])) for key in METRICS}


def comparison(rows, reference):
    assert len(rows) == len(reference) == 610
    assert [int(r["slice_index"]) for r in rows] == list(range(610))
    assert [int(r["slice_index"]) for r in reference] == list(range(610))
    assert [r["slice_name"] for r in rows] == [r["slice_name"] for r in reference]
    deltas = []
    for a, b in zip(rows, reference):
        deltas.append({"slice_index": int(a["slice_index"]),
                       **{"delta_" + key: float(a[key]) - float(b[key]) for key in METRICS}})
    delta = np.array([r["delta_psnr"] for r in deltas], dtype=np.float64)
    draws = np.random.default_rng(20260926).integers(0, 610, size=(10000, 610))
    boot = delta[draws].mean(axis=1)
    write_csv(HERE / "PAIRED_VS_BCA.csv", deltas)
    own, base = means(rows), means(reference)
    return {
        "reference": "E12_BCA",
        "candidate_metrics": own, "reference_metrics": base,
        "aggregate_delta": {k: own[k] - base[k] for k in METRICS},
        "paired_slice_mean_delta_psnr": float(delta.mean()),
        "paired_slice_bootstrap_95_ci_psnr": [float(v) for v in np.quantile(boot, [.025, .975])],
        "positive_slice_fraction": float((delta > 0).mean()),
        "bootstrap_seed": 20260926, "bootstrap_draws": 10000,
        "ci_scope": "single-patient slice-level, not patient-level significance",
        "inference_seconds_ratio": float(
            np.mean([float(r["inference_seconds"]) for r in rows]) /
            np.mean([float(r["inference_seconds"]) for r in reference])),
    }


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
    bca = read_csv(BCA_CSV)
    paired = comparison(rows, bca)
    u = model.model.unet0
    s = u.structure
    attention = {
        "alpha128": float(u.bca128.alpha.detach()),
        "alpha64": float(u.bca64.alpha.detach()),
        "attention128": {k: float(v) for k, v in u.bca128.last_activity.items()},
        "attention64": {k: float(v) for k, v in u.bca64.last_activity.items()},
    }
    summary = {
        "status": "COMPLETE", "arm": ARM, "weights": "step6000 EMA",
        "slices": 610, "optimizer_steps_during_full": 0,
        "metrics": means(rows), "comparison_vs_E12_BCA": paired,
        "runtime": runtime, "attention": attention,
        "beta128": float(s.beta128.detach()), "beta64": float(s.beta64.detach()),
        "structure_shapes": s.last_shapes,
        "structure_activity": numeric_activity(s),
        "train_seconds_per_step": training["seconds_per_step"],
        "access": access,
    }
    write_json(HERE / "FULL_L333_SUMMARY.json", summary)
    write_json(HERE / "FULL_L333_DATA_ACCESS.json", access)
    source_sha = sha256(ROOT / "src/rddm_rleb.py")
    split_sha = sha256(ROOT / "splits/L333_fast64.txt")
    eval_sha = sha256(ROOT / "meddino_full_l333_v1/evaluate.py")
    assert source_sha == FORMAL_SOURCE_SHA and split_sha == FAST64_SPLIT_SHA
    assert eval_sha == EVALUATOR_SHA
    audit = {
        "status": "PASS", "branch": "experiment/rddm-meddino-pdc-gated-bca-v1",
        "formal_source_sha256": source_sha, "fast64_split_sha256": split_sha,
        "full_evaluator_sha256": eval_sha,
        "pdc_upstream_audit_status": json.loads((HERE / "OFFICIAL_PDC_EQUIVALENCE.json").read_text())["status"],
        "checkpoint_sha256": sha256(HERE / "checkpoints/step_6000.pt"),
        "smoke_pass": True, "training_6000_complete": True,
        "full_l333_610_complete": True,
        "nonfinite_count": training["nonfinite_count"],
        "gradient_usage": training["grad_usage"],
        "conditioning_residual_nonzero": all(
            attention[key]["weighted_residual_rms"] > 0
            for key in ("attention128", "attention64")),
        "shape_fusion_nonzero": all(
            summary["structure_activity"][key]["weighted_shape_rms"] > 0
            for key in ("128", "64")),
        "gate_spatial_nonconstant": all(
            summary["structure_activity"][key]["gate_spatial_std"] > 0
            for key in ("128", "64")),
        "L506_opens": training["data_access"]["L506_opens"] + access["L506_opens"],
        "external_memory_opens": training["data_access"]["external_memory_opens"] + access["external_memory_opens"],
        "neighbor_target_opens": training["data_access"]["neighbor_target_opens"] + access["neighbor_target_opens"],
        "blocked_attempts": training["data_access"]["blocked_attempts"] + access["blocked_attempts"],
        "meddino_frozen": True, "pdc_pretrained": False,
        "comparison_caveat": "Single-patient slice-level CI; PDC+gating+fusion are bundled changes relative to E12-BCA.",
    }
    assert all(audit[k] == 0 for k in ("L506_opens", "external_memory_opens",
                                      "neighbor_target_opens", "blocked_attempts"))
    assert audit["conditioning_residual_nonzero"] and audit["shape_fusion_nonzero"]
    assert audit["gate_spatial_nonconstant"]
    write_json(HERE / "FINAL_AUDIT.json", audit)
    ci = paired["paired_slice_bootstrap_95_ci_psnr"]
    lines = [
        "# MedDINO-PDC-Gated-BCA-V1", "",
        "PiDiNet-Tiny carv4 PDC features from center LDCT are projected, anatomy-gated and added to the unchanged E12 H128/H64 hints.",
        "The E12-BCA window attention (Q=RDDM encoder, K/V=fused structure) and trainable main backbone are unchanged.", "",
        "## Full L333 (610 slices, step6000 EMA)", "",
        "| Arm | PSNR | SSIM | RMSE HU | MAE HU |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, m in (("PDC-gated BCA", summary["metrics"]), ("E12-BCA", paired["reference_metrics"])):
        lines.append(f'| {name} | {m["psnr"]:.6f} | {m["ssim"]:.6f} | {m["rmse_hu"]:.6f} | {m["mae_hu"]:.6f} |')
    lines += [
        "", f'Delta PSNR {paired["aggregate_delta"]["psnr"]:+.6f} dB; slice bootstrap 95% CI [{ci[0]:+.6f},{ci[1]:+.6f}]; positive fraction {paired["positive_slice_fraction"]:.3f}.',
        "The CI is within one L333 patient and is not patient-level significance.",
        f'beta128={summary["beta128"]:.6f}, beta64={summary["beta64"]:.6f}; alpha128={attention["alpha128"]:.6f}, alpha64={attention["alpha64"]:.6f}.',
        f'Train time {summary["train_seconds_per_step"]:.3f}s/step; inference {runtime["total_inference_wall_seconds"]:.1f}s; peak inference VRAM {runtime["peak_vram_bytes"] / 2**30:.2f} GiB.',
        "No training during Full-L333; L506, external memory and neighbor targets untouched.",
    ]
    (HERE / "PDC_GATED_BCA_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": audit["status"], "psnr": summary["metrics"]["psnr"],
                      "delta_bca": paired["aggregate_delta"]["psnr"]}), flush=True)


if __name__ == "__main__":
    main()
