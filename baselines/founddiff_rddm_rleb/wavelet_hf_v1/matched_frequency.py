"""Inference-only matched Wavelet versus Direct residual frequency diagnosis."""
import contextlib
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from context_2p5d.data import ContextDataGuard, ContextPairs
from context_2p5d.model import ARMS, construct, restored_slice
from context_2p5d.protocol import config as direct_config
from fast_dev.protocol import validation_indices
from mayo2016_smallset_v1.frozen_protocol import slice_metrics
from wavelet_hf_v1.haar import Haar, unit_test

ROOT = Path(__file__).resolve().parents[1]
WAVELET = ROOT / "experiments/wavelet_hf_v1"
DIRECT = Path("/home/xujian/projects/CoreDiff/experiments/rddm_2p5d_screen_v1_20260924/baselines/founddiff_rddm_rleb/experiments/rddm_2p5d_screen_v1")
OUT = WAVELET / "diagnostics/matched_frequency_v1"
STEPS = (2000, 4000, 6000, 8000)
BANDS = ("ll", "lh", "hl", "hh")
METRICS = ("psnr", "ssim", "rmse_hu", "mae_hu")
FREQ = ("ll_mse", "lh_mse", "hl_mse", "hh_mse", "mean_hf_mse", "spatial_residual_mse")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def save_csv(path, rows):
    assert rows
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def references():
    result = {}
    for name, path in (
        ("direct", DIRECT / "2P5D_DIRECT_CONCAT/validation_history.csv"),
        ("wavelet", WAVELET / "validation_history.csv"),
    ):
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert tuple(int(row["step"]) for row in rows[:4]) == STEPS
        assert all(row["patient"] == "L333" and int(row["slices"]) == 64 and
                   row["weights"] == "ema" for row in rows)
        result[name] = {int(row["step"]): row for row in rows}
    return result


def checkpoint(name, step):
    if name == "direct":
        return DIRECT / "runs/2P5D_DIRECT_CONCAT/checkpoints/step_{}.pt".format(step)
    return WAVELET / "fast8k/checkpoints/step_{}.pt".format(step)


def load_ema(name, step, device, cfg, init_audit, haar_hash):
    path = checkpoint(name, step)
    assert path.is_file(), str(path)
    payload = torch.load(str(path), map_location="cpu")
    assert payload["arm"] == ARMS[0]
    assert payload["optimizer_step"] == payload["ema_updates"] == step
    assert payload["config"] == cfg
    assert payload["initialization_audit"] == init_audit
    if name == "wavelet":
        assert payload["wavelet_hf"]["haar_source_sha256"] == haar_hash
    model, rebuilt_audit = construct(ARMS[0], device, cfg["seed"])
    assert rebuilt_audit == init_audit
    model.load_state_dict(payload["ema_state_dict"], strict=True)
    model.eval().requires_grad_(False)
    return model, {"path": str(path), "sha256": digest(path), "optimizer_step": step,
                   "ema_updates": payload["ema_updates"], "weights": "ema"}


def evaluate(model, dataset, cfg, device, haar, name, step):
    assert tuple((record[0], record[1]) for record in dataset.records) == tuple(
        ("L333", index) for index in validation_indices("FAST"))
    rows = []
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.no_grad(), torch.random.fork_rng(devices=devices):
        with open(os.devnull, "w") as quiet, contextlib.redirect_stderr(quiet):
            for position in range(len(dataset)):
                prev, center, next_, target, meta = dataset.load(position)
                prev = torch.from_numpy(prev).unsqueeze(0).to(device)
                center = torch.from_numpy(center).unsqueeze(0).to(device)
                next_ = torch.from_numpy(next_).unsqueeze(0).to(device)
                target = torch.from_numpy(target).unsqueeze(0).to(device)
                seed = cfg["evaluation"]["validation_seed"] + meta["slice_index"]
                torch.manual_seed(seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed(seed)
                captured = []
                original = model.model_predictions

                def capture(*args, **kwargs):
                    prediction = original(*args, **kwargs)
                    captured.append(prediction.pred_res.detach().clone())
                    return prediction

                model.model_predictions = capture
                try:
                    restored = restored_slice(model, center, prev, next_)
                finally:
                    model.model_predictions = original
                assert len(captured) == cfg["diffusion"]["sampling_timesteps"]
                assert restored.shape == captured[-1].shape == (1, 1, 512, 512)
                residual_gt = 2.0 * (center - target)
                residual_prediction = captured[-1]
                errors = haar.errors(residual_prediction, residual_gt)
                spatial = F.mse_loss(residual_prediction, residual_gt)
                metrics = slice_metrics(restored, target, cfg)
                row = {"model": name, "step": step, "slice_index": meta["slice_index"],
                       "sampling_seed": seed,
                       "ll_mse": float(errors["LL"]), "lh_mse": float(errors["LH"]),
                       "hl_mse": float(errors["HL"]), "hh_mse": float(errors["HH"]),
                       "mean_hf_mse": float(errors["HF"]),
                       "spatial_residual_mse": float(spatial)}
                row.update({key: float(metrics[key]) for key in METRICS})
                assert all(math.isfinite(value) for key, value in row.items()
                           if key not in ("model",))
                rows.append(row)
    assert len(rows) == 64
    assert [row["slice_index"] for row in rows] == list(validation_indices("FAST"))
    summary = {key: float(np.mean([row[key] for row in rows]))
               for key in FREQ + METRICS}
    return rows, summary


def comparison(step, direct, wavelet):
    row = {"step": step}
    for metric in FREQ:
        row["Direct_" + metric.upper()] = direct[metric]
        row["Wavelet_" + metric.upper()] = wavelet[metric]
        row["Delta_" + metric.upper()] = wavelet[metric] - direct[metric]
    row["HF_relative_improvement_percent"] = (
        (direct["mean_hf_mse"] - wavelet["mean_hf_mse"]) /
        direct["mean_hf_mse"] * 100.0)
    for metric in METRICS:
        row["Direct_" + metric.upper()] = direct[metric]
        row["Wavelet_" + metric.upper()] = wavelet[metric]
        row["Delta_" + metric.upper()] = wavelet[metric] - direct[metric]
    return row


def paired(step, direct, wavelet):
    assert [r["slice_index"] for r in direct] == [r["slice_index"] for r in wavelet]
    result = []
    for d, w in zip(direct, wavelet):
        row = {"step": step, "slice_index": d["slice_index"]}
        for metric in FREQ + METRICS:
            row["direct_" + metric] = d[metric]
            row["wavelet_" + metric] = w[metric]
            row["delta_" + metric] = w[metric] - d[metric]
        result.append(row)
    return result


def main():
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
    assert sys.version_info[:2] == (3, 8), sys.version
    assert torch.__version__ == "1.10.1+cu111", torch.__version__
    assert torch.version.cuda == "11.1"
    assert not OUT.exists(), str(OUT)
    for name in ("direct", "wavelet"):
        for step in STEPS:
            assert checkpoint(name, step).is_file(), str(checkpoint(name, step))
    cfg, base = direct_config()
    archived = json.loads((WAVELET / "CONFIG.json").read_text())
    for key in ("seed", "data", "model", "diffusion", "training", "ema",
                "augmentation", "evaluation", "fast_dev"):
        assert archived[key] == cfg[key]
    assert cfg["diffusion"]["sampling_timesteps"] == 2
    assert cfg["diffusion"]["sum_scale"] == 0.01
    assert cfg["evaluation"]["hu_window"] == [-1000, 1000]
    haar_hash = digest(ROOT / "wavelet_hf_v1/haar.py")
    prior = json.loads((WAVELET / "final_summary.json").read_text())
    assert haar_hash == prior["source_sha256"]["haar"]
    assert unit_test()["status"] == "PASS"
    init_audit = json.loads((DIRECT / "2P5D_DIRECT_CONCAT/INITIALIZATION_AUDIT.json").read_text())
    assert init_audit["canonical_state_sha256"] == "4ed6f3dfae251f4821a31cea8877557d0a0646026e51d8886ea9dcc19baf6e7e"
    refs = references()
    OUT.mkdir(parents=True)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    haar = Haar(device)
    guard = ContextDataGuard(cfg, "FAST", allow_training=False)
    all_rows = []
    main_rows = []
    checkpoint_audit = []
    with guard:
        dataset = ContextPairs(base, "validation", "FAST")
        dataset.bind_guard(guard)
        assert len(dataset) == 64
        for step in STEPS:
            by_name = {}
            for name in ("direct", "wavelet"):
                model, ckpt_audit = load_ema(name, step, device, cfg, init_audit, haar_hash)
                checkpoint_audit.append({"model": name, **ckpt_audit})
                rows, summary = evaluate(model, dataset, cfg, device, haar, name, step)
                expected = refs[name][step]
                for metric in METRICS:
                    assert abs(summary[metric] - float(expected[metric])) < 1e-5, (
                        "validation mismatch", name, step, metric, summary[metric], expected[metric])
                by_name[name] = (rows, summary)
                all_rows.extend(rows)
                print("VALIDATED", name, step, "PSNR", summary["psnr"],
                      "HF", summary["mean_hf_mse"], flush=True)
                del model
                torch.cuda.empty_cache()
            main_rows.append(comparison(step, by_name["direct"][1],
                                        by_name["wavelet"][1]))
    access = guard.report()
    assert access["L506_opens"] == access["external_memory_opens"] == 0
    assert access["neighbor_target_opens"] == access["blocked_attempts"] == 0
    assert access["data_directory_enumeration"] is False
    assert access["training_target_opens"] == 0
    paired_rows = paired(8000,
                         [r for r in all_rows if r["model"] == "direct" and r["step"] == 8000],
                         [r for r in all_rows if r["model"] == "wavelet" and r["step"] == 8000])
    hf_improvements = np.array([-r["delta_mean_hf_mse"] for r in paired_rows])
    psnr_improvements = np.array([r["delta_psnr"] for r in paired_rows])
    corr = float(np.corrcoef(hf_improvements, psnr_improvements)[0, 1])
    assert math.isfinite(corr)
    diagnostics = {
        "status": "PASS", "patient": "L333", "mode": "FAST", "slices": 64,
        "steps": list(STEPS), "weights": "ema", "sampling_timesteps": 2,
        "sum_scale": 0.01, "haar_source_sha256": haar_hash,
        "frequency_definition": "final clipped pred_res from the last deterministic sampling step versus 2*(center_LDCT-center_NDCT); both in [-1,1] protocol units",
        "historical_wavelet_frequency_metrics_note": "training-loss-timestep errors in the prior frequency_metrics.csv are a different quantity and are not substituted here",
        "hf_improved_slice_fraction_8k": float(np.mean(hf_improvements > 0)),
        "psnr_improved_slice_fraction_8k": float(np.mean(psnr_improvements > 0)),
        "pearson_hf_improvement_vs_psnr_delta_8k": corr,
        "pearson_is_causal": False,
        "comparison": main_rows, "checkpoint_audit": checkpoint_audit,
        "data_access": access,
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__,
                        "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(device)},
    }
    save_csv(OUT / "MATCHED_FREQUENCY_COMPARISON.csv", main_rows)
    save_csv(OUT / "MATCHED_FREQUENCY_PER_SLICE_8K.csv", paired_rows)
    save_csv(OUT / "MATCHED_FREQUENCY_PER_SLICE_ALL.csv", all_rows)
    save_json(OUT / "FREQUENCY_DIAGNOSIS.json", diagnostics)
    save_json(OUT / "DATA_ACCESS_AUDIT.json", access)
    print("DIAGNOSIS_PASS", flush=True)


if __name__ == "__main__":
    main()
