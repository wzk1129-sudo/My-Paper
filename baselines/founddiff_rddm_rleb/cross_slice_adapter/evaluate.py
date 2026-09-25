"""Frozen L333 seed/metric path with lightweight residual-correction diagnostics."""
import contextlib
import math
import os
import time
import numpy as np
import torch
from mayo2016_smallset_v1.runtime import METRICS, write_json
from mayo2016_smallset_v1.frozen_protocol import slice_metrics
from fast_dev.protocol import validation_indices
from context_2p5d.model import restored_slice, training_loss
from src.rddm_rleb import normalize_to_neg_one_to_one


@torch.no_grad()
def validate_adapter(diffusion, dataset, cfg, device, output=None):
    expected = validation_indices(dataset.mode)
    assert tuple((r[0], r[1]) for r in dataset.records) == tuple(("L333", i) for i in expected)
    diffusion.eval()
    wrapped = diffusion.model.unet0
    rows = []
    started = time.perf_counter()
    moments = [0.0] * 6  # n, sum(x), sum(y), sum(x*x), sum(y*y), sum(x*y)
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        with open(os.devnull, "w") as quiet, contextlib.redirect_stderr(quiet):
            for pos in range(len(dataset)):
                prev, center, next_, target, meta = dataset.load(pos)
                prev = torch.from_numpy(prev).unsqueeze(0).to(device)
                center = torch.from_numpy(center).unsqueeze(0).to(device)
                next_ = torch.from_numpy(next_).unsqueeze(0).to(device)
                target = torch.from_numpy(target).unsqueeze(0).to(device)
                seed = cfg["evaluation"]["validation_seed"] + meta["slice_index"]
                torch.manual_seed(seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed(seed)
                prediction = restored_slice(diffusion, center, prev, next_)
                assert prediction.shape == (1, 1, 512, 512)
                assert bool(torch.isfinite(prediction).all())
                metrics = slice_metrics(prediction, target, cfg)
                assert all(math.isfinite(metrics[k]) for k in METRICS)
                loss_seed = seed + cfg["evaluation"]["loss_seed_offset"]
                torch.manual_seed(loss_seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed(loss_seed)
                loss = float(training_loss(diffusion, target, center, prev, next_))
                r_gt = normalize_to_neg_one_to_one(center) - normalize_to_neg_one_to_one(target)
                r_base = wrapped.last_base
                delta = wrapped.last_delta
                d_abs = wrapped.last_d_abs
                baseline_mse = float(torch.mean((r_base - r_gt) ** 2))
                corrected_mse = float(torch.mean((r_base + delta - r_gt) ** 2))
                assert math.isfinite(loss) and abs(loss - corrected_mse) < 1e-7
                absolute = delta.abs()
                delta_rms = float(torch.sqrt(torch.mean(delta ** 2)))
                delta_mean_abs = float(absolute.mean())
                delta_p95 = float(torch.quantile(absolute.flatten(), 0.95))
                x = absolute.double()
                y = d_abs.double()
                values = (x.numel(), float(x.sum()), float(y.sum()),
                          float((x*x).sum()), float((y*y).sum()), float((x*y).sum()))
                moments = [a+b for a,b in zip(moments, values)]
                rows.append({
                    **meta, **{k: metrics[k] for k in METRICS},
                    "validation_loss": loss, "sampling_seed": seed,
                    "baseline_residual_mse": baseline_mse,
                    "corrected_residual_mse": corrected_mse,
                    "delta_r_rms": delta_rms,
                    "mean_abs_delta_r": delta_mean_abs,
                    "p95_abs_delta_r": delta_p95,
                })
                wrapped.last_base = wrapped.last_delta = wrapped.last_d_abs = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    n,sx,sy,sxx,syy,sxy = moments
    vx = max(0.0, sxx / n - (sx/n)**2)
    vy = max(0.0, syy / n - (sy/n)**2)
    corr = (sxy/n - (sx/n)*(sy/n)) / math.sqrt(vx*vy) if vx > 0 and vy > 0 else 0.0
    assert math.isfinite(corr)
    if output is not None:
        write_json(output, rows)
    keys = (*METRICS, "validation_loss", "baseline_residual_mse",
            "corrected_residual_mse", "delta_r_rms", "mean_abs_delta_r",
            "p95_abs_delta_r")
    summary = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    summary.update({
        "corr_abs_delta_r_D_abs": corr,
        "corr_undefined_zero_variance": vx == 0 or vy == 0,
        "p95_definition": "mean per-slice p95 absolute delta_r",
        "diagnostic_source": "fixed-seed validation loss forward",
        "slices": len(rows), "patient": "L333", "weights": "ema",
        "validation_mode": dataset.mode,
        "validation_seconds": time.perf_counter() - started,
    })
    return summary
