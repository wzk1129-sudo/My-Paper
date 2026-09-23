"""Read-only numerical definitions for the L333 60k bottleneck diagnostic."""
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter, sobel
from scipy.stats import spearmanr

BANDS = (("LOW", 0.0, 0.10), ("MID", 0.10, 0.25), ("HIGH", 0.25, 0.50))
N = 512
F = np.fft.fftshift(np.fft.fftfreq(N))
FX, FY = np.meshgrid(F, F, indexing="xy")
RADIAL = np.sqrt(FX ** 2 + FY ** 2)
MASKS = {name: (RADIAL >= lo) & ((RADIAL < hi) if name != "HIGH" else (RADIAL <= hi))
         for name, lo, hi in BANDS}
HANN = np.outer(np.hanning(N), np.hanning(N))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    if not rows:
        raise ValueError("No rows: " + str(path))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def stats(values):
    a = np.asarray([v for v in values if v is not None and math.isfinite(float(v))], dtype=float)
    if not len(a):
        return {"n": 0, "mean": None, "median": None, "std": None, "q25": None, "q75": None, "iqr": None}
    q25, q75 = np.percentile(a, [25, 75])
    return {"n": len(a), "mean": float(a.mean()), "median": float(np.median(a)),
            "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "q25": float(q25), "q75": float(q75), "iqr": float(q75-q25)}


def block_bootstrap(values, statistic="median", block=10, repetitions=1000, seed=20260923):
    a = np.asarray(values, dtype=float)
    if len(a) != 610 or not np.isfinite(a).any():
        raise ValueError("Block bootstrap requires 610 finite ordered L333 slices")
    rng = np.random.RandomState(seed)
    estimates = np.empty(repetitions)
    for i in range(repetitions):
        picks = []
        while len(picks) < len(a):
            start = int(rng.randint(0, len(a) - block + 1))
            picks.extend(range(start, start + block))
        sample = a[picks[:len(a)]]
        estimates[i] = np.nanmedian(sample) if statistic == "median" else np.nanmean(sample)
    return [float(x) for x in np.percentile(estimates, [2.5, 97.5])]


def gradient(image):
    smooth = gaussian_filter(image, sigma=1.0)
    gx = sobel(smooth, axis=1)
    gy = sobel(smooth, axis=0)
    return gx, gy, np.hypot(gx, gy)


def region_metrics(index, ld, pred, gt):
    body = gt > -900
    core = binary_erosion(body, iterations=5)
    ncore = int(core.sum())
    row = {"slice_index": index, "valid_for_structure": ncore >= 1000,
           "n_body": int(body.sum()), "n_body_core": ncore,
           "body_fraction": float(body.mean())}
    if ncore < 1000:
        return row
    gxg, gyg, gg = gradient(gt)
    gxl, gyl, gl = gradient(ld)
    gxp, gyp, gp = gradient(pred)
    lo, hi = np.percentile(gg[core], [50, 80])
    edge = core & (gg >= hi)
    flat = core & (gg <= lo)
    row.update(n_edge=int(edge.sum()), n_flat=int(flat.sum()),
               gt_gradient_energy=float(np.mean(gg[core] ** 2)))
    def err(a, mask):
        d = a[mask] - gt[mask]
        return float(np.mean(np.abs(d))), float(np.sqrt(np.mean(d*d)))
    for name, image in (("ldct", ld), ("pred", pred)):
        for label, mask in (("edge", edge), ("flat", flat)):
            mae, rmse = err(image, mask)
            row[f"mae_{name}_{label}"] = mae
            row[f"rmse_{name}_{label}"] = rmse
    row["edge_flat_mae_ratio_ldct"] = row["mae_ldct_edge"] / (row["mae_ldct_flat"] + 1e-12)
    row["edge_flat_mae_ratio_pred"] = row["mae_pred_edge"] / (row["mae_pred_flat"] + 1e-12)
    residual = (pred - gt)**2
    row["edge_error_concentration"] = (float(residual[edge].sum()) /
        (float(residual[core].sum()) + 1e-12)) / (float(edge.sum()) / ncore)
    for name, gx, gy, mag in (("ldct", gxl, gyl, gl), ("pred", gxp, gyp, gp)):
        row[f"gradient_magnitude_ratio_{name}"] = float(mag[edge].mean() / (gg[edge].mean() + 1e-12))
        cos = (gx[edge]*gxg[edge] + gy[edge]*gyg[edge]) / (mag[edge]*gg[edge] + 1e-8)
        row[f"orientation_cos_{name}"] = float(np.mean(cos))
    row["edge_rmse_remaining_ratio"] = row["rmse_pred_edge"] / (row["rmse_ldct_edge"] + 1e-12)
    row["flat_rmse_remaining_ratio"] = row["rmse_pred_flat"] / (row["rmse_ldct_flat"] + 1e-12)
    return row


def spectrum(image, subtract_mean=True):
    x = image - image.mean() if subtract_mean else image
    return np.abs(np.fft.fftshift(np.fft.fft2(x * HANN, norm="ortho")))**2


def frequency_metrics(index, ld, pred, gt):
    p_ld = spectrum(ld - gt)
    p_pred = spectrum(pred - gt)
    p_gt = spectrum(gt)
    rows = []
    for band, mask in MASKS.items():
        eld = float(p_ld[mask].sum())
        epred = float(p_pred[mask].sum())
        egt = float(p_gt[mask].sum())
        rows.append({"slice_index": index, "band": band, "e_ld_band": eld,
                     "e_pred_band": epred, "e_gt_band": egt,
                     "remaining_error_ratio_band": epred/(eld+1e-12),
                     "normalized_error_pred_band": epred/(egt+1e-12)})
    return rows


def correlations(rows, outcomes, predictors):
    result = {}
    for outcome in outcomes:
        result[outcome] = {}
        for predictor in predictors:
            pairs = [(r[outcome], r[predictor]) for r in rows
                     if r[outcome] is not None and r[predictor] is not None
                     and math.isfinite(float(r[outcome])) and math.isfinite(float(r[predictor]))]
            if len(pairs) < 3:
                result[outcome][predictor] = {"n": len(pairs), "rho": None, "p_naive": None}
                continue
            rho, p = spearmanr(*zip(*pairs))
            result[outcome][predictor] = {"n": len(pairs), "rho": float(rho), "p_naive": float(p)}
    return result
