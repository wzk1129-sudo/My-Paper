"""Inference-only bottleneck diagnostic of formal 60k EMA; L333 allowlist only."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import contextlib
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy
import skimage
import torch

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
CORE = Path("/home/xujian/projects/CoreDiff")
RUN = CORE / "experiments/founddiff_table_v_20260921/rddm_rleb/mayo2016_smallset_v1/runs/FoundDiff_RDDM_RLEB_Mayo2016_smallset_v1_60k_20260921"
OUT = BASE / "experiments/diagnostics/rddm_60k_bottleneck_v1"
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(HERE))

from mayo2016_smallset_v1.data import CenterPairs
from mayo2016_smallset_v1.frozen_protocol import slice_metrics
from mayo2016_smallset_v1.runtime import check_config, seed_everything, validate
from src.rddm_rleb import build_baseline
from protocol import (block_bootstrap, correlations, frequency_metrics, region_metrics,
                      sha256, stats, write_csv, write_json)

EXPECTED = {"psnr": 40.65805039, "ssim": 0.94613206,
            "rmse_hu": 18.6488788, "mae_hu": 12.1817366}
TOL = {"psnr": 1e-4, "ssim": 1e-5, "rmse_hu": 1e-3, "mae_hu": 1e-3}
SOURCE_SHA = "e8ed93223fbdf4a0e2521f73c7a4f213561f6a191a1a7b2a5f95934b85ebf574"
CHECKPOINT_SHA = "49aa8ce2ee5c98c4deeaf3067d7059f54f18d7715e30439cd9326b6177867469"
CKPT = RUN / "checkpoints/step_60000.pt"
SOURCE = BASE / "src/rddm_rleb.py"


def state_hash(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def hu_window(tensor):
    a = tensor.detach().float().cpu().squeeze().numpy().astype(np.float64)
    return np.clip(a * 4096.0 - 1024.0, -1000.0, 1000.0)


def finite_rows(rows, keys):
    for row in rows:
        for key in keys:
            value = row.get(key)
            if value is not None and not math.isfinite(float(value)):
                raise FloatingPointError(f"Nonfinite {key}: {row}")


def summarize_series(rows, fields, use_block=True):
    result = {}
    for field in fields:
        ordered = [r.get(field) for r in rows]
        entry = stats(ordered)
        if use_block and len(ordered) == 610:
            values = np.array([float(v) if v is not None else np.nan for v in ordered])
            entry["block_bootstrap_95ci_median"] = block_bootstrap(values, "median")
            entry["block_bootstrap_95ci_mean"] = block_bootstrap(values, "mean")
        result[field] = entry
    return result


def write_report(control, structure, frequency, timestep, hard, flags, runtime):
    lines = [
        "# RDDM 60k Bottleneck Diagnostic V1", "",
        "FoundDiff RDDM+RLEB backbone adapted to our fixed Mayo-2016 protocol.",
        "Read-only inference of step_60000.pt EMA on full L333 (610 slices); L506 and external memory were not opened.", "",
        f"Control reproduction: **{'PASS' if control['pass'] else 'FAIL'}**. "
        f"PSNR {control['observed']['psnr']:.8f}, SSIM {control['observed']['ssim']:.8f}, "
        f"RMSE {control['observed']['rmse_hu']:.5f} HU, MAE {control['observed']['mae_hu']:.5f} HU.", "",
        "## Structure and boundaries", "",
        "| Measure | Median | Block-bootstrap 95% CI (median) |", "|---|---:|---:|"
    ]
    for key in ("edge_error_concentration", "edge_flat_mae_ratio_pred", "edge_rmse_remaining_ratio",
                "flat_rmse_remaining_ratio", "gradient_magnitude_ratio_ldct",
                "gradient_magnitude_ratio_pred", "orientation_cos_ldct", "orientation_cos_pred"):
        s = structure["metrics"][key]
        lines.append(f"| {key} | {s['median']:.6f} | {s['block_bootstrap_95ci_median']} |")
    lines += ["", f"Invalid-for-structure slices: {structure['invalid_indices']}.",
              "Edge and flat masks use GT NDCT body core and GT gradient percentiles. Remaining ratios compare prediction with raw LDCT.", "",
              "## Frequency", "",
              "| Band | Median remaining E_pred/E_ld | 95% CI | Median normalized E_pred/E_gt | 95% CI |",
              "|---|---:|---:|---:|---:|"]
    for band in ("LOW", "MID", "HIGH"):
        s = frequency["bands"][band]
        a, b = s["remaining_error_ratio_band"], s["normalized_error_pred_band"]
        lines.append(f"| {band} | {a['median']:.6f} | {a['block_bootstrap_95ci_median']} | "
                     f"{b['median']:.6f} | {b['block_bootstrap_95ci_median']} |")
    lines += ["", "Hann-windowed, demeaned 2D FFT power is summed in fixed radial cycles/pixel bands. "
              "Ratios are per slice; absolute band energies are also in FREQUENCY_BANDS.csv.", "",
              "## Direct timestep probe", "",
              "| t | Sampler visit | Median residual NMSE | Median one-pass x0 PSNR | Replicate PSNR SD mean |",
              "|---:|:---:|---:|---:|---:|"]
    for t in (0, 100, 250, 499, 750, 999):
        s = timestep["timesteps"][str(t)]
        lines.append(f"| {t} | {'yes' if t in (499, 999) else 'no'} | "
                     f"{s['residual_nmse']['median']:.6f} | {s['psnr']['median']:.6f} | "
                     f"{s['noise_replicate_variability']['psnr_within_slice_std_mean']:.6f} |")
    lines += ["", "This is a one-pass residual probe at fixed q_sample states, not a sampler sweep or final sampled-image comparison.", "",
              "## Hard slices", "",
              f"Hardest 20 indices by prediction RMSE: {hard['hardest20_indices']}.",
              f"Easiest 20 indices by prediction RMSE: {hard['easiest20_indices']}.", "",
              "| Predictor | rho(prediction RMSE) | rho(PSNR gain) |", "|---|---:|---:|"]
    for predictor in hard["predictors"]:
        a = hard["spearman"]["prediction_rmse"][predictor]["rho"]
        b = hard["spearman"]["psnr_gain"][predictor]["rho"]
        lines.append(f"| {predictor} | {a:.4f} | {b:.4f} |")
    lines += ["", "These Spearman correlations are descriptive within one continuous stack; no patient-level significance is inferred.", "",
              "## Synthesis and limits", "",
              f"Triggered engineering flags: {flags if flags else ['none']}.",
              "Potential mechanism categories follow only from observed signal: "
              + ", ".join(flags["mechanism_categories"]) + "." if isinstance(flags, dict) else "",
              "Flags are engineering heuristics. Passing or missing a flag does not establish whether a future module will improve restoration.",
              "Contiguous block bootstrap uses 10-slice blocks, 1000 resamples, seed 20260923. "
              "Intervals reflect within-L333 stack stability, not patient-population statistical significance.",
              f"Total runtime: {runtime:.1f} s. No optimizer step, training, parameter modification, baseline-file change, or Git push.",
              ""]
    # Avoid mixing a dict into the flag listing above; report caller supplies both names and categories.
    (OUT / "DIAGNOSTIC_REPORT.md").write_text("\n".join(lines))


def main():
    started = time.time()
    assert OUT.exists() and not any(OUT.iterdir()), "Refuse to overwrite diagnostic output"
    cfg = json.loads((RUN / "CONFIG.json").read_text())
    check_config(cfg)
    assert cfg["data"]["validation_patients"] == ["L333"]
    assert sha256(SOURCE) == SOURCE_SHA
    assert sha256(CKPT) == CHECKPOINT_SHA
    assert cfg["diffusion"]["sampling_timesteps"] == 2 and cfg["diffusion"]["sum_scale"] == 0.01
    cfg_sha = sha256(RUN / "CONFIG.json")
    source_sha = sha256(SOURCE)
    ck_sha = sha256(CKPT)
    torch.set_num_threads(cfg["training"]["cpu_threads"])
    seed_everything(cfg["seed"])
    device = torch.device("cuda:0")
    assert torch.cuda.is_available()
    ck = torch.load(str(CKPT), map_location="cpu")
    assert ck["optimizer_step"] == 60000 and ck["ema_updates"] == 60000
    assert ck["config"] == cfg and ck["source_hashes"]["src/rddm_rleb.py"] == SOURCE_SHA
    model = build_baseline().eval().requires_grad_(False)
    model.load_state_dict(ck["ema_state_dict"], strict=True)
    del ck
    model = model.to(device)
    initial_state = state_hash(model)
    data = CenterPairs(cfg, "validation")
    assert len(data) == 610 and all(r[0] == "L333" for r in data.records)
    print("CONTROL_START", flush=True)
    with torch.no_grad():
        observed = validate(model, data, cfg, device)
    delta = {k: observed[k]-EXPECTED[k] for k in EXPECTED}
    passed = all(abs(delta[k]) <= TOL[k] for k in EXPECTED)
    control = {"pass": passed, "expected": EXPECTED, "observed": observed,
               "difference": delta, "tolerance": TOL, "patient": "L333",
               "slices": 610, "weights": "ema_state_dict",
               "sampling_timesteps": 2, "sum_scale": 0.01}
    write_json(OUT / "CONTROL.json", control)
    print("CONTROL_PASS" if passed else "CONTROL_FAIL", json.dumps(delta), flush=True)
    if not passed:
        return 2
    assert state_hash(model) == initial_state
    structure_rows, frequency_rows, difficulty_rows = [], [], []
    print("SPATIAL_FREQUENCY_START", flush=True)
    with torch.no_grad(), torch.random.fork_rng(devices=[0]):
        with open(os.devnull, "w") as quiet, contextlib.redirect_stderr(quiet):
            for index in range(610):
                ld, gt, meta = data.load(index)
                assert meta == {"patient_id": "L333", "slice_index": index}
                ld_t = torch.from_numpy(ld).unsqueeze(0).to(device)
                gt_t = torch.from_numpy(gt).unsqueeze(0).to(device)
                torch.manual_seed(20260829 + index)
                torch.cuda.manual_seed(20260829 + index)
                pred_t = model.sample([ld_t], last=True)[-1]
                assert pred_t.shape == gt_t.shape and torch.isfinite(pred_t).all()
                pred_metrics = slice_metrics(pred_t, gt_t, cfg)
                ld_metrics = slice_metrics(ld_t, gt_t, cfg)
                ld_hu, pred_hu, gt_hu = hu_window(ld_t), hu_window(pred_t), hu_window(gt_t)
                reg = region_metrics(index, ld_hu, pred_hu, gt_hu)
                freqs = frequency_metrics(index, ld_hu, pred_hu, gt_hu)
                structure_rows.append(reg)
                frequency_rows.extend(freqs)
                high = freqs[2]
                difficulty_rows.append({
                    "slice_index": index,
                    "raw_ldct_psnr": ld_metrics["psnr"],
                    "prediction_psnr": pred_metrics["psnr"],
                    "psnr_gain": pred_metrics["psnr"]-ld_metrics["psnr"],
                    "raw_ldct_rmse": ld_metrics["rmse_hu"],
                    "prediction_rmse": pred_metrics["rmse_hu"],
                    "input_residual_rms": ld_metrics["rmse_hu"],
                    "gt_gradient_energy": reg.get("gt_gradient_energy"),
                    "gt_high_frequency_energy": high["e_gt_band"],
                    "body_fraction": reg["body_fraction"],
                    "edge_error_concentration": reg.get("edge_error_concentration"),
                    "high_frequency_remaining_ratio": high["remaining_error_ratio_band"]})
                if (index+1) % 100 == 0:
                    print(f"SPATIAL_FREQUENCY_PROGRESS {index+1}/610", flush=True)
    # Exact seeded re-run must agree with the frozen validation means.
    for metric, key in (("psnr", "prediction_psnr"), ("rmse_hu", "prediction_rmse")):
        assert abs(float(np.mean([r[key] for r in difficulty_rows]))-observed[metric]) < TOL[metric]
    finite_rows(difficulty_rows, ("raw_ldct_psnr", "prediction_psnr", "prediction_rmse"))
    print("TIMESTEP_START", flush=True)
    probe_indices = [round(float(x)) for x in np.linspace(0, 609, 128)]
    assert len(probe_indices) == len(set(probe_indices)) == 128
    ts = (0, 100, 250, 499, 750, 999)
    probe_rows = []
    with torch.no_grad(), torch.random.fork_rng(devices=[0]):
        for position, index in enumerate(probe_indices):
            ld, gt, meta = data.load(index)
            assert meta["patient_id"] == "L333" and meta["slice_index"] == index
            ld_t = torch.from_numpy(ld).unsqueeze(0).to(device)
            gt_t = torch.from_numpy(gt).unsqueeze(0).to(device)
            x_input = ld_t*2-1
            x_start = gt_t*2-1
            x_res = x_input-x_start
            for t in ts:
                tt = torch.tensor([t], device=device, dtype=torch.long)
                for replicate in range(3):
                    noise_seed = 20260829 + index + replicate*1000000
                    torch.manual_seed(noise_seed)
                    torch.cuda.manual_seed(noise_seed)
                    noise = torch.randn_like(x_start)
                    xt = model.q_sample(x_start, x_res, tt, noise)
                    time_input = model.alphas_cumsum[tt]*model.num_timesteps
                    beta_input = model.betas_cumsum[tt]*model.num_timesteps
                    raw = model.model(torch.cat((xt, x_input), dim=1),
                                      [time_input, beta_input], None)[0]
                    assert raw.shape == x_res.shape and torch.isfinite(raw).all()
                    residual_mse = float(((raw-x_res)**2).mean())
                    residual_nmse = float(((raw-x_res)**2).sum()/(x_res.square().sum()+1e-12))
                    pred_x0 = torch.clamp(x_input-raw, -1, 1)
                    metrics = slice_metrics((pred_x0+1)/2, gt_t, cfg)
                    probe_rows.append({"slice_index": index, "timestep": t,
                                       "sampler_2step_visit": t in (499, 999),
                                       "noise_replicate": replicate, "noise_seed": noise_seed,
                                       "residual_mse": residual_mse, "residual_nmse": residual_nmse,
                                       **{k: metrics[k] for k in EXPECTED}})
            if (position+1) % 16 == 0:
                print(f"TIMESTEP_PROGRESS {position+1}/128", flush=True)
    finite_rows(probe_rows, ("residual_mse", "residual_nmse", *EXPECTED))
    assert len(probe_rows) == 128*6*3
    print("AGGREGATE_START", flush=True)
    write_csv(OUT / "STRUCTURE_REGION.csv", structure_rows)
    write_csv(OUT / "FREQUENCY_BANDS.csv", frequency_rows)
    write_csv(OUT / "TIMESTEP_PROBE.csv", probe_rows)
    write_csv(OUT / "SLICE_DIFFICULTY.csv", difficulty_rows)
    ordered = sorted(difficulty_rows, key=lambda r: r["prediction_rmse"], reverse=True)
    hard20 = [dict(group="hardest", **r) for r in ordered[:20]]
    easy20 = [dict(group="easiest", **r) for r in ordered[-20:]]
    write_csv(OUT / "HARD_SLICES.csv", hard20+easy20)
    valid = [r for r in structure_rows if r["valid_for_structure"]]
    structure_fields = [k for k in valid[0] if k not in
                        ("slice_index", "valid_for_structure", "n_body", "n_body_core", "n_edge", "n_flat")]
    structure = {"valid_slices": len(valid),
                 "invalid_indices": [r["slice_index"] for r in structure_rows if not r["valid_for_structure"]],
                 "definitions": {"erosion_pixels": 5, "gaussian_sigma": 1.0,
                                 "edge_top_percent": 20, "flat_bottom_percent": 50},
                 "metrics": summarize_series(structure_rows, structure_fields)}
    write_json(OUT / "STRUCTURE_REGION_SUMMARY.json", structure)
    frequency = {"bands": {}, "frequency_units": "cycles/pixel", "block_length": 10,
                 "bootstrap_replicates": 1000, "bootstrap_seed": 20260923}
    for band in ("LOW", "MID", "HIGH"):
        rows = [r for r in frequency_rows if r["band"] == band]
        assert [r["slice_index"] for r in rows] == list(range(610))
        frequency["bands"][band] = summarize_series(
            rows, ("e_ld_band", "e_pred_band", "e_gt_band",
                   "remaining_error_ratio_band", "normalized_error_pred_band"))
    write_json(OUT / "FREQUENCY_SUMMARY.json", frequency)
    timestep = {"indices": probe_indices, "replicates": 3,
                "sampler_2step_visits": [999, 499], "timesteps": {}}
    for t in ts:
        rows = [r for r in probe_rows if r["timestep"] == t]
        s = summarize_series(rows, ("residual_mse", "residual_nmse", *EXPECTED), False)
        variability = {}
        for field in ("residual_nmse", "psnr"):
            per_slice_std = [float(np.std([r[field] for r in rows if r["slice_index"] == i],
                                          ddof=1)) for i in probe_indices]
            variability[field+"_within_slice_std_mean"] = float(np.mean(per_slice_std))
            variability[field+"_within_slice_std_median"] = float(np.median(per_slice_std))
        timestep["timesteps"][str(t)] = {**s, "noise_replicate_variability": variability}
    write_json(OUT / "TIMESTEP_SUMMARY.json", timestep)
    predictors = ("raw_ldct_rmse", "input_residual_rms", "gt_gradient_energy",
                  "gt_high_frequency_energy", "body_fraction",
                  "edge_error_concentration", "high_frequency_remaining_ratio")
    hard = {"hardest20_indices": [r["slice_index"] for r in hard20],
            "easiest20_indices": [r["slice_index"] for r in easy20],
            "predictors": predictors,
            "spearman": correlations(difficulty_rows, ("prediction_rmse", "psnr_gain"), predictors),
            "metrics": summarize_series(difficulty_rows,
                                        ("raw_ldct_psnr", "prediction_psnr", "psnr_gain",
                                         "raw_ldct_rmse", "prediction_rmse", "body_fraction"))}
    write_json(OUT / "HARD_SLICE_SUMMARY.json", hard)
    sm = structure["metrics"]
    fm = frequency["bands"]
    tm = timestep["timesteps"]
    flags = []
    if (sm["edge_error_concentration"]["median"] > 1.5 and
        sm["edge_flat_mae_ratio_pred"]["median"] > 1.5 and
        sm["orientation_cos_ldct"]["median"] >= .9):
        flags.append("STRUCTURE_BOTTLENECK_SIGNAL")
    if (sm["gradient_magnitude_ratio_pred"]["median"] < .9 and
        sm["orientation_cos_ldct"]["median"] >= .9):
        flags.append("POSSIBLE_STRUCTURE_OVERSMOOTHING")
    if (fm["HIGH"]["remaining_error_ratio_band"]["median"] -
        fm["LOW"]["remaining_error_ratio_band"]["median"] >= .15 and
        fm["HIGH"]["normalized_error_pred_band"]["median"] ==
        max(fm[b]["normalized_error_pred_band"]["median"] for b in ("LOW", "MID", "HIGH"))):
        flags.append("HIGH_FREQUENCY_BOTTLENECK_SIGNAL")
    best_psnr = max(tm[str(t)]["psnr"]["median"] for t in ts)
    for t in (499, 999):
        other_nmse = np.median([tm[str(x)]["residual_nmse"]["median"] for x in ts if x != t])
        if (tm[str(t)]["residual_nmse"]["median"] >= 1.25*other_nmse or
            tm[str(t)]["psnr"]["median"] <= best_psnr-.2):
            flags.append(f"SAMPLER_TIMESTEP_WEAKNESS_SIGNAL_t{t}")
    strong = [(outcome, predictor, result["rho"]) for outcome, group in hard["spearman"].items()
              for predictor, result in group.items()
              if result["rho"] is not None and abs(result["rho"]) >= .5]
    if strong:
        flags.append("STRONG_DIAGNOSTIC_CORRELATION")
    categories = []
    if any("STRUCTURE" in f for f in flags):
        categories.append("Structure")
    if any("FREQUENCY" in f for f in flags):
        categories.append("Frequency")
    if any("TIMESTEP" in f for f in flags):
        categories.append("Timestep")
    if not categories:
        categories.append("none-clear")
    summary = {"control_pass": True, "flags": flags, "strong_correlations": strong,
               "mechanism_categories": categories,
               "within_stack_only": True, "patient_population_significance": False,
               "invalid_structure_slices": structure["invalid_indices"],
               "full_validation_slices": 610, "timestep_probe_rows": len(probe_rows),
               "seconds": time.time()-started}
    write_json(OUT / "DIAGNOSTIC_SUMMARY.json", summary)
    # Simple plain-text report, based on computed tables only.
    lines = ["# RDDM 60k Bottleneck Diagnostic V1", "",
             "FoundDiff RDDM+RLEB backbone adapted to our fixed Mayo-2016 protocol.",
             "Only L333 was opened. step_60000.pt EMA, 2-step / 0.01 control. No training.", "",
             "Control reproduction: PASS. " +
             ", ".join(f"{k}={observed[k]:.8f} (delta {delta[k]:+.8g})" for k in EXPECTED), "",
             "## Structure-region", "",
             "| Measure | Median | Block-bootstrap 95% CI |", "|---|---:|---:|"]
    for key in ("edge_error_concentration", "edge_flat_mae_ratio_pred",
                "edge_rmse_remaining_ratio", "flat_rmse_remaining_ratio",
                "gradient_magnitude_ratio_ldct", "gradient_magnitude_ratio_pred",
                "orientation_cos_ldct", "orientation_cos_pred"):
        s = sm[key]
        lines.append(f"| {key} | {s['median']:.6f} | {s['block_bootstrap_95ci_median']} |")
    lines += ["", f"Invalid structure indices: {structure['invalid_indices']}.", "",
              "## Frequency", "",
              "| Band | Median remaining Epred/Eld | 95% CI | Median normalized Epred/Egt | 95% CI |",
              "|---|---:|---:|---:|---:|"]
    for band in ("LOW", "MID", "HIGH"):
        a = fm[band]["remaining_error_ratio_band"]
        b = fm[band]["normalized_error_pred_band"]
        lines.append(f"| {band} | {a['median']:.6f} | {a['block_bootstrap_95ci_median']} | "
                     f"{b['median']:.6f} | {b['block_bootstrap_95ci_median']} |")
    lines += ["", "## Timestep direct probe (not sampler sweep)", "",
              "| t | 2-step visit | Median residual NMSE | Median one-pass x0 PSNR | Mean within-slice replicate PSNR SD |",
              "|---:|:---:|---:|---:|---:|"]
    for t in ts:
        s = tm[str(t)]
        lines.append(f"| {t} | {'yes' if t in (499,999) else 'no'} | "
                     f"{s['residual_nmse']['median']:.6f} | {s['psnr']['median']:.6f} | "
                     f"{s['noise_replicate_variability']['psnr_within_slice_std_mean']:.6f} |")
    lines += ["", "## Hard slices", "",
              f"Hardest 20 indices: {hard['hardest20_indices']}.",
              f"Easiest 20 indices: {hard['easiest20_indices']}.", "",
              "| Predictor | rho(prediction RMSE) | rho(PSNR gain) |",
              "|---|---:|---:|"]
    for predictor in predictors:
        a = hard["spearman"]["prediction_rmse"][predictor]["rho"]
        b = hard["spearman"]["psnr_gain"][predictor]["rho"]
        lines.append(f"| {predictor} | {a:.4f} | {b:.4f} |")
    lines += ["", "## Interpretation", "",
              f"Engineering flags: {flags if flags else ['none']}.",
              f"Mechanism categories worth testing: {categories}.",
              "Compare prediction and raw LDCT columns in the structure and frequency tables before choosing a mechanism.",
              "The direct timestep probe diagnoses residual prediction at fixed q_sample states. It is not a new sampler result.",
              "No flag proves a future model improvement, and absence of a flag does not rule one out.",
              "Block bootstrap: contiguous 10-slice blocks, 1000 repetitions, seed 20260923. "
              "Intervals reflect stability within this single L333 stack, not patient-population significance.",
              f"Elapsed time: {summary['seconds']:.1f} s. No optimizer step or Git push.", ""]
    (OUT / "DIAGNOSTIC_REPORT.md").write_text("\n".join(lines))
    assert state_hash(model) == initial_state
    assert sha256(SOURCE) == source_sha == SOURCE_SHA
    assert sha256(CKPT) == ck_sha == CHECKPOINT_SHA
    assert sha256(RUN/"CONFIG.json") == cfg_sha
    audit = {"training_patients_opened": [], "validation_patient": "L333",
             "L333_indices_opened": list(range(610)),
             "L333_pair_loads_by_phase": {"control": 610, "spatial_frequency": 610,
                                          "timestep": 128},
             "L506_opens": 0, "external_memory_opens": 0, "blocked_attempts": 0,
             "dataset_allowlist": ["L333"], "directory_enumeration": False,
             "optimizer_steps": 0, "model_parameters_unchanged": True,
             "model_state_sha256_before_after": initial_state,
             "formal_source_sha256_before_after": source_sha,
             "formal_checkpoint_sha256_before_after": ck_sha,
             "formal_config_sha256_before_after": cfg_sha}
    write_json(OUT / "DATA_ACCESS_AUDIT.json", audit)
    provenance = {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                   cwd=BASE, text=True).strip(),
                  "git_branch": subprocess.check_output(["git", "branch", "--show-current"],
                   cwd=BASE, text=True).strip(),
                  "source_sha256": source_sha, "checkpoint_sha256": ck_sha,
                  "diagnostic_code_sha256": {p.name: sha256(p) for p in
                    (HERE/"diagnose.py", HERE/"protocol.py")},
                  "python": sys.version, "pytorch": torch.__version__,
                  "cuda": torch.version.cuda, "scipy": scipy.__version__,
                  "skimage": skimage.__version__, "gpu": torch.cuda.get_device_name(0),
                  "seed_control": 20260829, "seed_bootstrap": 20260923,
                  "started_unix": started, "ended_unix": time.time(),
                  "runtime_seconds": time.time()-started}
    write_json(OUT / "RUN_PROVENANCE.json", provenance)
    print("DIAGNOSTIC_COMPLETE", json.dumps({"flags": flags, "seconds": summary["seconds"]}), flush=True)
    return 0


if __name__ == "__main__":
    try:
        status = main()
        sys.exit(status)
    except Exception:
        import traceback
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        raise
