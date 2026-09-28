"""Inference-only sampler consistency audit for existing 4k/8k EMA weights."""
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from fast_dev.data import DataAccessGuard, FastPairs
from fast_dev.protocol import CONFIG, FAST_INDICES, check_config
from mayo2016_smallset_v1.frozen_protocol import slice_metrics
from mayo2016_smallset_v1.runtime import seed_everything
from res_noise_beta_time_v1.model import build_joint

ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT = ROOT / "experiments/res_noise_beta_time_v1"
RUN = EXPERIMENT / "runs/beta_time_fast8k_20260929"
OUT = Path(__file__).resolve().parent
METRICS = ("psnr", "ssim", "rmse_hu", "mae_hu")
PATHS = ("CURRENT_JOINT", "RESIDUAL_ONLY", "RDDM_UPDATE_ONLY",
         "FULL_CONSISTENT_JOINT", "ORACLE_INIT_DIAGNOSTIC_ONLY")
TIMES = (999, 499)
BASELINE_8K = 40.484568141498166

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()

def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")

def write_csv(name, rows):
    assert rows
    with open(OUT / name, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def stats(a, b):
    x, y = a.double().flatten(), b.double().flatten()
    xm, ym = x.mean(), y.mean()
    xs, ys = x.std(unbiased=False), y.std(unbiased=False)
    corr = (((x - xm) * (y - ym)).mean() / (xs * ys + 1e-30)).item()
    return (x.mean().item(), xs.item(), y.mean().item(), ys.item(),
            torch.mean((x-y)**2).item(), corr)

def evaluate(model, x_input, target, z, cfg, step, index):
    ndct = 2 * target - 1
    residual_gt = x_input - ndct
    A = model.alphas_cumsum
    B = model.betas_cumsum
    results, trajectories = {}, []
    seed = cfg["evaluation"]["validation_seed"] + index
    initial = {
        "CURRENT_JOINT": x_input + math.sqrt(model.sum_scale) * z,
        "RDDM_UPDATE_ONLY": x_input + math.sqrt(model.sum_scale) * z,
        # A_T is 0.993647..., so this is a documented GT-free approximation.
        "FULL_CONSISTENT_JOINT": x_input + B[999] * z,
        # Diagnostic upper bound only. GT is never passed to the model.
        "ORACLE_INIT_DIAGNOSTIC_ONLY": ndct + A[999] * residual_gt + B[999] * z,
    }
    for path, initial_state in initial.items():
        state = initial_state
        for time_index, t in enumerate(TIMES):
            t_tensor = torch.full((1,), t, device=state.device, dtype=torch.long)
            preds = model.model_predictions(x_input, state, t_tensor)
            predicted_residual = preds.pred_res
            predicted_noise = preds.pred_noise
            x0 = preds.pred_x_start
            effective_noise = (state - ndct - A[t] * residual_gt) / (B[t] + 1e-12)
            pn_mean, pn_std, ef_mean, ef_std, mse, corr = stats(predicted_noise, effective_noise)
            row = {
                "checkpoint": step, "path": path, "slice_index": index,
                "sampling_seed": seed, "timestep": t, "A_t": A[t].item(),
                "B_t": B[t].item(), "delta_A": 0.0, "delta_B": 0.0,
                "pred_noise_mean": pn_mean, "pred_noise_std": pn_std,
                "epsilon_eff_mean": ef_mean, "epsilon_eff_std": ef_std,
                "noise_MSE": mse, "noise_corr": corr,
                "pred_res_MSE": torch.mean((predicted_residual-residual_gt)**2).item(),
                "x0_PSNR": slice_metrics((x0+1)/2, target, cfg)["psnr"],
            }
            if path == "CURRENT_JOINT":
                residual_x0 = (x_input-predicted_residual).clamp(-1,1)
                residual_row = dict(row, path="RESIDUAL_ONLY",
                    x0_PSNR=slice_metrics((residual_x0+1)/2, target, cfg)["psnr"])
                trajectories.append(residual_row)
                if t == TIMES[-1]:
                    results["RESIDUAL_ONLY"] = (residual_x0+1)/2
            if time_index == 0:
                delta_A = (A[t]-A[TIMES[1]]).item()
                delta_B = (B[t]-B[TIMES[1]]).item()
                row["delta_A"] = delta_A
                row["delta_B"] = delta_B
                if path == "CURRENT_JOINT":
                    residual_row["delta_A"] = delta_A
                    residual_row["delta_B"] = delta_B
                state = state - delta_A * predicted_residual
                if path != "CURRENT_JOINT":
                    state = state - delta_B * predicted_noise
            else:
                results[path] = (x0+1)/2
            trajectories.append(row)
    return results, trajectories

def main():
    cfg = json.loads(CONFIG.read_text())
    check_config(cfg)
    assert cfg["evaluation"]["validation_seed"] == 20260829
    assert tuple(FAST_INDICES) and len(FAST_INDICES) == 64
    assert not any((OUT/name).exists() for name in ("SCHEDULE_AUDIT.json","FINAL_DIAGNOSIS.json","trajectory_diagnostics.csv")), "Refusing to overwrite diagnostic outputs"
    torch.set_num_threads(cfg["training"]["cpu_threads"])
    seed_everything(cfg["seed"])
    device = torch.device("cuda:0")
    assert torch.cuda.is_available()
    model = build_joint().to(device).eval().requires_grad_(False)
    assert model.num_timesteps == 1000 and model.sampling_timesteps == 2
    assert model.is_ddim_sampling and model.ddim_sampling_eta == 0
    assert model.condition and model.sum_scale == 0.01
    assert len(list(model.parameters())) > 0
    schedule = {
        "source": str(ROOT/"src/rddm_rleb.py"),
        "source_sha256": sha(ROOT/"src/rddm_rleb.py"),
        "convert_to_ddim": True, "sum_scale": model.sum_scale,
        "sqrt_sum_scale": math.sqrt(model.sum_scale),
        "sampling_timesteps": model.sampling_timesteps,
        "ddim_sampling_eta": model.ddim_sampling_eta,
        "training_q_sample": "x_t = x0 + A_t * (LDCT-NDCT) + B_t * epsilon, epsilon~N(0,I)",
        "sampling_initialization_current": "LDCT + sqrt(sum_scale)*z",
        "timesteps": {},
    }
    for t in TIMES:
        schedule["timesteps"][str(t)] = {
            "A_t": model.alphas_cumsum[t].item(),
            "B_t": model.betas_cumsum[t].item(),
            "abs_A_t_minus_1": abs(model.alphas_cumsum[t].item()-1),
        }
    bt = schedule["timesteps"]["999"]["B_t"]
    bm = schedule["timesteps"]["499"]["B_t"]
    schedule["B_999_over_sqrt_sum_scale"] = bt/math.sqrt(model.sum_scale)
    schedule["sqrt_sum_scale_over_B_999"] = math.sqrt(model.sum_scale)/bt
    schedule["sqrt_sum_scale_over_B_499"] = math.sqrt(model.sum_scale)/bm
    schedule["A_999_init_approximation_error_coefficient"] = 1-model.alphas_cumsum[999].item()
    write_json("SCHEDULE_AUDIT.json", schedule)
    previous = json.loads((EXPERIMENT/"DIAGNOSTIC_SUMMARY.json").read_text())
    access = {}
    metric_rows, trajectory_rows, summary = [], [], []
    with DataAccessGuard(cfg, "FAST", allow_training=False) as guard:
        data = FastPairs(cfg, "validation", "FAST")
        assert len(data) == 64
        for step in (4000, 8000):
            checkpoint = RUN/"checkpoints"/f"step_{step}.pt"
            ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
            assert ck["optimizer_step"] == step and ck["ema_updates"] == step
            model.load_state_dict(ck["ema_state_dict"], strict=True)
            del ck
            model.test_res_or_noise = "res_noise"
            checkpoint_sha = sha(checkpoint)
            per_slice, start = [], time.perf_counter()
            with torch.inference_mode():
                for position in range(len(data)):
                    ldct, target_np, meta = data.load(position)
                    index = meta["slice_index"]
                    assert meta["patient_id"] == "L333" and index == FAST_INDICES[position]
                    x = torch.from_numpy(ldct).unsqueeze(0).to(device)
                    target = torch.from_numpy(target_np).unsqueeze(0).to(device)
                    assert x.shape == target.shape == (1,1,512,512)
                    assert torch.isfinite(x).all() and torch.isfinite(target).all()
                    seed = cfg["evaluation"]["validation_seed"] + index
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed(seed)
                    z = torch.randn(x.shape, device=device)
                    predictions, traces = evaluate(model, 2*x-1, target, z, cfg, step, index)
                    if step == 4000 and position == 0:
                        for name, mode in (("CURRENT_JOINT","res_noise"),("RESIDUAL_ONLY","res")):
                            model.test_res_or_noise = mode
                            torch.manual_seed(seed)
                            torch.cuda.manual_seed(seed)
                            reference = model.sample([x.clone()], last=True)[-1]
                            max_error = (reference-predictions[name]).abs().max().item()
                            assert max_error < 2e-6, (name,max_error)
                        model.test_res_or_noise = "res_noise"
                    trajectory_rows.extend(traces)
                    for path in PATHS:
                        metric = slice_metrics(predictions[path], target, cfg)
                        assert all(math.isfinite(metric[k]) for k in METRICS)
                        per_slice.append({"checkpoint":step, "path":path,
                                          "slice_index":index, "sampling_seed":seed,
                                          **{k:metric[k] for k in METRICS}})
            write_csv(f"sampler_paths_{step//1000}k_per_slice.csv", per_slice)
            elapsed = time.perf_counter()-start
            by_path = []
            for path in PATHS:
                selected = [r for r in per_slice if r["path"] == path]
                row = {"checkpoint":step, "path":path, "slices":len(selected),
                       "validation_seconds":elapsed,
                       **{k:float(np.mean([r[k] for r in selected])) for k in METRICS}}
                by_path.append(row)
                metric_rows.append(row)
            write_csv(f"sampler_paths_{step//1000}k.csv", by_path)
            old = next(s for s in previous["steps"] if s["step"] == step)
            assert checkpoint_sha == old["checkpoint_sha256"]
            for path, old_key in (("CURRENT_JOINT","learned"),("RESIDUAL_ONLY","residual")):
                new_row = next(r for r in by_path if r["path"] == path)
                for key in METRICS:
                    assert abs(new_row[key]-old["metrics"][old_key][key]) < 1e-5, (
                        step, path, key, new_row[key], old["metrics"][old_key][key])
            summary.append({"step":step, "checkpoint_sha256":checkpoint_sha,
                            "control_reproduced":True, "paths":by_path})
            print(f"STEP_{step}_PASS", " ".join(f"{r['path']}={r['psnr']:.6f}" for r in by_path), flush=True)
        access = guard.report()
    assert access["validation_indices_opened"] == list(FAST_INDICES)
    assert access.get("test_opens",0) == 0 and access.get("external_memory_opens",0) == 0
    assert access.get("blocked_attempts",0) == 0
    write_csv("trajectory_diagnostics.csv", trajectory_rows)
    write_json("DATA_ACCESS_AUDIT.json", access)
    rows8 = {r["path"]:r for r in summary[1]["paths"]}
    current, update, full = (rows8[k]["psnr"] for k in
        ("CURRENT_JOINT","RDDM_UPDATE_ONLY","FULL_CONSISTENT_JOINT"))
    if update > 30 and current < 15:
        case = "CASE_1_UPDATE_MISMATCH_IMPORTANT"
    elif update <= 30 and full > 30:
        case = "CASE_2_INIT_SCALE_MISMATCH_PRIMARY"
    elif full >= 38 and full < BASELINE_8K:
        case = "CASE_3_JOINT_RESTORATION_BELOW_BASELINE"
    else:
        case = "CASE_4_JOINT_TRAJECTORY_UNUSABLE"
    diagnosis = {
        "status":"PASS", "primary_case":case, "steps":summary,
        "baseline_fast64_8k_psnr":BASELINE_8K,
        "delta_8k_psnr_vs_current":{p:rows8[p]["psnr"]-current for p in PATHS},
        "full_consistent_vs_baseline_8k":full-BASELINE_8K,
        "full_consistent_gt_free_approximation":True,
        "oracle_is_diagnostic_only":True,
        "training_performed":False, "optimizer_steps":0,
        "checkpoints_modified":False, "L506_opens":0, "external_memory_opens":0,
        "validation_patient":"L333", "validation_indices":list(FAST_INDICES),
        "data_access":access,
    }
    write_json("FINAL_DIAGNOSIS.json", diagnosis)
    print("DIAGNOSIS_PASS", case, flush=True)

if __name__ == "__main__":
    main()
