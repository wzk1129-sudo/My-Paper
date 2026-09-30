from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import skimage
import torch

from evaluation import (
    METRIC_DEFINITIONS,
    METRIC_NAMES,
    PROTOCOL_VERSION,
    SSIM_SETTINGS,
    evaluate_dataset,
    evaluate_fixed_visualizations,
)
from rddm_ct import (
    PairedCTDataset,
    build_system,
    config_sha256,
    git_state,
    load_config,
    seed_everything,
    sha256_file,
    write_json,
)


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]


def _git_output(*args):
    return subprocess.check_output(["git"] + list(args), cwd=str(PROJECT_ROOT), text=True).strip()


def _text_sha256(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def working_tree_manifest():
    tracked_diff = _git_output("diff", "--no-ext-diff", "--binary")
    cached_diff = _git_output("diff", "--cached", "--no-ext-diff", "--binary")
    untracked = _git_output("ls-files", "--others", "--exclude-standard")
    return {
        **git_state(),
        "tracked_diff_sha256": _text_sha256(tracked_diff),
        "cached_diff_sha256": _text_sha256(cached_diff),
        "untracked_files": untracked.splitlines() if untracked else [],
    }


def model_parameter_fingerprint(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def build_sanity_check(rows, config, sample_count=5):
    if len(rows) < sample_count:
        raise RuntimeError("Not enough slices for sanity check")
    rng = random.Random(int(config["seed"]))
    selected = sorted(rng.sample(list(rows), sample_count), key=lambda row: int(row["slice_index"]))
    hu_span = float(config["data"]["hu_max"]) - float(config["data"]["hu_min"])
    window_min, window_max = [float(value) for value in config["evaluation"]["hu_window"]]
    psnr_range = window_max - window_min
    checks = []
    for row in selected:
        rmse_hu_from_normalized = float(row["rmse_normalized"]) * hu_span
        mae_hu_from_normalized = float(row["mae_normalized"]) * hu_span
        psnr_from_rmse = (
            float("inf") if float(row["rmse_hu"]) == 0.0
            else 20.0 * math.log10(psnr_range / float(row["rmse_hu"]))
        )
        checks.append({
            **row,
            "rmse_hu_from_normalized": rmse_hu_from_normalized,
            "mae_hu_from_normalized": mae_hu_from_normalized,
            "psnr_from_rmse_hu": psnr_from_rmse,
            "rmse_formula_abs_error": abs(rmse_hu_from_normalized - float(row["rmse_hu"])),
            "mae_formula_abs_error": abs(mae_hu_from_normalized - float(row["mae_hu"])),
            "psnr_formula_abs_error": abs(psnr_from_rmse - float(row["psnr"])),
        })
    max_error = max(
        max(item["rmse_formula_abs_error"], item["mae_formula_abs_error"], item["psnr_formula_abs_error"])
        for item in checks
    )
    return {
        "selection": "Python random.Random(config seed) sample without replacement, then sort by slice_index",
        "seed": int(config["seed"]),
        "sample_count": sample_count,
        "dataset_hu_span": hu_span,
        "metric_hu_window": [window_min, window_max],
        "checks": checks,
        "maximum_formula_absolute_error": max_error,
        "pass": bool(max_error < 1e-9),
        "aggregation_note": (
            "PSNR is computed per slice from that slice RMSE and then averaged. "
            "Because log is nonlinear, mean PSNR is not obtained by substituting mean RMSE into the PSNR formula."
        ),
    }


def source_hashes(config_path, visualization_path=None):
    paths = [
        EXPERIMENT_DIR / "evaluation.py",
        EXPERIMENT_DIR / "evaluate.py",
        EXPERIMENT_DIR / "rddm_ct.py",
        Path(config_path),
    ]
    if visualization_path is not None:
        paths.append(Path(visualization_path))
    return {str(path.resolve()): sha256_file(path.resolve()) for path in paths}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, default=None)
    parser.add_argument("--model-name", default="RDDM-B0")
    parser.add_argument("--visualization-protocol", type=Path, default=None)
    args = parser.parse_args()

    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    visualization_path = args.visualization_protocol.resolve() if args.visualization_protocol else None
    if not checkpoint_path.is_file():
        raise FileNotFoundError(str(checkpoint_path))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError("Refusing to overwrite evaluation output: {}".format(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    seed_everything(int(config["seed"]))
    device = torch.device("cuda:0")

    launch_manifest = {
        "started_at_unix": time.time(),
        "command": " ".join([sys.executable] + sys.argv),
        "cwd": os.getcwd(),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "numpy": np.__version__,
        "skimage": skimage.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "tmux_session": os.environ.get("RDDM_CT_TMUX_SESSION"),
        "log_path": os.environ.get("RDDM_CT_LOG_PATH"),
        "checkpoint": str(checkpoint_path),
        "output_dir": str(output_dir),
        "protocol_version": PROTOCOL_VERSION,
    }
    write_json(output_dir / "evaluation_run_manifest.json", launch_manifest)

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    checkpoint_step = int(checkpoint["step"])
    if args.expected_step is not None and checkpoint_step != int(args.expected_step):
        raise RuntimeError("Checkpoint step mismatch: {} != {}".format(checkpoint_step, args.expected_step))
    if checkpoint.get("config_sha256") != config_sha256(config_path):
        raise RuntimeError("Checkpoint/config hash mismatch")
    _, diffusion, parameter_count = build_system(config, device)
    diffusion.model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    gc.collect()
    diffusion.eval()
    parameter_fingerprint_before = model_parameter_fingerprint(diffusion.model)

    patients_key = "validation_patients" if args.split == "validation" else "test_patients"
    patients = config["data"][patients_key]
    dataset = PairedCTDataset(config["data"]["root"], patients, config["data"])

    visualization_result = None
    visualization_protocol = None
    if visualization_path is not None:
        visualization_protocol = json.loads(visualization_path.read_text(encoding="utf-8"))
        validation_dataset = PairedCTDataset(
            config["data"]["root"], config["data"]["validation_patients"], config["data"]
        )
        write_json(output_dir / "visualization_protocol_frozen.json", visualization_protocol)
        visualization_result = evaluate_fixed_visualizations(
            diffusion, validation_dataset, config, device, visualization_protocol,
            output_dir / "visualization",
        )

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    result = evaluate_dataset(
        diffusion, dataset, config, device, output_dir,
        stride=args.stride, save_images=False, secondary_mask_provider=None,
    )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak_vram_mib = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
    parameter_fingerprint_after = model_parameter_fingerprint(diffusion.model)
    if parameter_fingerprint_before != parameter_fingerprint_after:
        raise RuntimeError("Model parameter fingerprint changed during evaluation")

    sanity = build_sanity_check(result["slices"], config, sample_count=5)
    write_json(output_dir / "metric_sanity_check.json", sanity)
    statistics = result["summary"]["main_evaluation"]["statistics"]
    finite_metrics = all(
        math.isfinite(float(statistics[name][field]))
        for name in METRIC_NAMES
        for field in ("mean", "std", "min", "max")
    )
    if not finite_metrics or not sanity["pass"]:
        raise RuntimeError("Evaluation metric sanity check failed")

    is_formal_test = (
        args.split == "test"
        and list(patients) == ["L506"]
        and len(dataset) == 526
        and int(args.stride) == 1
        and checkpoint_step == 100000
        and int(config["diffusion"]["sampling_timesteps"]) == 10
        and float(config["diffusion"]["ddim_eta"]) == 0.0
        and visualization_protocol is not None
    )
    status = "FROZEN" if is_formal_test else "REGRESSION_ONLY"
    completed_at = time.time()
    summary = {
        **result["summary"],
        "model": args.model_name,
        "status": status,
        "split": args.split,
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "parameter_count": parameter_count,
        "parameter_fingerprint_before": parameter_fingerprint_before,
        "parameter_fingerprint_after": parameter_fingerprint_after,
        "model_parameters_unchanged": True,
        "dataset": dataset.audit(),
        "elapsed_seconds": elapsed,
        "seconds_per_slice": elapsed / len(result["slices"]),
        "peak_vram_mib": peak_vram_mib,
        "visualization": visualization_result,
    }
    write_json(output_dir / "summary.json", summary)

    evaluation_manifest = {
        "model": args.model_name,
        "status": status,
        "evaluation_type": "real_gt",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "checkpoint_step": checkpoint_step,
        "git": working_tree_manifest(),
        "source_hashes": source_hashes(config_path, visualization_path),
        "evaluation_script_hash": sha256_file(EXPERIMENT_DIR / "evaluation.py"),
        "evaluation_runner_hash": sha256_file(EXPERIMENT_DIR / "evaluate.py"),
        "train_ids": list(config["data"]["train_patients"]),
        "validation_id": list(config["data"]["validation_patients"]),
        "test_id": list(config["data"]["test_patients"]),
        "dataset_root": str(Path(config["data"]["root"]).resolve()),
        "dose_level": str(config["data"]["dose_label"]),
        "test_slice_list": [
            {"patient_id": row["patient_id"], "slice_index": int(row["slice_index"])}
            for row in result["slices"]
        ],
        "seed": int(config["seed"]),
        "per_slice_generator_seed": "config seed + integer slice_index",
        "sampling": {
            "diffusion_timesteps": int(config["diffusion"]["timesteps"]),
            "sampling_timesteps": int(config["diffusion"]["sampling_timesteps"]),
            "ddim_eta": float(config["diffusion"]["ddim_eta"]),
            "deterministic": float(config["diffusion"]["ddim_eta"]) == 0.0,
            "stride": int(args.stride),
        },
        "normalization": {
            "stored_to_hu": "HU = stored_value - 1024",
            "dataset_hu_clip": [float(config["data"]["hu_min"]), float(config["data"]["hu_max"])],
            "dataset_zero_one": "(clip(HU,-1024,3072)+1024)/4096",
            "diffusion_space": "2*x_zero_one-1",
            "hu_inverse": "HU = x_zero_one*4096-1024",
        },
        "clipping_policy": {
            "sampler_predicted_residual": "[-1,1] at each reverse step",
            "sampler_predicted_x_start": "[-1,1] at each reverse step",
            "sampler_final_zero_one": "[0,1]",
            "metric_prediction_and_target_hu": [
                float(config["evaluation"]["hu_window"][0]),
                float(config["evaluation"]["hu_window"][1]),
            ],
        },
        "metric_definitions": METRIC_DEFINITIONS,
        "ssim_settings": SSIM_SETTINGS,
        "aggregation": {
            "primary": "per-slice metrics followed by arithmetic slice mean",
            "std": "population std, ddof=0",
            "reported_statistics": ["mean", "std", "min", "max", "count"],
        },
        "main_evaluation": "whole_image",
        "secondary_evaluation": result["summary"]["secondary_evaluation"],
        "visualization_protocol": visualization_protocol,
        "metric_sanity_check": str(output_dir / "metric_sanity_check.json"),
        "per_slice_metrics": str(output_dir / "per_slice_metrics.csv"),
        "summary": str(output_dir / "summary.json"),
        "parameter_count": parameter_count,
        "model_parameters_unchanged": True,
        "started_at_unix": launch_manifest["started_at_unix"],
        "completed_at_unix": completed_at,
    }
    write_json(output_dir / "evaluation_manifest.json", evaluation_manifest)
    if is_formal_test:
        write_json(output_dir / "baseline_eval_manifest.json", evaluation_manifest)
    print(json.dumps({
        "status": status,
        "split": args.split,
        "checkpoint_step": checkpoint_step,
        "slice_count": len(result["slices"]),
        "statistics": statistics,
        "elapsed_seconds": elapsed,
        "peak_vram_mib": peak_vram_mib,
        "output_dir": str(output_dir),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
