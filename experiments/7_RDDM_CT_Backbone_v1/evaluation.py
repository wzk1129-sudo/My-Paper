from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import skimage
import torch
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity

from rddm_ct import PairedCTDataset, normalized_to_hu, write_json


PROTOCOL_VERSION = "RDDM_CT_MAIN_EVAL_V1_2026-09-04"
METRIC_NAMES = ("psnr", "ssim", "rmse_hu", "mae_hu")
CSV_FIELDS = (
    "patient_id", "slice_index", "dataset_position", "psnr", "ssim",
    "rmse_hu", "mae_hu", "rmse_normalized", "mae_normalized",
    "prediction_min", "prediction_max", "target_min", "target_max",
    "prediction_metric_hu_min", "prediction_metric_hu_max",
    "target_metric_hu_min", "target_metric_hu_max",
)
SSIM_SETTINGS = {
    "library": "skimage.metrics.structural_similarity",
    "library_version": skimage.__version__,
    "win_size": 7,
    "data_range": "evaluation.hu_window[1] - evaluation.hu_window[0]",
    "channel_axis": None,
    "gaussian_weights": False,
    "use_sample_covariance": True,
    "K1": 0.01,
    "K2": 0.03,
    "gradient": False,
    "full": False,
}
METRIC_DEFINITIONS = {
    "evaluation_type": "real_gt",
    "scope": "whole_image",
    "prediction_range_reporting": "model zero-one output before metric HU-window clipping",
    "metric_preprocessing": "convert prediction and dataset NDCT target to HU, then clip both to evaluation.hu_window",
    "psnr": "per-slice 10*log10(data_range^2/MSE_HU), data_range=HU window width; arithmetic mean across slices",
    "ssim": "per-slice skimage.metrics.structural_similarity on clipped HU arrays with explicit SSIM_SETTINGS; arithmetic mean across slices",
    "rmse_hu": "per-slice sqrt(mean((pred_hu_clipped-target_hu_clipped)^2)); arithmetic mean across slices",
    "mae_hu": "per-slice mean(abs(pred_hu_clipped-target_hu_clipped)); arithmetic mean across slices",
    "normalized_errors": "same clipped-HU error divided by dataset HU span; reported for sanity checking only",
    "statistics_std": "population standard deviation over per-slice values (numpy std, ddof=0)",
}


def _metric_tensors(prediction, target, config):
    data_config = config["data"]
    window_min, window_max = [float(value) for value in config["evaluation"]["hu_window"]]
    prediction_zero_one = prediction.detach().float().cpu()
    target_zero_one = target.detach().float().cpu()
    prediction_hu = normalized_to_hu(prediction_zero_one, data_config).clamp(window_min, window_max)
    target_hu = normalized_to_hu(target_zero_one, data_config).clamp(window_min, window_max)
    return prediction_zero_one, target_zero_one, prediction_hu, target_hu, window_min, window_max


def slice_metrics(prediction: torch.Tensor, target: torch.Tensor, config: Mapping[str, object]):
    pred_norm, target_norm, pred_hu, target_hu, window_min, window_max = _metric_tensors(
        prediction, target, config
    )
    difference_hu = pred_hu - target_hu
    mse_hu = float(difference_hu.square().mean())
    rmse_hu = math.sqrt(mse_hu)
    mae_hu = float(difference_hu.abs().mean())
    dataset_hu_span = float(config["data"]["hu_max"]) - float(config["data"]["hu_min"])
    rmse_normalized = rmse_hu / dataset_hu_span
    mae_normalized = mae_hu / dataset_hu_span
    data_range = window_max - window_min
    psnr = float("inf") if mse_hu == 0.0 else 10.0 * math.log10((data_range * data_range) / mse_hu)
    pred_array = pred_hu.squeeze().numpy().astype(np.float64)
    target_array = target_hu.squeeze().numpy().astype(np.float64)
    ssim = float(structural_similarity(
        target_array, pred_array, win_size=7, gradient=False, data_range=data_range,
        channel_axis=None, gaussian_weights=False, full=False,
        use_sample_covariance=True, K1=0.01, K2=0.03,
    ))
    return {
        "psnr": psnr,
        "ssim": ssim,
        "rmse_hu": rmse_hu,
        "mae_hu": mae_hu,
        "rmse_normalized": rmse_normalized,
        "mae_normalized": mae_normalized,
        "prediction_min": float(pred_norm.min()),
        "prediction_max": float(pred_norm.max()),
        "target_min": float(target_norm.min()),
        "target_max": float(target_norm.max()),
        "prediction_metric_hu_min": float(pred_hu.min()),
        "prediction_metric_hu_max": float(pred_hu.max()),
        "target_metric_hu_min": float(target_hu.min()),
        "target_metric_hu_max": float(target_hu.max()),
    }


def _mean_metrics(items):
    return {name: float(np.mean([float(item[name]) for item in items])) for name in METRIC_NAMES}


def descriptive_statistics(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    if not rows:
        raise RuntimeError("Cannot aggregate an empty metric list")
    result = {}
    for name in METRIC_NAMES:
        values = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        result[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "min": float(values.min()),
            "max": float(values.max()),
            "count": int(values.size),
        }
    return result


def aggregate(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    if not rows:
        raise RuntimeError("Cannot aggregate an empty metric list")
    by_patient = defaultdict(list)
    for row in rows:
        by_patient[str(row["patient_id"])].append(row)
    patient = {name: {**_mean_metrics(items), "slice_count": len(items)}
               for name, items in sorted(by_patient.items())}
    overall_slice = _mean_metrics(rows)
    overall_patient = {
        name: float(np.mean([values[name] for values in patient.values()]))
        for name in METRIC_NAMES
    }
    return {
        "slice_count": len(rows),
        "patient_count": len(patient),
        "per_patient": patient,
        "overall_slice_mean": overall_slice,
        "overall_patient_mean": overall_patient,
        "overall_slice_statistics": descriptive_statistics(rows),
    }


def _write_per_slice_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in CSV_FIELDS})
    temporary.replace(path)


def _window_uint8(value, data_config, window):
    low, high = [float(item) for item in window]
    hu = normalized_to_hu(value.detach().float().cpu(), data_config).squeeze().numpy()
    return (np.clip((hu - low) / (high - low), 0.0, 1.0) * 255.0).astype(np.uint8)


def _absolute_error_uint8(prediction, target, data_config, error_range_hu):
    low, high = [float(item) for item in error_range_hu]
    prediction_hu = normalized_to_hu(prediction.detach().float().cpu(), data_config).squeeze().numpy()
    target_hu = normalized_to_hu(target.detach().float().cpu(), data_config).squeeze().numpy()
    error = np.abs(prediction_hu - target_hu)
    return (np.clip((error - low) / (high - low), 0.0, 1.0) * 255.0).astype(np.uint8)


def _save_panel(path: Path, arrays, labels) -> None:
    height, width = arrays[0].shape
    canvas = Image.new("L", (width * len(arrays), height + 28), color=0)
    for index, array in enumerate(arrays):
        canvas.paste(Image.fromarray(array, mode="L"), (index * width, 28))
    draw = ImageDraw.Draw(canvas)
    for index, text in enumerate(labels):
        draw.text((index * width + 4, 7), text, fill=255)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(path))


def save_qualitative(path: Path, ldct, prediction, target, config, label: str) -> None:
    window = config["evaluation"]["qualitative_window"]
    arrays = [
        _window_uint8(ldct, config["data"], window),
        _window_uint8(prediction, config["data"], window),
        _window_uint8(target, config["data"], window),
        _absolute_error_uint8(prediction, target, config["data"], (0.0, 400.0)),
    ]
    _save_panel(path, arrays, ("LDCT", "RDDM-B0", "NDCT", "Abs error (0-400 HU)"))


def secondary_evaluation_interface(mask_provider=None) -> Dict[str, object]:
    if mask_provider is None:
        return {
            "status": "interface_only",
            "scope": "body_or_roi",
            "metrics": list(METRIC_NAMES),
            "reason": "No reliable body/ROI mask is frozen; no secondary metrics are computed.",
        }
    raise NotImplementedError("A body/ROI mask protocol must be frozen before secondary evaluation is enabled")


@torch.no_grad()
def evaluate_fixed_visualizations(diffusion, dataset, config, device, protocol, output_dir):
    records_by_key = {
        (record.patient_id, int(record.slice_index)): position
        for position, record in enumerate(dataset.records)
    }
    rows = []
    diffusion.eval()
    for item in protocol["slices"]:
        patient_id = str(item["patient_id"])
        slice_index = int(item["slice_id"])
        key = (patient_id, slice_index)
        if key not in records_by_key:
            raise RuntimeError("Visualization slice not found: {}:{}".format(patient_id, slice_index))
        sample = dataset[records_by_key[key]]
        context = sample["ldct_context"].unsqueeze(0).to(device)
        target = sample["ndct_target"].unsqueeze(0).to(device)
        generator = torch.Generator(device=device)
        generator.manual_seed(int(config["seed"]) + slice_index)
        prediction = diffusion.sample_ct(context, generator=generator)
        ldct = context[:, 1:2]
        window = item["ct_window_hu"]
        error_range = item["absolute_error_display_range_hu"]
        arrays = [
            _window_uint8(ldct, config["data"], window),
            _window_uint8(prediction, config["data"], window),
            _window_uint8(target, config["data"], window),
            _absolute_error_uint8(prediction, target, config["data"], error_range),
        ]
        labels = ("LDCT", "RDDM-B0", "NDCT", "Absolute error")
        stem = "{}_slice_{:03d}".format(patient_id, slice_index)
        _save_panel(Path(output_dir) / "full" / (stem + ".png"), arrays, labels)
        x0, y0, x1, y1 = [int(value) for value in item["roi_xyxy"]]
        if not (0 <= x0 < x1 <= arrays[0].shape[1] and 0 <= y0 < y1 <= arrays[0].shape[0]):
            raise ValueError("Invalid ROI for {}: {}".format(stem, item["roi_xyxy"]))
        _save_panel(
            Path(output_dir) / "roi" / (stem + "_roi.png"),
            [array[y0:y1, x0:x1] for array in arrays],
            labels,
        )
        rows.append({
            "patient_id": patient_id,
            "slice_id": slice_index,
            "roi_xyxy": [x0, y0, x1, y1],
            "ct_window_hu": [float(value) for value in window],
            "absolute_error_display_range_hu": [float(value) for value in error_range],
            "metrics": slice_metrics(prediction, target, config),
        })
        del context, target, prediction
    result = {
        "protocol_version": str(protocol["protocol_version"]),
        "selection_locked_before_test_evaluation": True,
        "slices": rows,
    }
    write_json(Path(output_dir) / "visualization_manifest.json", result)
    return result


@torch.no_grad()
def evaluate_dataset(
    diffusion,
    dataset: PairedCTDataset,
    config: Mapping[str, object],
    device: torch.device,
    output_dir: Optional[Path] = None,
    stride: int = 1,
    save_images: bool = False,
    secondary_mask_provider=None,
):
    if stride < 1:
        raise ValueError("stride must be positive")
    selected = list(range(0, len(dataset), stride))
    if not selected:
        raise RuntimeError("Evaluation selection is empty")
    qualitative_positions = set()
    if save_images:
        per_patient = defaultdict(list)
        for position in selected:
            per_patient[dataset.records[position].patient_id].append(position)
        count = int(config["evaluation"]["qualitative_slices_per_patient"])
        for positions in per_patient.values():
            choices = np.linspace(0, len(positions) - 1, num=min(count, len(positions)), dtype=np.int64)
            qualitative_positions.update(positions[int(choice)] for choice in choices)
    rows = []
    diffusion.eval()
    for ordinal, position in enumerate(selected, start=1):
        sample = dataset[position]
        context = sample["ldct_context"].unsqueeze(0).to(device)
        target = sample["ndct_target"].unsqueeze(0).to(device)
        generator = torch.Generator(device=device)
        generator.manual_seed(int(config["seed"]) + int(sample["slice_index"]))
        prediction = diffusion.sample_ct(context, generator=generator)
        metrics = slice_metrics(prediction, target, config)
        row = {
            "patient_id": sample["patient_id"],
            "slice_index": int(sample["slice_index"]),
            "dataset_position": int(position),
            **metrics,
        }
        rows.append(row)
        if save_images and position in qualitative_positions and output_dir is not None:
            name = "{}_slice_{:03d}.png".format(sample["patient_id"], int(sample["slice_index"]))
            save_qualitative(
                Path(output_dir) / "qualitative" / name,
                context[:, 1:2], prediction, target, config, name,
            )
        if ordinal == 1 or ordinal % 25 == 0 or ordinal == len(selected):
            print("evaluation_progress={}/{} patient={} slice={}".format(
                ordinal, len(selected), sample["patient_id"], int(sample["slice_index"])
            ), flush=True)
        del context, target, prediction
    legacy_metrics = aggregate(rows)
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "evaluation_type": "real_gt",
        "main_evaluation": {"scope": "whole_image", "statistics": descriptive_statistics(rows)},
        "secondary_evaluation": secondary_evaluation_interface(secondary_mask_provider),
        "slice_count": len(rows),
        "patient_count": legacy_metrics["patient_count"],
        "stride": int(stride),
        "ssim_settings": SSIM_SETTINGS,
        "metric_definitions": METRIC_DEFINITIONS,
        "legacy_metrics": legacy_metrics,
    }
    result = {"metrics": legacy_metrics, "slices": rows, "stride": int(stride), "summary": summary}
    if output_dir is not None:
        output_dir = Path(output_dir)
        _write_per_slice_csv(output_dir / "per_slice_metrics.csv", rows)
        write_json(output_dir / "summary.json", summary)
        write_json(output_dir / "metrics.json", result)
    return result
