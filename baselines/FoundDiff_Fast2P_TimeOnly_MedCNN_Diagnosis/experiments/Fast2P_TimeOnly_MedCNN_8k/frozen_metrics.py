"""Frozen RDDM-CT normalization / metric functions; copied verbatim, no model imports.
Protocol: RDDM_CT_MAIN_EVAL_V1_2026-09-04. See protocol_provenance.json.
"""
import math
from typing import Mapping
import numpy as np
import torch
from skimage.metrics import structural_similarity

def normalize_stored_ct(array: np.ndarray, data_config: Mapping[str, object]) -> np.ndarray:
    hu = array.astype(np.float32) + float(data_config["stored_value_offset_hu"])
    hu_min = float(data_config["hu_min"])
    hu_max = float(data_config["hu_max"])
    hu = np.clip(hu, hu_min, hu_max)
    return ((hu - hu_min) / (hu_max - hu_min)).astype(np.float32)

def normalized_to_hu(value, data_config: Mapping[str, object]):
    hu_min = float(data_config["hu_min"])
    hu_max = float(data_config["hu_max"])
    return value * (hu_max - hu_min) + hu_min

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
