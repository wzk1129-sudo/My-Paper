"""Read-only L333 soft-tissue evaluation for frozen Mayo checkpoints."""
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import skimage
from skimage.metrics import structural_similarity

from src.rddm_rleb import build_baseline
from mayo2016_smallset_v1.data import CenterPairs
from mayo2016_smallset_v1.frozen_protocol import (
    normalize_stored_ct, normalized_to_hu, slice_metrics,
)
from mayo2016_smallset_v1.runtime import seed_everything
from context_2p5d.data import ContextPairs
from context_2p5d.model import DirectUnet, restored_slice

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]
METHODS = ("RAW_QUARTER", "FORMAL_RDDM_60K", "DIRECT_CONCAT_DEV_12K")
METRICS = ("psnr", "ssim", "rmse_hu", "mae_hu")
EXPECTED_TRAIN = ("L067", "L096", "L109", "L143", "L192", "L286", "L291", "L310")
SOFT_MIN, SOFT_MAX = -160.0, 240.0
LITERATURE = {
    "LDCT": (22.556, 0.7895, 30.062),
    "RED-CNN": (27.935, 0.8500, 16.174),
    "DuGAN": (28.088, 0.8416, 15.894),
    "Uformer": (28.225, 0.8504, 15.647),
    "CTformer": (28.037, 0.8442, 15.990),
    "CT-Mamba": (27.932, 0.8477, 16.178),
    "FMDNet": (28.340, 0.8568, 15.322),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    with open(path, "x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


class L333Only:
    """Audit actual medical NPY opens; reject other patients, writes, and enumeration."""
    def __init__(self, root):
        self.root = os.path.abspath(root)
        self.active = False
        self.sample = None
        self.context = False
        self.reads = []
        self.blocked = 0

    def begin_sample(self, patient, index):
        assert self.active and self.sample is None
        assert patient == "L333" and 0 <= index < 610
        self.sample = index

    def end_sample(self):
        self.sample = None

    def _hook(self, event, args):
        if not self.active or event not in ("open", "os.listdir", "os.scandir") or not args:
            return
        value = args[0]
        if not isinstance(value, (str, bytes, os.PathLike)):
            return
        path = os.path.abspath(os.fsdecode(value))
        if path != self.root and not path.startswith(self.root + os.sep):
            return
        if event != "open":
            self.blocked += 1
            raise PermissionError("Medical data directory enumeration forbidden")
        match = re.fullmatch(r"(L\d{3})_(25|target)_(\d{3})_img\.npy", os.path.basename(path))
        flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
        writing = bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
        valid = False
        if match and os.path.dirname(path) == self.root and not writing and self.sample is not None:
            patient, view, number = match[1], match[2], int(match[3])
            allowed = {self.sample}
            if self.context:
                allowed.update((max(0, self.sample - 1), min(609, self.sample + 1)))
            valid = patient == "L333" and (
                (view == "target" and number == self.sample) or
                (view == "25" and number in allowed)
            )
        if not valid:
            self.blocked += 1
            raise PermissionError("Unapproved medical file open")
        self.reads.append((match[1], match[2], int(match[3])))

    def __enter__(self):
        self.active = True
        sys.addaudithook(self._hook)
        return self

    def __exit__(self, *_):
        self.active = False
        assert self.sample is None

    def report(self):
        return {
            "patients_opened": sorted({p for p, _, _ in self.reads}),
            "center_target_indices_opened": sorted({i for p, v, i in self.reads if v == "target"}),
            "opens": len(self.reads),
            "L506_opens": sum(p == "L506" for p, _, _ in self.reads),
            "external_memory_opens": sum(p != "L333" for p, _, _ in self.reads),
            "blocked_attempts": self.blocked,
            "data_directory_enumeration": False,
        }


def soft_metrics(prediction, target, data_config):
    pred_hu = normalized_to_hu(prediction.detach().float().cpu(), data_config).clamp(SOFT_MIN, SOFT_MAX)
    gt_hu = normalized_to_hu(target.detach().float().cpu(), data_config).clamp(SOFT_MIN, SOFT_MAX)
    p01 = (pred_hu - SOFT_MIN) / (SOFT_MAX - SOFT_MIN)
    g01 = (gt_hu - SOFT_MIN) / (SOFT_MAX - SOFT_MIN)
    for v in (pred_hu, gt_hu):
        assert bool(torch.isfinite(v).all()) and float(v.min()) >= SOFT_MIN and float(v.max()) <= SOFT_MAX
    for v in (p01, g01):
        assert float(v.min()) >= -1e-7 and float(v.max()) <= 1 + 1e-7
    diff01 = p01 - g01
    diff_hu = pred_hu - gt_hu
    mse01 = float(diff01.square().mean())
    assert mse01 > 0 and math.isfinite(mse01)
    psnr = 10.0 * math.log10(1.0 / mse01)
    rmse = math.sqrt(float(diff_hu.square().mean()))
    mae = float(diff_hu.abs().mean())
    ssim = float(structural_similarity(
        g01.squeeze().numpy().astype(np.float64),
        p01.squeeze().numpy().astype(np.float64),
        win_size=7, gradient=False, data_range=1.0,
        channel_axis=None, gaussian_weights=False, full=False,
        use_sample_covariance=True, K1=0.01, K2=0.03,
    ))
    assert all(math.isfinite(x) for x in (psnr, ssim, rmse, mae))
    return {"psnr": psnr, "ssim": ssim, "rmse_hu": rmse, "mae_hu": mae}, p01, g01, pred_hu, gt_hu


def independent_sanity(metrics, p01, g01, pred_hu, gt_hu):
    x = np.asarray(p01.squeeze().numpy(), dtype=np.float64)
    y = np.asarray(g01.squeeze().numpy(), dtype=np.float64)
    a = np.asarray(pred_hu.squeeze().numpy(), dtype=np.float64)
    b = np.asarray(gt_hu.squeeze().numpy(), dtype=np.float64)
    psnr = -10.0 * np.log10(np.mean(np.square(x - y)))
    rmse = np.sqrt(np.mean(np.square(a - b)))
    assert abs(psnr - metrics["psnr"]) < 1e-4
    assert abs(rmse - metrics["rmse_hu"]) < 1e-4
    return {
        "formula_psnr_data_range_1": float(psnr),
        "formula_rmse_hu": float(rmse),
        "psnr_abs_error": float(abs(psnr - metrics["psnr"])),
        "rmse_abs_error": float(abs(rmse - metrics["rmse_hu"])),
    }


def metric_rows(method, index, prediction, target, cfg, sanity_indices, sanity):
    original = slice_metrics(prediction, target, cfg)
    soft, p01, g01, pred_hu, gt_hu = soft_metrics(prediction, target, cfg["data"])
    status = "INPUT_BASELINE" if method == "RAW_QUARTER" else (
        "PAPER_COMPARABLE_8PATIENT" if method == "FORMAL_RDDM_60K" else "DEV_ONLY"
    )
    result = []
    for protocol, values in (("HU_MINUS1000_1000", original), ("SOFT_MINUS160_240", soft)):
        result.append({
            "method": method, "status": status, "train_patients": (
                "" if method == "RAW_QUARTER" else
                ",".join(EXPECTED_TRAIN if method == "FORMAL_RDDM_60K" else ("L067", "L143"))
            ),
            "patient": "L333", "slice_index": index,
            "common_interior": 1 <= index <= 608,
            "protocol": protocol, **{k:float(values[k]) for k in METRICS},
        })
    if index in sanity_indices:
        sanity.append({"method": method, "slice_index": index,
                       **independent_sanity(soft, p01, g01, pred_hu, gt_hu)})
    return result


def load_formal(path, cfg, device):
    ck = torch.load(path, map_location="cpu")
    assert ck["optimizer_step"] == 60000
    assert tuple(ck["config"]["data"]["train_patients"]) == EXPECTED_TRAIN
    assert ck["config"]["data"]["validation_patients"] == ["L333"]
    assert ck["config"]["data"] == cfg["data"]
    assert ck["config"]["diffusion"] == cfg["diffusion"]
    model = build_baseline()
    model.load_state_dict(ck["ema_state_dict"], strict=True)
    del ck
    model = model.to(device).eval().requires_grad_(False)
    assert model.sampling_timesteps == 2 and model.sum_scale == 0.01
    return model


def load_direct(path, device):
    ck = torch.load(path, map_location="cpu")
    assert ck["arm"] == "2P5D_DIRECT_CONCAT" and ck["optimizer_step"] == 12000
    assert ck["config"]["data"]["train_patients"] == ["L067", "L143"]
    assert ck["config"]["data"]["validation_patients"] == ["L333"]
    model = build_baseline()
    model.model.unet0 = DirectUnet()
    model.load_state_dict(ck["ema_state_dict"], strict=True)
    del ck
    model = model.to(device).eval().requires_grad_(False)
    assert model.sampling_timesteps == 2 and model.sum_scale == 0.01
    return model


def aggregate(rows):
    result = {}
    for method in METHODS:
        result[method] = {}
        for protocol in ("HU_MINUS1000_1000", "SOFT_MINUS160_240"):
            result[method][protocol] = {}
            for subset, condition in (("ALL", lambda i: True),
                                      ("COMMON_INTERIOR", lambda i: 1 <= i <= 608)):
                values = [r for r in rows if r["method"] == method and r["protocol"] == protocol
                          and condition(r["slice_index"])]
                stats = {"slices": len(values)}
                for k in METRICS:
                    arr = np.asarray([v[k] for v in values], dtype=np.float64)
                    stats[k] = {"mean": float(arr.mean()), "std": float(arr.std(ddof=0)),
                                "median": float(np.median(arr))}
                result[method][protocol][subset] = stats
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=HERE / "CONFIG.json")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    c = json.load(open(args.config))
    cfg = json.load(open(PACKAGE / "mayo2016_smallset_v1/CONFIG.json"))
    fast_cfg = json.load(open(PACKAGE / "configs/rddm_fast_dev.json"))
    assert tuple(cfg["data"]["train_patients"]) == EXPECTED_TRAIN
    assert cfg["data"]["validation_patients"] == ["L333"]
    assert cfg["data"] == {**fast_cfg["data"], "train_patients": list(EXPECTED_TRAIN)}
    assert cfg["evaluation"]["hu_window"] == [-1000.0, 1000.0]
    assert cfg["diffusion"]["sampling_timesteps"] == 2
    assert sha256(PACKAGE / "src/rddm_rleb.py") == c["expected_formal_source_sha256"]
    assert sha256(PACKAGE / "mayo2016_smallset_v1/frozen_protocol.py") == c["expected_frozen_protocol_sha256"]
    assert not (HERE / "SUMMARY.json").exists()
    assert not (HERE / "per_slice_metrics.csv").exists()
    assert not (HERE / "MAYO_SOFT_TISSUE_METRIC_REPORT.md").exists()
    assert not (HERE / "CHECKPOINT_AUDIT.json").exists()
    if args.smoke:
        assert not (HERE / "SMOKE_AUDIT.json").exists()
    device = torch.device(args.device)
    torch.set_num_threads(2)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    seed_everything(cfg["seed"])
    formal_path = Path(c["formal_checkpoint"])
    direct_path = Path(c["direct_checkpoint"])
    formal_before = sha256(formal_path)
    direct_before = sha256(direct_path)
    assert formal_before == c["expected_formal_checkpoint_sha256"]
    assert direct_before == c["expected_direct_checkpoint_sha256"]
    full = CenterPairs(cfg, "validation")
    context = ContextPairs(fast_cfg, "validation", "FULL")
    assert len(full) == len(context) == 610
    assert tuple((r[0],r[1]) for r in full.records) == tuple((r[0],r[1]) for r in context.records)
    assert tuple((r[0],r[1]) for r in full.records) == tuple(("L333",i) for i in range(610))
    sanity_indices = sorted(int(i) for i in np.random.RandomState(20260829).choice(610,3,replace=False))
    indices = sanity_indices if args.smoke else list(range(610))
    guard = L333Only(cfg["data"]["root"])
    context.bind_guard(guard)
    rows, sanity = [], []
    started = time.perf_counter()
    with guard, torch.no_grad(), torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
        guard.context = False
        model = load_formal(formal_path, cfg, device)
        for n, index in enumerate(indices, 1):
            guard.begin_sample("L333", index)
            try:
                center, target, meta = full.load(index)
            finally:
                guard.end_sample()
            assert meta["patient_id"] == "L333" and meta["slice_index"] == index
            assert center.shape == target.shape == (1,512,512)
            assert center.dtype == target.dtype == np.float32
            assert np.isfinite(center).all() and np.isfinite(target).all()
            assert 0 <= center.min() <= center.max() <= 1
            assert 0 <= target.min() <= target.max() <= 1
            center = torch.from_numpy(center).unsqueeze(0).to(device)
            target = torch.from_numpy(target).unsqueeze(0).to(device)
            rows.extend(metric_rows("RAW_QUARTER", index, center, target, cfg, sanity_indices, sanity))
            seed = cfg["evaluation"]["validation_seed"] + index
            torch.manual_seed(seed)
            if device.type == "cuda": torch.cuda.manual_seed(seed)
            prediction = model.sample([center], last=True)[-1]
            assert prediction.shape == (1,1,512,512) and bool(torch.isfinite(prediction).all())
            rows.extend(metric_rows("FORMAL_RDDM_60K", index, prediction, target, cfg, sanity_indices, sanity))
            if n % 100 == 0: print("FORMAL_PROGRESS", n, len(indices), flush=True)
        del model
        if device.type == "cuda": torch.cuda.empty_cache()
        guard.context = True
        model = load_direct(direct_path, device)
        for n, index in enumerate(indices, 1):
            prev, center, next_, target, meta = context.load(index)
            assert meta["patient_id"] == "L333" and meta["slice_index"] == index
            assert meta["prev_index"] == max(0,index-1) and meta["next_index"] == min(609,index+1)
            tensors = (prev,center,next_,target)
            assert all(t.shape == (1,512,512) and t.dtype == np.float32 and np.isfinite(t).all() for t in tensors)
            prev,center,next_,target = (torch.from_numpy(t).unsqueeze(0).to(device) for t in tensors)
            seed = cfg["evaluation"]["validation_seed"] + index
            torch.manual_seed(seed)
            if device.type == "cuda": torch.cuda.manual_seed(seed)
            prediction = restored_slice(model, center, prev, next_)
            assert prediction.shape == (1,1,512,512) and bool(torch.isfinite(prediction).all())
            rows.extend(metric_rows("DIRECT_CONCAT_DEV_12K", index, prediction, target, cfg, sanity_indices, sanity))
            if n % 100 == 0: print("DEV_PROGRESS", n, len(indices), flush=True)
        del model
    access = guard.report()
    assert access["patients_opened"] == ["L333"]
    assert access["L506_opens"] == access["external_memory_opens"] == access["blocked_attempts"] == 0
    if not args.smoke:
        assert access["center_target_indices_opened"] == list(range(610))
    assert len(sanity) == 3 * len(METHODS)
    assert max(x["psnr_abs_error"] for x in sanity) < 1e-4
    assert max(x["rmse_abs_error"] for x in sanity) < 1e-4
    assert sha256(formal_path) == formal_before and sha256(direct_path) == direct_before
    assert sha256(PACKAGE / "src/rddm_rleb.py") == c["expected_formal_source_sha256"]
    if args.smoke:
        save_json(HERE / "SMOKE_AUDIT.json", {
            "status":"PASS", "indices":indices, "rows":len(rows), "sanity":sanity,
            "data_access":access, "optimizer_steps":0, "elapsed_seconds":time.perf_counter()-started,
        })
        print("SMOKE_PASS", indices, flush=True)
        return
    stats = aggregate(rows)
    assert all(stats[m][p]["ALL"]["slices"] == 610 and
               stats[m][p]["COMMON_INTERIOR"]["slices"] == 608
               for m in METHODS for p in ("HU_MINUS1000_1000","SOFT_MINUS160_240"))
    formal_old = stats["FORMAL_RDDM_60K"]["HU_MINUS1000_1000"]["ALL"]
    dev_old = stats["DIRECT_CONCAT_DEV_12K"]["HU_MINUS1000_1000"]["ALL"]
    assert abs(formal_old["psnr"]["mean"] - c["archived_formal_original_psnr"]) < 1e-5
    assert abs(dev_old["psnr"]["mean"] - c["archived_direct_original_psnr"]) < 1e-5
    formal_soft = stats["FORMAL_RDDM_60K"]["SOFT_MINUS160_240"]["ALL"]
    deltas = {}
    for name,(psnr,ssim,rmse) in LITERATURE.items():
        deltas[name] = {
            "delta_psnr": formal_soft["psnr"]["mean"] - psnr,
            "delta_ssim": formal_soft["ssim"]["mean"] - ssim,
            "delta_rmse_hu": formal_soft["rmse_hu"]["mean"] - rmse,
        }
    summary = {
        "status":"PASS",
        "protocols":{
            "original":{"hu_window":[-1000,1000],"psnr_data_range_hu":2000,
                        "implementation":"frozen slice_metrics"},
            "soft":{"hu_window":[-160,240],"psnr_data_range_normalized":1.0,
                    "ssim_data_range_normalized":1.0,"rmse_mae_domain":"clipped HU",
                    "ssim_parameters":"same as frozen slice_metrics"},
        },
        "train_patient_status":{
            "FORMAL_RDDM_60K":{"patients":list(EXPECTED_TRAIN),"status":"PAPER_COMPARABLE_8PATIENT"},
            "DIRECT_CONCAT_DEV_12K":{"patients":["L067","L143"],"status":"DEV_ONLY_NOT_FAIR_PAPER_COMPARISON"},
        },
        "patient":"L333","total_slices":610,"common_interior_indices":[1,608],
        "excluded_interior_boundary_indices":[0,609],
        "per_slice_arithmetic_aggregation":{"std_ddof":0,"metrics":list(METRICS)},
        "results":stats,"literature_reference_source":"user-provided task; not independently verified",
        "literature_reference":{k:{"psnr":v[0],"ssim":v[1],"rmse_hu":v[2]} for k,v in LITERATURE.items()},
        "formal_deltas_vs_literature":deltas,
        "sanity":{"status":"PASS","random_seed":20260829,"indices":sanity_indices,"checks":sanity},
        "data_access":access,"optimizer_steps":0,"training_steps":0,
        "checkpoint_hashes_unchanged":True,
        "elapsed_seconds":time.perf_counter()-started,
    }
    checkpoint_audit = {
        "formal_checkpoint":str(formal_path),"formal_sha256_before_after":formal_before,
        "formal_optimizer_step":60000,"formal_weights":"EMA","formal_train_patients":list(EXPECTED_TRAIN),
        "direct_checkpoint":str(direct_path),"direct_sha256_before_after":direct_before,
        "direct_optimizer_step":12000,"direct_weights":"EMA","direct_train_patients":["L067","L143"],
        "validation_patient_not_in_training":True,
        "formal_source_sha256":sha256(PACKAGE / "src/rddm_rleb.py"),
        "frozen_protocol_sha256":sha256(PACKAGE / "mayo2016_smallset_v1/frozen_protocol.py"),
        "original_formal_psnr_reproduced":True,"original_direct_psnr_reproduced":True,
        "python":platform.python_version(),"torch":torch.__version__,"cuda":torch.version.cuda,
        "skimage":skimage.__version__,"gpu":torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "data_access":access,"optimizer_steps":0,"L506_opens":0,"external_memory_opens":0,
    }
    fields = ["method","status","train_patients","patient","slice_index","common_interior",
              "protocol",*METRICS]
    with open(HERE / "per_slice_metrics.csv", "x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    save_json(HERE / "SUMMARY.json", summary)
    save_json(HERE / "CHECKPOINT_AUDIT.json", checkpoint_audit)
    lines = [
        "# Mayo-2016 Soft-Tissue Metric Comparison V1","",
        "Evaluation-only on L333; original model input normalization, checkpoints, and 2-step sampler are unchanged.",
        "Per-slice arithmetic mean over all 610 slices; common interior uses indices 1 through 608 (608 slices).",
        "Soft-tissue values are clipped to [-160,240] HU, normalized to [0,1] for PSNR/SSIM, with data_range=1.",
        "RMSE and MAE are computed in clipped HU. Original-window values use frozen slice_metrics.",
        "",
        "| Method | Training patients | PSNR [-1000,1000] | PSNR soft | SSIM soft | RMSE soft HU | MAE soft HU | Status |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for method in METHODS:
        old = stats[method]["HU_MINUS1000_1000"]["ALL"]
        soft = stats[method]["SOFT_MINUS160_240"]["ALL"]
        patients = "-" if method == "RAW_QUARTER" else (
            "8 fixed train patients" if method == "FORMAL_RDDM_60K" else "L067, L143")
        status = "INPUT_BASELINE" if method == "RAW_QUARTER" else (
            "PAPER_COMPARABLE_8PATIENT" if method == "FORMAL_RDDM_60K" else "DEV_ONLY")
        lines.append("| {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {} |".format(
            method, patients, old["psnr"]["mean"], soft["psnr"]["mean"],
            soft["ssim"]["mean"], soft["rmse_hu"]["mean"], soft["mae_hu"]["mean"], status))
    lines += ["","Literature reference values below were supplied in the task and were not independently source-verified.",
              "Numerical positioning is descriptive; differences in acquisition and evaluation can affect comparability.",
              "",
              "| Reference | PSNR | SSIM | RMSE HU | RDDM Delta PSNR | RDDM Delta SSIM | RDDM Delta RMSE HU |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name,(psnr,ssim,rmse) in LITERATURE.items():
        d = deltas[name]
        lines.append("| {} | {:.3f} | {:.4f} | {:.3f} | {:+.4f} | {:+.4f} | {:+.4f} |".format(
            name,psnr,ssim,rmse,d["delta_psnr"],d["delta_ssim"],d["delta_rmse_hu"]))
    lines += ["","The 2-patient Direct Concat result is DEV_ONLY and is not a fair paper comparison.",
              "Full and common-interior mean/std/median for every metric are in SUMMARY.json.",
              "Three fixed random slices passed independent PSNR/RMSE formula checks.",
              "L506 opens = 0; external-memory opens = 0; training and optimizer steps = 0.",
              "No predictions or medical images were saved.",""]
    with open(HERE / "MAYO_SOFT_TISSUE_METRIC_REPORT.md", "x") as f:
        f.write("\n".join(lines))
    print("COMPLETE", json.dumps({
        "formal_soft":{k:formal_soft[k]["mean"] for k in METRICS},
        "raw_soft":{k:stats["RAW_QUARTER"]["SOFT_MINUS160_240"]["ALL"][k]["mean"] for k in METRICS},
        "dev_soft":{k:stats["DIRECT_CONCAT_DEV_12K"]["SOFT_MINUS160_240"]["ALL"][k]["mean"] for k in METRICS},
        "L506_opens":access["L506_opens"],
    }),flush=True)


if __name__ == "__main__":
    main()
