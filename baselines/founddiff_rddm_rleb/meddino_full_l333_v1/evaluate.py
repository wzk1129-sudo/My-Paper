"""Read-only matched EMA inference on all 610 L333 slices."""
import argparse
import csv
import gc
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch

from anatomy_controlnet_v1.evaluate import soft_metrics
from context_2p5d.data import ContextDataGuard, ContextPairs
from context_2p5d.model import restored_slice
from fast_dev.protocol import FAST_INDICES
from mayo2016_smallset_v1.frozen_protocol import slice_metrics
from mayo2016_smallset_v1.runtime import ROOT, sha256
from meddino_four_arm_v1.model import FourArmControlledUnet
from meddino_four_arm_v1.protocol import CONTROL_HASHES, frozen_config
from meddino_four_arm_v1.runtime import load_direct
from prosac_apgc_control_v1.model import ProsacControlledUnet

OUT = ROOT / "experiments/meddino_full_l333_validation_v1"
OLD = Path("/home/xujian/projects/CoreDiff/experiments/rddm_prosac_apgc_control_v1_20260925/baselines/founddiff_rddm_rleb")
NAMES = ("C3_FINAL", "D1_MULTI_512", "D3_MULTI_1024")
DIRS = {
    "C3_FINAL": OLD / "experiments/prosac_apgc_control_v1/runs/C3_ANATOMY",
    "D1_MULTI_512": ROOT / "experiments/meddino_four_arm_v1/runs/D1_MULTI_512",
    "D3_MULTI_1024": ROOT / "experiments/meddino_four_arm_v1/runs/D3_MULTI_1024",
}
STORED = {"C3_FINAL": "C3_ANATOMY", "D1_MULTI_512": "D1_MULTI_512",
          "D3_MULTI_1024": "D3_MULTI_1024"}
METRICS = ("psnr", "ssim", "rmse_hu", "mae_hu", "soft_psnr", "soft_ssim",
           "soft_rmse_hu", "soft_mae_hu")
SANITY_POS = (0, 31, 63)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    assert rows
    with path.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def checkpoint_audit():
    result = {}
    for name in NAMES:
        folder = DIRS[name]
        manifest = json.loads((folder / "CHECKPOINT_MANIFEST.json").read_text())
        matches = [row for row in manifest["matched"] if row["step"] == 6000]
        assert manifest["arm"] == STORED[name] and len(matches) == 1
        path = folder / "checkpoints/step_6000.pt"
        digest = file_hash(path)
        assert digest == matches[0]["sha256"], name
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        assert checkpoint["arm"] == STORED[name]
        assert checkpoint["optimizer_step"] == checkpoint["ema_updates"] == 6000
        assert checkpoint.get("ema_state_dict")
        result[name] = {"absolute_path": str(path), "bytes": path.stat().st_size,
                        "sha256": digest, "training_step": 6000,
                        "ema_state_dict_available": True,
                        "manifest_path": str(folder / "CHECKPOINT_MANIFEST.json")}
        del checkpoint
    return result


def dataset():
    cfg = frozen_config()
    data = ContextPairs(cfg, "validation", "FULL")
    assert len(data) == 610
    assert [(r[0], r[1]) for r in data.records] == [("L333", i) for i in range(610)]
    return cfg, data


def load_model(name, audit, device):
    started = time.perf_counter()
    diffusion = load_direct("cpu")
    main = diffusion.model.unet0
    if name == "C3_FINAL":
        diffusion.model.unet0 = ProsacControlledUnet(main, "C3_ANATOMY", 512)
    else:
        diffusion.model.unet0 = FourArmControlledUnet(main, name)
    path = Path(audit[name]["absolute_path"])
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert checkpoint["arm"] == STORED[name]
    diffusion.load_state_dict(checkpoint["ema_state_dict"], strict=True)
    del checkpoint, main
    assert diffusion.objective == "pred_res"
    assert diffusion.num_timesteps == 1000
    assert diffusion.sampling_timesteps == 2 and diffusion.sum_scale == 0.01
    assert diffusion.is_ddim_sampling
    diffusion = diffusion.to(device).eval().requires_grad_(False)
    assert not any(p.requires_grad for p in diffusion.parameters())
    return diffusion, time.perf_counter() - started


def evaluate(name, model, cfg, data, positions, device):
    assert cfg["evaluation"]["validation_seed"] == 20260829
    assert cfg["evaluation"]["hu_window"] == [-1000.0, 1000.0]
    assert cfg["data"]["hu_min"] == -1024.0 and cfg["data"]["hu_max"] == 3072.0
    guard = ContextDataGuard(cfg, "FULL", allow_training=False)
    data.bind_guard(guard)
    rows = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    devices = [device.index or 0] if device.type == "cuda" else []
    with guard, torch.inference_mode(), torch.random.fork_rng(devices=devices):
        for position in positions:
            prev, center, next_, target, meta = data.load(position)
            assert meta["patient_id"] == "L333" and meta["slice_index"] == position
            prev = torch.from_numpy(prev).unsqueeze(0).to(device)
            center = torch.from_numpy(center).unsqueeze(0).to(device)
            next_ = torch.from_numpy(next_).unsqueeze(0).to(device)
            target = torch.from_numpy(target).unsqueeze(0).to(device)
            for tensor in (prev, center, next_, target):
                assert tensor.shape == (1, 1, 512, 512)
                assert tensor.dtype == torch.float32 and bool(torch.isfinite(tensor).all())
                assert 0.0 <= float(tensor.min()) <= float(tensor.max()) <= 1.0
            seed = 20260829 + position
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed(seed)
                torch.cuda.synchronize(device)
            slice_started = time.perf_counter()
            prediction = restored_slice(model, center, prev, next_)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - slice_started
            assert prediction.shape == (1, 1, 512, 512)
            assert bool(torch.isfinite(prediction).all())
            wide = slice_metrics(prediction, target, cfg)
            soft = soft_metrics(prediction, target, cfg["data"])
            row = {"slice_index": position,
                   "slice_name": data.records[position][2].name,
                   **{key: wide[key] for key in ("psnr", "ssim", "rmse_hu", "mae_hu")},
                   **soft, "inference_seconds": elapsed}
            assert all(math.isfinite(float(row[key])) for key in METRICS + ("inference_seconds",))
            rows.append(row)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    runtime = {"total_inference_wall_seconds": time.perf_counter() - started,
               "mean_seconds_per_slice": float(np.mean([r["inference_seconds"] for r in rows])),
               "peak_vram_bytes": torch.cuda.max_memory_allocated(device)
                                  if device.type == "cuda" else 0}
    access = guard.report()
    assert access["L506_opens"] == access["external_memory_opens"] == 0
    assert access["neighbor_target_opens"] == access["blocked_attempts"] == 0
    assert access["training_target_opens"] == 0
    assert access["validation_center_indices_opened"] == list(positions)
    return rows, runtime, access


def archived(name):
    path = DIRS[name] / "per_slice_step_6000.csv"
    with path.open(newline="") as f:
        return {int(row["slice_index"]): row for row in csv.DictReader(f)}


def prepare():
    assert not any(OUT.iterdir()), "New output directory must be empty"
    for rel, expected in CONTROL_HASHES.items():
        assert sha256(ROOT / rel) == expected, rel
    for rel in ("prosac_apgc_control_v1/model.py",
                "prosac_apgc_control_v1/structure.py",
                "prosac_apgc_control_v1/runtime.py",
                "anatomy_controlnet_v1/evaluate.py"):
        assert sha256(ROOT / rel) == sha256(OLD / rel), rel
    cfg, data = dataset()
    listing = "".join(f"{i}\t{data.records[i][2].name}\n" for i in range(610))
    (OUT / "FULL_L333_SLICE_LIST.txt").write_text(listing)
    checkpoints = checkpoint_audit()
    write_json(OUT / "CHECKPOINT_AUDIT.json", checkpoints)
    config = {
        "experiment": "MedDINO Full-L333 Validation V1",
        "parent_commit": "f90c38cee8759d75177c3ccf69f02ecddca78528",
        "models": list(NAMES), "patient": "L333", "slices": 610, "stride": 1,
        "slice_list_sha256": sha256(OUT / "FULL_L333_SLICE_LIST.txt"),
        "weights": "EMA", "optimizer_steps": 0,
        "sampling_timesteps": 2, "sum_scale": 0.01, "batch_size": 1,
        "validation_seed_per_slice": "20260829 + slice_index",
        "preprocessing": {"stored_value_offset_hu": -1024, "clip_hu": [-1024, 3072],
                          "normalize": [0, 1]},
        "primary_window_hu": [-1000, 1000],
        "secondary_window_hu": [-160, 240],
        "secondary_label": "SECONDARY_SOFT_TISSUE_METRIC",
        "metric_implementation": {"wide": "mayo2016_smallset_v1.frozen_protocol.slice_metrics",
                                  "soft": "anatomy_controlnet_v1.evaluate.soft_metrics"},
        "frozen_evaluator_sha256": sha256(ROOT / "anatomy_controlnet_v1/evaluate.py"),
        "frozen_protocol_sha256": sha256(ROOT / "mayo2016_smallset_v1/frozen_protocol.py"),
        "fast64_split_sha256": sha256(ROOT / "splits/L333_fast64.txt"),
        "python": platform.python_version(), "pytorch": torch.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "dataset_manifest_sha256": data.manifest_hash,
        "source_hashes": {rel: sha256(ROOT / rel) for rel in
                          ("src/rddm_rleb.py", "meddino_four_arm_v1/model.py",
                           "meddino_four_arm_v1/structure.py", "prosac_apgc_control_v1/model.py",
                           "prosac_apgc_control_v1/structure.py")},
    }
    assert cfg["evaluation"]["validation_seed"] == 20260829
    write_json(OUT / "EXPERIMENT_CONFIG.json", config)
    return cfg, data, checkpoints


def sanity():
    if (OUT / "EXPERIMENT_CONFIG.json").exists():
        assert {p.name for p in OUT.iterdir()} == {
            "EXPERIMENT_CONFIG.json", "CHECKPOINT_AUDIT.json",
            "FULL_L333_SLICE_LIST.txt"}
        cfg, data = dataset()
        checkpoints = json.loads((OUT / "CHECKPOINT_AUDIT.json").read_text())
        assert sha256(OUT / "FULL_L333_SLICE_LIST.txt") == json.loads(
            (OUT / "EXPERIMENT_CONFIG.json").read_text())["slice_list_sha256"]
    else:
        cfg, data, checkpoints = prepare()
    device = torch.device("cuda:0")
    indices = [int(FAST_INDICES[p]) for p in SANITY_POS]
    assert all(index in [int(r["slice_index"]) for r in archived("D1_MULTI_512").values()]
               for index in indices)
    checks = {}
    for name in NAMES:
        model, load_seconds = load_model(name, checkpoints, device)
        rows, runtime, access = evaluate(name, model, cfg, data, indices, device)
        archive = archived(name)
        comparisons = []
        for row in rows:
            old = archive[row["slice_index"]]
            differences = {key: abs(row[key] - float(old[key])) for key in METRICS}
            assert max(differences.values()) <= 1e-4, (name, row["slice_index"], differences)
            comparisons.append({"slice_index": row["slice_index"], "absolute_difference": differences})
        checks[name] = {"status": "PASS", "indices": indices, "comparisons": comparisons,
                        "max_abs_metric_difference": max(max(c["absolute_difference"].values())
                                                         for c in comparisons),
                        "model_load_seconds": load_seconds,
                        "runtime": runtime, "data_access": access}
        del model
        gc.collect()
        torch.cuda.empty_cache()
    write_json(OUT / "EVALUATOR_REPRO_AUDIT.json",
               {"status": "PASS", "tolerance": 1e-4, "models": checks,
                "full_evaluation_permitted": True})
    print("EVALUATOR_REPRO_PASS", flush=True)


def full():
    repro = json.loads((OUT / "EVALUATOR_REPRO_AUDIT.json").read_text())
    assert repro["status"] == "PASS" and repro["full_evaluation_permitted"]
    cfg, data = dataset()
    checkpoints = json.loads((OUT / "CHECKPOINT_AUDIT.json").read_text())
    assert all(file_hash(checkpoints[name]["absolute_path"]) == checkpoints[name]["sha256"]
               for name in NAMES)
    assert sha256(OUT / "FULL_L333_SLICE_LIST.txt") == json.loads(
        (OUT / "EXPERIMENT_CONFIG.json").read_text())["slice_list_sha256"]
    assert not any((OUT / (name + "_FULL_L333.csv")).exists() for name in NAMES)
    device = torch.device("cuda:0")
    runtimes, accesses = {}, {}
    for name in NAMES:
        model, load_seconds = load_model(name, checkpoints, device)
        rows, runtime, access = evaluate(name, model, cfg, data, range(610), device)
        assert len(rows) == 610 and [r["slice_index"] for r in rows] == list(range(610))
        runtime["model_load_seconds"] = load_seconds
        write_csv(OUT / (name + "_FULL_L333.csv"), rows)
        runtimes[name] = runtime
        accesses[name] = access
        write_json(OUT / "RUNTIME_COMPARISON.json", runtimes)
        write_json(OUT / "DATA_ACCESS_AUDIT.json",
                   {"optimizer_steps": 0, "by_model": accesses,
                    "L506_opens": sum(v["L506_opens"] for v in accesses.values()),
                    "external_memory_opens": sum(v["external_memory_opens"] for v in accesses.values()),
                    "neighbor_target_opens": sum(v["neighbor_target_opens"] for v in accesses.values()),
                    "blocked_attempts": sum(v["blocked_attempts"] for v in accesses.values())})
        print("FULL_MODEL_COMPLETE", name, flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    assert set(runtimes) == set(accesses) == set(NAMES)
    print("FULL_INFERENCE_COMPLETE", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("sanity", "full"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.mode == "sanity":
        sanity()
    else:
        full()


if __name__ == "__main__":
    main()
