"""Fixed Fast64 protocol and verified archived Direct Concat EMA provenance."""
import copy
import csv
import json
from pathlib import Path
from fast_dev.protocol import CONFIG, FAST_TRAIN, FAST_INDICES, protocol as frozen_protocol, check_config
from mayo2016_smallset_v1.runtime import ROOT, sha256
from .model import ARM, DIRECT_ARM, SOURCE_PIN, CHECKPOINT_PIN

HERE = ROOT / "experiments/cross_slice_error_adapter_v1"
STEPS = (0, 500, 1000, 2000, 4000, 6000)
DIRECT_ROOT = Path("/home/xujian/projects/CoreDiff/experiments/rddm_2p5d_screen_v1_20260924/baselines/founddiff_rddm_rleb/experiments/rddm_2p5d_screen_v1")
DIRECT_RUN = DIRECT_ROOT / "runs" / DIRECT_ARM
DIRECT_CHECKPOINT = DIRECT_RUN / "checkpoints/step_12000.pt"
DIRECT_HISTORY = ROOT / "experiments/rddm_2p5d_screen_v1/2P5D_DIRECT_CONCAT/validation_history.csv"
DIRECT_FULL = ROOT / "experiments/rddm_2p5d_screen_v1/full_l333/2P5D_DIRECT_CONCAT/full_l333_paired.csv"
ADJACENCY = ROOT / "experiments/rddm_2p5d_screen_v1/ADJACENCY_AUDIT.json"
BASE_PSNR = 40.76762730415604


def config():
    frozen_protocol()
    base = json.loads(CONFIG.read_text())
    check_config(base)
    assert tuple(base["data"]["train_patients"]) == FAST_TRAIN == ("L067", "L143")
    assert len(FAST_INDICES) == 64
    assert sha256(ROOT / "src/rddm_rleb.py") == SOURCE_PIN
    assert json.loads(ADJACENCY.read_text())["result"] == "PASS"
    manifest = json.loads((DIRECT_RUN / "checkpoint_manifest.json").read_text())
    assert manifest["arm"] == DIRECT_ARM
    assert manifest["matched"][-1] == {"step": 12000, "sha256": CHECKPOINT_PIN}
    assert manifest["latest"] == {"step": 12000, "sha256": CHECKPOINT_PIN}
    assert sha256(DIRECT_CHECKPOINT) == CHECKPOINT_PIN
    with DIRECT_HISTORY.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[-1]["step"] == "12000" and rows[-1]["weights"] == "ema"
    assert rows[-1]["patient"] == "L333" and rows[-1]["slices"] == "64"
    assert abs(float(rows[-1]["psnr"]) - BASE_PSNR) < 1e-10
    cfg = {
        "experiment_name": ARM,
        "description": "Frozen Direct Concat 12k EMA plus additive cross-slice residual-error adapter",
        "parent_git_commit": "2d63c4339f00917b54ae341686d3d7c7cc32e703",
        "base_protocol": base,
        "frozen_direct_checkpoint": str(DIRECT_CHECKPOINT),
        "frozen_direct_checkpoint_sha256": CHECKPOINT_PIN,
        "direct_fast64_psnr": BASE_PSNR,
        "adapter_input": ["r_base.detach", "center_ldct", "D_mean", "D_slope", "D_abs"],
        "adapter_training": {
            "seed": 20260829, "optimizer": "Adam", "learning_rate": 1e-4,
            "betas": [0.9, 0.99], "weight_decay": 1e-9,
            "batch_size": 2, "amp": False, "gradient_clip": 1.0,
            "ema_decay": 0.995, "max_optimizer_steps": 6000,
            "validation_steps": list(STEPS), "checkpoint_steps": [500, 1000, 2000, 4000, 6000],
            "sampling_timesteps": 2, "sum_scale": 0.01,
            "diagnostic_source": "fixed-seed validation loss forward on L333-fast64",
        },
        "promotion": "2k/4k/6k delta PSNR positive and 6k >= +0.03 dB",
    }
    return cfg, base


def source_manifest():
    names = [
        "src/rddm_rleb.py", "context_2p5d/model.py", "context_2p5d/data.py",
        "context_2p5d/evaluate.py", "mayo2016_smallset_v1/frozen_protocol.py",
        "mayo2016_smallset_v1/runtime.py", "fast_dev/data.py", "fast_dev/protocol.py",
        "configs/rddm_fast_dev.json", "splits/L333_fast64.txt",
        "splits/fast_train_2patients.txt",
        "experiments/rddm_2p5d_screen_v1/ADJACENCY_AUDIT.json",
        "experiments/rddm_2p5d_screen_v1/2P5D_DIRECT_CONCAT/validation_history.csv",
        "cross_slice_adapter/model.py", "cross_slice_adapter/protocol.py",
        "cross_slice_adapter/evaluate.py", "cross_slice_adapter/experiment.py",
    ]
    return {name: sha256(ROOT / name) for name in names}
