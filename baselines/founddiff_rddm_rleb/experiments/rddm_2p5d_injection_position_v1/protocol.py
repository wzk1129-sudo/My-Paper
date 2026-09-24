"""Frozen Mayo Fast protocol and matched Direct Concat 12k reference."""
import copy
import csv
import json
from mayo2016_smallset_v1.runtime import ROOT, sha256
from fast_dev.protocol import CONFIG, FAST_TRAIN, FAST_INDICES, protocol as frozen_protocol, check_config
from .model import ARMS, SOURCE_PIN

HERE = ROOT / "experiments/rddm_2p5d_injection_position_v1/attempt2"
STEPS = (2000, 4000, 6000, 8000, 10000, 12000)
BASE_HISTORY = ROOT / "experiments/rddm_2p5d_screen_v1/2P5D_DIRECT_CONCAT/validation_history.csv"
BASE_FULL = ROOT / "experiments/rddm_2p5d_screen_v1/full_l333/2P5D_DIRECT_CONCAT/full_l333_paired.csv"
ADJACENCY = ROOT / "experiments/rddm_2p5d_screen_v1/ADJACENCY_AUDIT.json"


def config():
    frozen_protocol()
    base = json.loads(CONFIG.read_text())
    check_config(base)
    assert sha256(ROOT / "src/rddm_rleb.py") == SOURCE_PIN
    assert json.loads(ADJACENCY.read_text())["result"] == "PASS"
    cfg = copy.deepcopy(base)
    cfg["experiment_name"] = "RDDM_2P5D_Context_Injection_Position_Screen_V1"
    cfg["description"] = "A/B/C independently initialized from canonical RDDM step-0; same LDCT neighbor encoder; only injection sites differ"
    cfg["training"]["max_optimizer_steps"] = 12000
    cfg["training"]["matched_checkpoint_steps"] = list(STEPS)
    cfg["safety"]["maximum_steps"] = 12000
    assert tuple(base["data"]["train_patients"]) == FAST_TRAIN
    assert len(FAST_INDICES) == 64
    for key in base["training"]:
        if key not in ("max_optimizer_steps", "matched_checkpoint_steps"):
            assert cfg["training"][key] == base["training"][key]
    for key in base["safety"]:
        if key != "maximum_steps":
            assert cfg["safety"][key] == base["safety"][key]
    for key in base:
        if key not in ("experiment_name", "description", "training", "safety"):
            assert cfg[key] == base[key]
    return cfg, base


def references():
    with BASE_HISTORY.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert [int(row["step"]) for row in rows] == list(STEPS)
    assert all(row["weights"] == "ema" and row["patient"] == "L333"
               and row["slices"] == "64" for row in rows)
    return {
        int(row["step"]): {
            **{metric: float(row[metric]) for metric in
               ("psnr", "ssim", "rmse_hu", "mae_hu", "validation_loss")},
            "train_stream_sha256": row["train_stream_sha256"],
        }
        for row in rows
    }


def source_manifest():
    names = [
        "src/rddm_rleb.py", "configs/rddm_fast_dev.json", "configs/fast_dev_protocol.json",
        "splits/L333_fast64.txt", "splits/fast_train_2patients.txt",
        "fast_dev/data.py", "fast_dev/evaluate.py", "fast_dev/protocol.py",
        "mayo2016_smallset_v1/data.py", "mayo2016_smallset_v1/runtime.py",
        "mayo2016_smallset_v1/frozen_protocol.py",
        "experiments/rddm_2p5d_screen_v1/ADJACENCY_AUDIT.json",
        "experiments/rddm_2p5d_screen_v1/DICOM_SPATIAL_AUDIT.json",
        "experiments/rddm_2p5d_screen_v1/2P5D_DIRECT_CONCAT/INITIALIZATION_AUDIT.json",
        "experiments/rddm_2p5d_screen_v1/2P5D_DIRECT_CONCAT/validation_history.csv",
        "experiments/rddm_2p5d_screen_v1/full_l333/2P5D_DIRECT_CONCAT/full_l333_paired.csv",
    ]
    names += [
        "experiments/rddm_2p5d_injection_position_v1/" + item
        for item in ("model.py", "data.py", "protocol.py", "runtime.py",
                     "evaluate.py", "train.py", "queue.py", "report.py")
    ]
    return {name: sha256(ROOT / name) for name in names}
