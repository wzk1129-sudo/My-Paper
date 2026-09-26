"""Pinned four-arm protocol; no 2k performance gate."""
import copy
import json
from pathlib import Path
from mayo2016_smallset_v1.runtime import ROOT, sha256
from prosac_add_fusion_v1.protocol import (
    verify as verify_existing, DIRECT_CK, DIRECT_SHA, CONTROL_HASHES,
    OLD_PROSAC,
)
from prosac_add_fusion_v1.structure import (
    OFFICIAL_ROOT, OFFICIAL_COMMIT, WEIGHT_PATH, WEIGHT_SHA256,
)
from anatomy_controlnet_v1.protocol import frozen_config

HERE = ROOT / "experiments/meddino_four_arm_v1"
ARMS = ("D1_MULTI_512", "D2_MULTI_ADD_512", "D3_MULTI_1024", "D4_MULTI_FT_512")
STEPS = (0, 500, 1000, 2000, 4000, 6000)
BASE_COMMIT = "38582515ea26ac84fe3f76a9eb7345e753230bd7"


def verify():
    verify_existing()
    assert sha256(ROOT / "src/rddm_rleb.py") == CONTROL_HASHES["src/rddm_rleb.py"]
    assert sha256(DIRECT_CK) == DIRECT_SHA
    return True


def config(arm):
    assert arm in ARMS
    verify()
    cfg = copy.deepcopy(frozen_config())
    cfg["experiment_name"] = "RDDM_MedDINO_Four_Arm_V1_" + arm
    cfg["description"] = "Multi-layer MedDINO [2,5,8,11] on frozen Direct RDDM ControlNet"
    cfg["arm"] = arm
    cfg["meddino_layers"] = [2, 5, 8, 11]
    cfg["meddino_extraction"] = "get_intermediate_layers(n=[2,5,8,11],reshape=True,norm=True)"
    cfg["meddino_input_size"] = 1024 if arm == "D3_MULTI_1024" else 512
    cfg["meddino_official_commit"] = OFFICIAL_COMMIT
    cfg["meddino_checkpoint_sha256"] = WEIGHT_SHA256
    cfg["direct_checkpoint_sha256"] = DIRECT_SHA
    cfg["fusion"] = "Sobel plus Laplacian add, APGC product" if arm == "D2_MULTI_ADD_512" else "anatomy only"
    cfg["training"]["max_optimizer_steps"] = 6000
    cfg["training"]["matched_checkpoint_steps"] = list(STEPS[1:])
    cfg["safety"]["maximum_steps"] = 6000
    cfg["evaluation"]["checkpoints"] = list(STEPS if arm != "D4_MULTI_FT_512" else (2000, 4000, 6000))
    cfg["fine_tune"] = {
        "enabled": arm == "D4_MULTI_FT_512",
        "fork_from": "D1_MULTI_512/step_2000.pt" if arm == "D4_MULTI_FT_512" else None,
        "trainable_blocks": [8, 9, 10, 11] if arm == "D4_MULTI_FT_512" else [],
        "meddino_learning_rate": 3e-5 if arm == "D4_MULTI_FT_512" else None,
    }
    return cfg


def archived_c3():
    path = OLD_PROSAC / "runs/C3_ANATOMY"
    with (path / "validation_history.csv").open() as f:
        import csv
        rows = {int(r["step"]): r for r in csv.DictReader(f)}
    assert all(step in rows for step in (2000, 4000, 6000))
    return path, rows
