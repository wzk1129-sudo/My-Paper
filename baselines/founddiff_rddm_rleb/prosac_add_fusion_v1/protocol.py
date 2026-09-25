"""Pinned C0/C1/C2 controls and unchanged Mayo Fast64 data/metric protocol."""
import copy
import json
import subprocess
from pathlib import Path

from anatomy_controlnet_v1.protocol import frozen_config, DIRECT_CK, DIRECT_SHA
from mayo2016_smallset_v1.runtime import ROOT, sha256
from .structure import OFFICIAL_ROOT, OFFICIAL_COMMIT, WEIGHT_PATH, WEIGHT_SHA256

HERE = ROOT / "experiments/prosac_add_fusion_v1"
OLD = Path("/home/xujian/projects/CoreDiff/experiments/rddm_anatomy_controlnet_v1_20260925/baselines/founddiff_rddm_rleb/experiments/anatomy_controlnet_v1")
OLD_PROSAC = Path("/home/xujian/projects/CoreDiff/experiments/rddm_prosac_apgc_control_v1_20260925/baselines/founddiff_rddm_rleb/experiments/prosac_apgc_control_v1")
ARMS = ("C4_ADD",)
STEPS = (0, 500, 1000, 2000, 4000, 6000)
SOURCE_SHA = "e8ed93223fbdf4a0e2521f73c7a4f213561f6a191a1a7b2a5f95934b85ebf574"
CONTROL_HASHES = {
    "configs/fast_dev_protocol.json": "c97165401015661b92f1fdcf857ad0fe202aba48d9ae5ae909a8530f3f39987a",
    "splits/L333_fast64.txt": "0d3170fd2462c7b19d1cc0f187b12e03bac2c9b15babca7c1d881568adde6895",
    "anatomy_controlnet_v1/evaluate.py": "eacbede48f38d47b810ad432802a528ccc64db77cdeb549cf7edd8cc477d564f",
    "src/rddm_rleb.py": SOURCE_SHA,
}


def verify():
    frozen_config()
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=OFFICIAL_ROOT,
                                   text=True).strip() == OFFICIAL_COMMIT
    for path, expected in CONTROL_HASHES.items():
        assert sha256(ROOT / path) == expected, path
    assert sha256(DIRECT_CK) == DIRECT_SHA
    for arm in ("C1_CAPACITY", "C2_EDGE"):
        p = OLD / "runs" / arm
        summary = json.loads((p / "final_summary.json").read_text())
        assert summary["status"] == "COMPLETE" and summary["optimizer_steps"] == 6000
        assert summary["data_access"]["L506_opens"] == 0
        assert summary["data_access"]["external_memory_opens"] == 0
        for step in STEPS:
            assert (p / ("per_slice_step_%d.csv" % step)).is_file()
    assert sha256(WEIGHT_PATH) == WEIGHT_SHA256
    old_prov = json.loads((OLD_PROSAC / "PROVENANCE.json").read_text())
    assert old_prov["official_source_commit"] == OFFICIAL_COMMIT
    assert old_prov["weight_sha256"] == WEIGHT_SHA256
    assert old_prov["direct_checkpoint_sha256"] == DIRECT_SHA
    import csv, sys, torch
    with (OLD_PROSAC / "VALIDATION_HISTORY.csv").open(newline="") as f:
        history = {(r["arm"], int(r["step"])): r for r in csv.DictReader(f)}
    assert abs(float(history["C3_ANATOMY", 2000]["psnr"]) - 40.78820919045856) < 1.e-8
    assert abs(float(history["C4_APGC", 2000]["psnr"]) - 40.782183568995336) < 1.e-8
    assert sys.version.split()[0] == "3.11.16"
    assert torch.__version__ == "2.8.0+cu128" and torch.version.cuda == "12.8"
    return True


def config(arm, input_size):
    assert arm in ARMS
    verify()
    cfg = copy.deepcopy(frozen_config())
    cfg["experiment_name"] = "RDDM_ProSAC_Eq7_Add_Fusion_V1_" + arm
    cfg["description"] = "ProSAC Eq.(7) Sobel plus Laplacian on frozen Direct RDDM ControlNet"
    cfg["arm"] = arm
    cfg["edge_aggregation"] = "Sobel plus Laplacian before Conv1x1_1toC"
    cfg["meddino_input_size"] = input_size
    cfg["meddino_official_commit"] = OFFICIAL_COMMIT
    cfg["meddino_checkpoint_sha256"] = WEIGHT_SHA256
    cfg["direct_checkpoint_sha256"] = DIRECT_SHA
    cfg["training"]["max_optimizer_steps"] = 6000
    cfg["training"]["matched_checkpoint_steps"] = list(STEPS[1:])
    cfg["safety"]["maximum_steps"] = 6000
    cfg["evaluation"]["checkpoints"] = list(STEPS)
    return cfg

