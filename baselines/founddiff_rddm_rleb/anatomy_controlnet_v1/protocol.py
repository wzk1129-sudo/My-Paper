"""Pinned trained control, frozen Fast64 protocol and independent output paths."""
import copy
import json
from pathlib import Path
from fast_dev.protocol import protocol as check_fast, check_config, FAST_INDICES
from mayo2016_smallset_v1.runtime import ROOT, sha256
from context_2p5d.model import SOURCE_PIN

HERE=ROOT/"experiments/anatomy_controlnet_v1"
DIRECT_RUN=Path("/home/xujian/projects/CoreDiff/experiments/rddm_2p5d_screen_v1_20260924/baselines/founddiff_rddm_rleb/experiments/rddm_2p5d_screen_v1/runs/2P5D_DIRECT_CONCAT")
DIRECT_CK=DIRECT_RUN/"checkpoints/step_12000.pt"
DIRECT_SHA="151b7e58d2ca0d936be0842434e3d17809b325f144d0d48e47c8565287bcdcdf"
STEPS=(0,500,1000,2000,4000,6000)
ARMS=("C0_DIRECT","C1_CAPACITY","C2_EDGE")

def frozen_config():
    check_fast()
    cfg=json.loads((ROOT/"configs/rddm_fast_dev.json").read_text())
    check_config(cfg)
    assert len(FAST_INDICES)==64
    assert sha256(ROOT/"src/rddm_rleb.py")==SOURCE_PIN
    manifest=json.loads((DIRECT_RUN/"checkpoint_manifest.json").read_text())
    assert manifest["arm"]=="2P5D_DIRECT_CONCAT"
    assert any(r["step"]==12000 and r["sha256"]==DIRECT_SHA for r in manifest["matched"])
    assert sha256(DIRECT_CK)==DIRECT_SHA
    return cfg

def experiment_config(arm):
    assert arm in ARMS
    cfg=copy.deepcopy(frozen_config())
    cfg["experiment_name"]="RDDM_Anatomy_ControlNet_V1_"+arm
    cfg["description"]="Frozen trained 2.5D Direct RDDM plus capacity/edge control branch"
    cfg["training"]["max_optimizer_steps"]=6000 if arm!="C0_DIRECT" else 0
    cfg["training"]["matched_checkpoint_steps"]=list(STEPS[1:])
    cfg["safety"]["maximum_steps"]=6000 if arm!="C0_DIRECT" else 0
    cfg["evaluation"]["checkpoints"]=list(STEPS)
    cfg["direct_checkpoint_sha256"]=DIRECT_SHA
    cfg["arm"]=arm
    return cfg
