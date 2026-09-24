"""Frozen Fast protocol and matched 12k CONTROL_RDDM reference."""
import copy
import csv
import json
from mayo2016_smallset_v1.runtime import ROOT, sha256
from fast_dev.protocol import CONFIG, FAST_TRAIN, FAST_INDICES, protocol as frozen_protocol, check_config
from .model import ARMS, SOURCE_PIN

HERE=ROOT/"experiments/rddm_2p5d_screen_v1"
STEPS=(2000,4000,6000,8000,10000,12000)
BASE_HISTORY=ROOT/"experiments/rddm_vs_ldct_regression_v1/CONTROL_HISTORY.csv"
BASE_FULL=ROOT/"experiments/rddm_vs_ldct_regression_v1/FULL_L333_PER_SLICE.csv"
ADJACENCY=HERE/"ADJACENCY_AUDIT.json"


def config():
    frozen_protocol()
    base=json.loads(CONFIG.read_text())
    check_config(base)
    assert sha256(ROOT/"src/rddm_rleb.py")==SOURCE_PIN
    assert json.loads(ADJACENCY.read_text())["result"]=="PASS"
    cfg=copy.deepcopy(base)
    cfg["experiment_name"]="RDDM_2P5D_Context_Fast_Screen_V1"
    cfg["description"]="Two independent 2.5D LDCT-only context candidates on frozen RDDM Fast protocol"
    cfg["training"]["max_optimizer_steps"]=12000
    cfg["training"]["matched_checkpoint_steps"]=list(STEPS)
    cfg["safety"]["maximum_steps"]=12000
    assert {k:v for k,v in cfg.items() if k not in ("experiment_name","description","training","safety")}=={
        k:v for k,v in base.items() if k not in ("experiment_name","description","training","safety")}
    for key in base["training"]:
        if key not in ("max_optimizer_steps","matched_checkpoint_steps"):
            assert cfg["training"][key]==base["training"][key]
    for key in base["safety"]:
        if key!="maximum_steps":assert cfg["safety"][key]==base["safety"][key]
    assert tuple(base["data"]["train_patients"])==FAST_TRAIN
    assert len(FAST_INDICES)==64
    return cfg,base


def references():
    with BASE_HISTORY.open(newline="") as f:rows=list(csv.DictReader(f))
    assert [int(r["step"]) for r in rows]==list(STEPS)
    assert all(r["weights"]=="ema" and r["patient"]=="L333" and r["slices"]=="64" for r in rows)
    return {int(r["step"]):{
        **{k:float(r[k]) for k in ("psnr","ssim","rmse_hu","mae_hu","validation_loss")},
        "train_stream_sha256":r["train_stream_sha256"]
    } for r in rows}


def source_manifest():
    names=[
      "src/rddm_rleb.py","configs/rddm_fast_dev.json","configs/fast_dev_protocol.json",
      "splits/L333_fast64.txt","splits/fast_train_2patients.txt",
      "fast_dev/data.py","fast_dev/evaluate.py","fast_dev/protocol.py",
      "mayo2016_smallset_v1/data.py","mayo2016_smallset_v1/runtime.py",
      "mayo2016_smallset_v1/frozen_protocol.py",
      "experiments/rddm_vs_ldct_regression_v1/CONTROL_HISTORY.csv",
      "experiments/rddm_vs_ldct_regression_v1/FULL_L333_PER_SLICE.csv",
      "experiments/rddm_2p5d_screen_v1/ADJACENCY_AUDIT.json",
      "experiments/rddm_2p5d_screen_v1/DICOM_SPATIAL_AUDIT.json",
      "context_2p5d/model.py","context_2p5d/data.py","context_2p5d/protocol.py",
      "context_2p5d/runtime.py","context_2p5d/evaluate.py","context_2p5d/train.py",
      "context_2p5d/queue.py","context_2p5d/report.py",
    ]
    return {name:sha256(ROOT/name) for name in names}
