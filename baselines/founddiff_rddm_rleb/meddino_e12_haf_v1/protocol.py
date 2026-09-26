"""Frozen D1 Fast64 protocol with E12-only architecture changes."""
import copy
import json
from mayo2016_smallset_v1.runtime import ROOT, sha256
from meddino_salf_v1.protocol import config as e1_config, verify as verify_e1, D1_ROOT

HERE = ROOT / "experiments/meddino_e12_haf_v1"
ARM = "E12_HAF"
STEPS = (0, 500, 1000, 2000, 4000, 6000)
PARENT_COMMIT = "a75418c4d3465f425f6cced85cb4676331cf70ec"
D1_FULL_ROOT = ROOT / "experiments/meddino_full_l333_validation_v1"


def verify():
    verify_e1()
    e1 = json.loads((ROOT / "experiments/meddino_salf_v1/E1_INCOMPLETE_AUDIT.json").read_text())
    assert e1["status"] == "INCOMPLETE_CHECKPOINT_ARCHIVAL"
    assert e1["checkpoint_saved"] is False and e1["full_l333_run"] is False
    assert sha256(ROOT / "src/rddm_rleb.py") == "e8ed93223fbdf4a0e2521f73c7a4f213561f6a191a1a7b2a5f95934b85ebf574"
    assert sha256(ROOT / "splits/L333_fast64.txt") == "0d3170fd2462c7b19d1cc0f187b12e03bac2c9b15babca7c1d881568adde6895"
    return True


def config():
    verify()
    cfg = copy.deepcopy(e1_config())
    cfg["experiment_name"] = "RDDM_MedDINO_E12_HAF_V1"
    cfg["description"] = "SALF + zero-initialized hierarchy + local256/cross128/cross64 Control hint fusion"
    cfg["arm"] = ARM
    cfg["parent_commit"] = PARENT_COMMIT
    cfg["independent_from_E1_checkpoint"] = True
    cfg["hierarchy"] = {
        "H64": "A64", "H128": "A128 + ZeroConv(upsample H64)",
        "H256": "A256 + ZeroConv(upsample H128)",
        "upsample": "bilinear align_corners=False",
    }
    cfg["hybrid_injection"] = {
        "s256": "local 1x1/GELU/depthwise3x3/ZeroConv",
        "s128": "16x16 anatomy KV via deterministic exact avg-pool equivalent to adaptive pooling, 128 channels, 4 heads, ZeroConv",
        "s64": "16x16 anatomy KV via deterministic exact avg-pool equivalent to adaptive pooling, 256 channels, 8 heads, ZeroConv",
        "existing_controlnet_residual_taps_unchanged": True,
    }
    assert cfg["training"]["max_optimizer_steps"] == 6000
    assert cfg["training"]["batch_size"] == 2
    assert cfg["training"]["learning_rate"] == 2e-4
    assert cfg["training"]["amp"] is False
    assert cfg["ema"]["decay"] == .995
    assert cfg["diffusion"]["sampling_timesteps"] == 2
    assert cfg["diffusion"]["sum_scale"] == .01
    return cfg

