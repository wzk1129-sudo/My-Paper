"""Frozen Mayo Fast6k protocol with one new BCA injection mechanism."""
import copy
from mayo2016_smallset_v1.runtime import ROOT, sha256
from meddino_salf_v1.protocol import config as e1_config, verify as verify_e1

HERE = ROOT / "experiments/meddino_e12_bca_v1"
ARM = "E12_BCA"
STEPS = (0, 500, 1000, 2000, 4000, 6000)
PARENT_COMMIT = "b9e31c586f88b3796173190f0ed5f38f698f2f5c"
FORMAL_SOURCE_SHA = "e8ed93223fbdf4a0e2521f73c7a4f213561f6a191a1a7b2a5f95934b85ebf574"
FAST64_SPLIT_SHA = "0d3170fd2462c7b19d1cc0f187b12e03bac2c9b15babca7c1d881568adde6895"


def verify():
    verify_e1()
    assert sha256(ROOT / "src/rddm_rleb.py") == FORMAL_SOURCE_SHA
    assert sha256(ROOT / "splits/L333_fast64.txt") == FAST64_SPLIT_SHA
    return True


def config():
    verify()
    cfg = copy.deepcopy(e1_config())
    cfg["experiment_name"] = "RDDM_MedDINO_E12_BCA_V1"
    cfg["description"] = "Canonical Direct RDDM + E12 topology + encoder window cross-attention"
    cfg["arm"] = ARM
    cfg["parent_commit"] = PARENT_COMMIT
    cfg["structure_topology"] = "E12 frozen MedDINO 2/5/8/11 + SALF + zero-init hierarchical reassembly; no Edge branch in actual E12 source"
    cfg["initialization"] = "same Direct 12k EMA canonical state as D1/E12; no trained D1/E12 6k weights"
    cfg["backbone_trainable"] = True
    cfg["meddino_frozen"] = True
    cfg["controlnet"] = False
    cfg["backbone_cross_attention"] = {
        "tap": "post-downsample encoder stages at 128x128 and 64x64",
        "query": "RDDM backbone",
        "key_value": "E12 structure",
        "window_128": 8, "window_64": 16, "heads": 4, "dim_head": 32,
        "prenorm": True, "alpha_init": 0.1, "zero_conv": False,
    }
    assert cfg["training"]["max_optimizer_steps"] == 6000
    assert cfg["training"]["batch_size"] == 2
    assert cfg["training"]["learning_rate"] == 2e-4
    assert cfg["training"]["amp"] is False
    assert cfg["ema"]["decay"] == .995
    assert cfg["diffusion"]["sampling_timesteps"] == 2
    assert cfg["diffusion"]["sum_scale"] == .01
    return cfg
