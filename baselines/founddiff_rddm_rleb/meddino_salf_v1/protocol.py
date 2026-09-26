"""Frozen D1 Fast64 protocol plus one SALF-only model variable."""
import copy
import json

from mayo2016_smallset_v1.runtime import ROOT, sha256
from meddino_four_arm_v1.protocol import config as d1_config, verify as verify_base

HERE = ROOT / "experiments/meddino_salf_v1"
ARM = "E1_SALF_512"
STEPS = (0, 500, 1000, 2000, 4000, 6000)
PARENT_COMMIT = "a4c2cf837743f869d4e6baffa00df14a91a61378"
D1_ROOT = ROOT / "experiments/meddino_four_arm_v1/runs/D1_MULTI_512"
D1_FULL_ROOT = ROOT / "experiments/meddino_full_l333_validation_v1"


def verify():
    verify_base()
    four = json.loads((ROOT / "experiments/meddino_four_arm_v1/FINAL_AUDIT.json").read_text())
    full = json.loads((D1_FULL_ROOT / "FINAL_AUDIT.json").read_text())
    assert four["status"] == full["status"] == "PASS"
    assert full["parent_commit"] == PARENT_COMMIT or full["parent_commit"] == "f90c38cee8759d75177c3ccf69f02ecddca78528"
    assert full["model_counts"]["D1_MULTI_512"] == 610
    assert full["L506_opens"] == full["external_memory_opens"] == 0
    assert sha256(ROOT / "src/rddm_rleb.py") == "e8ed93223fbdf4a0e2521f73c7a4f213561f6a191a1a7b2a5f95934b85ebf574"
    assert sha256(ROOT / "splits/L333_fast64.txt") == "0d3170fd2462c7b19d1cc0f187b12e03bac2c9b15babca7c1d881568adde6895"
    summary = json.loads((D1_ROOT / "final_summary.json").read_text())
    assert summary["status"] == "COMPLETE" and summary["optimizer_steps"] == 6000
    return True


def config():
    verify()
    cfg = copy.deepcopy(d1_config("D1_MULTI_512"))
    cfg["experiment_name"] = "RDDM_MedDINO_SALF_V1_E1_SALF_512"
    cfg["description"] = "D1 Multi-512 plus scale-aware zero-initialized layer scorers"
    cfg["arm"] = ARM
    cfg["layer_fusion"] = {
        "name": "SALF", "layers": [2, 5, 8, 11],
        "scales": ["s256", "s128", "s64"],
        "scorer_per_scale_layer": "Linear(768,1)",
        "scorer_weight_and_bias_initialization": 0,
        "normalized_weights": "softmax over four layers",
        "effective_multiplier": "4 * softmax weight",
        "extra_trainable_parameters": 9228,
        "meddino_trainable": False,
    }
    tr = cfg["training"]
    assert tr["max_optimizer_steps"] == 6000 and tr["learning_rate"] == 2e-4
    assert tr["betas"] == [0.9, 0.99] and tr["batch_size"] == 2
    assert tr["amp"] is False and cfg["ema"]["decay"] == 0.995
    assert cfg["diffusion"]["sampling_timesteps"] == 2
    assert cfg["diffusion"]["sum_scale"] == 0.01
    return cfg
