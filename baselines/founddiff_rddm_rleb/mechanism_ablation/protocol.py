"""Frozen fast-dev compatibility checks for the two-arm mechanism ablation."""
import copy
import hashlib
import json
from pathlib import Path
import torch
from fast_dev.protocol import CONFIG, FAST_TRAIN, FAST_INDICES, protocol
from mayo2016_smallset_v1.runtime import ROOT, seed_everything, sha256, write_json
from src.rddm_rleb import UnetRes, build_baseline

HERE = ROOT / "experiments/rddm_vs_ldct_regression_v1"
EXPERIMENT_CONFIG = HERE / "EXPERIMENT_CONFIG.json"
STEPS = (2000, 4000, 6000, 8000, 10000, 12000)
ARMS = ("CONTROL_RDDM", "MATCHED_LDCT_REGRESSION")
EXPECTED_SOURCE = "e8ed93223fbdf4a0e2521f73c7a4f213561f6a191a1a7b2a5f95934b85ebf574"

def frozen_config():
    from fast_dev.protocol import check_config
    cfg = json.loads(CONFIG.read_text())
    check_config(cfg)
    assert protocol()["fast_train"] == list(FAST_TRAIN)
    assert len(FAST_INDICES) == 64
    assert sha256(ROOT / "src/rddm_rleb.py") == EXPECTED_SOURCE
    return cfg

def experiment_config():
    base = frozen_config()
    desired = copy.deepcopy(base)
    desired["experiment_name"] = "RDDM_vs_Matched_LDCT_only_Residual_Regression_V1"
    desired["description"] = "Two-arm training mechanism ablation on frozen Mayo fast protocol"
    desired["training"]["max_optimizer_steps"] = 12000
    desired["training"]["matched_checkpoint_steps"] = list(STEPS)
    desired["safety"]["maximum_steps"] = 12000
    desired["arms"] = list(ARMS)
    desired["training_order"] = list(ARMS)
    desired["regression_definition"] = {
        "input": "[LDCT,LDCT]", "time": "fixed_zero",
        "target": "LDCT_minus_NDCT_in_minus_one_one",
        "loss": "MSE_pred_residual", "inference": "direct_one_pass",
    }
    if EXPERIMENT_CONFIG.exists():
        actual = json.loads(EXPERIMENT_CONFIG.read_text())
        assert actual == desired, "Experiment config changed"
    else:
        write_json(EXPERIMENT_CONFIG, desired)
    return desired, base

def make_model(arm):
    if arm == "CONTROL_RDDM":
        return build_baseline()
    if arm == "MATCHED_LDCT_REGRESSION":
        return UnetRes(dim=64, dim_mults=(1, 2, 4, 8), channels=1,
                       num_unet=1, condition=True, input_condition=False,
                       objective="pred_res", test_res_or_noise="res")
    raise ValueError(arm)

def state_hash(model):
    h = hashlib.sha256()
    for key, tensor in model.state_dict().items():
        array = tensor.detach().cpu().contiguous().numpy()
        h.update(key.encode() + b"\0" + str(array.dtype).encode() + b"\0")
        h.update(str(tuple(array.shape)).encode() + b"\0")
        h.update(array.tobytes())
    return h.hexdigest()

def initialization_audit():
    cfg, _ = experiment_config()
    seed_everything(cfg["seed"])
    control = make_model(ARMS[0]).model
    seed_everything(cfg["seed"])
    regression = make_model(ARMS[1])
    a, b = control.state_dict(), regression.state_dict()
    assert a.keys() == b.keys()
    for key in a:
        assert a[key].shape == b[key].shape and a[key].dtype == b[key].dtype
        assert torch.equal(a[key], b[key]), key
    counts = [sum(p.numel() for p in model.parameters()) for model in (control, regression)]
    assert counts == [13130945, 13130945], counts
    hashes = [state_hash(control), state_hash(regression)]
    assert hashes[0] == hashes[1]
    result = {"bitwise_identical": True, "per_tensor_equal": True,
              "control_unetres_sha256": hashes[0], "regression_unetres_sha256": hashes[1],
              "parameter_count_each": counts[0], "seed": cfg["seed"],
              "fresh_random_initialization": True, "pretrained_checkpoint_used": False,
              "formal_source_sha256": EXPECTED_SOURCE}
    write_json(HERE / "INITIALIZATION_AUDIT.json", result)
    return result
