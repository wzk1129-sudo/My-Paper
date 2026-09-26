"""Frozen PDC-Gated-BCA Fast6k protocol plus NDCT-only side structure loss."""
import copy
import json
from mayo2016_smallset_v1.runtime import ROOT, sha256
from meddino_pdc_gated_bca_v1.protocol import config as pdc_config, verify as verify_pdc

HERE = ROOT / "experiments/meddino_pdc_structsup_v1"
ARM = "PDC_STRUCTSUP_BCA"
STEPS = (0, 500, 1000, 2000, 4000, 6000)
PARENT_COMMIT = "33d32b0d42b207ac5e8726ac427e2fc06093cffc"
FORMAL_SOURCE_SHA = "e8ed93223fbdf4a0e2521f73c7a4f213561f6a191a1a7b2a5f95934b85ebf574"
FAST64_SPLIT_SHA = "0d3170fd2462c7b19d1cc0f187b12e03bac2c9b15babca7c1d881568adde6895"
EVALUATOR_SHA = "94ddc8b2d9e9bae08457bb0f4e57c20f2d282eb80ef5b77c50304818f3f29366"


def verify():
    verify_pdc()
    assert sha256(ROOT / "src/rddm_rleb.py") == FORMAL_SOURCE_SHA
    assert sha256(ROOT / "splits/L333_fast64.txt") == FAST64_SPLIT_SHA
    assert sha256(ROOT / "meddino_full_l333_v1/evaluate.py") == EVALUATOR_SHA
    return True


def config():
    verify()
    cfg = copy.deepcopy(pdc_config())
    cfg["experiment_name"] = "PDC-StructSup-BCA-V1"
    cfg["description"] = "Unchanged PDC-Gated-BCA backbone with NDCT-derived 128/64 side structure supervision"
    cfg["arm"] = ARM
    cfg["parent_commit"] = PARENT_COMMIT
    cfg["initialization"] = "same canonical Direct 12k EMA as PDC-Gated-BCA; no PDC 6k warm start"
    calibration_path = HERE / "LAMBDA_CALIBRATION_FINAL.json"
    if calibration_path.exists():
        calibration = json.loads(calibration_path.read_text())
        assert calibration["status"] == "PASS" and calibration["smoke_steps"] == 20
        lam128 = float(calibration["lambda128_final"])
        lamimg = float(calibration["lambda_img_final"])
        lam64 = float(calibration["lambda64_final"])
    else:
        lam128 = lam64 = lamimg = 0.05
    assert lam128 == lam64 and lam128 in (0.02, 0.05, 0.1)
    cfg["structure_supervision"] = {
        "head128": "1x1 Conv(64,1) on gated PDC shape",
        "head64": "1x1 Conv(128,1) on gated PDC shape",
        "target": "NDCT HU window[-1000,1000] /1000 -> Gaussian sigma1 kernel5 -> Scharr -> magnitude/sqrt2 -> clamp[0,1] -> area downsample",
        "lambda128": lam128, "lambda64": lam64,
        "loss128": "L1(pred_struct128,target_struct128)",
        "loss64": "L1(pred_struct64,target_struct64)",
        "image_auxiliary_loss": True, "lambda_img": lamimg,
        "image_auxiliary_definition": "L1((2*center-1)-pred_res, 2*NDCT-1); pred_res captured from existing diffusion model forward",
        "target_used_for_validation": False,
    }
    assert lamimg in (0.02, 0.05, 0.1)
    cfg["training"]["matched_checkpoint_steps"] = list(STEPS)
    assert cfg["training"]["max_optimizer_steps"] == 6000
    assert cfg["training"]["batch_size"] == 2
    assert cfg["training"]["learning_rate"] == 2e-4
    assert cfg["training"]["amp"] is False
    assert cfg["ema"]["decay"] == .995
    assert cfg["diffusion"]["sampling_timesteps"] == 2 and cfg["diffusion"]["sum_scale"] == .01
    return cfg
