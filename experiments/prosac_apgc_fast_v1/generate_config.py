"""Frozen fast matched M0/M1 protocol; generated from the RDDM-CT source snapshot."""
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
source = json.loads((ROOT / "experiments/7_RDDM_CT_Backbone_v1/config.json").read_text())
config = copy.deepcopy(source)
config["name"] = "ProSAC-style RDDM-CT M0/M1 APGC fast matched reproduction"
config["version"] = "fast-v1"
config["seed"] = 20260829
config["data"]["train_patients"] = ["L067", "L096", "L109", "L143", "L192", "L286", "L291", "L310"]
config["data"]["validation_patients"] = ["L333"]
config["data"]["test_patients"] = ["L506"]
config["data"]["hu_min"] = -160.0
config["data"]["hu_max"] = 240.0
config["data"]["normalization"] = "linear_zero_one"
config["diffusion"]["timesteps"] = 1000
config["diffusion"]["sampling_timesteps"] = 10
config["diffusion"]["sum_scale"] = 1.0
config["diffusion"]["objective"] = "pred_res_noise"
config["diffusion"]["loss_type"] = "l1"
config["diffusion"]["ddim_eta"] = 0.0
config["training"]["optimizer"] = "Adam"
config["training"]["learning_rate"] = 0.0002
config["training"]["batch_size"] = 4
config["training"]["gradient_accumulation"] = 1
config["training"]["total_steps"] = 8000
config["training"]["validation_every"] = 2000
config["training"]["checkpoint_every"] = 2000
config["training"]["validation_slice_stride"] = 1
config["training"]["num_workers"] = 0
config["training"]["validation_steps"] = [2000, 4000, 8000]
config["training"]["checkpoint_steps"] = [2000, 4000, 8000]
config["training"]["amp"] = True
config["training"]["gradient_clip"] = 1.0
config["training"]["ema_decay"] = 0.995
config["training"]["augmentation"] = {"horizontal_flip_p": 0.5, "rotation_degrees": [0, 90, 180, 270]}
config["evaluation"]["hu_window"] = [-160.0, 240.0]
config["evaluation"]["metrics"] = ["psnr", "ssim", "rmse_hu", "mae_hu", "edge_rmse_hu", "edge_mae_hu"]
config["evaluation"]["validation_patient"] = "L333"
config["evaluation"]["validation_subset"] = "L333_fast64"
config["evaluation"]["validation_seed"] = 20260829
config["experiment_notes"] = {
    "loss": "L1 residual plus L1 noise, equal weights",
    "paper_status": "RECONSTRUCTION_CHOICE_NOT_EXPLICIT_IN_PAPER",
    "model_selection": "matched M1 minus M0 on fixed L333-fast64",
    "meddino": "official checkpoint frozen; center LDCT only",
    "sampling_timesteps": "10 retained from frozen RDDM-CT; not specified by ProSAC paper",
    "evaluation_window": "same soft-tissue [-160,240] for both arms",
    "M1_injection": "single 64x64 shared-encoder output residual injection; reconstruction choice",
}
for arm in ("M0", "M1"):
    current = copy.deepcopy(config)
    current["arm"] = arm
    current["training"]["output_dir"] = str(HERE / "runs" / arm)
    (HERE / ("CONFIG_" + arm + ".json")).write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
