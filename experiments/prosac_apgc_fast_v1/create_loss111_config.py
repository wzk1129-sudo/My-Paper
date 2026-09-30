"""Derive loss111 arm configs; every other training/model/data/sampler field is copied."""
import json
from pathlib import Path

here = Path(__file__).resolve().parent
for short, name in (("M0", "M0_LOSS111"), ("M1", "M1_APGC_LOSS111")):
    baseline = json.loads((here / ("CONFIG_" + short + ".json")).read_text())
    config = json.loads((here / ("CONFIG_" + short + ".json")).read_text())
    config["arm"] = name
    config["diffusion"]["loss_type"] = "l2"
    config["training"]["output_dir"] = str(here / "runs" / name)
    config["experiment_notes"]["loss"] = (
        "MSE(pred_res,r)+MSE(pred_noise,eps)+MSE(x0_hat,x_start)")
    config["experiment_notes"]["loss_weights"] = {
        "lambda_res": 1.0, "lambda_noise": 1.0, "lambda_recon": 1.0}
    config["experiment_notes"]["loss_weight_status"] = (
        "RECONSTRUCTION_CHOICE_NOT_EXPLICIT_IN_PAPER")
    changed = []
    def diff(a, b, prefix=""):
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                diff(a.get(key), b.get(key), prefix + "." + key)
        elif a != b:
            changed.append(prefix[1:])
    diff(baseline, config)
    allowed = {
        "arm", "diffusion.loss_type", "training.output_dir",
        "experiment_notes.loss", "experiment_notes.loss_weights",
        "experiment_notes.loss_weight_status"}
    assert set(changed) == allowed, changed
    (here / ("CONFIG_" + name + ".json")).write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n")
print("LOSS111_CONFIG_DIFF_PASS")
