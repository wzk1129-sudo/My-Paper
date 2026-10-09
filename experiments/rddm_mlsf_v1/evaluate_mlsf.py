"""Matched EMA validation for MLSF; step 0 is an initialization gate."""
import argparse
import csv
import json
import math
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent
V1 = ROOT.parent / "rddm_official_2p5d_meddino_condition_v1"
for path in (str(ROOT), str(V1)):
    if path not in sys.path:
        sys.path.insert(0, path)
from mlsf_model import build_model, parameter_hash
from model import load_state_without_med
from context_data import ContextPairs, ContextDataGuard
from fast_dev.protocol import CONFIG, check_config
from mayo2016_smallset_v1.runtime import seed_everything
from evaluate import validate
from provenance import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--mode", choices=("FAST", "FULL"), default="FAST")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    cfg = json.loads(CONFIG.read_text())
    check_config(cfg)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed_everything(20260829)
    with ContextDataGuard(cfg, mode=args.mode, allow_training=False) as guard:
        data = ContextPairs(cfg, "validation", mode=args.mode)
        data.bind_guard(guard)
        model = build_model()
        checkpoint_step = 0
        if args.checkpoint is not None:
            checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            checkpoint_step = checkpoint["optimizer_step"]
            assert checkpoint_step in (1000, 2000, 4000, 6000, 8000)
            load_state_without_med(model, checkpoint["ema_state_dict"])
            del checkpoint
        med_before = parameter_hash(model.model.unet0.med.state_dict())
        model = model.to("cuda:0").eval()
        row, activity = validate(model, data, cfg, torch.device("cuda:0"),
                                 ROOT / ("validation_step_%d.json" % checkpoint_step))
        med_after = parameter_hash(model.model.unet0.med.state_dict())
        assert med_before == med_after
        access = guard.report()
        assert access["L506_opens"] == access["external_memory_opens"] == 0
        assert access["neighbor_target_opens"] == access["blocked_attempts"] == 0
        result = {"step": checkpoint_step, "mode": args.mode, "metrics": row,
                  "condition_activity": activity, "access": access,
                  "meddino_frozen_hash_unchanged": True}
        if checkpoint_step == 0:
            history = next(r for r in csv.DictReader((V1 / "VALIDATION_HISTORY.csv").open())
                           if int(r["step"]) == 0)
            expected = {"psnr": float(history["candidate_psnr"]),
                        "ssim": float(history["candidate_ssim"]),
                        "rmse_hu": float(history["candidate_rmse_hu"]),
                        "mae_hu": float(history["candidate_mae_hu"])}
            result["historical_v1_step0"] = expected
            result["delta_from_v1_step0"] = {name: row[name] - value
                                            for name, value in expected.items()}
            assert all(math.isfinite(v) and abs(v) < 1e-8
                       for v in result["delta_from_v1_step0"].values()), result["delta_from_v1_step0"]
            assert abs(row["psnr"] - 40.623374647) < 1e-8
            assert abs(row["ssim"] - 0.946055567) < 1e-8
        write_json(args.output, result)
        print("VALIDATION_PASS", checkpoint_step, row["psnr"], row["ssim"], flush=True)


if __name__ == "__main__":
    main()
