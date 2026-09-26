"""Independent Fast6k train/EMA validation with resumable checkpoints."""
import argparse
import csv
import json
import math
from pathlib import Path

import torch

from anatomy_controlnet_v1.evaluate import validate
from context_2p5d.data import ContextDataGuard
from mayo2016_smallset_v1.runtime import write_json
from meddino_salf_v1.train import write_csv, d1_history, METRIC_KEYS
from .protocol import ARM, HERE, STEPS, config
from .runtime import StructSupTrainState
from meddino_pdc_gated_bca_v1.model import numeric_activity


def history_file():
    return HERE / "VALIDATION_HISTORY.csv"


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    cfg = config()
    smoke = json.loads((HERE / "SMOKE_RESULT_FINAL.json").read_text())
    assert smoke["status"] == "PASS" and smoke["steps"] == 20
    assert cfg["training"]["max_optimizer_steps"] == 6000
    checkpoint_dir = HERE / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    guard = ContextDataGuard(cfg, "FAST", allow_training=True)
    d1 = d1_history()
    pdc_path = HERE.parent / "meddino_pdc_gated_bca_v1/ARCHIVE_VALIDATION_HISTORY.csv"
    bca_path = HERE.parent / "meddino_e12_bca_v1/VALIDATION_HISTORY.csv"
    pdc = {int(r["step"]): r for r in read_csv(pdc_path)}
    bca = {int(r["step"]): r for r in read_csv(bca_path)}
    assert set(pdc) == set(bca) == set(STEPS)
    with guard:
        state = StructSupTrainState("cuda:0", guard)
        assert state.init_audit == smoke["initialization"]
        if args.resume:
            assert args.resume.is_file()
            state.resume(args.resume)
            assert state.step in STEPS[1:] and state.step < 6000
            history = read_csv(history_file())
            assert history and int(history[-1]["step"]) == state.step
            slices = read_csv(HERE / "FAST64_PER_SLICE.csv")
            log_mode = "a"
        else:
            assert not history_file().exists()
            assert not (HERE / "train.jsonl").exists()
            assert not any(checkpoint_dir.iterdir())
            history, slices, log_mode = [], [], "x"
            write_json(HERE / "EXPERIMENT_CONFIG.json", cfg)
            write_json(HERE / "INITIALIZATION_AUDIT.json", state.init_audit)

        def evaluate(step):
            metrics, samples, _ = validate(
                state.ema.model, state.validation, cfg, state.device, activity=False)
            assert metrics["weights"] == "ema" and metrics["slices"] == 64
            assert all(math.isfinite(float(metrics[k])) for k in METRIC_KEYS)
            reference = d1[step]
            matched_pdc = pdc[step]
            matched_bca = bca[step]
            assert state.stream_digest.hex() == reference["train_stream_sha256"]
            assert state.stream_digest.hex() == matched_pdc["train_stream_sha256"]
            assert state.stream_digest.hex() == matched_bca["train_stream_sha256"]
            row = {
                "arm": ARM, "step": step, **metrics,
                "pdc_psnr": float(matched_pdc["psnr"]),
                "delta_psnr_vs_pdc": float(metrics["psnr"]) - float(matched_pdc["psnr"]),
                "bca_psnr": float(matched_bca["psnr"]),
                "delta_psnr_vs_bca": float(metrics["psnr"]) - float(matched_bca["psnr"]),
                "d1_psnr": float(reference["psnr"]),
                "delta_psnr_vs_d1": float(metrics["psnr"]) - float(reference["psnr"]),
                "train_loss": None if state.last is None else state.last["train_loss"],
                "base_loss": None if state.last is None else state.last["base_loss"],
                "structure_loss_128": None if state.last is None else state.last["structure_loss_128"],
                "structure_loss_64": None if state.last is None else state.last["structure_loss_64"],
                "weighted_structure_128": None if state.last is None else state.last["weighted_structure_128"],
                "weighted_structure_64": None if state.last is None else state.last["weighted_structure_64"],
                "loss_running": state.loss_running,
                "image_aux_loss": None if state.last is None else state.last["image_aux_loss"],
                "weighted_image_aux": None if state.last is None else state.last["weighted_image_aux"],
                "gradient_norm_pre_clip": None if state.last is None else
                    state.last["gradient_norm_pre_clip"],
                "learning_rate": state.optimizer.param_groups[0]["lr"],
                "peak_vram_bytes": state.peak_vram,
                "seconds_per_step": state.train_seconds / step if step else None,
                "train_stream_sha256": state.stream_digest.hex(),
            }
            history.append(row)
            write_csv(history_file(), history)
            slices.extend({"step": step, **sample} for sample in samples)
            write_csv(HERE / "FAST64_PER_SLICE.csv", slices)
            write_json(HERE / "DATA_ACCESS_AUDIT.json", guard.report())
            ck = checkpoint_dir / f"step_{step}.pt"
            assert not ck.exists()
            state.checkpoint(ck)
            print("VALIDATION", step, metrics["psnr"], row["delta_psnr_vs_pdc"], flush=True)

        if not args.resume:
            evaluate(0)
        with (HERE / "train.jsonl").open(log_mode, buffering=1) as log:
            while state.step < 6000:
                row = state.update()
                log.write(json.dumps(row, allow_nan=False) + "\n")
                if state.step % 100 == 0:
                    write_json(HERE / "progress.json", {
                        "status": "RUNNING", "step": state.step, "stop": 6000,
                        "last": row})
                    print("TRAIN", state.step, row["train_loss"], flush=True)
                if state.step in STEPS[1:]:
                    evaluate(state.step)
        assert [int(row["step"]) for row in history] == list(STEPS)
        access = guard.report()
        assert access["L506_opens"] == access["external_memory_opens"] == 0
        assert access["neighbor_target_opens"] == access["blocked_attempts"] == 0
        assert state.nonfinite_count == 0
        assert all(v > 0 for v in state.grad_usage.values())
        u = state.ema.model.model.unet0
        summary = {
            "status": "COMPLETE", "arm": ARM, "optimizer_step": state.step,
            "ema_updates": state.ema.updates, "fast64_history": history,
            "grad_usage": state.grad_usage,
            "last_aux": state.last_aux,
            "loss_running": state.loss_running,
            "lambda128": state.lambda128, "lambda64": state.lambda64,
            "image_auxiliary_loss": True, "lambda_img": state.lambda_img,
            "alpha128": float(u.bca128.alpha.detach()),
            "alpha64": float(u.bca64.alpha.detach()),
            "attention128": {k: float(v) for k, v in u.bca128.last_activity.items()},
            "attention64": {k: float(v) for k, v in u.bca64.last_activity.items()},
            "beta128": float(u.structure.beta128.detach()),
            "beta64": float(u.structure.beta64.detach()),
            "structure_shapes": u.structure.last_shapes,
            "structure_activity": numeric_activity(u.structure),
            "train_seconds": state.train_seconds,
            "seconds_per_step": state.train_seconds / state.step,
            "peak_vram_bytes": state.peak_vram,
            "nonfinite_count": state.nonfinite_count,
            "train_stream_sha256": state.stream_digest.hex(),
            "data_access": access,
            "checkpoint": str(checkpoint_dir / "step_6000.pt"),
        }
        write_json(HERE / "FINAL_TRAIN_SUMMARY.json", summary)
        write_json(HERE / "DATA_ACCESS_AUDIT.json", access)
        write_json(HERE / "progress.json", {"status": "COMPLETE", "step": 6000, "stop": 6000})
        print("TRAIN COMPLETE", history[-1]["psnr"], flush=True)


if __name__ == "__main__":
    main()
