"""Guarded four-arm smoke and full 6k Fast64 runs; no performance gate."""
import argparse
import csv
import json
import math
from pathlib import Path

import torch

from anatomy_controlnet_v1.evaluate import validate
from anatomy_controlnet_v1.model import parameter_counts
from context_2p5d.data import ContextDataGuard
from mayo2016_smallset_v1.runtime import ROOT, write_json, sha256, rng_state
from prosac_add_fusion_v1.runtime import rng_hash
from .runtime import TrainState, med_block_hash
from .protocol import HERE, ARMS, STEPS, config


def write_csv(path, rows):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def access_ok(access, full=True):
    assert access["L506_opens"] == access["external_memory_opens"] == 0
    assert access["neighbor_target_opens"] == access["blocked_attempts"] == 0
    if full:
        expected = list(map(int, (ROOT / "splits/L333_fast64.txt").read_text().splitlines()))
        assert access["validation_center_indices_opened"] == expected
        assert access["training_patients_opened"] == ["L067", "L143"]


def smoke(arm, device):
    out = HERE / "smoke" / (arm + "_BF16_RETRY" if arm == "D4_MULTI_FT_512" else arm)
    out.mkdir(parents=True, exist_ok=False)
    guard = ContextDataGuard(config(arm), "FAST", allow_training=True)
    state = None
    fork_metrics = post_smoke_metrics = None
    try:
        with guard:
            state = TrainState(arm, device, guard)
            if arm == "D4_MULTI_FT_512":
                audit = state.fork_d1(HERE / "runs/D1_MULTI_512/checkpoints/step_2000.pt")
                assert audit["fork_step"] == state.step == 2000
                assert audit["output_max_abs_difference"] <= 1.e-6
                before_rng = rng_hash(rng_state())
                fork_metrics, _, _ = validate(
                    state.ema.model, state.validation, state.cfg, state.device, activity=False)
                assert rng_hash(rng_state()) == before_rng
                with (HERE / "runs/D1_MULTI_512/validation_history.csv").open(newline="") as f:
                    d1_2k = next(r for r in csv.DictReader(f) if int(r["step"]) == 2000)
                assert abs(fork_metrics["psnr"] - float(d1_2k["psnr"])) <= 1.e-6
            smoke_steps = 20 if arm == "D4_MULTI_FT_512" else 2
            rows = [state.update() for _ in range(smoke_steps)]
            if arm == "D4_MULTI_FT_512":
                assert state.step == 2020 and state.grad_audit["nonzero_count"] > 0
                post_smoke_metrics, _, _ = validate(
                    state.ema.model, state.validation, state.cfg, state.device, activity=False)
                assert all(math.isfinite(float(post_smoke_metrics[k])) for k in (
                    "psnr", "ssim", "rmse_hu", "mae_hu", "soft_psnr", "soft_ssim"))
            assert all(math.isfinite(r["train_loss"]) and
                       math.isfinite(r["gradient_norm_pre_clip"]) for r in rows)
            assert state.ema.updates == state.step
        access = guard.report()
        access_ok(access, full=(arm == "D4_MULTI_FT_512"))
        write_json(out / "SMOKE_RESULT.json", {
            "status": "PASS", "arm": arm, "steps": smoke_steps,
            "optimizer_step_after_smoke": state.step,
            "fork_metrics": fork_metrics, "post_smoke_metrics": post_smoke_metrics,
            "meddino_nonzero_gradient_steps": state.grad_audit["nonzero_count"],
            "initialization": state.init_audit, "fork_audit": state.fork_audit,
            "loss": [r["train_loss"] for r in rows],
            "grad_norm": [r["gradient_norm_pre_clip"] for r in rows],
            "peak_vram_bytes": state.peak_vram,
            "data_access": access,
        })
        print("SMOKE_PASS", arm, flush=True)
    except BaseException as exc:
        write_json(out / "SMOKE_RESULT.json", {
            "status": "FAILED", "error": repr(exc), "data_access": guard.report(),
            "cuda_allocated_bytes": torch.cuda.memory_allocated() if torch.cuda.is_available() else None,
            "cuda_reserved_bytes": torch.cuda.memory_reserved() if torch.cuda.is_available() else None})
        raise
    finally:
        del state


def train(arm, device):
    smoke_dir = arm + "_BF16_RETRY" if arm == "D4_MULTI_FT_512" else arm
    smoke_record = json.loads((HERE / "smoke" / smoke_dir / "SMOKE_RESULT.json").read_text())
    assert smoke_record["status"] == "PASS"
    out = HERE / "runs" / arm
    out.mkdir(parents=True, exist_ok=False)
    (out / "checkpoints").mkdir()
    cfg = config(arm)
    guard = ContextDataGuard(cfg, "FAST", allow_training=True)
    state = None
    history, activity, checkpoint_manifest = [], [], []
    try:
        with guard:
            state = TrainState(arm, device, guard)
            assert state.init_audit == smoke_record["initialization"]
            if arm == "D4_MULTI_FT_512":
                fork = state.fork_d1(HERE / "runs/D1_MULTI_512/checkpoints/step_2000.pt")
                assert fork == smoke_record["fork_audit"]
            else:
                fork = None
            write_json(out / "CONFIG.json", cfg)
            write_json(out / "INITIALIZATION_AUDIT.json", state.init_audit)
            write_json(out / "TRAIN_RNG_AUDIT.json", {
                "initial_rng_sha256": state.initial_rng_sha256,
                "smoke_initial_rng_sha256": smoke_record["initialization"]["canonical_training_rng_sha256"],
                "fork": fork,
            })
            write_json(out / "PARAMETER_AUDIT.json", parameter_counts(state.model))

            def evaluate(step):
                before = rng_hash(rng_state())
                metrics, slices, diagnostic = validate(
                    state.ema.model, state.validation, cfg, state.device, activity=True)
                assert rng_hash(rng_state()) == before
                assert metrics["slices"] == 64 and metrics["weights"] == "ema"
                write_csv(out / ("per_slice_step_%d.csv" % step), slices)
                row = {
                    "arm": arm, "step": step, **metrics,
                    "train_loss": None if state.last is None else state.last["train_loss"],
                    "gradient_norm_pre_clip": None if state.last is None else state.last["gradient_norm_pre_clip"],
                    "learning_rate": state.optimizer.param_groups[0]["lr"],
                    "meddino_learning_rate": state.optimizer.param_groups[1]["lr"]
                    if len(state.optimizer.param_groups) > 1 else None,
                    "peak_vram_bytes": state.peak_vram,
                    "seconds_per_step": None if state.step == 0 else
                    state.train_seconds / state.step,
                    "train_stream_sha256": state.stream_digest.hex(),
                }
                history.append(row)
                activity.append({"arm": arm, "step": step, "features": diagnostic})
                write_csv(out / "validation_history.csv", history)
                write_json(out / "feature_activity.json", activity)
                print("VALIDATION", arm, step, metrics["psnr"], flush=True)
                return row

            if arm == "D4_MULTI_FT_512":
                row = evaluate(2000)
                d1_rows = list(csv.DictReader(
                    (HERE / "runs/D1_MULTI_512/validation_history.csv").open()))
                d1_2k = next(r for r in d1_rows if int(r["step"]) == 2000)
                assert abs(float(row["psnr"]) - float(d1_2k["psnr"])) <= 1.e-6
            else:
                evaluate(0)
            with (out / "train.jsonl").open("x", buffering=1) as f:
                while state.step < 6000:
                    row = state.update()
                    f.write(json.dumps(row, allow_nan=False) + "\n")
                    if state.step % 100 == 0:
                        write_json(out / "progress.json", {
                            "status": "RUNNING", "step": state.step, "stop": 6000})
                        print("TRAIN", arm, state.step, row["train_loss"], flush=True)
                    if state.step in (500, 1000, 2000, 4000, 6000):
                        if arm != "D4_MULTI_FT_512" or state.step in (4000, 6000):
                            evaluate(state.step)
                    save = ((arm == "D1_MULTI_512" and state.step in (2000, 6000)) or
                            (arm != "D1_MULTI_512" and state.step == 6000))
                    if save:
                        ck_path = out / "checkpoints" / ("step_%d.pt" % state.step)
                        state.checkpoint(ck_path)
                        checkpoint_manifest.append({"step": state.step, "sha256": sha256(ck_path)})
                        write_json(out / "CHECKPOINT_MANIFEST.json", {
                            "arm": arm, "matched": checkpoint_manifest})
        access = guard.report()
        access_ok(access)
        assert state.step == 6000 and state.ema.updates == 6000
        assert state.nonfinite_count == 0
        u = state.model.model.unet0
        assert u.main_state_hash() == state.main_hash
        assert state.ema.model.model.unet0.main_state_hash() == state.main_hash
        write_json(out / "DATA_ACCESS_AUDIT.json", access)
        write_json(out / "TRAIN_STREAM_AUDIT.json", {
            "sha256": state.stream_digest.hex(), "samples": 12000,
            "initial_rng_sha256": state.initial_rng_sha256,
            "fork_from_D1_at_2k": arm == "D4_MULTI_FT_512",
        })
        final = {
            "status": "COMPLETE", "arm": arm, "optimizer_steps": state.step,
            "history": history, "parameter_audit": parameter_counts(state.model),
            "peak_vram_bytes": state.peak_vram,
            "training_seconds_including_D1_shared": state.train_seconds,
            "training_seconds_after_fork": state.fork_train_seconds if fork else None,
            "seconds_per_step": (state.fork_train_seconds / 4000 if fork
                                 else state.train_seconds / 6000),
            "nonfinite_count": state.nonfinite_count,
            "main_hash_before": state.main_hash,
            "main_hash_after": u.main_state_hash(),
            "ema_main_hash_after": state.ema.model.model.unet0.main_state_hash(),
            "train_stream_sha256": state.stream_digest.hex(),
            "data_access": access,
            "fork_audit": state.fork_audit,
            "grad_audit": state.grad_audit if fork else None,
        }
        if fork:
            med = u.structure.med
            delta_sq = 0.
            final_sq = 0.
            for i in range(8, 12):
                for name, parameter in med.blocks[i].named_parameters():
                    current = parameter.detach().cpu().double()
                    initial = state.finetune_initial[f"{i}.{name}"].double()
                    delta_sq += float((current - initial).square().sum())
                    final_sq += float(current.square().sum())
            final["finetune_audit"] = {
                **fork,
                "blocks_8_11_parameter_delta_norm": math.sqrt(delta_sq),
                "blocks_8_11_relative_delta_norm": math.sqrt(delta_sq) / fork["blocks_8_11_initial_parameter_norm"],
                "blocks_8_11_final_parameter_norm": math.sqrt(final_sq),
                "blocks_0_7_final_hash": med_block_hash(med, 0, 8),
                "blocks_8_11_final_hash": med_block_hash(med, 8, 12),
                "gradient_rms_mean": state.grad_audit["rms_sum"] / state.grad_audit["count"],
                "gradient_norm_mean": state.grad_audit["norm_sum"] / state.grad_audit["count"],
                "nonzero_gradient_steps": state.grad_audit["nonzero_count"],
            }
            assert final["finetune_audit"]["blocks_0_7_final_hash"] == fork["blocks_0_7_initial_hash"]
            assert final["finetune_audit"]["blocks_8_11_final_hash"] != fork["blocks_8_11_initial_hash"]
            assert state.grad_audit["nonzero_count"] > 0
        write_json(out / "final_summary.json", final)
        write_json(out / "progress.json", {"status": "COMPLETE", "step": 6000})
        print("ARM_COMPLETE", arm, flush=True)
    except BaseException as exc:
        write_json(out / "DATA_ACCESS_AUDIT.json", guard.report())
        write_json(out / "progress.json", {
            "status": "FAILED", "step": 0 if state is None else state.step,
            "error": repr(exc),
            "cuda_allocated_bytes": torch.cuda.memory_allocated() if torch.cuda.is_available() else None,
            "cuda_reserved_bytes": torch.cuda.memory_reserved() if torch.cuda.is_available() else None,
        })
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("smoke", "train"))
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.mode == "smoke":
        smoke(args.arm, args.device)
    else:
        train(args.arm, args.device)


if __name__ == "__main__":
    main()
