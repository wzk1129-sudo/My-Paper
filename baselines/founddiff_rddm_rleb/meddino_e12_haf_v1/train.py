"""E12 independent smoke and 6k Fast64 training on frozen two-patient protocol."""
import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from anatomy_controlnet_v1.evaluate import validate
from anatomy_controlnet_v1.model import parameter_counts
from context_2p5d.data import ContextDataGuard
from context_2p5d.model import restored_slice
from mayo2016_smallset_v1.runtime import ROOT, sha256, write_json
from meddino_salf_v1.train import write_csv, d1_history, gate_rows, check_access, METRIC_KEYS
from .protocol import ARM, HERE, STEPS, config
from .runtime import E12TrainState


def checkpoint_preflight():
    folder = HERE / "checkpoints"
    folder.mkdir(parents=True, exist_ok=True)
    probe = folder / ".write_test.tmp"
    assert not probe.exists()
    try:
        with probe.open("xb") as f:
            f.write(b"E12 checkpoint directory writable\n")
            f.flush()
            os.fsync(f.fileno())
        assert probe.read_bytes() == b"E12 checkpoint directory writable\n"
    finally:
        probe.unlink(missing_ok=True)
    assert folder.is_dir() and os.access(folder, os.W_OK)
    write_json(HERE / "CHECKPOINT_PREFLIGHT.json",
               {"status": "PASS", "path": str(folder),
                "temporary_write_read_delete": True,
                "checked_before_training": True})


def smoke(device):
    cfg = config()
    guard = ContextDataGuard(cfg, "FAST", allow_training=True)
    with guard:
        state = E12TrainState(device, guard)
        rows = [state.update() for _ in range(20)]
        assert state.step == state.ema.updates == 20
        assert all(math.isfinite(float(r["train_loss"])) for r in rows)
        assert all(math.isfinite(float(r["gradient_norm_pre_clip"])) for r in rows)
        needed = ("SALF", "E2_64to128_zero", "E2_128to256_zero",
                  "local256_zero", "cross128_zero", "cross64_zero", "ControlNet")
        assert all(state.gradient_nonzero_steps[k] > 0 for k in needed), state.gradient_nonzero_steps
        attention_qkv_active = all(
            state.gradient_nonzero_steps[k] > 0
            for k in ("cross128_QKV", "cross64_QKV"))
        assert attention_qkv_active, state.gradient_nonzero_steps
        assert state.nonfinite_count == 0
    access = guard.report()
    check_access(access, full=False)
    write_json(HERE / "SMOKE_RESULT.json", {
        "status": "PASS", "steps": 20,
        "initialization": state.init_audit,
        "loss_first": rows[0]["train_loss"], "loss_last": rows[-1]["train_loss"],
        "gradient_nonzero_steps": state.gradient_nonzero_steps,
        "attention_QKV_active": attention_qkv_active,
        "last_group_gradient_norms": state.last_group_gradients,
        "peak_vram_bytes": state.peak_vram, "data_access": access})
    print("E12_SMOKE_PASS", flush=True)


def flatten_numeric(prefix, value, out):
    if isinstance(value, dict):
        for key, item in value.items():
            flatten_numeric(prefix + "." + key if prefix else key, item, out)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(float(value)):
            out[prefix] = float(value)


@torch.no_grad()
def diagnostics(state, step):
    model = state.ema.model.eval()
    u = model.model.unet0
    rows = []
    devices = [state.device.index or 0] if state.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        for pos in range(64):
            prev, center, next_, _, meta = state.validation.load(pos)
            assert meta["patient_id"] == "L333"
            tensors = [torch.from_numpy(v).unsqueeze(0).to(state.device)
                       for v in (prev, center, next_)]
            seed = state.cfg["evaluation"]["validation_seed"] + meta["slice_index"]
            torch.manual_seed(seed)
            if state.device.type == "cuda":
                torch.cuda.manual_seed(seed)
            u.collect_activity = True
            prediction = restored_slice(model, tensors[1], tensors[0], tensors[2])
            assert bool(torch.isfinite(prediction).all())
            data = {}
            flatten_numeric("", u.activity, data)
            rows.append(data)
    u.collect_activity = False
    keys = sorted(set.intersection(*(set(row) for row in rows)))
    summary = {k: {"mean": float(np.mean([r[k] for r in rows])),
                   "std": float(np.std([r[k] for r in rows])),
                   "min": float(np.min([r[k] for r in rows])),
                   "max": float(np.max([r[k] for r in rows]))}
               for k in keys}
    write_json(HERE / f"ACTIVITY_STEP_{step}.json",
               {"step": step, "patient": "L333", "slices": 64,
                "weights": "EMA", "summary": summary,
                "attention_entropy": "not computed: avoids dense attention weights"})
    return summary


def train(device):
    smoke_result = json.loads((HERE / "SMOKE_RESULT.json").read_text())
    assert smoke_result["status"] == "PASS" and smoke_result["steps"] == 20
    preflight = json.loads((HERE / "CHECKPOINT_PREFLIGHT.json").read_text())
    assert preflight["status"] == "PASS"
    cfg = config()
    guard = ContextDataGuard(cfg, "FAST", allow_training=True)
    d1 = d1_history()
    history, all_slices, all_gates = [], [], []
    checkpoint = HERE / "checkpoints/step_6000.pt"
    assert not checkpoint.exists() and checkpoint.parent.is_dir()
    with guard:
        state = E12TrainState(device, guard)
        assert state.init_audit == smoke_result["initialization"]
        write_json(HERE / "EXPERIMENT_CONFIG.json", cfg)
        write_json(HERE / "INITIALIZATION_PARITY_AUDIT.json", state.init_audit)
        write_json(HERE / "PARAMETER_AUDIT.json", parameter_counts(state.model))

        def evaluate(step):
            metrics, slices, _ = validate(
                state.ema.model, state.validation, cfg, state.device, activity=False)
            assert metrics["weights"] == "ema" and metrics["slices"] == 64
            assert all(math.isfinite(float(metrics[k])) for k in METRIC_KEYS)
            reference = d1[step]
            assert state.stream_digest.hex() == reference["train_stream_sha256"]
            row = {
                "arm": ARM, "step": step, **metrics,
                "d1_psnr": float(reference["psnr"]),
                "delta_psnr_vs_d1": metrics["psnr"] - float(reference["psnr"]),
                "train_loss": None if state.last is None else state.last["train_loss"],
                "gradient_norm_pre_clip": None if state.last is None else
                    state.last["gradient_norm_pre_clip"],
                "learning_rate": state.optimizer.param_groups[0]["lr"],
                "peak_vram_bytes": state.peak_vram,
                "seconds_per_step": state.train_seconds / step if step else None,
                "train_stream_sha256": state.stream_digest.hex(),
            }
            history.append(row)
            write_csv(HERE / "VALIDATION_HISTORY.csv", history)
            all_slices.extend({"step": step, **sample} for sample in slices)
            write_csv(HERE / "FAST64_PER_SLICE.csv", all_slices)
            all_gates.extend(gate_rows(
                state.ema.model, state.validation, state.device, step))
            write_csv(HERE / "FAST64_GATE_WEIGHTS.csv", all_gates)
            if step in (0, 6000):
                diagnostics(state, step)
            print("VALIDATION", step, metrics["psnr"], row["delta_psnr_vs_d1"],
                  flush=True)

        evaluate(0)
        with (HERE / "train.jsonl").open("x", buffering=1) as log:
            while state.step < 6000:
                row = state.update()
                log.write(json.dumps(row, allow_nan=False) + "\n")
                if state.step % 100 == 0:
                    write_json(HERE / "progress.json",
                               {"status": "RUNNING", "step": state.step, "stop": 6000})
                    print("TRAIN", state.step, row["train_loss"], flush=True)
                if state.step in STEPS[1:]:
                    evaluate(state.step)
        assert state.step == state.ema.updates == 6000
        assert state.nonfinite_count == 0
        assert all(v > 0 for v in state.gradient_nonzero_steps.values())
        state.checkpoint(checkpoint)
        write_json(HERE / "CHECKPOINT_AUDIT.json", {
            "arm": ARM, "step": 6000, "sha256": sha256(checkpoint),
            "bytes": checkpoint.stat().st_size, "ema": True,
            "optimizer_steps": 6000,
            "preflight": "PASS"})
    access = guard.report()
    check_access(access)
    write_json(HERE / "DATA_ACCESS_AUDIT.json", access)
    write_json(HERE / "TRAIN_STREAM_AUDIT.json", {
        "sha256": state.stream_digest.hex(), "samples": 12000,
        "matched_to_D1_at_every_validation": True,
        "gradient_nonzero_steps": state.gradient_nonzero_steps})
    write_json(HERE / "training_summary.json", {
        "status": "COMPLETE", "arm": ARM, "optimizer_steps": state.step,
        "ema_updates": state.ema.updates, "nonfinite_count": state.nonfinite_count,
        "history": history, "peak_vram_bytes": state.peak_vram,
        "seconds_per_step": state.train_seconds / 6000,
        "total_training_seconds": state.train_seconds,
        "parameter_audit": parameter_counts(state.model),
        "main_hash_before": state.main_hash,
        "main_hash_after": state.model.model.unet0.main_state_hash(),
        "ema_main_hash_after": state.ema.model.model.unet0.main_state_hash(),
        "gradient_nonzero_steps": state.gradient_nonzero_steps,
        "data_access": access})
    write_json(HERE / "progress.json", {"status": "COMPLETE", "step": 6000})
    print("E12_TRAIN_COMPLETE", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("smoke", "train", "preflight"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.mode == "preflight":
        checkpoint_preflight()
    else:
        device = torch.device("cuda:0")
        smoke(device) if args.mode == "smoke" else train(device)


if __name__ == "__main__":
    main()

