"""Independent E1 smoke and 6k Fast64 training, reusing D1 update/evaluator."""
import argparse
import csv
import json
import math
from pathlib import Path

import torch

from anatomy_controlnet_v1.evaluate import validate
from anatomy_controlnet_v1.model import parameter_counts
from context_2p5d.data import ContextDataGuard
from mayo2016_smallset_v1.runtime import ROOT, sha256, write_json
from src.rddm_rleb import normalize_to_neg_one_to_one
from meddino_four_arm_v1.protocol import frozen_config
from .protocol import ARM, HERE, STEPS, D1_ROOT, config
from .runtime import SALFTrainState

METRIC_KEYS = ("psnr", "ssim", "rmse_hu", "mae_hu", "soft_psnr",
               "soft_ssim", "soft_rmse_hu", "soft_mae_hu", "validation_loss")


def write_csv(path, rows):
    assert rows
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def d1_history():
    with (D1_ROOT / "validation_history.csv").open(newline="") as f:
        return {int(r["step"]): r for r in csv.DictReader(f)}


@torch.no_grad()
def gate_rows(model, dataset, device, step):
    provider = model.model.unet0.structure
    assert not provider.med.training
    provider.capture_weights = True
    model.eval()
    rows = []
    try:
        for pos in range(64):
            _, center, _, _, meta = dataset.load(pos)
            assert meta["patient_id"] == "L333"
            tensor = torch.from_numpy(center).unsqueeze(0).to(device)
            provider(normalize_to_neg_one_to_one(tensor))
            record = {"step": step, "slice_index": meta["slice_index"]}
            assert provider.last_weights is not None
            for scale, tag in (("s256", "w256"), ("s128", "w128"), ("s64", "w64")):
                weights = provider.last_weights[scale][0].tolist()
                assert len(weights) == 4 and abs(sum(weights) - 1.0) <= 1e-6
                for layer, weight in zip((2, 5, 8, 11), weights):
                    record[f"{tag}_{layer}"] = weight
                record[f"{tag}_entropy"] = -sum(w * math.log(max(w, 1e-12)) for w in weights)
            rows.append(record)
    finally:
        provider.capture_weights = False
        provider.last_weights = None
    assert len(rows) == 64
    return rows


def check_access(access, full=True):
    assert all(access[k] == 0 for k in ("L506_opens", "external_memory_opens",
                                        "neighbor_target_opens", "blocked_attempts"))
    if full:
        assert access["training_patients_opened"] == ["L067", "L143"]
        assert access["validation_center_indices_opened"] == list(map(
            int, (ROOT / "splits/L333_fast64.txt").read_text().splitlines()))


def smoke(device):
    cfg = config()
    guard = ContextDataGuard(cfg, "FAST", allow_training=True)
    with guard:
        state = SALFTrainState(device, guard)
        before = state.init_audit
        assert before["step_zero_equivalent"]
        assert before["added_parameters"] == 9228
        rows = [state.update() for _ in range(2)]
        assert state.step == state.ema.updates == 2
        assert all(math.isfinite(float(r["train_loss"])) for r in rows)
        assert all(math.isfinite(float(r["salf_gradient_norm"])) for r in rows)
    access = guard.report()
    check_access(access, full=False)
    write_json(HERE / "SMOKE_RESULT.json", {
        "status": "PASS", "steps": 2, "initialization": before,
        "train_rows": rows, "salf_nonzero_gradient_steps": state.salf_gradient_steps,
        "peak_vram_bytes": state.peak_vram, "data_access": access})
    print("SALF_SMOKE_PASS", flush=True)


def train(device):
    smoke_result = json.loads((HERE / "SMOKE_RESULT.json").read_text())
    assert smoke_result["status"] == "PASS"
    cfg = config()
    guard = ContextDataGuard(cfg, "FAST", allow_training=True)
    history, all_slices, all_gates = [], [], []
    d1 = d1_history()
    checkpoint_path = HERE / "checkpoints/step_6000.pt"
    assert not checkpoint_path.exists()
    with guard:
        state = SALFTrainState(device, guard)
        assert state.init_audit == smoke_result["initialization"]
        write_json(HERE / "EXPERIMENT_CONFIG.json", cfg)
        write_json(HERE / "INITIALIZATION_PARITY_AUDIT.json", state.init_audit)
        write_json(HERE / "PARAMETER_AUDIT.json", parameter_counts(state.model))

        def evaluate(step):
            metrics, slices, _ = validate(
                state.ema.model, state.validation, cfg, state.device, activity=False)
            assert metrics["weights"] == "ema" and metrics["slices"] == 64
            assert all(math.isfinite(float(metrics[key])) for key in METRIC_KEYS)
            reference = d1[step]
            assert state.stream_digest.hex() == reference["train_stream_sha256"]
            row = {
                "arm": ARM, "step": step, **metrics,
                "d1_psnr": float(reference["psnr"]),
                "delta_psnr_vs_d1": metrics["psnr"] - float(reference["psnr"]),
                "train_loss": None if state.last is None else state.last["train_loss"],
                "gradient_norm_pre_clip": None if state.last is None else
                                          state.last["gradient_norm_pre_clip"],
                "salf_gradient_norm": None if state.last is None else
                                      state.last["salf_gradient_norm"],
                "learning_rate": state.optimizer.param_groups[0]["lr"],
                "peak_vram_bytes": state.peak_vram,
                "seconds_per_step": state.train_seconds / step if step else None,
                "train_stream_sha256": state.stream_digest.hex(),
            }
            history.append(row)
            write_csv(HERE / "VALIDATION_HISTORY.csv", history)
            for sample in slices:
                all_slices.append({"step": step, **sample})
            write_csv(HERE / "FAST64_PER_SLICE.csv", all_slices)
            all_gates.extend(gate_rows(state.ema.model, state.validation, state.device, step))
            write_csv(HERE / "FAST64_GATE_WEIGHTS.csv", all_gates)
            print("VALIDATION", step, metrics["psnr"], row["delta_psnr_vs_d1"], flush=True)

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
        assert state.nonfinite_count == 0 and state.salf_gradient_steps > 0
        state.checkpoint(checkpoint_path)
        write_json(HERE / "CHECKPOINT_MANIFEST.json", {
            "arm": ARM, "step": 6000, "sha256": sha256(checkpoint_path),
            "ema": True, "optimizer_steps": 6000})
    access = guard.report()
    check_access(access)
    write_json(HERE / "DATA_ACCESS_AUDIT.json", access)
    write_json(HERE / "TRAIN_STREAM_AUDIT.json", {
        "sha256": state.stream_digest.hex(), "samples": 12000,
        "matched_to_D1_at_every_validation": True,
        "salf_nonzero_gradient_steps": state.salf_gradient_steps,
        "mean_salf_gradient_norm": state.salf_gradient_norm_sum / 6000})
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
        "data_access": access})
    write_json(HERE / "progress.json", {"status": "COMPLETE", "step": 6000})
    print("SALF_TRAIN_COMPLETE", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("smoke", "train"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device("cuda:0")
    if args.mode == "smoke":
        smoke(device)
    else:
        train(device)


if __name__ == "__main__":
    main()
