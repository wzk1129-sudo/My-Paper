"""Independent C3/C4 first-stage and resumed 6k training."""
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
from .runtime import TrainState, rng_hash
from .protocol import HERE, OLD, OLD_PROSAC, ARMS, config


def write_csv(path, rows):
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def merge_access(a, b):
    if a is None:
        return b
    x = dict(a)
    for k in ("training_patients_opened", "validation_center_indices_opened",
              "validation_ldct_indices_opened"):
        x[k] = sorted(set(a[k]) | set(b[k]))
    for k in ("center_target_opens", "center_ldct_opens", "neighbor_ldct_opens",
              "neighbor_target_opens", "training_target_opens", "total_opens",
              "L506_opens", "external_memory_opens", "blocked_attempts"):
        x[k] = a[k] + b[k]
    return x


def access_ok(a, full=True):
    assert a["L506_opens"] == a["external_memory_opens"] == 0
    assert a["neighbor_target_opens"] == a["blocked_attempts"] == 0
    if full:
        expected = list(map(int, (ROOT / "splits/L333_fast64.txt").read_text().splitlines()))
        assert a["validation_center_indices_opened"] == expected


def smoke(arm, input_size, device):
    out = HERE / "smoke" / arm
    out.mkdir(parents=True, exist_ok=False)
    guard = ContextDataGuard(config(arm, input_size), "FAST", allow_training=True)
    try:
        with guard:
            state = TrainState(arm, input_size, device, guard)
            rows = [state.update() for _ in range(20)]
            assert all(math.isfinite(r["train_loss"]) and
                       math.isfinite(r["gradient_norm_pre_clip"]) for r in rows)
            assert state.ema.updates == 20
            assert all(p.grad is None for p in state.model.model.unet0.structure.med.parameters())
            assert all(float(z.weight.detach().norm()) > 0
                       for z in state.model.model.unet0.zero_convs)
        access = guard.report()
        access_ok(access, False)
        write_json(out / "SMOKE_RESULT.json", {
            "status": "PASS", "arm": arm, "steps": 20,
            "initialization": state.init_audit,
            "loss": [r["train_loss"] for r in rows],
            "grad_norm": [r["gradient_norm_pre_clip"] for r in rows],
            "peak_vram_bytes": state.peak_vram,
            "seconds_per_step": state.train_seconds / 20,
            "data_access": access,
        })
        print("SMOKE_PASS", arm, flush=True)
    except BaseException as exc:
        write_json(out / "SMOKE_RESULT.json", {
            "status": "FAILED", "error": repr(exc), "data_access": guard.report()})
        raise


def train(arm, input_size, device, resume):
    smoke_record = json.loads((HERE / "smoke" / arm / "SMOKE_RESULT.json").read_text())
    assert smoke_record["status"] == "PASS"
    out = HERE / "runs" / arm
    if resume:
        assert json.loads((out / "progress.json").read_text())["status"] == "FIRST_STAGE_COMPLETE"
        history = read_csv(out / "validation_history.csv")
        activity = read_csv(out / "control_activity.csv")
        old_access = json.loads((out / "DATA_ACCESS_AUDIT.json").read_text())
        stop = 6000
    else:
        out.mkdir(parents=True, exist_ok=False)
        (out / "checkpoints").mkdir()
        history, activity, old_access = [], [], None
        stop = 2000
    cfg = config(arm, input_size)
    guard = ContextDataGuard(cfg, "FAST", allow_training=True)
    state = None
    try:
        with guard:
            state = TrainState(arm, input_size, device, guard)
            assert state.init_audit == smoke_record["initialization"]
            if resume:
                state.restore(out / "checkpoints/step_2000.pt")
                assert state.step == 2000
            else:
                write_json(out / "CONFIG.json", cfg)
                write_json(out / "INITIALIZATION_AUDIT.json", state.init_audit)
                write_json(out / "PARAMETER_AUDIT.json", parameter_counts(state.model))
                write_json(out / "TRAIN_RNG_AUDIT.json", {
                    "initial_rng_sha256": state.initial_rng_sha256,
                    "smoke_rng_sha256": smoke_record["initialization"]["canonical_training_rng_sha256"],
                    "canonical_restored": True,
                })

            def evaluate(step):
                before = rng_hash(rng_state())
                metrics, rows, act = validate(
                    state.ema.model, state.validation, cfg, state.device, activity=True)
                assert rng_hash(rng_state()) == before
                assert metrics["slices"] == 64 and metrics["weights"] == "ema"
                write_csv(out / ("per_slice_step_%d.csv" % step), rows)
                history.append({
                    "arm": arm, "step": step, **metrics,
                    "train_loss": None if state.last is None else state.last["train_loss"],
                    "gradient_norm_pre_clip": None if state.last is None else
                    state.last["gradient_norm_pre_clip"],
                    "learning_rate": state.optimizer.param_groups[0]["lr"],
                    "peak_vram_bytes": state.peak_vram,
                    "seconds_per_step": None if step == 0 else state.train_seconds / step,
                    "train_stream_sha256": state.stream_digest.hex(),
                })
                activity.append({
                    "arm": arm, "step": step,
                    **{k: json.dumps(v) for k, v in act.items()},
                })
                write_csv(out / "validation_history.csv", history)
                write_csv(out / "control_activity.csv", activity)
                print("VALIDATION", arm, step, metrics["psnr"], flush=True)

            if not resume:
                evaluate(0)
            with (out / "train.jsonl").open("a" if resume else "x", buffering=1) as f:
                while state.step < stop:
                    row = state.update()
                    f.write(json.dumps(row, allow_nan=False) + "\n")
                    if state.step % 100 == 0:
                        write_json(out / "progress.json", {
                            "status": "RUNNING", "step": state.step, "stop": stop})
                        print("TRAIN", arm, state.step, row["train_loss"], flush=True)
                    nodes = (4000, 6000) if resume else (500, 1000, 2000)
                    if state.step in nodes:
                        evaluate(state.step)
                        if state.step in (2000, 6000):
                            path = out / "checkpoints" / ("step_%d.pt" % state.step)
                            state.checkpoint(path)
                            manifest = out / "CHECKPOINT_MANIFEST.json"
                            rows = (json.loads(manifest.read_text())["matched"]
                                    if manifest.is_file() else [])
                            rows.append({"step": state.step, "sha256": sha256(path)})
                            write_json(manifest, {"arm": arm, "matched": rows})
        access = merge_access(old_access, guard.report())
        access_ok(access)
        assert state.step == stop and state.ema.updates == stop
        assert state.nonfinite_count == 0
        assert state.model.model.unet0.main_state_hash() == state.main_hash
        assert state.ema.model.model.unet0.main_state_hash() == state.main_hash
        write_json(out / "DATA_ACCESS_AUDIT.json", access)
        write_json(out / "TRAIN_STREAM_AUDIT.json", {
            "sha256": state.stream_digest.hex(), "samples": stop * 2,
            "initial_rng_sha256": state.initial_rng_sha256})
        status = "COMPLETE" if resume else "FIRST_STAGE_COMPLETE"
        write_json(out / "final_summary.json", {
            "arm": arm, "status": status, "optimizer_steps": stop,
            "history": history, "parameter_audit": parameter_counts(state.model),
            "peak_vram_bytes": state.peak_vram,
            "training_seconds": state.train_seconds,
            "seconds_per_step": state.train_seconds / stop,
            "nonfinite_count": state.nonfinite_count,
            "main_hash_before": state.main_hash,
            "main_hash_after": state.model.model.unet0.main_state_hash(),
            "ema_main_hash_after": state.ema.model.model.unet0.main_state_hash(),
            "train_stream_sha256": state.stream_digest.hex(),
            "data_access": access,
        })
        write_json(out / "progress.json", {"status": status, "step": stop})
        print("ARM", status, arm, flush=True)
    except BaseException as exc:
        write_json(out / "DATA_ACCESS_AUDIT.json", merge_access(old_access, guard.report()))
        write_json(out / "progress.json", {
            "status": "FAILED", "step": 0 if state is None else state.step,
            "error": repr(exc)})
        raise


def gate():
    arm = "C4_ADD"
    out = HERE / "runs" / arm
    history = {int(r["step"]): r for r in read_csv(out / "validation_history.csv")}
    assert sorted(history) == [0, 500, 1000, 2000]
    references = {
        "C3_ANATOMY": OLD_PROSAC / "runs/C3_ANATOMY",
        "C4_APGC": OLD_PROSAC / "runs/C4_APGC",
        "C1_CAPACITY": OLD / "runs/C1_CAPACITY",
    }
    deltas = {}
    slice_stats = {}
    for name, directory in references.items():
        ref_history = {int(r["step"]): r for r in read_csv(directory / "validation_history.csv")}
        deltas[name] = {
            step: float(history[step]["psnr"]) - float(ref_history[step]["psnr"])
            for step in (500, 1000, 2000)}
        ref_slices = read_csv(directory / "per_slice_step_2000.csv")
        new_slices = read_csv(out / "per_slice_step_2000.csv")
        assert len(ref_slices) == len(new_slices) == 64
        assert [r["slice_index"] for r in ref_slices] == [r["slice_index"] for r in new_slices]
        slice_stats[name] = sum(
            float(a["psnr"]) > float(b["psnr"])
            for a, b in zip(new_slices, ref_slices)) / 64.
    d = deltas["C3_ANATOMY"]
    upward = d[500] < d[1000] < d[2000]
    promote = d[2000] >= .005 or (upward and slice_stats["C3_ANATOMY"] >= .60)
    stop_clear = d[2000] <= -.005 and not upward
    decisions = {arm: {
        "delta_psnr": deltas,
        "positive_slice_fraction": slice_stats,
        "upward_vs_C3": upward,
        "clear_no_promotion": stop_clear,
        "continue_to_6k": promote,
    }}
    write_json(HERE / "PROMOTION_DECISION.json", decisions)
    return decisions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("smoke", "first", "resume", "gate"))
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--input-size", type=int, default=512)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    if a.mode == "gate":
        print(json.dumps(gate()), flush=True)
    elif a.mode == "smoke":
        smoke(a.arm, a.input_size, a.device)
    else:
        train(a.arm, a.input_size, a.device, a.mode == "resume")


if __name__ == "__main__":
    main()

