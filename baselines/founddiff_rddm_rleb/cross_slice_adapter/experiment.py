"""Frozen Direct Concat control, adapter-only smoke and 6k training."""
import argparse
import copy
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
import traceback

import torch

from fast_dev.protocol import FAST_INDICES
from context_2p5d.data import ContextPairs, ContextBatchStream, ContextDataGuard
from context_2p5d.evaluate import validate as frozen_validate
from context_2p5d.model import construct as construct_direct, state_hash, training_loss
from context_2p5d.runtime import rng_hash
from mayo2016_smallset_v1.runtime import (
    ROOT, EMA, atomic_save, rng_state, restore_rng, sha256, write_json)
from src.rddm_rleb import normalize_to_neg_one_to_one
from .evaluate import validate_adapter
from .model import ARM, DIRECT_ARM, SOURCE_PIN, CHECKPOINT_PIN, attach_adapter, step_zero_equivalence, usage
from .protocol import HERE, STEPS, DIRECT_CHECKPOINT, DIRECT_FULL, DIRECT_HISTORY, BASE_PSNR, config, source_manifest


def csv_write(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def check_access(access, mode="FAST", allow_training=False):
    assert access["L506_opens"] == access["external_memory_opens"] == 0
    assert access["neighbor_target_opens"] == access["blocked_attempts"] == 0
    assert access["data_directory_enumeration"] is False
    expected = list(FAST_INDICES) if mode == "FAST" else list(range(610))
    assert access["validation_center_indices_opened"] == expected
    if allow_training:
        assert access["training_patients_opened"] == ["L067", "L143"]
    else:
        assert access["training_patients_opened"] == []


def reference():
    with DIRECT_HISTORY.open(newline="") as f:
        rows = {int(r["step"]): r for r in csv.DictReader(f)}
    assert float(rows[12000]["psnr"]) == BASE_PSNR
    return rows


def load_frozen(device):
    cfg, _ = config()
    diffusion, init = construct_direct(DIRECT_ARM, device, cfg["adapter_training"]["seed"])
    ck = torch.load(str(DIRECT_CHECKPOINT), map_location="cpu")
    assert ck["arm"] == DIRECT_ARM and ck["optimizer_step"] == 12000
    assert ck["ema_updates"] == 12000 and ck["nonfinite_count"] == 0
    assert ck["source_manifest"]["src/rddm_rleb.py"] == SOURCE_PIN
    assert ck["source_manifest"]["context_2p5d/model.py"] == sha256(ROOT / "context_2p5d/model.py")
    diffusion.load_state_dict(ck["ema_state_dict"], strict=True)
    diffusion.eval().requires_grad_(False)
    backbone_hash = state_hash(diffusion.model.unet0.state_dict())
    assert sha256(DIRECT_CHECKPOINT) == CHECKPOINT_PIN
    return diffusion, init, backbone_hash


def control(device):
    assert not (HERE / "CONTROL_REPRO.json").exists()
    cfg, base = config()
    device = torch.device(device)
    torch.set_num_threads(base["training"]["cpu_threads"])
    diffusion, init, backbone_hash = load_frozen(device)
    guard = ContextDataGuard(base, "FAST", allow_training=False)
    with guard:
        dataset = ContextPairs(base, "validation", "FAST")
        dataset.bind_guard(guard)
        baseline = frozen_validate(diffusion, dataset, base, device)
        assert baseline["slices"] == 64 and baseline["weights"] == "ema"
        if abs(baseline["psnr"] - BASE_PSNR) > 1e-6:
            raise RuntimeError("Frozen Direct Concat Fast64 control did not reproduce")
        diffusion, attached_hash = attach_adapter(diffusion, cfg["adapter_training"]["seed"])
        assert attached_hash == backbone_hash
        difference = step_zero_equivalence(diffusion)
        step0 = validate_adapter(diffusion, dataset, base, device, HERE / "step0_fast64_rows.json")
        assert abs(step0["psnr"] - baseline["psnr"]) <= 1e-6
        for key in ("ssim", "rmse_hu", "mae_hu"):
            assert abs(step0[key] - baseline[key]) <= 1e-6
    access = guard.report()
    check_access(access)
    assert state_hash(diffusion.model.unet0.base.state_dict()) == backbone_hash
    result = {
        "status": "CONTROL_PASS", "checkpoint_path": str(DIRECT_CHECKPOINT),
        "checkpoint_sha256": CHECKPOINT_PIN,
        "archived_direct_fast64_psnr": BASE_PSNR,
        "reproduced_direct_fast64": baseline, "adapter_step0_fast64": step0,
        "step_zero_max_abs_residual_difference": difference,
        "backbone_state_sha256": backbone_hash,
        "adapter_initial_state_sha256": state_hash(diffusion.model.unet0.adapter.state_dict()),
        "source_manifest": source_manifest(), "data_access": access,
        "direct_initialization_audit": init,
    }
    write_json(HERE / "CONTROL_REPRO.json", result)
    write_json(HERE / "INITIALIZATION_AUDIT.json", result)
    print("CONTROL_PASS", baseline["psnr"], step0["psnr"], flush=True)
    return result


class State:
    def __init__(self, device):
        self.cfg, self.base = config()
        self.device = torch.device(device)
        torch.set_num_threads(self.base["training"]["cpu_threads"])
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        self.guard = ContextDataGuard(self.base, "FAST", allow_training=True)
        self.train_data = ContextPairs(self.base, "train")
        self.val_data = ContextPairs(self.base, "validation", "FAST")
        self.train_data.bind_guard(self.guard)
        self.val_data.bind_guard(self.guard)
        assert len(self.train_data) == 1145 and len(self.val_data) == 64
        self.stream = ContextBatchStream(self.train_data, self.cfg["adapter_training"]["seed"])
        frozen, self.direct_audit, self.backbone_hash = load_frozen(self.device)
        self.model, attached_hash = attach_adapter(frozen, self.cfg["adapter_training"]["seed"])
        assert attached_hash == self.backbone_hash
        self.adapter = self.model.model.unet0.adapter
        self.initial_adapter_hash = state_hash(self.adapter.state_dict())
        self.initial_rng_sha256 = rng_hash(rng_state())
        tr = self.cfg["adapter_training"]
        self.optimizer = torch.optim.Adam(self.adapter.parameters(), lr=tr["learning_rate"],
                                          betas=tuple(tr["betas"]), weight_decay=tr["weight_decay"])
        self.ema = EMA(self.adapter, tr["ema_decay"])
        self.eval_model = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.step = 0
        self.train_seconds = 0.0
        self.peak_vram_bytes = 0
        self.nonfinite_count = 0
        self.gradient_min = None
        self.gradient_max = None
        self.stream_digest = hashlib.sha256(b"").digest()
        self.manifest = source_manifest()
        self.gradient_audit = {
            "first_final_gradient_abs": None, "first_hidden_gradient_abs": None,
            "first_hidden_nonzero_step": None, "max_final_gradient_abs": 0.0,
            "max_hidden_gradient_abs": 0.0,
        }
        assert all(not p.requires_grad for p in self.model.model.unet0.base.parameters())
        assert {id(p) for group in self.optimizer.param_groups for p in group["params"]} == {
            id(p) for p in self.adapter.parameters()}
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def update(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        prev, center, next_, target, metadata = self.stream.next(2)
        assert prev.shape == center.shape == next_.shape == target.shape == (2, 1, 512, 512)
        for tensor in (prev, center, next_, target):
            assert tensor.dtype == torch.float32 and bool(torch.isfinite(tensor).all())
            assert 0 <= float(tensor.min()) <= float(tensor.max()) <= 1
        for meta in metadata:
            assert meta["patient_id"] in ("L067", "L143")
            assert meta["prev_index"] == max(0, meta["slice_index"] - 1)
            maximum = 559 if meta["patient_id"] == "L067" else 584
            assert meta["next_index"] == min(maximum, meta["slice_index"] + 1)
            record = {k: meta[k] for k in ("patient_id", "slice_index",
                                           "horizontal_flip", "rotation_degrees")}
            self.stream_digest = hashlib.sha256(
                self.stream_digest + json.dumps(record, sort_keys=True,
                                                separators=(",", ":")).encode()).digest()
        prev, center, next_, target = (x.to(self.device) for x in (prev, center, next_, target))
        self.model.eval()
        self.adapter.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss = training_loss(self.model, target, center, prev, next_)
        wrapped = self.model.model.unet0
        r_gt = normalize_to_neg_one_to_one(center) - normalize_to_neg_one_to_one(target)
        baseline_mse = float(torch.mean((wrapped.last_base - r_gt) ** 2))
        corrected_mse = float(torch.mean((wrapped.last_base + wrapped.last_delta - r_gt) ** 2))
        delta_rms = float(torch.sqrt(torch.mean(wrapped.last_delta.detach() ** 2)))
        if not math.isfinite(float(loss)) or abs(float(loss) - corrected_mse) > 1e-7:
            self.nonfinite_count += 1
            raise FloatingPointError("Invalid corrected residual loss")
        loss.backward()
        assert all(p.grad is None for p in wrapped.base.parameters())
        assert all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                   for p in self.adapter.parameters())
        final_grad = float(self.adapter.final.weight.grad.abs().sum() +
                           self.adapter.final.bias.grad.abs().sum())
        hidden_grad = sum(float(p.grad.abs().sum()) for name, p in self.adapter.named_parameters()
                          if not name.startswith("final."))
        if self.step == 0:
            self.gradient_audit["first_final_gradient_abs"] = final_grad
            self.gradient_audit["first_hidden_gradient_abs"] = hidden_grad
        if hidden_grad > 0 and self.gradient_audit["first_hidden_nonzero_step"] is None:
            self.gradient_audit["first_hidden_nonzero_step"] = self.step + 1
        self.gradient_audit["max_final_gradient_abs"] = max(
            self.gradient_audit["max_final_gradient_abs"], final_grad)
        self.gradient_audit["max_hidden_gradient_abs"] = max(
            self.gradient_audit["max_hidden_gradient_abs"], hidden_grad)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), 1.0))
        if not math.isfinite(grad_norm):
            self.nonfinite_count += 1
            raise FloatingPointError("Nonfinite adapter gradient")
        self.gradient_min = grad_norm if self.gradient_min is None else min(self.gradient_min, grad_norm)
        self.gradient_max = grad_norm if self.gradient_max is None else max(self.gradient_max, grad_norm)
        self.optimizer.step()
        self.ema.update(self.adapter)
        self.step += 1
        assert all(bool(torch.isfinite(p).all()) for p in self.adapter.parameters())
        if self.step <= 20:
            assert state_hash(wrapped.base.state_dict()) == self.backbone_hash
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            self.peak_vram_bytes = max(self.peak_vram_bytes,
                                       torch.cuda.max_memory_allocated(self.device))
        elapsed = time.perf_counter() - started
        self.train_seconds += elapsed
        wrapped.last_base = wrapped.last_delta = wrapped.last_d_abs = None
        return {
            "step": self.step, "train_loss": float(loss.detach()),
            "baseline_error_mse": baseline_mse, "corrected_error_mse": corrected_mse,
            "delta_r_rms": delta_rms, "gradient_norm_pre_clip": grad_norm,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "seconds_per_step": elapsed, "peak_vram_bytes": self.peak_vram_bytes,
            "samples": metadata,
        }

    def sync_eval_ema(self):
        self.eval_model.model.unet0.adapter.load_state_dict(self.ema.model.state_dict(), strict=True)
        assert state_hash(self.eval_model.model.unet0.base.state_dict()) == self.backbone_hash
        self.eval_model.eval()

    def payload(self):
        return {
            "arm": ARM, "optimizer_step": self.step,
            "adapter_state_dict": self.adapter.state_dict(),
            "ema_adapter_state_dict": self.ema.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "ema_updates": self.ema.updates,
            "rng": rng_state(), "data_stream": self.stream.state_dict(),
            "stream_digest_hex": self.stream_digest.hex(),
            "config": self.cfg, "source_manifest": self.manifest,
            "frozen_checkpoint_sha256": CHECKPOINT_PIN,
            "backbone_state_sha256": self.backbone_hash,
            "initial_adapter_sha256": self.initial_adapter_hash,
            "initial_rng_sha256": self.initial_rng_sha256,
            "train_seconds": self.train_seconds,
            "peak_vram_bytes": self.peak_vram_bytes,
            "gradient_min": self.gradient_min, "gradient_max": self.gradient_max,
            "nonfinite_count": self.nonfinite_count,
            "gradient_audit": self.gradient_audit,
        }

    def resume(self, checkpoint):
        saved = torch.load(str(checkpoint), map_location="cpu")
        assert saved["arm"] == ARM and saved["optimizer_step"] < 6000
        assert saved["config"] == self.cfg and saved["source_manifest"] == self.manifest
        assert saved["frozen_checkpoint_sha256"] == CHECKPOINT_PIN
        assert saved["backbone_state_sha256"] == self.backbone_hash
        assert saved["initial_adapter_sha256"] == self.initial_adapter_hash
        self.adapter.load_state_dict(saved["adapter_state_dict"], strict=True)
        self.ema.model.load_state_dict(saved["ema_adapter_state_dict"], strict=True)
        self.optimizer.load_state_dict(saved["optimizer_state_dict"])
        self.stream.load_state_dict(saved["data_stream"])
        self.stream_digest = bytes.fromhex(saved["stream_digest_hex"])
        self.step = saved["optimizer_step"]
        self.ema.updates = saved["ema_updates"]
        self.train_seconds = saved["train_seconds"]
        self.peak_vram_bytes = saved["peak_vram_bytes"]
        self.gradient_min = saved["gradient_min"]
        self.gradient_max = saved["gradient_max"]
        self.nonfinite_count = saved["nonfinite_count"]
        self.gradient_audit = saved["gradient_audit"]
        restore_rng(saved["rng"])
        assert self.ema.updates == self.step



def smoke(device):
    preflight = json.loads((HERE / "CONTROL_REPRO.json").read_text())
    assert preflight["status"] == "CONTROL_PASS"
    assert preflight["source_manifest"] == source_manifest()
    assert not (HERE / "SMOKE_RESULT.json").exists()
    state = State(device)
    assert state.initial_adapter_hash == preflight["adapter_initial_state_sha256"]
    assert state.backbone_hash == preflight["backbone_state_sha256"]
    with state.guard:
        rows = [state.update() for _ in range(20)]
    access = state.guard.report()
    assert access["L506_opens"] == access["external_memory_opens"] == 0
    assert access["neighbor_target_opens"] == access["blocked_attempts"] == 0
    assert access["training_patients_opened"] == ["L067", "L143"]
    assert access["validation_center_indices_opened"] == []
    assert all(math.isfinite(r["train_loss"]) and math.isfinite(r["gradient_norm_pre_clip"])
               for r in rows)
    assert state.gradient_audit["first_final_gradient_abs"] > 0
    assert state.gradient_audit["first_hidden_gradient_abs"] == 0
    assert state.gradient_audit["first_hidden_nonzero_step"] in range(2, 21)
    assert rows[0]["delta_r_rms"] == 0 and rows[-1]["delta_r_rms"] > 0
    assert state_hash(state.model.model.unet0.base.state_dict()) == state.backbone_hash
    assert all(not p.requires_grad for p in state.model.model.unet0.base.parameters())
    assert len(state.optimizer.state) == len(list(state.adapter.parameters()))
    result = {
        "status": "SMOKE_PASS", "optimizer_steps": 20,
        "training_state_discarded": True,
        "losses": [r["train_loss"] for r in rows],
        "baseline_error_mse": [r["baseline_error_mse"] for r in rows],
        "corrected_error_mse": [r["corrected_error_mse"] for r in rows],
        "delta_r_rms": [r["delta_r_rms"] for r in rows],
        "gradient_norms": [r["gradient_norm_pre_clip"] for r in rows],
        "gradient_audit": state.gradient_audit,
        "backbone_state_sha256_before_after": state.backbone_hash,
        "backbone_requires_grad_false": True, "backbone_optimizer_state_count": 0,
        "adapter_optimizer_state_count": len(state.optimizer.state),
        "initial_adapter_sha256": state.initial_adapter_hash,
        "initial_rng_sha256": state.initial_rng_sha256,
        "source_manifest": state.manifest, "data_access": access,
    }
    write_json(HERE / "SMOKE_RESULT.json", result)
    print("SMOKE_PASS", rows[0]["train_loss"], rows[-1]["train_loss"], flush=True)
    return result


def full_l333(state, base):
    out = HERE / "full_l333"
    out.mkdir(exist_ok=False)
    guard = ContextDataGuard(base, "FULL", allow_training=False)
    state.sync_eval_ema()
    with guard:
        dataset = ContextPairs(base, "validation", "FULL")
        dataset.bind_guard(guard)
        candidate = validate_adapter(state.eval_model, dataset, base, state.device,
                                     out / "candidate_per_slice.json")
    access = guard.report()
    check_access(access, mode="FULL")
    observed = json.loads((out / "candidate_per_slice.json").read_text())
    with DIRECT_FULL.open(newline="") as f:
        control = list(csv.DictReader(f))
    assert len(observed) == len(control) == 610
    paired = []
    for a, b in zip(observed, control):
        assert a["patient_id"] == b["patient_id"] == "L333"
        assert a["slice_index"] == int(b["slice_index"])
        item = {"patient_id": "L333", "slice_index": a["slice_index"]}
        for metric in ("psnr", "ssim", "rmse_hu", "mae_hu"):
            baseline = float(b["candidate_" + metric])
            item["candidate_" + metric] = a[metric]
            item["direct_concat_" + metric] = baseline
            item["delta_" + metric] = a[metric] - baseline
        paired.append(item)
    csv_write(out / "paired_per_slice.csv", paired)
    summary = {
        "candidate": candidate,
        "direct_concat": {
            metric: float(sum(r["direct_concat_" + metric] for r in paired) / 610)
            for metric in ("psnr", "ssim", "rmse_hu", "mae_hu")
        },
        "delta_mean": {
            metric: float(sum(r["delta_" + metric] for r in paired) / 610)
            for metric in ("psnr", "ssim", "rmse_hu", "mae_hu")
        },
        "data_access": access,
        "interpretation": "within-L333-stack stability; not patient-level significance",
    }
    write_json(out / "full_l333_summary.json", summary)
    return summary


def report(summary):
    rows = summary["history"]
    lines = [
        "# Cross-Slice Residual Error Adapter V1", "",
        "Frozen 12k EMA Direct Concat; only the five-channel additive adapter was optimized.",
        "L067/L143 train, fixed L333-fast64 EMA, original HU and 2-step sampler.", "",
        "| Step | PSNR | Delta PSNR | SSIM | RMSE HU | MAE HU | delta_r RMS | Baseline MSE | Corrected MSE | corr(abs(delta_r), D_abs) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {step} | {psnr:.6f} | {delta_psnr:+.6f} | {ssim:.6f} | "
            "{rmse_hu:.4f} | {mae_hu:.4f} | {delta_r_rms:.6g} | "
            "{baseline_residual_mse:.6g} | {corrected_residual_mse:.6g} | "
            "{corr_abs_delta_r_D_abs:.5f} |".format(**row))
    lines += [
        "", "Promotion: **{}**. Strong signal: **{}**.".format(
            summary["flags"]["promotion"], summary["flags"]["strong_signal"]),
        "Failure interpretation: **{}**.".format(summary["flags"]["failure_interpretation"]),
        "Backbone state SHA256 unchanged: {}.".format(summary["backbone_state_sha256"]),
        "Original Direct Concat checkpoint SHA256 unchanged: {}.".format(CHECKPOINT_PIN),
        "No L506, external-memory, or neighbor-NDCT reads.",
        "Full L333 performed only if promoted; paired analysis is within-stack stability.",
        "",
    ]
    (HERE / "CROSS_SLICE_ERROR_ADAPTER_V1_REPORT.md").write_text("\n".join(lines))



def history_record(step, metrics, row, state):
    return {
        "step": step, **metrics,
        "direct_concat_psnr": BASE_PSNR,
        "delta_psnr": metrics["psnr"] - BASE_PSNR,
        "train_loss": row["train_loss"] if row else None,
        "gradient_norm_pre_clip": row["gradient_norm_pre_clip"] if row else None,
        "learning_rate": state.cfg["adapter_training"]["learning_rate"],
        "adapter_parameter_count": usage(state.model)["adapter_parameter_count"],
        "peak_vram_bytes": state.peak_vram_bytes,
        "seconds_per_step": state.train_seconds / step if step else 0.0,
        "train_stream_sha256": state.stream_digest.hex(),
    }


def train(device, resume=None):
    cfg, base = config()
    preflight = json.loads((HERE / "CONTROL_REPRO.json").read_text())
    smoked = json.loads((HERE / "SMOKE_RESULT.json").read_text())
    assert preflight["status"] == "CONTROL_PASS" and smoked["status"] == "SMOKE_PASS"
    assert preflight["source_manifest"] == smoked["source_manifest"] == source_manifest()
    out = HERE / "runs" / ARM
    if resume:
        assert out.is_dir()
    else:
        out.mkdir(parents=True, exist_ok=False)
        (out / "checkpoints").mkdir()
    queue = {"status": "RUNNING", "start_unix": time.time(), "max_steps": 6000,
             "source_manifest": source_manifest(), "checkpoint_sha256": CHECKPOINT_PIN}
    write_json(HERE / "queue_status.json", queue)
    state = None
    try:
        state = State(device)
        assert state.initial_adapter_hash == smoked["initial_adapter_sha256"]
        assert state.initial_rng_sha256 == smoked["initial_rng_sha256"]
        assert state.backbone_hash == preflight["backbone_state_sha256"]
        history = []
        checkpoint_manifest = {"arm": ARM, "matched": [], "latest": None}
        if resume:
            state.resume(resume)
            with (out / "validation_history.csv").open(newline="") as f:
                history = list(csv.DictReader(f))
            checkpoint_manifest = json.loads((out / "checkpoint_manifest.json").read_text())
        else:
            write_json(out / "CONFIG.json", cfg)
            write_json(out / "INITIALIZATION_AUDIT.json", preflight)
            write_json(out / "provenance.json", {
                "git_parent": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip(),
                "source_manifest": state.manifest, "python": sys.version,
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(state.device),
                "seed": cfg["adapter_training"]["seed"],
                "fast_indices": list(FAST_INDICES),
                "frozen_checkpoint_sha256": CHECKPOINT_PIN,
            })
            history.append(history_record(0, preflight["adapter_step0_fast64"], None, state))
            csv_write(out / "validation_history.csv", history)
        with state.guard:
            mode = "a" if resume else "x"
            with (out / "train.jsonl").open(mode, buffering=1) as log:
                while state.step < 6000:
                    row = state.update()
                    log.write(json.dumps(row, allow_nan=False) + "\n")
                    if state.step % 100 == 0:
                        write_json(out / "progress.json",
                                   {"status": "RUNNING", "step": state.step, "stop": 6000})
                        print("TRAIN", state.step, row["train_loss"], flush=True)
                    if state.step in STEPS[1:]:
                        before = rng_hash(rng_state())
                        state.sync_eval_ema()
                        metrics = validate_adapter(
                            state.eval_model, state.val_data, base, state.device,
                            out / ("validation_step_%d.json" % state.step))
                        assert rng_hash(rng_state()) == before
                        assert metrics["slices"] == 64 and metrics["weights"] == "ema"
                        references = reference()
                        if state.step in references:
                            assert state.stream_digest.hex() == references[state.step]["train_stream_sha256"]
                        record = history_record(state.step, metrics, row, state)
                        assert all(math.isfinite(record[k]) for k in
                                   ("psnr", "ssim", "rmse_hu", "mae_hu", "delta_psnr",
                                    "delta_r_rms", "baseline_residual_mse",
                                    "corrected_residual_mse", "corr_abs_delta_r_D_abs"))
                        history.append(record)
                        csv_write(out / "validation_history.csv", history)
                        print("VALIDATION", state.step, record["psnr"], record["delta_psnr"], flush=True)
                        checkpoint = out / "checkpoints" / ("step_%d.pt" % state.step)
                        atomic_save(checkpoint, state.payload())
                        checkpoint_manifest["matched"].append({
                            "step": state.step, "sha256": sha256(checkpoint)})
                        checkpoint_manifest["latest"] = {
                            "step": state.step, "sha256": sha256(checkpoint)}
                        write_json(out / "checkpoint_manifest.json", checkpoint_manifest)
        access = state.guard.report()
        check_access(access, allow_training=True)
        assert [int(r["step"]) for r in history] == list(STEPS)
        assert state.step == state.ema.updates == 6000
        assert state_hash(state.model.model.unet0.base.state_dict()) == state.backbone_hash
        assert sha256(DIRECT_CHECKPOINT) == CHECKPOINT_PIN
        assert state.stream_digest.hex() == reference()[6000]["train_stream_sha256"]
        deltas = {int(r["step"]): float(r["delta_psnr"]) for r in history}
        promotion = (all(deltas[s] > 0 for s in (2000, 4000, 6000))
                     and deltas[6000] >= 0.03)
        strong = promotion and deltas[6000] >= 0.05
        last = history[-1]
        if float(last["delta_r_rms"]) < 1e-5 and float(last["delta_psnr"]) <= 0:
            interpretation = "CROSS_SLICE_RELATION_NOT_ADDITIVE"
        elif float(last["delta_r_rms"]) >= 1e-5 and float(last["delta_psnr"]) <= 0:
            interpretation = "ADAPTER_ACTIVE_BUT_NOT_USEFUL"
        else:
            interpretation = "LATE_POSITIVE_SIGNAL" if promotion else "INCONCLUSIVE"
        full = full_l333(state, base) if promotion else None
        flags = {"promotion": "PROMOTE_FULL_L333" if promotion else "NO_PROMOTION",
                 "strong_signal": bool(strong), "failure_interpretation": interpretation,
                 "delta_r_active_threshold_rms": 1e-5}
        state.sync_eval_ema()
        usage_summary = {
            **usage(state.model), "ema_adapter": usage(state.eval_model),
            "delta_r_rms_at_6000": last["delta_r_rms"],
            "mean_abs_delta_r_at_6000": last["mean_abs_delta_r"],
            "p95_abs_delta_r_at_6000": last["p95_abs_delta_r"],
            "corr_abs_delta_r_D_abs_at_6000": last["corr_abs_delta_r_D_abs"],
            "gradient_audit": state.gradient_audit,
        }
        write_json(out / "ADAPTER_USAGE_SUMMARY.json", usage_summary)
        write_json(out / "TRAIN_STREAM_AUDIT.json", {
            "samples": 12000, "stream_sha256": state.stream_digest.hex(),
            "archived_direct_sha256": reference()[6000]["train_stream_sha256"],
            "matched": True})
        write_json(out / "DATA_ACCESS_AUDIT.json", access)
        summary = {
            "status": "COMPLETE", "optimizer_steps": 6000, "history": history,
            "flags": flags, "full_l333": full,
            "train_seconds": state.train_seconds,
            "seconds_per_step": state.train_seconds / 6000,
            "peak_vram_bytes": state.peak_vram_bytes,
            "nonfinite_count": state.nonfinite_count,
            "gradient_min": state.gradient_min,
            "gradient_max": state.gradient_max,
            "gradient_audit": state.gradient_audit,
            "adapter_usage": usage_summary,
            "backbone_state_sha256": state.backbone_hash,
            "direct_checkpoint_sha256": CHECKPOINT_PIN,
            "data_access": access, "source_manifest": state.manifest,
            "initial_adapter_sha256": state.initial_adapter_hash,
            "initial_rng_sha256": state.initial_rng_sha256,
        }
        write_json(out / "final_summary.json", summary)
        report(summary)
        final_audit = {
            "status": "PASS", "training_steps": 6000,
            "control_fast64_psnr": preflight["reproduced_direct_fast64"]["psnr"],
            "step0_fast64_psnr": history[0]["psnr"],
            "smoke_status": smoked["status"],
            "frozen_backbone_hash_before_after": state.backbone_hash,
            "checkpoint_sha256_before_after": CHECKPOINT_PIN,
            "formal_source_sha256": sha256(ROOT / "src/rddm_rleb.py"),
            "adapter_only_optimizer": True, "backbone_requires_grad_false": True,
            "training_stream_matched": True, "nonfinite_count": state.nonfinite_count,
            "promotion": flags["promotion"],
            "L506_opens": access["L506_opens"],
            "external_memory_opens": access["external_memory_opens"],
            "neighbor_target_opens": access["neighbor_target_opens"],
            "blocked_attempts": access["blocked_attempts"],
            "training_access": access, "control_access": preflight["data_access"],
            "full_access": full["data_access"] if full else None,
        }
        assert final_audit["formal_source_sha256"] == SOURCE_PIN
        assert all(final_audit[k] == 0 for k in
                   ("L506_opens", "external_memory_opens", "neighbor_target_opens", "blocked_attempts"))
        write_json(HERE / "FINAL_AUDIT.json", final_audit)
        light = HERE / ARM
        light.mkdir(exist_ok=True)
        for name in ("CONFIG.json", "INITIALIZATION_AUDIT.json", "provenance.json",
                     "TRAIN_STREAM_AUDIT.json", "DATA_ACCESS_AUDIT.json",
                     "validation_history.csv", "ADAPTER_USAGE_SUMMARY.json",
                     "final_summary.json", "checkpoint_manifest.json"):
            source = out / name
            if source.is_file():
                (light / name).write_bytes(source.read_bytes())
        write_json(out / "progress.json", {"status": "COMPLETE", "step": 6000, "stop": 6000})
        queue["status"] = "COMPLETE"
        queue["finished_unix"] = time.time()
        write_json(HERE / "queue_status.json", queue)
        print("COMPLETE", flags, flush=True)
    except BaseException as exc:
        if state is not None:
            write_json(out / "DATA_ACCESS_AUDIT.json", state.guard.report())
            write_json(out / "progress.json", {"status": "FAILED", "step": state.step,
                                               "error": repr(exc)})
        queue["status"] = "FAILED"
        queue["error"] = repr(exc)
        queue["traceback"] = traceback.format_exc()
        queue["finished_unix"] = time.time()
        write_json(HERE / "queue_status.json", queue)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("control", "smoke", "train"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume")
    args = parser.parse_args()
    HERE.mkdir(parents=True, exist_ok=True)
    if args.mode == "control":
        control(args.device)
    elif args.mode == "smoke":
        smoke(args.device)
    else:
        train(args.device, args.resume)


if __name__ == "__main__":
    main()
