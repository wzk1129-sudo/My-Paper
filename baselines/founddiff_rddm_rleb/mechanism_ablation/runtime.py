"""Matched optimization state. Data, augmentations, metrics, and control diffusion are frozen imports."""
import hashlib
import json
import math
import time
import torch
import torch.nn.functional as F
from fast_dev.data import FastPairs
from mayo2016_smallset_v1.data import BatchStream
from mayo2016_smallset_v1.runtime import EMA, seed_everything, rng_state, restore_rng
from .protocol import ARMS, experiment_config, make_model, state_hash, HERE

def direct_predict(model, ldct):
    assert ldct.ndim == 4 and ldct.shape[1:] == (1, 512, 512)
    x_input = ldct * 2 - 1
    x_in = torch.cat((x_input, x_input), dim=1)
    assert torch.equal(x_in[:, :1], x_in[:, 1:])
    fixed_time = torch.zeros(x_input.shape[0], device=x_input.device, dtype=x_input.dtype)
    assert torch.count_nonzero(fixed_time) == 0
    output = model(x_in, [fixed_time, fixed_time], None)[0]
    assert output.shape == x_input.shape
    return output

def direct_loss(model, ldct, target):
    residual = (ldct * 2 - 1) - (target * 2 - 1)
    prediction = direct_predict(model, ldct)
    return F.mse_loss(prediction, residual), prediction

def direct_restore(model, ldct):
    x_input = ldct * 2 - 1
    prediction = direct_predict(model, ldct)
    return ((x_input - prediction).clamp(-1, 1) + 1) * 0.5

class MatchedState:
    def __init__(self, arm, device):
        assert arm in ARMS
        self.cfg, self.data_cfg = experiment_config()
        self.arm = arm
        self.device = torch.device(device)
        torch.set_num_threads(self.cfg["training"]["cpu_threads"])
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        seed_everything(self.cfg["seed"])
        self.train_data = FastPairs(self.data_cfg, "train")
        self.validation_data = FastPairs(self.data_cfg, "validation", "FAST")
        assert len(self.train_data) == 1145 and len(self.validation_data) == 64
        self.stream = BatchStream(self.train_data, self.cfg["seed"])
        self.model = make_model(arm).to(self.device)
        self.backbone = self.model.model if arm == ARMS[0] else self.model
        self.initial_hash = state_hash(self.backbone)
        expected = json.loads((HERE / "INITIALIZATION_AUDIT.json").read_text())["control_unetres_sha256"]
        assert self.initial_hash == expected, "Initialization mismatch"
        assert sum(p.numel() for p in self.model.parameters()) == 13130945
        tr = self.cfg["training"]
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=tr["learning_rate"],
                                          betas=tuple(tr["betas"]), weight_decay=tr["weight_decay"])
        self.ema = EMA(self.model, self.cfg["ema"]["decay"])
        self.step = 0
        self.stream_digest = hashlib.sha256(b"").digest()
        self.elapsed_train_seconds = 0.
        self.peak_vram = 0
        self.last = None
        self.gradient_min = None
        self.gradient_max = None
        self.nonfinite_count = 0
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    @property
    def stream_sha256(self):
        return self.stream_digest.hex()

    def update(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        ldct, target, metadata = self.stream.next(2)
        assert all(m["patient_id"] in ("L067", "L143") for m in metadata)
        assert ldct.shape == target.shape == (2, 1, 512, 512)
        assert ldct.dtype == target.dtype == torch.float32
        assert torch.isfinite(ldct).all() and torch.isfinite(target).all()
        assert ldct.min() >= 0 and ldct.max() <= 1 and target.min() >= 0 and target.max() <= 1
        for m in metadata:
            record = {key: m[key] for key in ("patient_id", "slice_index", "horizontal_flip", "rotation_degrees")}
            self.stream_digest = hashlib.sha256(
                self.stream_digest + json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
            ).digest()
        ldct, target = ldct.to(self.device), target.to(self.device)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        if self.arm == ARMS[0]:
            loss = self.model([target, ldct])[0]
        else:
            loss, prediction = direct_loss(self.model, ldct, target)
            assert prediction.shape == (2, 1, 512, 512)
        if not torch.isfinite(loss):
            self.nonfinite_count += 1
            raise FloatingPointError("Nonfinite loss")
        loss.backward()
        missing = [n for n, p in self.model.named_parameters() if p.requires_grad and p.grad is None]
        if missing:
            raise RuntimeError("Missing gradients: " + repr(missing))
        norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0))
        if not math.isfinite(norm):
            self.nonfinite_count += 1
            raise FloatingPointError("Nonfinite gradient")
        self.gradient_min = norm if self.gradient_min is None else min(norm, self.gradient_min)
        self.gradient_max = norm if self.gradient_max is None else max(norm, self.gradient_max)
        self.optimizer.step()
        self.ema.update(self.model)
        self.step += 1
        assert self.ema.updates == self.step
        if any(not torch.isfinite(p).all() for p in self.model.parameters()):
            self.nonfinite_count += 1
            raise FloatingPointError("Nonfinite parameter")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            self.peak_vram = max(self.peak_vram, torch.cuda.max_memory_allocated(self.device))
        elapsed = time.perf_counter() - started
        self.elapsed_train_seconds += elapsed
        self.last = {"step": self.step, "train_loss": float(loss.detach()),
                     "gradient_norm_pre_clip": norm,
                     "learning_rate": self.optimizer.param_groups[0]["lr"],
                     "seconds_per_step": elapsed, "peak_vram_bytes": self.peak_vram,
                     "ema_updates": self.ema.updates, "samples": metadata}
        return self.last

    def checkpoint(self):
        return {"arm": self.arm, "optimizer_step": self.step,
                "model_state_dict": self.model.state_dict(),
                "ema_state_dict": self.ema.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "ema_updates": self.ema.updates, "rng": rng_state(),
                "data_stream": self.stream.state_dict(),
                "stream_digest_hex": self.stream_sha256,
                "config": self.cfg, "initial_hash": self.initial_hash,
                "elapsed_train_seconds": self.elapsed_train_seconds,
                "peak_vram_bytes": self.peak_vram,
                "gradient_min": self.gradient_min, "gradient_max": self.gradient_max,
                "nonfinite_count": self.nonfinite_count, "last": self.last}

    def resume(self, path):
        ck = torch.load(path, map_location="cpu")
        assert ck["config"] == self.cfg and ck["arm"] == self.arm
        assert ck["initial_hash"] == self.initial_hash
        assert ck["optimizer_step"] < 12000
        self.model.load_state_dict(ck["model_state_dict"], strict=True)
        self.ema.model.load_state_dict(ck["ema_state_dict"], strict=True)
        self.optimizer.load_state_dict(ck["optimizer_state_dict"])
        self.ema.updates = ck["ema_updates"]
        self.step = ck["optimizer_step"]
        assert self.ema.updates == self.step
        self.stream.load_state_dict(ck["data_stream"])
        self.stream_digest = bytes.fromhex(ck["stream_digest_hex"])
        self.elapsed_train_seconds = ck["elapsed_train_seconds"]
        self.peak_vram = ck["peak_vram_bytes"]
        self.gradient_min = ck["gradient_min"]
        self.gradient_max = ck["gradient_max"]
        self.nonfinite_count = ck["nonfinite_count"]
        self.last = ck["last"]
        restore_rng(ck["rng"])
