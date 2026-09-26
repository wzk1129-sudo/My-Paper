"""Independent BCA train state; D1 stream and canonical Direct EMA initialization."""
import hashlib
import json
import math
import time

import torch

from anatomy_controlnet_v1.model import MAIN_NAMES, parameter_counts, trainable_parameters
from context_2p5d.model import state_hash, training_loss
from mayo2016_smallset_v1.runtime import rng_state, restore_rng, atomic_save
from anatomy_controlnet_v1.runtime import rng_hash
from meddino_four_arm_v1.runtime import TrainState, load_direct
from prosac_add_fusion_v1.runtime import FrozenParameterEMA
from .model import graft_bca
from .protocol import ARM, config


class BCATrainState(TrainState):
    def __init__(self, device, guard):
        super().__init__("D1_MULTI_512", device, guard)
        d1 = self.model
        canonical_rng = rng_state()
        provider = d1.model.unet0.structure
        del self.optimizer, self.ema
        candidate = load_direct("cpu")
        direct_hash = state_hash(candidate.model.unet0.state_dict())
        assert direct_hash == self.main_hash
        graft_bca(candidate.model.unet0, provider)
        self.model = candidate.to(self.device).eval()
        del d1, candidate
        restore_rng(canonical_rng)
        self.arm = ARM
        self.cfg = config()
        tr = self.cfg["training"]
        self.optimizer = torch.optim.Adam(
            trainable_parameters(self.model),
            lr=tr["learning_rate"],
            betas=tuple(tr["betas"]),
            weight_decay=tr["weight_decay"],
        )
        self.ema = FrozenParameterEMA(self.model, self.cfg["ema"]["decay"])
        u = self.model.model.unet0
        assert not hasattr(u, "control")
        assert all(p.requires_grad for name in MAIN_NAMES for p in getattr(u, name).parameters())
        assert all(not p.requires_grad for p in u.structure.med.parameters())
        assert float(u.bca128.alpha.detach()) == float(u.bca64.alpha.detach())
        self.initial_rng_sha256 = rng_hash(canonical_rng)
        self.init_audit = {
            "canonical_main_state_hash": direct_hash,
            "direct_checkpoint": "same canonical Direct 12k EMA as D1/E12",
            "warm_start_from_trained_D1_or_E12_6k": False,
            "alpha128_init": float(u.bca128.alpha.detach()),
            "alpha64_init": float(u.bca64.alpha.detach()),
            "meddino_frozen": True,
            "main_trainable": True,
            "parameter_counts": parameter_counts(self.model),
        }
        self.grad_usage = {"bca128": 0, "bca64": 0, "alpha128": 0, "alpha64": 0,
                           "main": 0, "structure": 0}
        self.last = None

    def update(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        prev, center, next_, target, metadata = self.stream.next(2)
        assert prev.shape == center.shape == next_.shape == target.shape == (2, 1, 512, 512)
        for tensor in (prev, center, next_, target):
            assert tensor.dtype == torch.float32
            assert bool(torch.isfinite(tensor).all())
            assert 0 <= float(tensor.min()) <= float(tensor.max()) <= 1
        for meta in metadata:
            assert meta["patient_id"] in ("L067", "L143")
            record = {k: meta[k] for k in ("patient_id", "slice_index", "horizontal_flip", "rotation_degrees")}
            self.stream_digest = hashlib.sha256(
                self.stream_digest + json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
            ).digest()
        prev, center, next_, target = (v.to(self.device) for v in (prev, center, next_, target))
        self.model.train()
        u = self.model.model.unet0
        assert not u.structure.med.training
        self.optimizer.zero_grad(set_to_none=True)
        loss = training_loss(self.model, target, center, prev, next_)
        if not bool(torch.isfinite(loss)):
            self.nonfinite_count += 1
            raise FloatingPointError("nonfinite loss")
        loss.backward()
        groups = {
            "bca128": list(u.bca128.parameters()),
            "bca64": list(u.bca64.parameters()),
            "alpha128": [u.bca128.alpha],
            "alpha64": [u.bca64.alpha],
            "main": [p for name in MAIN_NAMES for p in getattr(u, name).parameters()],
            "structure": [p for name, p in u.structure.named_parameters() if not name.startswith("med.")],
        }
        assert all(p.grad is None for p in u.structure.med.parameters())
        for name, group in groups.items():
            grads = [p.grad for p in group if p.grad is not None]
            if not grads or any(not bool(torch.isfinite(g).all()) for g in grads):
                self.nonfinite_count += 1
                raise FloatingPointError("nonfinite/missing " + name + " gradient")
            if any(bool(g.abs().max() > 0) for g in grads):
                self.grad_usage[name] += 1
        params = trainable_parameters(self.model)
        norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        if not math.isfinite(norm):
            self.nonfinite_count += 1
            raise FloatingPointError("nonfinite gradient norm")
        self.optimizer.step()
        self.ema.update(self.model)
        self.step += 1
        if any(not bool(torch.isfinite(p).all()) for p in params):
            self.nonfinite_count += 1
            raise FloatingPointError("nonfinite parameter")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            self.peak_vram = max(self.peak_vram, torch.cuda.max_memory_allocated(self.device))
        elapsed = time.perf_counter() - started
        self.train_seconds += elapsed
        self.last = {
            "step": self.step, "train_loss": float(loss.detach()),
            "gradient_norm_pre_clip": norm,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "seconds_per_step": elapsed, "peak_vram_bytes": self.peak_vram,
            "train_stream_sha256": self.stream_digest.hex(),
        }
        return self.last

    def checkpoint(self, path):
        atomic_save(path, {
            "arm": ARM, "optimizer_step": self.step,
            "model_state_dict": self.model.state_dict(),
            "ema_state_dict": self.ema.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "ema_updates": self.ema.updates,
            "rng": rng_state(), "stream": self.stream.state_dict(),
            "stream_sha256": self.stream_digest.hex(),
            "config": self.cfg, "main_hash": self.main_hash,
            "initialization": self.init_audit,
            "train_seconds": self.train_seconds, "peak_vram": self.peak_vram,
            "nonfinite_count": self.nonfinite_count,
            "grad_usage": self.grad_usage, "last": self.last,
        })

    def resume(self, path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        assert ck["arm"] == ARM and ck["config"] == self.cfg
        self.model.load_state_dict(ck["model_state_dict"], strict=True)
        self.ema.model.load_state_dict(ck["ema_state_dict"], strict=True)
        self.ema.updates = ck["ema_updates"]
        self.optimizer.load_state_dict(ck["optimizer_state_dict"])
        self.stream.load_state_dict(ck["stream"])
        restore_rng(ck["rng"])
        self.step = ck["optimizer_step"]
        self.stream_digest = bytes.fromhex(ck["stream_sha256"])
        self.train_seconds = ck["train_seconds"]
        self.peak_vram = ck["peak_vram"]
        self.nonfinite_count = ck["nonfinite_count"]
        self.grad_usage = ck["grad_usage"]
        self.last = ck["last"]
        return self
