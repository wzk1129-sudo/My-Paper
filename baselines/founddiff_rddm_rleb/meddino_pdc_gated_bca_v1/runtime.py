"""Independent PDC-gated BCA state; canonical Direct EMA initialization and D1 stream."""
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
from .model import graft_pdc, numeric_activity
from .protocol import ARM, config


class PDCTrainState(TrainState):
    def __init__(self, device, guard):
        super().__init__("D1_MULTI_512", device, guard)
        d1 = self.model
        canonical_rng = rng_state()
        provider = d1.model.unet0.structure
        del self.optimizer, self.ema
        candidate = load_direct("cpu")
        direct_hash = state_hash(candidate.model.unet0.state_dict())
        assert direct_hash == self.main_hash
        graft_pdc(candidate.model.unet0, provider)
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
        assert all(not p.requires_grad for p in u.structure.anatomy.med.parameters())
        assert float(u.structure.beta128.detach()) == float(u.structure.beta64.detach()) < 0.101
        self.initial_rng_sha256 = rng_hash(canonical_rng)
        added = sum(p.numel() for name, p in u.named_parameters()
                    if name.startswith(("structure.shape.", "structure.shape_proj",
                                        "structure.gate", "structure.beta")))
        self.init_audit = {
            "canonical_main_state_hash": direct_hash,
            "direct_checkpoint": "same canonical Direct 12k EMA as E12-BCA",
            "warm_start_from_trained_E12_BCA_6k": False,
            "alpha128_init": float(u.bca128.alpha.detach()),
            "alpha64_init": float(u.bca64.alpha.detach()),
            "beta128_init": float(u.structure.beta128.detach()),
            "beta64_init": float(u.structure.beta64.detach()),
            "meddino_frozen": True, "pdc_pretrained": False,
            "main_trainable": True,
            "parameter_counts": parameter_counts(self.model),
            "pdc_parameters": sum(p.numel() for p in u.structure.shape.parameters()),
            "projection_parameters": sum(p.numel() for p in u.structure.shape_proj128.parameters()) +
                                     sum(p.numel() for p in u.structure.shape_proj64.parameters()),
            "gate_parameters": sum(p.numel() for p in u.structure.gate128.parameters()) +
                               sum(p.numel() for p in u.structure.gate64.parameters()),
            "added_parameters_vs_E12_BCA": added,
        }
        self.grad_usage = {name: 0 for name in (
            "pdc", "shape_proj128", "shape_proj64", "gate128", "gate64",
            "beta128", "beta64", "bca128", "bca64", "alpha128", "alpha64",
            "main", "anatomy")}
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
        assert not u.structure.anatomy.med.training
        self.optimizer.zero_grad(set_to_none=True)
        loss = training_loss(self.model, target, center, prev, next_)
        if not bool(torch.isfinite(loss)):
            self.nonfinite_count += 1
            raise FloatingPointError("nonfinite loss")
        loss.backward()
        s = u.structure
        groups = {
            "pdc": list(s.shape.parameters()),
            "shape_proj128": list(s.shape_proj128.parameters()),
            "shape_proj64": list(s.shape_proj64.parameters()),
            "gate128": list(s.gate128.parameters()),
            "gate64": list(s.gate64.parameters()),
            "beta128": [s.beta128], "beta64": [s.beta64],
            "bca128": list(u.bca128.parameters()),
            "bca64": list(u.bca64.parameters()),
            "alpha128": [u.bca128.alpha], "alpha64": [u.bca64.alpha],
            "main": [p for name in MAIN_NAMES for p in getattr(u, name).parameters()],
            "anatomy": [p for p in s.anatomy.parameters() if p.requires_grad],
        }
        assert all(p.grad is None for p in s.anatomy.med.parameters())
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
