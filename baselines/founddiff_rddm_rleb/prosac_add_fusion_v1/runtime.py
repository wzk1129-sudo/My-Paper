"""Independent, resumable C3/C4 training with frozen Direct and MedDINO."""
import copy
import hashlib
import json
import math
import time

import torch

from src.rddm_rleb import build_baseline
from context_2p5d.model import DirectUnet, state_hash, training_loss
from context_2p5d.data import ContextPairs, ContextBatchStream
from mayo2016_smallset_v1.runtime import seed_everything, rng_state, restore_rng, atomic_save
from anatomy_controlnet_v1.model import MAIN_NAMES, ENCODER_NAMES, parameter_counts, trainable_parameters
from anatomy_controlnet_v1.runtime import rng_hash, substate
from .model import ProsacControlledUnet
from .protocol import config, frozen_config, DIRECT_CK, DIRECT_SHA


def load_direct(device):
    frozen_config()
    seed_everything(20260829)
    diffusion = build_baseline()
    diffusion.model.unet0 = DirectUnet()
    ck = torch.load(str(DIRECT_CK), map_location="cpu", weights_only=False)
    assert ck["arm"] == "2P5D_DIRECT_CONCAT" and ck["optimizer_step"] == 12000
    assert ck["ema_updates"] == 12000
    diffusion.load_state_dict(ck["ema_state_dict"], strict=True)
    del ck
    assert diffusion.objective == "pred_res"
    assert diffusion.num_timesteps == 1000
    assert diffusion.sampling_timesteps == 2 and diffusion.sum_scale == .01
    return diffusion.to(device).eval()


def construct(arm, input_size, device="cpu", check_step_zero=True):
    reference = load_direct("cpu")
    canonical = rng_state()
    main = reference.model.unet0
    main_hash = state_hash(main.state_dict())
    encoder_hash = state_hash(substate(main, ENCODER_NAMES))
    candidate = copy.deepcopy(reference)
    candidate.model.unet0 = ProsacControlledUnet(main, arm, input_size)
    u = candidate.model.unet0
    assert u.main_state_hash() == main_hash
    assert state_hash(substate(u.control, ENCODER_NAMES)) == encoder_hash
    assert all(not p.requires_grad for p in u.structure.med.parameters())
    assert not u.structure.med.training
    restore_rng(canonical)
    initial_rng = rng_hash(rng_state())
    device = torch.device(device)
    candidate = candidate.to(device).eval()
    difference = None
    if check_step_zero:
        reference = reference.to(device).eval()
        with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []), torch.no_grad():
            torch.manual_seed(109)
            if device.type == "cuda":
                torch.cuda.manual_seed(109)
            x = torch.randn(1, 2, 512, 512, device=device)
            p = torch.randn(1, 1, 512, 512, device=device)
            n = torch.randn(1, 1, 512, 512, device=device)
            t = torch.tensor([10.], device=device)
            u0 = reference.model.unet0
            u1 = candidate.model.unet0
            u0.set_neighbors(p, n)
            u1.set_neighbors(p, n)
            try:
                a = u0(x, t)
                b = u1(x, t)
            finally:
                u0.clear_neighbors()
                u1.clear_neighbors()
            difference = float((a - b).abs().max())
            assert a.shape == b.shape == (1, 1, 512, 512)
            assert difference <= 1e-6, difference
        del reference
    restore_rng(canonical)
    zero_norms = [
        {"weight_norm": float(z.weight.detach().norm()),
         "bias_norm": float(z.bias.detach().norm())}
        for z in u.zero_convs
    ]
    assert all(v["weight_norm"] == v["bias_norm"] == 0 for v in zero_norms)
    audit = {
        "arm": arm, "input_size": input_size,
        "direct_checkpoint_sha256": DIRECT_SHA,
        "main_state_hash": main_hash,
        "control_copy_hash": encoder_hash,
        "zero_conv_norms": zero_norms,
        "step_zero_max_abs_difference": difference,
        "step_zero_equivalent": difference is not None and difference <= 1e-6,
        "canonical_training_rng_restored": True,
        "canonical_training_rng_sha256": initial_rng,
        "meddino_frozen": True, "meddino_eval": True,
        **parameter_counts(candidate),
    }
    return candidate, audit


class FrozenParameterEMA:
    def __init__(self, model, decay=.995):
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model):
        source = dict(model.named_parameters())
        for name, target in self.model.named_parameters():
            original = source[name]
            if original.requires_grad:
                target.mul_(self.decay).add_(original, alpha=1. - self.decay)
            else:
                target.copy_(original)
        buffers = dict(model.named_buffers())
        for name, target in self.model.named_buffers():
            target.copy_(buffers[name])
        self.updates += 1


class TrainState:
    def __init__(self, arm, input_size, device, guard):
        self.arm = arm
        self.device = torch.device(device)
        self.cfg = config(arm, input_size)
        torch.set_num_threads(2)
        self.data = ContextPairs(frozen_config(), "train")
        self.validation = ContextPairs(frozen_config(), "validation", "FAST")
        self.data.bind_guard(guard)
        self.validation.bind_guard(guard)
        assert len(self.data) == 1145 and len(self.validation) == 64
        self.stream = ContextBatchStream(self.data, self.cfg["seed"])
        self.model, self.init_audit = construct(arm, input_size, self.device)
        self.main_hash = self.init_audit["main_state_hash"]
        self.initial_rng_sha256 = rng_hash(rng_state())
        assert self.initial_rng_sha256 == self.init_audit["canonical_training_rng_sha256"]
        tr = self.cfg["training"]
        self.optimizer = torch.optim.Adam(
            trainable_parameters(self.model), lr=tr["learning_rate"],
            betas=tuple(tr["betas"]), weight_decay=tr["weight_decay"])
        self.ema = FrozenParameterEMA(self.model, self.cfg["ema"]["decay"])
        assert self.ema.model.model.unet0.main_state_hash() == self.main_hash
        self.step = 0
        self.nonfinite_count = 0
        self.train_seconds = 0.
        self.peak_vram = 0
        self.stream_digest = hashlib.sha256(b"").digest()
        self.last = None
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
            record = {k: meta[k] for k in
                      ("patient_id", "slice_index", "horizontal_flip", "rotation_degrees")}
            self.stream_digest = hashlib.sha256(
                self.stream_digest +
                json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).digest()
        prev = prev.to(self.device)
        center = center.to(self.device)
        next_ = next_.to(self.device)
        target = target.to(self.device)
        pre_rng = rng_hash(rng_state()) if self.step in (0, 499, 999, 1999, 3999, 5999) else None
        self.model.train()
        assert not self.model.model.unet0.structure.med.training
        self.optimizer.zero_grad(set_to_none=True)
        loss = training_loss(self.model, target, center, prev, next_)
        if not bool(torch.isfinite(loss)):
            self.nonfinite_count += 1
            raise FloatingPointError("nonfinite loss")
        loss.backward()
        u = self.model.model.unet0
        for name in MAIN_NAMES:
            assert all(p.grad is None for p in getattr(u, name).parameters())
        assert all(p.grad is None for p in u.structure.med.parameters())
        grads = [p.grad for p in trainable_parameters(self.model) if p.grad is not None]
        if not grads or any(not bool(torch.isfinite(g).all()) for g in grads):
            self.nonfinite_count += 1
            raise FloatingPointError("nonfinite or missing trainable gradients")
        norm = float(torch.nn.utils.clip_grad_norm_(trainable_parameters(self.model), 1.))
        if not math.isfinite(norm):
            self.nonfinite_count += 1
            raise FloatingPointError("nonfinite gradient norm")
        self.optimizer.step()
        self.ema.update(self.model)
        self.step += 1
        if any(not bool(torch.isfinite(p).all()) for p in trainable_parameters(self.model)):
            self.nonfinite_count += 1
            raise FloatingPointError("nonfinite parameter")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            self.peak_vram = max(self.peak_vram, torch.cuda.max_memory_allocated(self.device))
            if self.peak_vram > 22 * 1024 ** 3:
                raise MemoryError("peak VRAM exceeded 22 GiB")
        elapsed = time.perf_counter() - started
        self.train_seconds += elapsed
        self.last = {
            "step": self.step, "train_loss": float(loss.detach()),
            "gradient_norm_pre_clip": norm,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "seconds_per_step": elapsed, "peak_vram_bytes": self.peak_vram,
            "pre_step_rng_sha256": pre_rng,
            "stream_sha256": self.stream_digest.hex(),
        }
        return self.last

    def checkpoint(self, path):
        atomic_save(path, {
            "arm": self.arm, "optimizer_step": self.step,
            "model_state_dict": self.model.state_dict(),
            "ema_state_dict": self.ema.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "ema_updates": self.ema.updates, "rng": rng_state(),
            "stream": self.stream.state_dict(),
            "stream_sha256": self.stream_digest.hex(),
            "config": self.cfg, "main_hash": self.main_hash,
            "initial_rng_sha256": self.initial_rng_sha256,
            "train_seconds": self.train_seconds, "peak_vram": self.peak_vram,
            "nonfinite_count": self.nonfinite_count, "last": self.last,
        })

    def restore(self, path):
        ck = torch.load(str(path), map_location="cpu", weights_only=False)
        assert ck["arm"] == self.arm
        assert ck["config"] == self.cfg
        assert ck["main_hash"] == self.main_hash
        assert ck["initial_rng_sha256"] == self.initial_rng_sha256
        self.model.load_state_dict(ck["model_state_dict"], strict=True)
        self.ema.model.load_state_dict(ck["ema_state_dict"], strict=True)
        self.optimizer.load_state_dict(ck["optimizer_state_dict"])
        self.ema.updates = ck["ema_updates"]
        self.stream.load_state_dict(ck["stream"])
        self.step = ck["optimizer_step"]
        self.nonfinite_count = ck["nonfinite_count"]
        self.train_seconds = ck["train_seconds"]
        self.peak_vram = ck["peak_vram"]
        self.stream_digest = bytes.fromhex(ck["stream_sha256"])
        self.last = ck["last"]
        restore_rng(ck["rng"])
        assert self.ema.updates == self.step
        assert self.model.model.unet0.main_state_hash() == self.main_hash
        assert self.ema.model.model.unet0.main_state_hash() == self.main_hash

