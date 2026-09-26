"""Independent state, exact D1@2k fork, and guarded Fast64 training."""
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import torch

from src.rddm_rleb import build_baseline
from context_2p5d.model import DirectUnet, state_hash, training_loss
from context_2p5d.data import ContextPairs, ContextBatchStream
from mayo2016_smallset_v1.runtime import seed_everything, rng_state, restore_rng, atomic_save
from anatomy_controlnet_v1.model import MAIN_NAMES, ENCODER_NAMES, parameter_counts, trainable_parameters
from anatomy_controlnet_v1.runtime import rng_hash, substate
from prosac_add_fusion_v1.runtime import FrozenParameterEMA
from .model import FourArmControlledUnet
from .protocol import config, frozen_config, DIRECT_CK, DIRECT_SHA


def load_direct(device):
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


def construct(arm, device="cpu", check_step_zero=True):
    reference = load_direct("cpu")
    canonical = rng_state()
    main = reference.model.unet0
    main_hash = state_hash(main.state_dict())
    encoder_hash = state_hash(substate(main, ENCODER_NAMES))
    candidate = copy.deepcopy(reference)
    candidate.model.unet0 = FourArmControlledUnet(main, arm)
    u = candidate.model.unet0
    assert u.main_state_hash() == main_hash
    assert state_hash(substate(u.control, ENCODER_NAMES)) == encoder_hash
    assert all(not p.requires_grad for p in u.structure.med.parameters())
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
            u0, u1 = reference.model.unet0, candidate.model.unet0
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
            assert difference <= 1.e-6, difference
        del reference
    restore_rng(canonical)
    zero_norms = [
        {"weight_norm": float(z.weight.detach().norm()), "bias_norm": float(z.bias.detach().norm())}
        for z in u.zero_convs
    ]
    assert all(v["weight_norm"] == v["bias_norm"] == 0 for v in zero_norms)
    return candidate, {
        "arm": arm, "input_size": u.structure.input_size,
        "direct_checkpoint_sha256": DIRECT_SHA,
        "main_state_hash": main_hash, "control_copy_hash": encoder_hash,
        "zero_conv_norms": zero_norms, "step_zero_max_abs_difference": difference,
        "step_zero_equivalent": difference is not None and difference <= 1.e-6,
        "canonical_training_rng_restored": True,
        "canonical_training_rng_sha256": initial_rng,
        "meddino_frozen_at_initialization": True,
        **parameter_counts(candidate),
    }


def audit_state_hash(state):
    """Dtype-safe diagnostic hash; never alters model or checkpoint tensors."""
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update(name.encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(str(tensor.dtype).encode())
        h.update(tensor.detach().to(dtype=torch.float32).cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def med_block_hash(med, start, stop):
    return audit_state_hash({f"blocks.{i}.{k}": v
                             for i in range(start, stop)
                             for k, v in med.blocks[i].state_dict().items()})


class TrainState:
    def __init__(self, arm, device, guard):
        self.arm = arm
        self.device = torch.device(device)
        self.cfg = config(arm)
        torch.set_num_threads(2)
        self.data = ContextPairs(frozen_config(), "train")
        self.validation = ContextPairs(frozen_config(), "validation", "FAST")
        self.data.bind_guard(guard)
        self.validation.bind_guard(guard)
        assert len(self.data) == 1145 and len(self.validation) == 64
        self.stream = ContextBatchStream(self.data, self.cfg["seed"])
        self.model, self.init_audit = construct(arm, self.device)
        self.main_hash = self.init_audit["main_state_hash"]
        self.initial_rng_sha256 = rng_hash(rng_state())
        assert self.initial_rng_sha256 == self.init_audit["canonical_training_rng_sha256"]
        tr = self.cfg["training"]
        self.optimizer = torch.optim.Adam(
            trainable_parameters(self.model), lr=tr["learning_rate"],
            betas=tuple(tr["betas"]), weight_decay=tr["weight_decay"])
        self.ema = FrozenParameterEMA(self.model, self.cfg["ema"]["decay"])
        self.step = 0
        self.nonfinite_count = 0
        self.train_seconds = 0.
        self.fork_train_seconds = 0.
        self.peak_vram = 0
        self.stream_digest = hashlib.sha256(b"").digest()
        self.last = None
        self.fork_audit = None
        self.finetune_initial = None
        self.grad_audit = {"count": 0, "rms_sum": 0., "norm_sum": 0., "nonzero_count": 0}
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def fork_d1(self, path):
        assert self.arm == "D4_MULTI_FT_512" and self.step == 0
        ck = torch.load(str(path), map_location="cpu", weights_only=False)
        assert ck["arm"] == "D1_MULTI_512" and ck["optimizer_step"] == 2000
        assert ck["ema_updates"] == 2000 and ck["main_hash"] == self.main_hash
        assert ck["initial_rng_sha256"] == self.initial_rng_sha256
        assert ck["config"] == config("D1_MULTI_512")
        self.model.load_state_dict(ck["model_state_dict"], strict=True)
        self.ema.model.load_state_dict(ck["ema_state_dict"], strict=True)
        assert audit_state_hash(self.model.state_dict()) == audit_state_hash(ck["model_state_dict"])
        assert audit_state_hash(self.ema.model.state_dict()) == audit_state_hash(ck["ema_state_dict"])
        self.optimizer.load_state_dict(ck["optimizer_state_dict"])
        self.stream.load_state_dict(ck["stream"])
        self.step = ck["optimizer_step"]
        self.ema.updates = ck["ema_updates"]
        self.stream_digest = bytes.fromhex(ck["stream_sha256"])
        self.train_seconds = ck["train_seconds"]
        self.peak_vram = ck["peak_vram"]
        self.last = ck["last"]
        med = self.model.model.unet0.structure.med
        before_0_7 = med_block_hash(med, 0, 8)
        before_8_11 = med_block_hash(med, 8, 12)
        self.finetune_initial = {f"{i}.{name}": p.detach().cpu().clone()
                                 for i in range(8, 12)
                                 for name, p in med.blocks[i].named_parameters()}
        initial_norm = math.sqrt(sum(float(v.double().square().sum())
                                     for v in self.finetune_initial.values()))
        reference = copy.deepcopy(self.model).eval()
        self.model.eval()
        with torch.random.fork_rng(devices=[self.device.index or 0] if self.device.type == "cuda" else []), torch.no_grad():
            torch.manual_seed(127)
            if self.device.type == "cuda":
                torch.cuda.manual_seed(127)
            x = torch.randn(1, 2, 512, 512, device=self.device)
            p = torch.randn(1, 1, 512, 512, device=self.device)
            n = torch.randn(1, 1, 512, 512, device=self.device)
            t = torch.tensor([10.], device=self.device)
            a, b = reference.model.unet0, self.model.model.unet0
            a.set_neighbors(p, n); b.set_neighbors(p, n)
            try:
                difference = float((a(x, t) - b(x, t)).abs().max())
            finally:
                a.clear_neighbors(); b.clear_neighbors()
        del reference
        assert difference <= 1.e-6
        self.model.model.unet0.structure.enable_finetune()
        med_params = [p for block in med.blocks[8:12] for p in block.parameters()]
        assert med_params and all(p.requires_grad for p in med_params)
        assert all(not p.requires_grad for block in med.blocks[:8] for p in block.parameters())
        assert all(not p.requires_grad for p in med.patch_embed.parameters())
        self.optimizer.add_param_group({
            "params": med_params, "lr": 3.e-5,
            "betas": tuple(self.cfg["training"]["betas"]),
            "weight_decay": self.cfg["training"]["weight_decay"],
        })
        self.fork_audit = {
            "fork_step": 2000, "output_max_abs_difference": difference,
            "blocks_0_7_initial_hash": before_0_7,
            "blocks_8_11_initial_hash": before_8_11,
            "blocks_8_11_initial_parameter_norm": initial_norm,
            "D1_model_state_hash_at_fork": audit_state_hash(ck["model_state_dict"]),
            "D1_ema_state_hash_at_fork": audit_state_hash(ck["ema_state_dict"]),
            "d1_train_seconds_before_fork": self.train_seconds,
            "d1_stream_sha256_at_fork": self.stream_digest.hex(),
            "existing_optimizer_state_restored": bool(self.optimizer.state),
            "rng_and_stream_restored": True,
        }
        restore_rng(ck["rng"])
        return self.fork_audit

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
            record = {k: meta[k] for k in ("patient_id", "slice_index", "horizontal_flip", "rotation_degrees")}
            self.stream_digest = hashlib.sha256(
                self.stream_digest + json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).digest()
        prev, center, next_, target = (v.to(self.device) for v in (prev, center, next_, target))
        self.model.train()
        u = self.model.model.unet0
        med = u.structure.med
        assert not med.training
        if self.arm == "D4_MULTI_FT_512":
            assert all(m.training for m in med.blocks[8:12])
            assert all(not m.training for m in med.blocks[:8])
        else:
            assert all(not p.requires_grad for p in med.parameters())
        self.optimizer.zero_grad(set_to_none=True)
        loss = training_loss(self.model, target, center, prev, next_)
        if not bool(torch.isfinite(loss)):
            self.nonfinite_count += 1
            raise FloatingPointError("nonfinite loss")
        loss.backward()
        for name in MAIN_NAMES:
            assert all(p.grad is None for p in getattr(u, name).parameters())
        assert all(p.grad is None for block in med.blocks[:8] for p in block.parameters())
        if self.arm != "D4_MULTI_FT_512":
            assert all(p.grad is None for p in med.parameters())
        else:
            med_grads = [p.grad for block in med.blocks[8:12] for p in block.parameters() if p.grad is not None]
            assert med_grads and all(bool(torch.isfinite(g).all()) for g in med_grads)
            med_norm = math.sqrt(sum(float(g.square().sum()) for g in med_grads))
            med_rms = math.sqrt(sum(float(g.square().sum()) for g in med_grads) /
                                sum(g.numel() for g in med_grads))
            self.grad_audit["count"] += 1
            self.grad_audit["norm_sum"] += med_norm
            self.grad_audit["rms_sum"] += med_rms
            self.grad_audit["nonzero_count"] += int(med_norm > 0)
        params = trainable_parameters(self.model)
        grads = [p.grad for p in params if p.grad is not None]
        if not grads or any(not bool(torch.isfinite(g).all()) for g in grads):
            self.nonfinite_count += 1
            raise FloatingPointError("nonfinite or missing trainable gradients")
        norm = float(torch.nn.utils.clip_grad_norm_(params, 1.))
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
        if self.arm == "D4_MULTI_FT_512":
            self.fork_train_seconds += elapsed
        self.last = {
            "step": self.step, "train_loss": float(loss.detach()),
            "gradient_norm_pre_clip": norm,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "meddino_learning_rate": self.optimizer.param_groups[1]["lr"] if len(self.optimizer.param_groups) > 1 else None,
            "seconds_per_step": elapsed, "peak_vram_bytes": self.peak_vram,
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
            "train_seconds": self.train_seconds,
            "fork_train_seconds": self.fork_train_seconds,
            "peak_vram": self.peak_vram,
            "nonfinite_count": self.nonfinite_count, "last": self.last,
            "fork_audit": self.fork_audit, "grad_audit": self.grad_audit,
        })
