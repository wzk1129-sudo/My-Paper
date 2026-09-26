"""Fresh D1 initialization with only 9,228 zero-initialized SALF parameters."""
import copy
import math

import torch

from mayo2016_smallset_v1.runtime import rng_state, restore_rng
from anatomy_controlnet_v1.model import parameter_counts, trainable_parameters
from prosac_add_fusion_v1.runtime import FrozenParameterEMA
from meddino_four_arm_v1.runtime import TrainState
from .model import SALFProvider
from .protocol import ARM, config


class SALFTrainState(TrainState):
    def __init__(self, device, guard):
        super().__init__("D1_MULTI_512", device, guard)
        d1 = self.model
        d1_counts = parameter_counts(d1)
        canonical_rng = rng_state()
        del self.ema, self.optimizer
        candidate = copy.deepcopy(d1)
        u = candidate.model.unet0
        u.structure = SALFProvider(u.structure).to(self.device)
        u.arm = ARM
        restore_rng(canonical_rng)
        old_state = d1.state_dict()
        new_state = candidate.state_dict()
        shared = set(old_state)
        new_keys = sorted(set(new_state) - shared)
        assert all(torch.equal(old_state[key], new_state[key]) for key in shared)
        assert len(new_keys) == 24
        assert sum(new_state[key].numel() for key in new_keys) == 9228
        assert all(".structure.scorers." in key for key in new_keys)
        counts = parameter_counts(candidate)
        assert counts["total_parameters"] - d1_counts["total_parameters"] == 9228
        assert counts["trainable_parameters"] - d1_counts["trainable_parameters"] == 9228
        assert all(not p.requires_grad for p in u.structure.med.parameters())
        candidate.eval()
        d1.eval()
        with torch.random.fork_rng(devices=[self.device.index or 0] if self.device.type == "cuda" else []), torch.no_grad():
            torch.manual_seed(20260829)
            if self.device.type == "cuda":
                torch.cuda.manual_seed(20260829)
            x = torch.randn(1, 2, 512, 512, device=self.device)
            prev = torch.randn(1, 1, 512, 512, device=self.device)
            next_ = torch.randn(1, 1, 512, 512, device=self.device)
            time = torch.tensor([37.], device=self.device)
            d1u, e1u = d1.model.unet0, candidate.model.unet0
            d1u.set_neighbors(prev, next_)
            e1u.set_neighbors(prev, next_)
            try:
                left = d1u(x, time)
                right = e1u(x, time)
            finally:
                d1u.clear_neighbors()
                e1u.clear_neighbors()
            difference = float((left - right).abs().max())
        assert difference <= 1e-6, difference
        assert u.main_state_hash() == self.main_hash
        assert rng_state()["torch"].equal(canonical_rng["torch"])
        self.model = candidate
        self.arm = ARM
        self.cfg = config()
        self.optimizer = torch.optim.Adam(
            trainable_parameters(self.model),
            lr=self.cfg["training"]["learning_rate"],
            betas=tuple(self.cfg["training"]["betas"]),
            weight_decay=self.cfg["training"]["weight_decay"])
        assert len(self.optimizer.param_groups) == 1
        self.ema = FrozenParameterEMA(self.model, self.cfg["ema"]["decay"])
        self.salf_gradient_steps = 0
        self.salf_gradient_norm_sum = 0.0
        self.init_audit = {
            "arm": ARM,
            "fresh_d1_initialization": True,
            "D1_trained_6k_checkpoint_loaded_as_initialization": False,
            "shared_state_tensors_bitwise_identical": True,
            "shared_state_tensor_count": len(shared),
            "new_tensor_keys": new_keys,
            "added_parameters": 9228,
            "baseline_parameter_count": d1_counts,
            "E1_parameter_count": counts,
            "step_zero_max_abs_output_difference": difference,
            "step_zero_equivalent": True,
            "scorer_weights_zero": True,
            "scorer_biases_zero": True,
            "initial_softmax_weight": [0.25] * 4,
            "initial_effective_gate": [1.0] * 4,
            "main_hash": self.main_hash,
            "meddino_frozen": True,
            "canonical_training_rng_sha256": self.initial_rng_sha256,
        }
        del d1, old_state, new_state, left, right
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)

    def update(self):
        row = super().update()
        grads = [p.grad for p in self.model.model.unet0.structure.scorers.parameters()]
        assert grads and all(g is not None and bool(torch.isfinite(g).all()) for g in grads)
        norm = math.sqrt(sum(float(g.square().sum()) for g in grads))
        assert math.isfinite(norm)
        self.salf_gradient_steps += int(norm > 0)
        self.salf_gradient_norm_sum += norm
        row["salf_gradient_norm"] = norm
        return row
