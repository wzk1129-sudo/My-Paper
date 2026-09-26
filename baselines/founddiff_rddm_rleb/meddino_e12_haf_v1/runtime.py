"""Independent E12 initialization from canonical Direct, with exact D1 parity."""
import copy
import math
import torch

from mayo2016_smallset_v1.runtime import rng_state, restore_rng
from anatomy_controlnet_v1.model import parameter_counts, trainable_parameters
from prosac_add_fusion_v1.runtime import FrozenParameterEMA
from meddino_four_arm_v1.runtime import TrainState
from .model import graft_e12
from .protocol import ARM, config


def group_gradients(unet):
    groups = {
        "SALF": unet.structure.scorers,
        "E2_64to128_zero": unet.structure.up64_128.zero,
        "E2_128to256_zero": unet.structure.up128_256.zero,
        "local256_zero": unet.control.local256.zero,
        "cross128_zero": unet.control.cross128.zero,
        "cross64_zero": unet.control.cross64.zero,
        "cross128_QKV": (unet.control.cross128.q, unet.control.cross128.k,
                         unet.control.cross128.v),
        "cross64_QKV": (unet.control.cross64.q, unet.control.cross64.k,
                        unet.control.cross64.v),
        "ControlNet": unet.control.hint_projections,
    }
    result = {}
    for name, group in groups.items():
        modules = group if isinstance(group, tuple) else (group,)
        params = [p for module in modules for p in module.parameters()]
        grads = [p.grad for p in params if p.grad is not None]
        assert all(bool(torch.isfinite(g).all()) for g in grads), name
        result[name] = math.sqrt(sum(float(g.square().sum()) for g in grads))
    return result


class E12TrainState(TrainState):
    def __init__(self, device, guard):
        super().__init__("D1_MULTI_512", device, guard)
        d1 = self.model
        d1_counts = parameter_counts(d1)
        canonical_rng = rng_state()
        del self.ema, self.optimizer
        candidate = copy.deepcopy(d1)
        candidate = graft_e12(candidate).to(self.device).eval()
        restore_rng(canonical_rng)
        old_state, new_state = d1.state_dict(), candidate.state_dict()
        shared = set(old_state)
        assert all(torch.equal(old_state[k], new_state[k]) for k in shared)
        new_keys = sorted(set(new_state) - shared)
        assert len(new_keys) > 24 and all(
            ".structure.scorers." in key or
            ".structure.up64_128." in key or
            ".structure.up128_256." in key or
            ".control.local256." in key or
            ".control.cross128." in key or
            ".control.cross64." in key for key in new_keys)
        u, d1u = candidate.model.unet0, d1.model.unet0
        for z in (u.structure.up64_128.zero, u.structure.up128_256.zero,
                  u.control.local256.zero, u.control.cross128.zero,
                  u.control.cross64.zero):
            assert float(z.weight.detach().abs().max()) == 0.
            assert float(z.bias.detach().abs().max()) == 0.
        assert all(not p.requires_grad for p in u.structure.med.parameters())
        with torch.random.fork_rng(
                devices=[self.device.index or 0] if self.device.type == "cuda" else []), torch.no_grad():
            torch.manual_seed(20260829)
            if self.device.type == "cuda":
                torch.cuda.manual_seed(20260829)
            center = torch.randn(1, 1, 512, 512, device=self.device)
            d1u.eval(); u.eval()
            d1_hint = d1u.structure(center)
            e12_hint = u.structure(center)
            structure_diff = max(float((d1_hint[k] - e12_hint[k]).abs().max())
                                 for k in ("s256", "s128", "s64"))
            x = torch.randn(1, 2, 512, 512, device=self.device)
            prev = torch.randn(1, 1, 512, 512, device=self.device)
            next_ = torch.randn(1, 1, 512, 512, device=self.device)
            t = torch.tensor([37.], device=self.device)
            d1u.set_neighbors(prev, next_)
            u.set_neighbors(prev, next_)
            try:
                left, right = d1u(x, t), u(x, t)
            finally:
                d1u.clear_neighbors(); u.clear_neighbors()
            output_diff = float((left - right).abs().max())
        assert structure_diff <= 1e-6, structure_diff
        assert output_diff <= 1e-6, output_diff
        assert u.main_state_hash() == self.main_hash
        assert rng_state()["torch"].equal(canonical_rng["torch"])
        counts = parameter_counts(candidate)
        added = counts["trainable_parameters"] - d1_counts["trainable_parameters"]
        assert counts["total_parameters"] - d1_counts["total_parameters"] == added
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
        self.gradient_nonzero_steps = {key: 0 for key in group_gradients(u)}
        self.last_group_gradients = None
        self.init_audit = {
            "arm": ARM, "canonical_direct_base_only": True,
            "E1_checkpoint_loaded": False,
            "shared_state_tensors_bitwise_identical": True,
            "shared_state_tensor_count": len(shared),
            "new_state_keys": new_keys, "added_trainable_parameters": added,
            "d1_parameters": d1_counts, "e12_parameters": counts,
            "step_zero_structure_max_abs_difference": structure_diff,
            "step_zero_output_max_abs_difference": output_diff,
            "step_zero_equivalent": True,
            "meddino_frozen": True, "salf_zero_init": True,
            "hierarchy_and_fusion_zero_init": True,
            "main_hash": self.main_hash,
            "canonical_training_rng_sha256": self.initial_rng_sha256,
        }
        del d1, left, right, old_state, new_state
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)

    def update(self):
        row = super().update()
        norms = group_gradients(self.model.model.unet0)
        self.last_group_gradients = norms
        for name, value in norms.items():
            assert math.isfinite(value)
            self.gradient_nonzero_steps[name] += int(value > 0)
        row["new_group_gradient_norms"] = norms
        return row

