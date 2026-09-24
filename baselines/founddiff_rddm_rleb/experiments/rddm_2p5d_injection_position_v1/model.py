"""Position-only 2.5D LDCT context injection on the frozen RDDM U-Net."""
import hashlib
import json
from contextlib import contextmanager
import torch
from torch import nn
from mayo2016_smallset_v1.runtime import ROOT, seed_everything, rng_state, restore_rng
from src.rddm_rleb import Unet, ResnetBlock, build_baseline, normalize_to_neg_one_to_one

ARMS = ("ENCODER_CONTEXT_INJECTION", "BOTTLENECK_CONTEXT_INJECTION",
        "DECODER_CONTEXT_INJECTION")
SOURCE_PIN = "e8ed93223fbdf4a0e2521f73c7a4f213561f6a191a1a7b2a5f95934b85ebf574"
DIRECT_AUDIT = ROOT / "experiments/rddm_2p5d_screen_v1/2P5D_DIRECT_CONCAT/INITIALIZATION_AUDIT.json"


def state_hash(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update(name.encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class PositionUnet(Unet):
    """One identical context encoder; only the zero-initialized fusion sites differ."""
    def __init__(self, arm):
        assert arm in ARMS
        super().__init__(dim=64, dim_mults=(1, 2, 4, 8), channels=1, condition=True)
        self.arm = arm
        self._neighbors = None
        self.context0 = nn.Sequential(nn.Conv2d(2, 32, 3, padding=1), nn.SiLU())
        self.context1 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU())
        self.context2 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.SiLU())
        self.context3 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.SiLU())
        channels = {
            ARMS[0]: ((64, 64), (128, 128), (256, 256)),
            ARMS[1]: ((256, 512),),
            ARMS[2]: ((128, 256), (64, 128), (32, 64)),
        }[arm]
        self.projections = nn.ModuleList([nn.Conv2d(a, b, 1) for a, b in channels])
        for projection in self.projections:
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    def set_neighbors(self, prev, next_):
        assert self._neighbors is None and prev.shape == next_.shape
        self._neighbors = (prev, next_)

    def clear_neighbors(self):
        self._neighbors = None

    def forward(self, x, time, x_self_cond=None):
        assert x.shape[1] == 2 and self._neighbors is not None
        prev, next_ = self._neighbors
        assert prev.shape == next_.shape == x[:, :1].shape
        c512 = self.context0(torch.cat((prev, next_), dim=1))
        c256 = self.context1(c512)
        c128 = self.context2(c256)
        c64 = self.context3(c128)
        x = self.init_conv(x)
        residual = x.clone()
        timestep = self.time_mlp(time)
        skips = []
        for i, (block, down) in enumerate(self.downs):
            x = block(x, timestep)
            skips.append(x)
            x = down(x)
            if self.arm == ARMS[0] and i < 3:
                context = (c256, c128, c64)[i]
                assert x.shape[-2:] == context.shape[-2:]
                x = x + self.projections[i](context)
        x = self.mid_block(x, timestep)
        if self.arm == ARMS[1]:
            assert x.shape[-2:] == c64.shape[-2:]
            x = x + self.projections[0](c64)
        for i, (block, up) in enumerate(self.ups):
            x = torch.cat((x, skips.pop()), dim=1)
            x = block(x, timestep)
            x = up(x)
            if self.arm == ARMS[2] and i < 3:
                context = (c128, c256, c512)[i]
                assert x.shape[-2:] == context.shape[-2:]
                x = x + self.projections[i](context)
        x = torch.cat((x, residual), dim=1)
        x = self.final_res_block(x, timestep)
        return self.final_conv(x)


def construct(arm, device="cpu", seed=20260829):
    assert arm in ARMS
    seed_everything(seed)
    baseline = build_baseline()
    original = baseline.state_dict()
    original_hash = state_hash(original)
    direct = json.loads(DIRECT_AUDIT.read_text())
    assert original_hash == direct["canonical_state_sha256"]
    canonical_rng = rng_state()
    model = build_baseline()
    model.model.unet0 = PositionUnet(arm)
    observed = model.state_dict()
    with torch.no_grad():
        for name, tensor in original.items():
            assert name in observed and observed[name].shape == tensor.shape
            observed[name].copy_(tensor)
    assert all(torch.equal(tensor, observed[name]) for name, tensor in original.items())
    new = sorted(set(observed) - set(original))
    assert new and sum(isinstance(m, ResnetBlock) for m in model.modules()) == 10
    assert model.objective == "pred_res" and model.num_timesteps == 1000
    assert model.sampling_timesteps == 2 and model.sum_scale == 0.01
    restore_rng(canonical_rng)
    device = torch.device(device)
    baseline = baseline.to(device).eval()
    model = model.to(device).eval()
    with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []), torch.no_grad():
        torch.manual_seed(41)
        x = torch.randn(1, 2, 64, 64, device=device)
        prev = torch.randn(1, 1, 64, 64, device=device)
        next_ = torch.randn(1, 1, 64, 64, device=device)
        t = torch.tensor([10.], device=device)
        reference = baseline.model.unet0(x, t)
        model.model.unet0.set_neighbors(prev, next_)
        try:
            output = model.model.unet0(x, t)
        finally:
            model.model.unet0.clear_neighbors()
        difference = float((output - reference).abs().max())
        assert output.shape == reference.shape and difference <= 1e-6
    restore_rng(canonical_rng)
    u = model.model.unet0
    projection_norms = [float(p.weight.norm()) for p in u.projections]
    assert all(value == 0 for value in projection_norms)
    audit = {
        "arm": arm, "seed": seed, "canonical_state_sha256": original_hash,
        "direct_concat_step0_canonical_hash_match": True,
        "shared_tensor_count": len(original), "shared_bitwise_identical": True,
        "new_tensor_keys": new,
        "new_parameter_count": sum(p.numel() for n, p in model.named_parameters()
                                   if n.startswith("model.unet0.context") or n.startswith("model.unet0.projections")),
        "initial_projection_norms": projection_norms,
        "step_zero_max_abs_output_difference": difference,
        "step_zero_equivalent": True, "baseline_source_sha256": SOURCE_PIN,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "rleb_count": 10, "canonical_training_rng_restored": True,
        "context_stage4_active": arm != ARMS[2],
    }
    del baseline
    return model, audit


@contextmanager
def neighbor_context(diffusion, prev, next_):
    u = diffusion.model.unet0
    u.set_neighbors(normalize_to_neg_one_to_one(prev),
                    normalize_to_neg_one_to_one(next_))
    try:
        yield
    finally:
        u.clear_neighbors()


def training_loss(diffusion, target, center, prev, next_):
    with neighbor_context(diffusion, prev, next_):
        return diffusion([target, center])[0]


def restored_slice(diffusion, center, prev, next_):
    with neighbor_context(diffusion, prev, next_):
        return diffusion.sample([center], last=True)[-1]


def parameter_usage(diffusion, arm):
    u = diffusion.model.unet0
    return {
        "context_stage_weight_norms": [
            float(getattr(u, "context" + str(i))[0].weight.detach().norm())
            for i in range(4)
        ],
        "projection_weight_norms": [float(p.weight.detach().norm()) for p in u.projections],
        "context_stage4_active": arm != ARMS[2],
    }
