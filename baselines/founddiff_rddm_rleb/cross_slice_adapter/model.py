"""Frozen Direct Concat with an additive cross-slice residual-error adapter."""
import torch
from torch import nn
from context_2p5d.model import construct as construct_direct, state_hash
from mayo2016_smallset_v1.runtime import rng_state, restore_rng
from src.rddm_rleb import normalize_to_neg_one_to_one

ARM = "CROSS_SLICE_ERROR_ADAPTER_V1"
DIRECT_ARM = "2P5D_DIRECT_CONCAT"
SOURCE_PIN = "e8ed93223fbdf4a0e2521f73c7a4f213561f6a191a1a7b2a5f95934b85ebf574"
CHECKPOINT_PIN = "151b7e58d2ca0d936be0842434e3d17809b325f144d0d48e47c8565287bcdcdf"


class ResidualBlock(nn.Module):
    def __init__(self, dilation):
        super().__init__()
        self.conv1 = nn.Conv2d(32, 32, 3, padding=dilation, dilation=dilation)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(32, 32, 3, padding=dilation, dilation=dilation)

    def forward(self, x):
        return x + self.conv2(self.act(self.conv1(x)))


class CrossSliceAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.entry = nn.Conv2d(5, 32, 3, padding=1)
        self.act = nn.SiLU()
        self.blocks = nn.Sequential(ResidualBlock(1), ResidualBlock(2), ResidualBlock(4))
        self.final = nn.Conv2d(32, 1, 1)
        nn.init.zeros_(self.final.weight)
        nn.init.zeros_(self.final.bias)

    def forward(self, x):
        assert x.shape[1] == 5
        return self.final(self.blocks(self.act(self.entry(x))))


class CorrectedUnet(nn.Module):
    """DirectUnet executes unchanged; adapter reads its detached residual and LDCT relations."""
    def __init__(self, base, adapter):
        super().__init__()
        self.base = base.eval().requires_grad_(False)
        self.adapter = adapter
        self.last_base = None
        self.last_delta = None
        self.last_d_abs = None

    def set_neighbors(self, prev, next_):
        self.base.set_neighbors(prev, next_)

    def clear_neighbors(self):
        self.base.clear_neighbors()

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, x, time, x_self_cond=None):
        assert x.shape[1] == 2
        prev, next_ = self.base.neighbors(x)
        center = x[:, 1:2]
        with torch.no_grad():
            r_base = self.base(x, time, x_self_cond).detach()
        d_mean = 0.5 * (prev + next_) - center
        d_slope = 0.5 * (next_ - prev)
        d_abs = 0.5 * ((prev - center).abs() + (next_ - center).abs())
        inputs = torch.cat((r_base, center, d_mean, d_slope, d_abs), dim=1)
        delta_r = self.adapter(inputs)
        self.last_base = r_base
        self.last_delta = delta_r
        self.last_d_abs = d_abs
        return r_base + delta_r


def attach_adapter(diffusion, seed=20260829):
    assert diffusion.model.unet0.__class__.__name__ == "DirectUnet"
    before = state_hash(diffusion.model.unet0.state_dict())
    state = rng_state()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 101)
        adapter = CrossSliceAdapter()
    restore_rng(state)
    device = next(diffusion.parameters()).device
    diffusion.model.unet0 = CorrectedUnet(diffusion.model.unet0, adapter).to(device)
    wrapped = diffusion.model.unet0
    assert state_hash(wrapped.base.state_dict()) == before
    assert all(not p.requires_grad for p in wrapped.base.parameters())
    assert all(p.requires_grad for p in wrapped.adapter.parameters())
    assert float(wrapped.adapter.final.weight.norm()) == 0.0
    assert float(wrapped.adapter.final.bias.norm()) == 0.0
    return diffusion, before


def step_zero_equivalence(diffusion, seed=20260829):
    wrapped = diffusion.model.unet0
    device = next(diffusion.parameters()).device
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices), torch.no_grad():
        torch.manual_seed(seed + 102)
        x = torch.randn(1, 2, 64, 64, device=device)
        prev = torch.randn(1, 1, 64, 64, device=device)
        next_ = torch.randn(1, 1, 64, 64, device=device)
        t = torch.tensor([10.0], device=device)
        wrapped.set_neighbors(prev, next_)
        try:
            r_base = wrapped.base(x, t)
            corrected = wrapped(x, t)
            delta = wrapped.last_delta
        finally:
            wrapped.clear_neighbors()
        difference = float((r_base - corrected).abs().max())
        assert difference <= 1e-6 and float(delta.abs().max()) == 0.0
    return difference


def usage(diffusion):
    u = diffusion.model.unet0
    return {
        "adapter_parameter_count": sum(p.numel() for p in u.adapter.parameters()),
        "adapter_parameter_norm": float(torch.sqrt(sum((p.detach() ** 2).sum()
                                                       for p in u.adapter.parameters()))),
        "final_weight_norm": float(u.adapter.final.weight.detach().norm()),
        "final_bias_norm": float(u.adapter.final.bias.detach().norm()),
        "hidden_entry_weight_norm": float(u.adapter.entry.weight.detach().norm()),
        "backbone_parameter_count": sum(p.numel() for p in u.base.parameters()),
        "backbone_state_sha256": state_hash(u.base.state_dict()),
    }
