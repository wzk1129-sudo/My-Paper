"""E12: scale-aware anatomy, coarse-to-fine reassembly, hybrid Control injection."""
import time
import torch
from torch import nn
from torch.nn import functional as F

from meddino_four_arm_v1.model import FourArmControlledUnet
from anatomy_controlnet_v1.model import ControlEncoder
from meddino_salf_v1.model import SALFProvider


def zero_conv(cin, cout):
    layer = nn.Conv2d(cin, cout, 1)
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


def rms(x):
    return float(x.detach().square().mean().sqrt())


class Reassembly(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.reduce = nn.Conv2d(cin, cout, 1)
        self.activate = nn.GELU()
        self.depthwise = nn.Conv2d(cout, cout, 3, padding=1, groups=cout)
        self.zero = zero_conv(cout, cout)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.zero(self.depthwise(self.activate(self.reduce(x))))


class HierarchicalSALFProvider(SALFProvider):
    def __init__(self, d1_provider):
        super().__init__(d1_provider)
        self.arm = "E12_HAF"
        self.up64_128 = Reassembly(128, 64)
        self.up128_256 = Reassembly(64, 32)
        self.last_hierarchy = {}

    def forward(self, center):
        anatomy = super().forward(center)
        a64, a128, a256 = anatomy["s64"], anatomy["s128"], anatomy["s256"]
        h64 = a64
        r128 = self.up64_128(h64)
        h128 = a128 + r128
        r256 = self.up128_256(h128)
        h256 = a256 + r256
        assert h64.shape[1:] == (128, 64, 64)
        assert h128.shape[1:] == (64, 128, 128)
        assert h256.shape[1:] == (32, 256, 256)
        if self.collect_diagnostics:
            self.last_hierarchy = {
                "A64_rms": rms(a64), "H64_rms": rms(h64),
                "A128_rms": rms(a128), "hier64to128_residual_rms": rms(r128),
                "H128_rms": rms(h128), "A256_rms": rms(a256),
                "hier128to256_residual_rms": rms(r256), "H256_rms": rms(h256),
            }
            self.last_activity.update(self.last_hierarchy)
        return {"s256": h256, "s128": h128, "s64": h64}


class LocalFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Conv2d(32, 64, 1)
        self.activate = nn.GELU()
        self.depthwise = nn.Conv2d(64, 64, 3, padding=1, groups=64)
        self.zero = zero_conv(64, 64)

    def forward(self, anatomy):
        return self.zero(self.depthwise(self.activate(self.project(anatomy))))


class PooledCrossAttention(nn.Module):
    def __init__(self, control_channels, anatomy_channels, heads):
        super().__init__()
        assert control_channels // heads == 32
        self.channels = control_channels
        self.heads = heads
        self.pool = nn.AdaptiveAvgPool2d((16, 16))
        self.q = nn.Conv2d(control_channels, control_channels, 1)
        self.k = nn.Conv2d(anatomy_channels, control_channels, 1)
        self.v = nn.Conv2d(anatomy_channels, control_channels, 1)
        self.attention = nn.MultiheadAttention(
            embed_dim=control_channels, num_heads=heads, batch_first=True)
        self.zero = zero_conv(control_channels, control_channels)
        self.collect_diagnostics = False
        self.last_activity = {}

    def forward(self, control, anatomy):
        batch, channels, height, width = control.shape
        assert channels == self.channels and height == width
        assert height in (128, 64)
        pooled = F.avg_pool2d(anatomy, kernel_size=height // 16, stride=height // 16)
        assert pooled.shape[-2:] == (16, 16)
        q = self.q(control).flatten(2).transpose(1, 2)
        k = self.k(pooled).flatten(2).transpose(1, 2)
        v = self.v(pooled).flatten(2).transpose(1, 2)
        assert k.shape[1] == v.shape[1] == 256
        assert q.shape[1] == height * width
        result, weights = self.attention(q, k, v, need_weights=False)
        assert weights is None
        result = result.transpose(1, 2).reshape(batch, channels, height, width)
        residual = self.zero(result)
        if self.collect_diagnostics:
            self.last_activity = {
                "Q_rms": rms(q), "K_rms": rms(k), "V_rms": rms(v),
                "attention_output_rms": rms(result),
                "conditioning_residual_rms": rms(residual),
                "query_tokens": height * width, "key_value_tokens": 256,
                "heads": self.heads,
                "attention_entropy": None,
            }
        return residual


class HybridControlEncoder(ControlEncoder):
    """Keeps original encoder state keys and residual-tap locations."""
    def add_hybrid(self):
        self.local256 = LocalFusion()
        self.cross128 = PooledCrossAttention(128, 64, 4)
        self.cross64 = PooledCrossAttention(256, 128, 8)
        self.collect_diagnostics = False
        self.last_activity = {}

    def forward(self, inputs, time, hints):
        x = self.init_conv(inputs)
        t = self.time_mlp(time)
        outputs = []
        details = {}
        for i, (block, down) in enumerate(self.downs):
            x = down(block(x, t))
            if i < 3:
                key = ("s256", "s128", "s64")[i]
                x = x + self.hint_projections[key](hints[key])
                if i == 0:
                    residual = self.local256(hints[key])
                elif i == 1:
                    self.cross128.collect_diagnostics = self.collect_diagnostics
                    residual = self.cross128(x, hints[key])
                else:
                    self.cross64.collect_diagnostics = self.collect_diagnostics
                    residual = self.cross64(x, hints[key])
                x = x + residual
                outputs.append(x)
                if self.collect_diagnostics:
                    details[key + "_conditioning_residual_rms"] = rms(residual)
        x = self.mid_block(x, t)
        outputs.append(x)
        if self.collect_diagnostics:
            details["s128_attention"] = self.cross128.last_activity
            details["s64_attention"] = self.cross64.last_activity
            self.last_activity = details
        return outputs


class HybridControlledUnet(FourArmControlledUnet):
    def forward(self, x, time, x_self_cond=None):
        self.control.collect_diagnostics = self.collect_activity
        out = super().forward(x, time, x_self_cond)
        if self.collect_activity:
            self.activity["hybrid"] = self.control.last_activity
        return out


def graft_e12(candidate):
    unet = candidate.model.unet0
    assert isinstance(unet, FourArmControlledUnet)
    unet.structure = HierarchicalSALFProvider(unet.structure)
    unet.control.__class__ = HybridControlEncoder
    unet.control.add_hybrid()
    unet.__class__ = HybridControlledUnet
    unet.arm = "E12_HAF"
    return candidate

