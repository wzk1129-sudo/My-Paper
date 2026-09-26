"""PDC shape + frozen MedDINO anatomy fusion, retaining the archived E12-BCA U-Net."""
import torch
from torch import nn

from meddino_e12_bca_v1.model import graft_bca
from .pdc import PDCShapeStream


def rms(value):
    return value.detach().square().mean().sqrt()


class AnatomyShapeGate(nn.Module):
    """Small-batch-stable local gate; GroupNorm replaces ordinary BatchNorm."""
    def __init__(self, channels):
        super().__init__()
        self.anatomy_gate = nn.Conv2d(channels, 1, 1)
        self.norm = nn.GroupNorm(1, channels + 1)
        hidden = max(16, channels // 4)
        self.conv1 = nn.Conv2d(channels + 1, hidden, 1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(hidden, 1, 1)

    def forward(self, shape, anatomy):
        cue = self.anatomy_gate(anatomy)
        joined = torch.cat((shape, cue), dim=1)
        return torch.sigmoid(self.conv2(self.act(self.conv1(self.norm(joined)))))


class PDCFusionProvider(nn.Module):
    """H + beta * projected PDC * (1 + MedDINO-guided gate)."""
    def __init__(self, anatomy):
        super().__init__()
        self.anatomy = anatomy
        self.shape = PDCShapeStream()
        self.shape_proj128 = nn.Sequential(nn.Conv2d(80, 64, 1), nn.GELU())
        self.shape_proj64 = nn.Sequential(nn.Conv2d(80, 128, 1), nn.GELU())
        self.gate128 = AnatomyShapeGate(64)
        self.gate64 = AnatomyShapeGate(128)
        self.beta128 = nn.Parameter(torch.tensor(0.1))
        self.beta64 = nn.Parameter(torch.tensor(0.1))
        self.last_shapes = {}
        self.last_activity = {}

    def forward(self, center):
        anatomy = self.anatomy(center)
        shape = self.shape(center)
        h128, h64 = anatomy["s128"], anatomy["s64"]
        x3, x4 = shape["x3"], shape["x4"]
        p128, p64 = self.shape_proj128(x3), self.shape_proj64(x4)
        g128, g64 = self.gate128(p128, h128), self.gate64(p64, h64)
        y128, y64 = p128 * (1 + g128), p64 * (1 + g64)
        w128, w64 = self.beta128 * y128, self.beta64 * y64
        s128, s64 = h128 + w128, h64 + w64
        assert h128.shape == s128.shape == (center.shape[0], 64, 128, 128)
        assert h64.shape == s64.shape == (center.shape[0], 128, 64, 64)
        assert bool(torch.isfinite(s128).all()) and bool(torch.isfinite(s64).all())
        self.last_shapes = {
            **self.shape.last_shapes,
            "H128": list(h128.shape), "H64": list(h64.shape),
            "S128": list(s128.shape), "S64": list(s64.shape),
        }
        self.last_activity = {
            "128": self._activity(h128, x3, p128, g128, y128, w128, s128),
            "64": self._activity(h64, x4, p64, g64, y64, w64, s64),
        }
        return {"s128": s128, "s64": s64}

    @staticmethod
    def _activity(h, x, p, g, y, w, s):
        gd = g.detach()
        return {
            "anatomy_rms": rms(h),
            "raw_shape_rms": rms(x),
            "projected_shape_rms": rms(p),
            "gate_mean": gd.mean(),
            "gate_std": gd.std(unbiased=False),
            "gate_min": gd.min(),
            "gate_max": gd.max(),
            "gate_spatial_std": gd.flatten(2).std(dim=-1, unbiased=False).mean(),
            "gated_shape_rms": rms(y),
            "weighted_shape_rms": rms(w),
            "fused_structure_rms": rms(s),
        }


def graft_pdc(direct_unet, d1_provider):
    """Preserve exactly BCAUnet forward and attention; replace only its structure hints."""
    unet = graft_bca(direct_unet, d1_provider)
    unet.structure = PDCFusionProvider(unet.structure)
    unet.structure.anatomy.med.requires_grad_(False)
    return unet


def numeric_activity(provider):
    return {stage: {key: float(value) for key, value in values.items()}
            for stage, values in provider.last_activity.items()}
