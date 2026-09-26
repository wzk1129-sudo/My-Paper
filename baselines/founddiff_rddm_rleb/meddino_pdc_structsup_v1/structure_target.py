"""Deterministic NDCT-only clean structure targets at 512, 128 and 64."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class CleanStructureTarget(nn.Module):
    def __init__(self):
        super().__init__()
        axis = torch.arange(-2, 3, dtype=torch.float32)
        gaussian = torch.exp(-axis.square() / 2)  # sigma=1 pixel
        gaussian = gaussian / gaussian.sum()
        self.register_buffer("gauss_x", gaussian.view(1, 1, 1, 5))
        self.register_buffer("gauss_y", gaussian.view(1, 1, 5, 1))
        self.register_buffer("scharr_x", torch.tensor(
            [[-3., 0., 3.], [-10., 0., 10.], [-3., 0., 3.]],
            dtype=torch.float32).view(1, 1, 3, 3) / 32.)
        self.register_buffer("scharr_y", torch.tensor(
            [[-3., -10., -3.], [0., 0., 0.], [3., 10., 3.]],
            dtype=torch.float32).view(1, 1, 3, 3) / 32.)
        self.last_stats = {}

    @torch.no_grad()
    def forward(self, ndct_zero_one):
        assert ndct_zero_one.ndim == 4 and ndct_zero_one.shape[1:] == (1, 512, 512)
        assert ndct_zero_one.dtype == torch.float32
        assert bool(torch.isfinite(ndct_zero_one).all())
        assert 0 <= float(ndct_zero_one.min()) <= float(ndct_zero_one.max()) <= 1
        hu = (ndct_zero_one * 4096. - 1024.).clamp(-1000., 1000.)
        x = hu / 1000.  # fixed [-1,1] soft-tissue HU window
        smooth = F.conv2d(F.pad(x, (2, 2, 0, 0), mode="reflect"), self.gauss_x)
        smooth = F.conv2d(F.pad(smooth, (0, 0, 2, 2), mode="reflect"), self.gauss_y)
        padded = F.pad(smooth, (1, 1, 1, 1), mode="reflect")
        gx = F.conv2d(padded, self.scharr_x)
        gy = F.conv2d(padded, self.scharr_y)
        magnitude = torch.sqrt(gx.square() + gy.square() + 1e-12)
        full = (magnitude / math.sqrt(2.)).clamp(0., 1.)
        target128 = F.interpolate(full, size=(128, 128), mode="area")
        target64 = F.interpolate(full, size=(64, 64), mode="area")
        assert target128.shape == (x.shape[0], 1, 128, 128)
        assert target64.shape == (x.shape[0], 1, 64, 64)
        assert all(bool(torch.isfinite(v).all()) and 0 <= float(v.min()) <= float(v.max()) <= 1
                   for v in (full, target128, target64))
        self.last_stats = {
            key: {"mean": float(value.mean()), "std": float(value.std(unbiased=False)),
                  "max": float(value.max())}
            for key, value in (("full", full), ("128", target128), ("64", target64))
        }
        return {"full": full, "128": target128, "64": target64}
