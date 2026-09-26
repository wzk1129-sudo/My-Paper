"""Side heads supervise only PDC-gated shape, preserving parent restoration path."""
import torch
from torch import nn

from meddino_pdc_gated_bca_v1.model import PDCFusionProvider, graft_pdc


class SupervisedPDCFusionProvider(PDCFusionProvider):
    def add_side_heads(self):
        self.head128 = nn.Conv2d(64, 1, 1)
        self.head64 = nn.Conv2d(128, 1, 1)
        # Keep initial auxiliary output near zero without changing S128/S64.
        for head in (self.head128, self.head64):
            nn.init.normal_(head.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(head.bias)
        self.last_predictions = {}

    def forward(self, center):
        self.last_predictions = {}
        result = super().forward(center)
        assert set(self.last_predictions) == {"128", "64"}
        return result

    def _activity(self, h, x, p, g, y, w, s):
        activity = super()._activity(h, x, p, g, y, w, s)
        stage = str(y.shape[-1])
        assert stage in ("128", "64")
        pred = getattr(self, "head" + stage)(y)
        assert pred.shape == (y.shape[0], 1, y.shape[-2], y.shape[-1])
        self.last_predictions[stage] = pred
        activity["pred_struct_rms"] = pred.detach().square().mean().sqrt()
        activity["pred_struct_spatial_std"] = pred.detach().flatten(2).std(dim=-1, unbiased=False).mean()
        return activity


def graft_structsup(direct_unet, d1_provider):
    unet = graft_pdc(direct_unet, d1_provider)
    unet.structure.__class__ = SupervisedPDCFusionProvider
    unet.structure.add_side_heads()
    return unet
