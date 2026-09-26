"""Official MedDINOv3 four-layer features for the four controlled arms."""
import contextlib
import torch
from torch import nn
from torch.nn import functional as F
from prosac_add_fusion_v1.structure import (
    official_meddino, MEAN_HU, STD_HU, OFFICIAL_ROOT, OFFICIAL_COMMIT,
    WEIGHT_PATH, WEIGHT_SHA256,
)

LAYERS = (2, 5, 8, 11)
SCALES = {"s256": (32, 256), "s128": (64, 128), "s64": (128, 64)}


class MultiProvider(nn.Module):
    def __init__(self, arm):
        super().__init__()
        assert arm in ("D1_MULTI_512", "D2_MULTI_ADD_512", "D3_MULTI_1024", "D4_MULTI_FT_512")
        self.arm = arm
        self.input_size = 1024 if arm == "D3_MULTI_1024" else 512
        self.edge_enabled = arm == "D2_MULTI_ADD_512"
        self.finetune_enabled = False
        self.collect_diagnostics = False
        self.med = official_meddino()
        self.projections = nn.ModuleDict({
            k: nn.Conv2d(3072, channels, 1) for k, (channels, _) in SCALES.items()
        })
        if self.edge_enabled:
            sobel_x = torch.tensor([[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]]) / 8.
            lap = torch.tensor([[[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]])
            self.register_buffer("sobel_x", sobel_x[None])
            self.register_buffer("sobel_y", sobel_x.transpose(-1, -2)[None].contiguous())
            self.register_buffer("laplacian", lap[None])
            self.edge_projections = nn.ModuleDict({
                k: nn.Conv2d(1, channels, 1) for k, (channels, _) in SCALES.items()
            })
            self.alpha = nn.ParameterDict({
                k: nn.Parameter(torch.ones(channels)) for k, (channels, _) in SCALES.items()
            })
            self.gamma = nn.ParameterDict({
                k: nn.Parameter(torch.ones(channels)) for k, (channels, _) in SCALES.items()
            })
        self.last_activity = {}

    def enable_finetune(self):
        assert self.arm == "D4_MULTI_FT_512" and len(self.med.blocks) == 12
        assert not self.finetune_enabled
        for p in self.med.parameters():
            p.requires_grad_(False)
        for block in self.med.blocks[8:12]:
            block.requires_grad_(True)
        self.finetune_enabled = True
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        self.med.eval()
        if mode and self.finetune_enabled:
            for block in self.med.blocks[8:12]:
                block.train()
        return self

    @staticmethod
    def _rms(x):
        return float(x.detach().square().mean().sqrt())

    @staticmethod
    def _cos(a, b):
        return float(F.cosine_similarity(a.detach().flatten(1), b.detach().flatten(1), dim=1).mean())

    def forward(self, center):
        assert center.ndim == 4 and center.shape[1:] == (1, 512, 512)
        with torch.no_grad():
            hu = ((center.float() + 1.) * 2048. - 1024.).clamp(-1000., 1000.)
            med_input = ((hu - MEAN_HU) / STD_HU).repeat(1, 3, 1, 1)
            if self.input_size == 1024:
                med_input = F.interpolate(med_input, size=(1024, 1024),
                                          mode="bilinear", align_corners=False, antialias=True)
            if self.edge_enabled:
                edge_input = (hu + 1000.) / 2000.
                gx = F.conv2d(edge_input, self.sobel_x, padding=1)
                gy = F.conv2d(edge_input, self.sobel_y, padding=1)
                sobel = torch.sqrt(gx.square() + gy.square() + 1.e-12)
                laplacian = F.conv2d(edge_input, self.laplacian, padding=1)
                edge = sobel + laplacian
        assert self.med.training is False
        grad_context = contextlib.nullcontext() if self.finetune_enabled and self.training else torch.no_grad()
        with grad_context:
            features = self.med.get_intermediate_layers(
                med_input, n=[2, 5, 8, 11], reshape=True, norm=True)
        side = self.input_size // 16
        assert len(features) == 4
        for feature in features:
            assert feature.shape == (center.shape[0], 768, side, side)
            assert bool(torch.isfinite(feature).all())
        multi = torch.cat(features, dim=1)
        assert multi.shape == (center.shape[0], 3072, side, side)
        hints = {}
        activity = {}
        if self.collect_diagnostics:
            activity["feature_rms"] = [self._rms(v) for v in features]
            activity["F2_F5_cosine"] = self._cos(features[0], features[1])
            activity["F5_F8_cosine"] = self._cos(features[1], features[2])
            activity["F8_F11_cosine"] = self._cos(features[2], features[3])
            activity["F2_F11_cosine"] = self._cos(features[0], features[3])
            activity["concat_rms"] = self._rms(multi)
            if self.edge_enabled:
                activity.update(sobel_rms=self._rms(sobel),
                                laplacian_rms=self._rms(laplacian),
                                edge_sum_rms=self._rms(edge))
        for key, (channels, size) in SCALES.items():
            anatomy = self.projections[key](multi)
            anatomy = F.interpolate(anatomy, size=(size, size),
                                    mode="bilinear", align_corners=False)
            if self.edge_enabled:
                edge_scale = F.interpolate(edge, size=(size, size),
                                           mode="bilinear", align_corners=False)
                edge_feature = self.edge_projections[key](edge_scale)
                hints[key] = (self.alpha[key][None, :, None, None] * anatomy) * (
                    self.gamma[key][None, :, None, None] * edge_feature)
                if self.collect_diagnostics:
                    activity[key + "_edge_rms"] = self._rms(edge_feature)
            else:
                hints[key] = anatomy
            assert hints[key].shape == (center.shape[0], channels, size, size)
            if self.collect_diagnostics:
                activity[key + "_anatomy_rms"] = self._rms(anatomy)
                activity[key + "_hint_rms"] = self._rms(hints[key])
        self.last_activity = activity
        return hints
