"""Scale-specific 4-way MedDINO layer weights; all D1 components retained."""
import torch
from torch import nn
from torch.nn import functional as F

from meddino_four_arm_v1.structure import MultiProvider, SCALES, MEAN_HU, STD_HU

LAYERS = (2, 5, 8, 11)
SCALE_NAMES = ("s256", "s128", "s64")


class SALFProvider(MultiProvider):
    def __init__(self, d1_provider):
        nn.Module.__init__(self)
        assert isinstance(d1_provider, MultiProvider)
        assert d1_provider.arm == "D1_MULTI_512"
        assert d1_provider.input_size == 512 and not d1_provider.edge_enabled
        self.arm = "E1_SALF_512"
        self.input_size = 512
        self.edge_enabled = False
        self.finetune_enabled = False
        self.collect_diagnostics = False
        self.capture_weights = False
        self.med = d1_provider.med
        self.projections = d1_provider.projections
        self.scorers = nn.ModuleDict({
            scale: nn.ModuleList([nn.Linear(768, 1) for _ in LAYERS])
            for scale in SCALE_NAMES
        })
        for scorer in self.scorers.values():
            for linear in scorer:
                nn.init.zeros_(linear.weight)
                nn.init.zeros_(linear.bias)
        self.last_activity = {}
        self.last_weights = None
        assert sum(p.numel() for p in self.scorers.parameters()) == 9228
        assert all(float(p.detach().abs().max()) == 0 for p in self.scorers.parameters())
        assert not any(p.requires_grad for p in self.med.parameters())

    def forward(self, center):
        assert center.ndim == 4 and center.shape[1:] == (1, 512, 512)
        with torch.no_grad():
            hu = ((center.float() + 1.) * 2048. - 1024.).clamp(-1000., 1000.)
            med_input = ((hu - MEAN_HU) / STD_HU).repeat(1, 3, 1, 1)
            features = self.med.get_intermediate_layers(
                med_input, n=[2, 5, 8, 11], reshape=True, norm=True)
        assert not self.med.training and len(features) == 4
        assert all(f.shape == (center.shape[0], 768, 32, 32) and
                   bool(torch.isfinite(f).all()) for f in features)
        descriptors = [f.mean(dim=(2, 3)) for f in features]
        hints, weight_record, activity = {}, {}, {}
        for scale in SCALE_NAMES:
            scores = torch.cat([
                linear(descriptor) for linear, descriptor in
                zip(self.scorers[scale], descriptors)
            ], dim=1)
            weights = torch.softmax(scores, dim=1)
            assert weights.shape == (center.shape[0], 4)
            assert bool(torch.isfinite(weights).all())
            assert bool(torch.allclose(weights.sum(dim=1),
                                       torch.ones_like(weights[:, 0]), atol=1e-6))
            gated = torch.cat([
                (4.0 * weights[:, i, None, None, None]) * feature
                for i, feature in enumerate(features)
            ], dim=1)
            assert gated.shape == (center.shape[0], 3072, 32, 32)
            channels, target = SCALES[scale]
            anatomy = self.projections[scale](gated)
            anatomy = F.interpolate(anatomy, size=(target, target),
                                    mode="bilinear", align_corners=False)
            assert anatomy.shape == (center.shape[0], channels, target, target)
            hints[scale] = anatomy
            if self.capture_weights or self.collect_diagnostics:
                weight_record[scale] = weights.detach().cpu()
            if self.collect_diagnostics:
                activity[scale + "_anatomy_rms"] = float(anatomy.detach().square().mean().sqrt())
                activity[scale + "_hint_rms"] = activity[scale + "_anatomy_rms"]
                activity[scale + "_weight_mean"] = weights.detach().mean(dim=0).cpu().tolist()
                activity[scale + "_entropy"] = float(
                    (-(weights * weights.clamp_min(1e-12).log()).sum(dim=1)).mean())
        self.last_weights = weight_record if weight_record else None
        self.last_activity = activity
        return hints
