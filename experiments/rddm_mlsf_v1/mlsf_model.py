"""MLSF extension of the verified V1 Unet; the RDDM forward topology is inherited."""
import hashlib
import sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

V1 = Path(__file__).resolve().parent.parent / "rddm_official_2p5d_meddino_condition_v1"
if str(V1) not in sys.path:
    sys.path.insert(0, str(V1))
from model import ContextUnet, build_model as build_v1_model, rms
from local_detail_encoder import LocalDetailEncoder
from local_cross_attention import LocalMedCrossAttention

CORRECTED = {
    "E1": (64, 256, 32, 128, "F256"),
    "E2": (128, 128, 64, 256, "F128"),
    "D1": (256, 128, 64, 512, "F128"),
    "D2": (128, 256, 32, 256, "F256"),
}


def parameter_hash(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


class MLSFUnet(ContextUnet):
    def attach_mlsf(self):
        assert self.anatomy and not hasattr(self, "local_encoder")
        # New CPU module initialization has its own RNG and cannot perturb V1 or training RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261009)
            self.local_encoder = LocalDetailEncoder()
            self.med_l8 = nn.Conv2d(768, 32, 1)
            self.med_l11 = nn.Conv2d(768, 32, 1)
            self.cross_attention = LocalMedCrossAttention()
            self.up_projection = nn.Conv2d(64, 32, 1)
            self.merge = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1),
                                       nn.GroupNorm(8, 32), nn.SiLU())
            self.film_corrections = nn.ModuleDict({
                stage: nn.Conv2d(spec[2], spec[3], 1) for stage, spec in CORRECTED.items()
            })
            for projection in self.film_corrections.values():
                nn.init.zeros_(projection.weight)
                nn.init.zeros_(projection.bias)
        self._mlsf_features = None
        self.last_shapes = {}

    def set_context(self, prev, center, next_, patches=None):
        assert self._mlsf_features is None, "clear_context() required between samples"
        super().set_context(prev, center, next_, patches)
        l256, l128 = self.local_encoder(prev, center, next_)
        assert len(self._patches) == 4
        raw_l8, raw_l11 = self._patches[2], self._patches[3]
        assert raw_l8.shape == raw_l11.shape == (center.shape[0], 768, 32, 32)
        m32 = torch.cat((self.med_l8(raw_l8), self.med_l11(raw_l11)), dim=1)
        f128 = self.cross_attention(l128, m32)
        up = F.interpolate(f128, size=(256, 256), mode="bilinear", align_corners=False)
        up32 = self.up_projection(up)
        f256 = l256 + self.merge(torch.cat((l256, up32), dim=1))
        assert f128.shape == (center.shape[0], 64, 128, 128)
        assert f256.shape == (center.shape[0], 32, 256, 256)
        self._mlsf_features = {"F128": f128, "F256": f256}
        self.last_shapes = {"L256": list(l256.shape), "L128": list(l128.shape),
                            "M32": list(m32.shape), "F128": list(f128.shape),
                            "F256": list(f256.shape),
                            "neighbors_per_query": int(self.cross_attention.neighbor_index.shape[-1])}

    def clear_context(self):
        super().clear_context()
        self._mlsf_features = None

    def modulate(self, stage, x, a):
        if stage not in CORRECTED:
            return super().modulate(stage, x, a)
        channels, spatial, _, _, feature = CORRECTED[stage]
        assert x.shape[1:] == (channels, spatial, spatial)
        assert self._mlsf_features is not None
        detail = self._mlsf_features[feature]
        condition = self.conditions[stage]
        if condition.global_mode:
            source = F.adaptive_avg_pool2d(a, 1).flatten(1)
            affine = condition.projection(condition.adapter(source))[:, :, None, None]
        else:
            source = F.interpolate(a, size=x.shape[-2:], mode="bilinear", align_corners=False)
            affine = condition.projection(condition.adapter(source))
        gamma_old, beta_old = affine.chunk(2, dim=1)
        delta_gamma, delta_beta = self.film_corrections[stage](detail).chunk(2, dim=1)
        assert delta_gamma.shape == delta_beta.shape == x.shape
        gamma = gamma_old + delta_gamma
        beta = beta_old + delta_beta
        delta = gamma * condition.norm(x) + beta
        if condition.force_off:
            delta = torch.zeros_like(delta)
        if condition.record:
            base = rms(x)
            delta_rms = rms(delta)
            condition.rows.append({"delta_feature_rms": delta_rms,
                                   "base_feature_rms": base,
                                   "delta_base_ratio": delta_rms / max(base, 1e-30),
                                   "gamma_rms": rms(gamma), "beta_rms": rms(beta)})
        return x + delta


def build_model():
    # This exact V1 builder loads the common Official 2D@8k EMA parent and V1 stem.
    model = build_v1_model(anatomy=True)
    unet = model.model.unet0
    assert type(unet) is ContextUnet
    shared_hash = parameter_hash(unet.state_dict())
    unet.__class__ = MLSFUnet
    unet.attach_mlsf()
    original_keys = set(ContextUnet.state_dict(unet)) - {
        key for key in unet.state_dict() if key.startswith(("local_encoder.", "med_l8.",
            "med_l11.", "cross_attention.", "up_projection.", "merge.",
            "film_corrections."))
    }
    current = {key: value for key, value in unet.state_dict().items() if key in original_keys}
    assert parameter_hash(current) == shared_hash
    return model
