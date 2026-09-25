"""Frozen official MedDINOv3 dense tokens and minimal APGC reconstruction."""
import hashlib
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

OFFICIAL_ROOT = Path(__file__).resolve().parents[3] / "official_source/MedDINOv3"
OFFICIAL_COMMIT = "a2460c2547899283446e520b84523454f7e25608"
WEIGHT_PATH = Path("/home/xujian/models/ct_foundation/meddinov3/model.pth")
WEIGHT_SHA256 = "492b2b96c00e9d3b93ba3b07a8d387aa2196b5e04698378fcc23049b447966d2"
DINO_PARENT = OFFICIAL_ROOT / "nnUNet/nnunetv2/training/nnUNetTrainer/dinov3"
MEAN_HU = 65.1084213256836
STD_HU = 178.01663208007812


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def official_meddino():
    if file_sha256(WEIGHT_PATH) != WEIGHT_SHA256:
        raise RuntimeError("MedDINO checkpoint SHA mismatch")
    if str(DINO_PARENT) not in sys.path:
        sys.path.insert(0, str(DINO_PARENT))
    from dinov3.models.vision_transformer import vit_base
    model = vit_base(drop_path_rate=0.2, layerscale_init=1.0e-5,
                     n_storage_tokens=4, qkv_bias=False, mask_k_bias=True)
    ck = torch.load(str(WEIGHT_PATH), map_location="cpu", weights_only=False)
    state = {k.replace("backbone.", ""): v for k, v in ck["teacher"].items()
             if "ibot" not in k and "dino_head" not in k}
    model.load_state_dict(state, strict=True)
    del ck, state
    model.eval().requires_grad_(False)
    return model


class AnatomyProvider(nn.Module):
    """One final normalized patch-token map; no intermediate ViT layers."""
    def __init__(self, arm, input_size=512):
        super().__init__()
        assert arm == "C4_ADD"
        assert input_size > 0 and input_size % 16 == 0
        self.arm = arm
        self.input_size = input_size
        self.med = official_meddino()
        self.projections = nn.ModuleDict({
            "s256": nn.Conv2d(768, 32, 1),
            "s128": nn.Conv2d(768, 64, 1),
            "s64": nn.Conv2d(768, 128, 1),
        })
        self.sizes = {"s256": 256, "s128": 128, "s64": 64}
        if arm == "C4_ADD":
            sobel_x = torch.tensor([[[-1., 0., 1.], [-2., 0., 2.],
                                     [-1., 0., 1.]]]) / 8.
            lap = torch.tensor([[[0., 1., 0.], [1., -4., 1.],
                                 [0., 1., 0.]]])
            self.register_buffer("sobel_x", sobel_x[None])
            self.register_buffer("sobel_y", sobel_x.transpose(-1, -2)[None].contiguous())
            self.register_buffer("laplacian", lap[None])
            self.edge_projections = nn.ModuleDict({
                "s256": nn.Conv2d(1, 32, 1),
                "s128": nn.Conv2d(1, 64, 1),
                "s64": nn.Conv2d(1, 128, 1),
            })
            self.alpha = nn.ParameterDict({
                key: nn.Parameter(torch.ones(ch))
                for key, ch in (("s256", 32), ("s128", 64), ("s64", 128))
            })
            self.gamma = nn.ParameterDict({
                key: nn.Parameter(torch.ones(ch))
                for key, ch in (("s256", 32), ("s128", 64), ("s64", 128))
            })
        self.last_activity = {}

    def train(self, mode=True):
        super().train(mode)
        self.med.eval()
        return self

    def forward(self, center):
        assert center.ndim == 4 and center.shape[1:] == (1, 512, 512)
        # RDDM [-1,1] <- normalized HU [-1024,3072]. This is a copy only.
        with torch.no_grad():
            hu = (center.float() + 1.) * 2048. - 1024.
            hu = hu.clamp(-1000., 1000.)
            med_input = ((hu - MEAN_HU) / (STD_HU + 1.0e-8)).repeat(1, 3, 1, 1)
            med_input = F.interpolate(
                med_input, size=(self.input_size, self.input_size),
                mode="bilinear", align_corners=False, antialias=True)
            self.med.eval()
            tokens = self.med(med_input, is_training=True)["x_norm_patchtokens"]
            patch_side = self.input_size // self.med.patch_size
            assert tokens.shape == (center.shape[0], patch_side * patch_side, 768)
            afm = tokens.transpose(1, 2).reshape(center.shape[0], 768, patch_side, patch_side)
            assert bool(torch.isfinite(afm).all())
            if self.arm == "C4_ADD":
                edge_input = (hu + 1000.) / 2000.
                gx = F.conv2d(edge_input, self.sobel_x, padding=1)
                gy = F.conv2d(edge_input, self.sobel_y, padding=1)
                sobel = torch.sqrt(gx.square() + gy.square() + 1.0e-12)
                lap = F.conv2d(edge_input, self.laplacian, padding=1)
                edge = sobel + lap
                edge_stats = {
                    "sobel_rms": float(sobel.square().mean().sqrt()),
                    "laplacian_rms": float(lap.square().mean().sqrt()),
                    "edge_sum_rms": float(edge.square().mean().sqrt()),
                }
        hints = {}
        activity = {"afm_rms": float(afm.detach().square().mean().sqrt()), **edge_stats}
        for key, target in self.sizes.items():
            anatomy = self.projections[key](afm)
            anatomy = F.interpolate(anatomy, size=(target, target),
                                    mode="bilinear", align_corners=False)
            if self.arm == "C4_ADD":
                edge_at_scale = F.interpolate(edge, size=(target, target),
                                              mode="bilinear", align_corners=False)
                efm = self.edge_projections[key](edge_at_scale)
                alpha = self.alpha[key][None, :, None, None]
                gamma = self.gamma[key][None, :, None, None]
                hints[key] = (alpha * anatomy) * (gamma * efm)
                activity[key + "_efm_rms"] = float(efm.detach().square().mean().sqrt())
            else:
                hints[key] = anatomy
            activity[key + "_hint_rms"] = float(hints[key].detach().square().mean().sqrt())
            assert hints[key].shape[1:] == (
                self.projections[key].out_channels, target, target)
            assert bool(torch.isfinite(hints[key]).all())
        self.last_activity = activity
        return hints

