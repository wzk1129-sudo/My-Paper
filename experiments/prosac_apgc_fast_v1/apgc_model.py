"""M1 APGC on the frozen RDDM-CT shared encoder; M0 uses RDDMCTBackbone unchanged."""
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from rddm_ct import RDDMCTBackbone

MEDDINO_ROOT = Path("/home/xujian/projects/CoreDiff/experiments/rddm_prosac_apgc_control_v1_20260925/official_source/MedDINOv3")
MEDDINO_WEIGHT = Path("/home/xujian/models/ct_foundation/meddinov3/model.pth")
MEDDINO_SHA256 = "492b2b96c00e9d3b93ba3b07a8d387aa2196b5e04698378fcc23049b447966d2"
MEDDINO_SOURCE_SHA256 = "a0e79d4506a9b166845446268bafeb802cace6e5c008a0b4a1da183befe46e7f"
MEAN_HU = 65.1084213256836
STD_HU = 178.01663208007812


def make_frozen_meddino():
    from rddm_ct import sha256_file
    if sha256_file(MEDDINO_WEIGHT) != MEDDINO_SHA256:
        raise RuntimeError("MedDINO checkpoint SHA mismatch")
    parent = MEDDINO_ROOT / "nnUNet/nnunetv2/training/nnUNetTrainer/dinov3"
    source = parent / "dinov3/models/vision_transformer.py"
    if sha256_file(source) != MEDDINO_SOURCE_SHA256:
        raise RuntimeError("MedDINO source SHA mismatch")
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))
    from dinov3.models.vision_transformer import vit_base
    med = vit_base(drop_path_rate=0.2, layerscale_init=1.0e-5,
                   n_storage_tokens=4, qkv_bias=False, mask_k_bias=True)
    checkpoint = torch.load(str(MEDDINO_WEIGHT), map_location="cpu", weights_only=False)
    state = {key.removeprefix("backbone."): value
             for key, value in checkpoint["teacher"].items()
             if "ibot" not in key and "dino_head" not in key}
    med.load_state_dict(state, strict=True)
    del checkpoint, state
    med.eval().requires_grad_(False)
    return med


class APGCProvider(nn.Module):
    """Single AEFM at the 64x64 encoder output; projections are reconstruction choices."""
    def __init__(self):
        super().__init__()
        self.med = make_frozen_meddino()
        self.anatomy_projection = nn.Conv2d(768, 512, 1)
        self.edge_projection = nn.Conv2d(1, 512, 1)
        self.alpha = nn.Parameter(torch.ones(512))
        self.gamma = nn.Parameter(torch.ones(512))
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8.
        laplacian = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])
        self.register_buffer("sobel_x", sobel_x[None, None])
        self.register_buffer("sobel_y", sobel_x.T.contiguous()[None, None])
        self.register_buffer("laplacian", laplacian[None, None])
        self.last_activity = {}

    def train(self, mode=True):
        super().train(mode)
        self.med.eval()
        return self

    def forward(self, center_diffusion):
        if center_diffusion.ndim != 4 or center_diffusion.shape[1:] != (1, 512, 512):
            raise ValueError("APGC requires center LDCT [B,1,512,512]")
        center = (center_diffusion.float() + 1.) * .5
        hu = center * 400. - 160.
        with torch.no_grad():
            med_input = ((hu - MEAN_HU) / STD_HU).repeat(1, 3, 1, 1)
            tokens = self.med(med_input, is_training=True)["x_norm_patchtokens"]
            if tokens.shape[1:] != (1024, 768):
                raise RuntimeError("Unexpected MedDINO token grid")
            afm = tokens.transpose(1, 2).reshape(center.shape[0], 768, 32, 32)
            gx = F.conv2d(center, self.sobel_x, padding=1)
            gy = F.conv2d(center, self.sobel_y, padding=1)
            sobel = torch.sqrt(gx.square() + gy.square() + 1e-12)
            lap = F.conv2d(center, self.laplacian, padding=1)
            edge = sobel + lap
        anatomy = F.interpolate(self.anatomy_projection(afm), size=(64, 64),
                                mode="bilinear", align_corners=False)
        efm = self.edge_projection(F.interpolate(edge, size=(64, 64),
                                                  mode="bilinear", align_corners=False))
        aefm = (self.alpha[None, :, None, None] * anatomy) * (
            self.gamma[None, :, None, None] * efm)
        self.last_activity = {
            "afm_rms": float(anatomy.detach().square().mean().sqrt()),
            "efm_rms": float(efm.detach().square().mean().sqrt()),
            "aefm_rms": float(aefm.detach().square().mean().sqrt()),
            "aefm_mean": float(aefm.detach().mean()),
            "aefm_std": float(aefm.detach().std()),
        }
        return aefm


class APGCBackbone(RDDMCTBackbone):
    """Creates canonical M0 backbone first, then adds APGC without changing its initialization."""
    def __init__(self, model_config):
        super().__init__(model_config)
        self.apgc = APGCProvider()

    def forward(self, x_t, ldct_context, t):
        if x_t.ndim != 4 or x_t.shape[1] != 1 or ldct_context.ndim != 4 or ldct_context.shape[1] != 3:
            raise ValueError("Expected state [B,1,H,W] and context [B,3,H,W]")
        if x_t.shape[0] != ldct_context.shape[0] or x_t.shape[-2:] != ldct_context.shape[-2:]:
            raise ValueError("State and context shapes differ")
        state_feature = self.state_stem(x_t)
        context_feature = self.context_stem(ldct_context)
        x = self.fusion(torch.cat((state_feature, context_feature), dim=1))
        root = x.clone()
        time_embedding = self.time_mlp(t)
        skips = []
        for block1, block2, attention, downsample in self.downs:
            x = block1(x, time_embedding)
            skips.append(x)
            x = block2(x, time_embedding)
            x = attention(x)
            skips.append(x)
            x = downsample(x)
        if x.shape[1:] != (512, 64, 64):
            raise RuntimeError("Unexpected encoder output shape")
        x = x + self.apgc(ldct_context[:, 1:2]).to(dtype=x.dtype)
        x = self.mid_block1(x, time_embedding)
        x = self.mid_attn(x)
        x = self.mid_block2(x, time_embedding)
        for block1, block2, attention, upsample in self.ups:
            x = torch.cat((x, skips.pop()), dim=1)
            x = block1(x, time_embedding)
            x = torch.cat((x, skips.pop()), dim=1)
            x = block2(x, time_embedding)
            x = attention(x)
            x = upsample(x)
        x = self.final_res_block(torch.cat((x, root), dim=1), time_embedding)
        return self.residual_head(x), self.noise_head(x)
