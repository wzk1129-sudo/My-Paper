"""E12-guided local cross-attention in the RDDM encoder (new experiment only)."""
import math
import torch
from torch import nn
from torch.nn import functional as F

from context_2p5d.model import DirectUnet
from meddino_e12_haf_v1.model import HierarchicalSALFProvider


class WindowCrossAttention(nn.Module):
    def __init__(self, main_channels, structure_channels, window_size):
        super().__init__()
        self.window_size = int(window_size)
        self.heads = 4
        self.dim_head = 32
        self.inner = self.heads * self.dim_head
        self.q_norm = nn.LayerNorm(main_channels)
        self.kv_norm = nn.LayerNorm(structure_channels)
        self.q = nn.Linear(main_channels, self.inner)
        self.k = nn.Linear(structure_channels, self.inner)
        self.v = nn.Linear(structure_channels, self.inner)
        self.out = nn.Linear(self.inner, main_channels)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.last_activity = {}

    def _windows(self, image):
        b, c, h, w = image.shape
        s = self.window_size
        assert h % s == 0 and w % s == 0
        return image.view(b, c, h // s, s, w // s, s).permute(0, 2, 4, 3, 5, 1).reshape(-1, s * s, c)

    def _restore(self, windows, b, h, w):
        s = self.window_size
        c = windows.shape[-1]
        return windows.view(b, h // s, w // s, s, s, c).permute(0, 5, 1, 3, 2, 4).reshape(b, c, h, w)

    def forward(self, main, structure):
        b, _, h, w = main.shape
        assert structure.shape[0] == b and structure.shape[-2:] == (h, w)
        q = self.q(self.q_norm(self._windows(main)))
        kv = self.kv_norm(self._windows(structure))
        k, v = self.k(kv), self.v(kv)
        n, tokens, _ = q.shape
        def heads(x):
            return x.view(n, tokens, self.heads, self.dim_head).transpose(1, 2)
        qh, kh, vh = heads(q), heads(k), heads(v)
        attended = F.scaled_dot_product_attention(qh, kh, vh, dropout_p=0.0)
        joined = attended.transpose(1, 2).reshape(n, tokens, self.inner)
        residual = self._restore(self.out(joined), b, h, w)
        self.last_activity = {
            "q_rms": q.detach().square().mean().sqrt(),
            "k_rms": k.detach().square().mean().sqrt(),
            "v_rms": v.detach().square().mean().sqrt(),
            "output_rms": residual.detach().square().mean().sqrt(),
            "weighted_residual_rms": (self.alpha.detach() * residual.detach()).square().mean().sqrt(),
        }
        return main + self.alpha * residual


class BCAUnet(DirectUnet):
    """Canonical Direct U-Net with two post-downsample encoder cross-attention taps."""
    def forward(self, x, time, x_self_cond=None):
        assert x.shape[1] == 2
        prev, next_ = self.neighbors(x)
        center = x[:, 1:2]
        hints = self.structure(center)
        x = torch.cat((x[:, :1], prev, center, next_), dim=1)
        if self.self_condition:
            x_self_cond = x_self_cond if x_self_cond is not None else torch.zeros_like(x)
            x = torch.cat((x_self_cond, x), dim=1)
        x = self.init_conv(x)
        r = x.clone()
        t = self.time_mlp(time)
        skips = []
        for i, (res_block, downsample) in enumerate(self.downs):
            x = res_block(x, t)
            skips.append(x)
            x = downsample(x)
            if i == 1:
                x = self.bca128(x, hints["s128"])
            elif i == 2:
                x = self.bca64(x, hints["s64"])
        x = self.mid_block(x, t)
        for res_block, upsample in self.ups:
            x = torch.cat((x, skips.pop()), dim=1)
            x = res_block(x, t)
            x = upsample(x)
        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        return self.final_conv(x)


def graft_bca(direct_unet, d1_provider):
    assert type(direct_unet) is DirectUnet
    direct_unet.__class__ = BCAUnet
    direct_unet.structure = HierarchicalSALFProvider(d1_provider)
    direct_unet.bca128 = WindowCrossAttention(128, 64, 8)
    direct_unet.bca64 = WindowCrossAttention(256, 128, 16)
    # MedDINO is strictly frozen; SALF and hierarchy are fresh trainable modules.
    direct_unet.structure.med.requires_grad_(False)
    return direct_unet
