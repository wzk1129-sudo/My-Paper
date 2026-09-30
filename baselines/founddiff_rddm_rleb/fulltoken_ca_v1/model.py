"""Full MedDINO final-layer patch-token cross-attention on Direct RDDM."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from context_2p5d.model import DirectUnet
from knowledge_prior_v1.core import load_direct, official_med, MEAN_HU, STD_HU

class GlobalPatchCrossAttention(nn.Module):
    """All spatial RDDM queries attend to all 1024 MedDINO patch tokens."""
    def __init__(self, channels, query_chunk=0):
        super().__init__()
        self.channels = channels
        self.query_chunk = int(query_chunk)
        self.q = nn.Linear(channels, 256)
        self.k = nn.Linear(768, 256)
        self.v = nn.Linear(768, 256)
        self.out = nn.Linear(256, channels)
        self.gate = nn.Parameter(torch.zeros(()))
        self.capture = False
        self._sums = {}
        self._count = 0

    def reset_activity(self):
        self._sums = {}
        self._count = 0
        self.capture = True

    def activity_summary(self):
        n = max(1, self._count)
        return {k: v / n for k, v in self._sums.items()} | {"calls": self._count}

    def forward(self, main, patch_tokens):
        b, c, h, w = main.shape
        assert c == self.channels
        assert patch_tokens.shape == (b, 1024, 768)
        q = self.q(main.flatten(2).transpose(1, 2))
        k = self.k(patch_tokens)
        v = self.v(patch_tokens)
        q = q.reshape(b, h*w, 4, 64).transpose(1, 2)
        k = k.reshape(b, 1024, 4, 64).transpose(1, 2)
        v = v.reshape(b, 1024, 4, 64).transpose(1, 2)
        if self.query_chunk:
            chunks = [F.scaled_dot_product_attention(q[:, :, start:start+self.query_chunk], k, v, dropout_p=0.0)
                      for start in range(0, h*w, self.query_chunk)]
            attended = torch.cat(chunks, dim=2)
        else:
            attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        attended = attended.transpose(1, 2).contiguous().reshape(b, h*w, 256)
        residual = self.out(attended).transpose(1, 2).reshape(b, c, h, w)
        weighted = self.gate * residual
        if self.capture:
            values = {
                "q_rms": q.detach().float().square().mean().sqrt(),
                "k_rms": k.detach().float().square().mean().sqrt(),
                "v_rms": v.detach().float().square().mean().sqrt(),
                "output_rms": residual.detach().float().square().mean().sqrt(),
                "weighted_residual_rms": weighted.detach().float().square().mean().sqrt(),
            }
            for name, val in values.items():
                self._sums[name] = self._sums.get(name, 0.0) + float(val)
            self._count += 1
        return main + weighted

class FullTokenUnet(DirectUnet):
    _token_cache = None

    def clear_neighbors(self):
        super().clear_neighbors()
        self._token_cache = None

    def train(self, mode=True):
        super().train(mode)
        self.med.eval()
        return self

    def dense_tokens(self, center):
        if self._token_cache is None:
            assert center.shape[1:] == (1, 512, 512)
            with torch.no_grad():
                hu = ((center.float() + 1.0) * 2048.0 - 1024.0).clamp(-1000.0, 1000.0)
                med_input = ((hu - MEAN_HU) / STD_HU).repeat(1, 3, 1, 1)
                tokens = self.med.get_intermediate_layers(
                    med_input, n=[11], reshape=False, norm=True,
                    return_class_token=False, return_extra_tokens=False
                )[0]
            assert tokens.shape == (center.shape[0], 1024, 768)
            assert bool(torch.isfinite(tokens).all())
            self._token_cache = tokens.detach()
        return self._token_cache

    def forward(self, x, time, x_self_cond=None):
        assert x.shape[1] == 2
        prev, next_ = self.neighbors(x)
        tokens = self.dense_tokens(x[:, 1:2])
        x = torch.cat((x[:, :1], prev, x[:, 1:2], next_), dim=1)
        x = self.init_conv(x)
        r = x.clone()
        t = self.time_mlp(time)
        skips = []
        for i, (block, down) in enumerate(self.downs):
            x = block(x, t)
            skips.append(x)
            x = down(x)
            if i == 1:
                assert x.shape[1:] == (128, 128, 128)
                x = self.ca128(x, tokens)
            elif i == 2:
                assert x.shape[1:] == (256, 64, 64)
                x = self.ca64(x, tokens)
        x = self.mid_block(x, t)
        for block, up in self.ups:
            x = torch.cat((x, skips.pop()), dim=1)
            x = block(x, t)
            x = up(x)
        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        return self.final_conv(x)

def load_candidate(device="cpu", query_chunk=0):
    model = load_direct("cpu")
    unet = model.model.unet0
    unet.__class__ = FullTokenUnet
    unet.med = official_med()
    unet.ca128 = GlobalPatchCrossAttention(128, query_chunk)
    unet.ca64 = GlobalPatchCrossAttention(256, query_chunk)
    unet.med.eval().requires_grad_(False)
    unet._token_cache = None
    return model.to(device).eval()
