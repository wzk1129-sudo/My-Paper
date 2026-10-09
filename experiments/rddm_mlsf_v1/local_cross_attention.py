"""Aligned 3x3 MedDINO token attention for 128x128 LDCT queries."""
import math
import torch
from torch import nn


class LocalMedCrossAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_norm = nn.GroupNorm(8, 64)
        self.key_norm = nn.GroupNorm(8, 64)
        self.query = nn.Conv2d(64, 64, 1)
        self.key = nn.Conv2d(64, 64, 1)
        self.value = nn.Conv2d(64, 64, 1)
        self.output = nn.Conv2d(64, 64, 1)
        qy, qx = torch.meshgrid(torch.arange(128), torch.arange(128), indexing="ij")
        cy, cx = qy.reshape(-1) // 4, qx.reshape(-1) // 4
        offsets = ((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1))
        indices = [((cy + dy).clamp(0, 31) * 32 + (cx + dx).clamp(0, 31))
                   for dy, dx in offsets]
        self.register_buffer("neighbor_index", torch.stack(indices, dim=-1), persistent=False)
        self.last_stats = None
        self.record = False

    def forward(self, local: torch.Tensor, med: torch.Tensor) -> torch.Tensor:
        assert local.ndim == med.ndim == 4
        assert local.shape[1:] == (64, 128, 128)
        assert med.shape == (local.shape[0], 64, 32, 32)
        batch = local.shape[0]
        q = self.query(self.query_norm(local)).reshape(batch, 4, 16, 16384)
        k_map = self.key(self.key_norm(med)).flatten(2)
        v_map = self.value(med).flatten(2)
        # Only nine keys/values per query. No all-to-all 16384x1024 matrix.
        k = k_map[:, :, self.neighbor_index].reshape(batch, 4, 16, 16384, 9)
        v = v_map[:, :, self.neighbor_index].reshape(batch, 4, 16, 16384, 9)
        score = (q.unsqueeze(-1) * k).sum(dim=2) / math.sqrt(16)
        weight = score.softmax(dim=-1)
        if self.record:
            with torch.no_grad():
                entropy = -(weight * weight.clamp_min(1e-12).log()).sum(-1) / math.log(9)
                self.last_stats = {
                    "normalized_entropy": float(entropy.mean()),
                    "max_probability": float(weight.max(-1).values.mean()),
                    "center_mass": float(weight[..., 4].mean()),
                    "border_mass": float((weight[..., :3].sum(-1) + weight[..., 6:].sum(-1)).mean()),
                }
        attended = (weight.unsqueeze(2) * v).sum(-1).reshape(batch, 64, 128, 128)
        out = self.output(attended)
        assert torch.isfinite(out).all()
        return local + out
