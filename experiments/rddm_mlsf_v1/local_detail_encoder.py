"""MLSF local LDCT encoder. Inputs are the existing [0, 1] context tensors."""
import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return x + y


class LocalDetailEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage_a = nn.Sequential(
            nn.Conv2d(4, 32, 3, stride=2, padding=1),
            nn.GroupNorm(8, 32), nn.SiLU(), ResidualBlock(32),
        )
        self.stage_b = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64), nn.SiLU(), ResidualBlock(64), ResidualBlock(64),
        )

    def forward(self, prev: torch.Tensor, center: torch.Tensor, next_: torch.Tensor):
        assert prev.shape == center.shape == next_.shape
        assert center.ndim == 4 and center.shape[1:] == (1, 512, 512)
        p, c, n = (2 * z - 1 for z in (prev, center, next_))
        difference = c - 0.5 * (p + n)
        x = torch.cat((p, c, n, difference), dim=1)
        l256 = self.stage_a(x)
        l128 = self.stage_b(l256)
        assert l256.shape == (center.shape[0], 32, 256, 256)
        assert l128.shape == (center.shape[0], 64, 128, 128)
        return l256, l128
