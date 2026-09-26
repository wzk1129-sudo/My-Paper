"""PiDiNet-Tiny carv4 PDC shape stream adapted to single-channel Mayo CT.

Adapted from PiDiNet / Pixel Difference Networks, ICCV 2021:
https://github.com/hellozhuo/pidinet (commit d21aa881ed9c628571636fad39acfe1fad517ebd).
See PIDINET_LICENSE.txt and PIDINET_UPSTREAM_AUDIT.json. This module omits all edge heads,
CDCM, CSAM and natural-image weights; only the official PDC and Tiny backbone topology remain.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

CARV4 = ("cd", "ad", "rd", "cv") * 4
AD_PERMUTATION = (3, 0, 1, 6, 4, 2, 7, 8, 5)
RD_OUTER = (0, 2, 4, 10, 14, 20, 22, 24)
RD_INNER = (6, 7, 8, 11, 13, 16, 17, 18)


def create_conv_func(op_type):
    """Adapted createConvFunc from official models/ops.py, preserving cd/ad/rd/cv mathematics."""
    assert op_type in ("cv", "cd", "ad", "rd")
    if op_type == "cv":
        return F.conv2d
    if op_type == "cd":
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in (1, 2) and weights.shape[-2:] == (3, 3) and padding == dilation
            center = weights.sum(dim=(2, 3), keepdim=True)
            yc = F.conv2d(x, center, stride=stride, padding=0, groups=groups)
            y = F.conv2d(x, weights, bias, stride=stride, padding=padding,
                         dilation=dilation, groups=groups)
            return y - yc
        return func
    if op_type == "ad":
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in (1, 2) and weights.shape[-2:] == (3, 3) and padding == dilation
            shape = weights.shape
            flat = weights.reshape(shape[0], shape[1], 9)
            transformed = (flat - flat[:, :, AD_PERMUTATION]).reshape(shape)
            return F.conv2d(x, transformed, bias, stride=stride, padding=padding,
                            dilation=dilation, groups=groups)
        return func
    def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
        assert dilation in (1, 2) and weights.shape[-2:] == (3, 3)
        shape = weights.shape
        flat = weights.reshape(shape[0], shape[1], 9)
        # Official rd fills a 5x5 kernel from the eight non-center weights.
        # new_zeros preserves input dtype/device (official hard-coded float32 CUDA/CPU).
        expanded = weights.new_zeros(shape[0], shape[1], 25)
        expanded[:, :, RD_OUTER] = flat[:, :, 1:]
        expanded[:, :, RD_INNER] = -flat[:, :, 1:]
        return F.conv2d(x, expanded.reshape(shape[0], shape[1], 5, 5), bias,
                        stride=stride, padding=2 * dilation, dilation=dilation, groups=groups)
    return func


class PDCConv2d(nn.Module):
    """Adapted official Conv2d PDC wrapper with the same parameter shape/init."""
    def __init__(self, op_type, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1, bias=False):
        super().__init__()
        assert kernel_size == 3 and in_channels % groups == out_channels % groups == 0
        self.op_type = op_type
        self.pdc = create_conv_func(op_type)
        self.stride, self.padding, self.dilation, self.groups = stride, padding, dilation, groups
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, 3, 3))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return self.pdc(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class PDCBlock(nn.Module):
    """Official MaxPool stride, depthwise PDC -> ReLU -> pointwise -> residual."""
    def __init__(self, op_type, in_channels, out_channels, stride=1):
        super().__init__()
        self.stride = stride
        if stride > 1:
            self.pool = nn.MaxPool2d(2, 2)
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        self.conv1 = PDCConv2d(op_type, in_channels, in_channels, groups=in_channels)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        if self.stride > 1:
            x = self.pool(x)
        y = self.conv2(self.relu2(self.conv1(x)))
        if self.stride > 1:
            x = self.shortcut(x)
        return y + x


class PDCShapeStream(nn.Module):
    """PiDiNet-Tiny carv4 feature trunk; CT 1-channel init, no edge classifier."""
    def __init__(self):
        super().__init__()
        self.init_block = PDCConv2d(CARV4[0], 1, 20)
        self.stage1 = nn.ModuleList([PDCBlock(CARV4[i], 20, 20) for i in range(1, 4)])
        self.stage2 = nn.ModuleList([PDCBlock(CARV4[4], 20, 40, stride=2)] +
                                    [PDCBlock(CARV4[i], 40, 40) for i in range(5, 8)])
        self.stage3 = nn.ModuleList([PDCBlock(CARV4[8], 40, 80, stride=2)] +
                                    [PDCBlock(CARV4[i], 80, 80) for i in range(9, 12)])
        self.stage4 = nn.ModuleList([PDCBlock(CARV4[12], 80, 80, stride=2)] +
                                    [PDCBlock(CARV4[i], 80, 80) for i in range(13, 16)])
        self.last_shapes = {}
        self.last_activity = {}

    def forward(self, center):
        assert center.ndim == 4 and center.shape[1:] == (1, 512, 512)
        # center is RDDM [-1,1]. Reconstruct frozen protocol HU, clip soft-tissue,
        # then divide by 1000: exactly [-1,1] for HU in [-1000,1000].
        hu = ((center.float() + 1.) * 2048. - 1024.).clamp(-1000., 1000.)
        x = hu / 1000.
        x = self.init_block(x)
        for block in self.stage1:
            x = block(x)
        x1 = x
        for block in self.stage2:
            x = block(x)
        x2 = x
        for block in self.stage3:
            x = block(x)
        x3 = x
        for block in self.stage4:
            x = block(x)
        x4 = x
        self.last_shapes = {name: list(value.shape) for name, value in
                            (("x1", x1), ("x2", x2), ("x3", x3), ("x4", x4))}
        self.last_activity = {"x3_rms": x3.detach().square().mean().sqrt(),
                              "x4_rms": x4.detach().square().mean().sqrt()}
        return {"x3": x3, "x4": x4}
