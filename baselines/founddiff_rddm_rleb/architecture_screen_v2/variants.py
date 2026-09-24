"""Four independent, zero-residual additions to the frozen RDDM U-Net.

The base encoder, decoder, ten ResnetBlocks, diffusion wrapper, and sampler
are inherited unchanged. Only the selected branch is allocated in each arm.
"""
import torch
from torch import nn
from src.rddm_rleb import Unet, build_baseline
from mayo2016_smallset_v1.runtime import seed_everything

ARMS = ("GATED_SKIP", "BOTTLENECK_CONTEXT",
        "LDCT_CONTENT_MODULATION", "PARTIAL_NAF_REFINEMENT")
SPECS = {
 "GATED_SKIP":{"injection":"all four pre-concat encoder skips","skip_channels":[64,64,128,256],"delta_path":"1x1 -> DW3x3 -> SiLU -> zero 1x1"},
 "BOTTLENECK_CONTEXT":{"injection":"after middle ResnetBlock","channels":512,"branch_kernels":[3,7,11],"delta_path":"DWConv -> GN -> SiLU, concat, zero 1x1"},
 "LDCT_CONTENT_MODULATION":{"injection":"encoder blocks 0/1 and decoder blocks 2/3","condition":"raw center LDCT shallow two-scale pyramid","formula":"F*(1+gamma)+beta","gamma_beta_projection":"zero initialized"},
 "PARTIAL_NAF_REFINEMENT":{"injection":"after middle and decoder blocks 0/1","channels":[512,512,256],"delta_path":"1x1 -> DW3x3 -> SimpleGate -> SCA -> zero 1x1"}}


def zero_last(conv):
    nn.init.zeros_(conv.weight)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)
    return conv

class SkipRefineBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.first = nn.Conv2d(channels, channels, 1)
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.activation = nn.SiLU()
        self.last = zero_last(nn.Conv2d(channels, channels, 1))
    def forward(self, skip):
        delta = self.last(self.activation(self.depthwise(self.first(skip))))
        return skip + delta

class MultiScaleContextBlock(nn.Module):
    def __init__(self, channels=512):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(channels, channels, kernel, padding=kernel//2,
                                    groups=channels), nn.GroupNorm(8, channels), nn.SiLU())
            for kernel in (3, 7, 11)])
        self.last = zero_last(nn.Conv2d(channels * 3, channels, 1))
    def forward(self, x):
        return x + self.last(torch.cat([branch(x) for branch in self.branches], dim=1))

class ContentModulation(nn.Module):
    def __init__(self, condition_channels, feature_channels):
        super().__init__()
        self.to_gamma_beta = zero_last(nn.Conv2d(condition_channels, feature_channels * 2, 1))
    def forward(self, feature, condition):
        gamma, beta = self.to_gamma_beta(condition).chunk(2, dim=1)
        assert feature.shape == gamma.shape == beta.shape
        return feature * (1 + gamma) + beta

class ConditionPyramid(nn.Module):
    def __init__(self):
        super().__init__()
        self.high = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.SiLU())
        self.half_scale = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU())
    def forward(self, ldct):
        high = self.high(ldct)
        return high, self.half_scale(high)

class NAFRefine(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.first = nn.Conv2d(channels, channels * 2, 1)
        self.depthwise = nn.Conv2d(channels * 2, channels * 2, 3,
                                   padding=1, groups=channels * 2)
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels, 1))
        self.last = zero_last(nn.Conv2d(channels, channels, 1))
    def forward(self, x):
        left, right = self.depthwise(self.first(x)).chunk(2, dim=1)
        gated = left * right
        return x + self.last(gated * self.sca(gated))

class VariantUnet(Unet):
    def __init__(self, arm):
        assert arm in ARMS
        super().__init__(dim=64, dim_mults=(1, 2, 4, 8), channels=1,
                         condition=True, input_condition=False)
        self.arm = arm
        self.variant_modules = nn.ModuleDict()
        if arm == "GATED_SKIP":
            self.variant_modules["skip"] = nn.ModuleList(
                [SkipRefineBlock(c) for c in (64, 64, 128, 256)])
        elif arm == "BOTTLENECK_CONTEXT":
            self.variant_modules["context"] = MultiScaleContextBlock(512)
        elif arm == "LDCT_CONTENT_MODULATION":
            self.variant_modules["pyramid"] = ConditionPyramid()
            self.variant_modules["encoder"] = nn.ModuleList(
                [ContentModulation(32, 64), ContentModulation(64, 64)])
            self.variant_modules["decoder"] = nn.ModuleList(
                [ContentModulation(64, 128), ContentModulation(32, 64)])
        elif arm == "PARTIAL_NAF_REFINEMENT":
            self.variant_modules["naf"] = nn.ModuleList(
                [NAFRefine(512), NAFRefine(512), NAFRefine(256)])

    def forward(self, x, time, x_self_cond=None):
        assert not self.self_condition and x.shape[1] == 2
        if self.arm == "LDCT_CONTENT_MODULATION":
            high, half = self.variant_modules["pyramid"](x[:, 1:2])
        x = self.init_conv(x)
        root_skip = x.clone()
        time_embedding = self.time_mlp(time)
        skips = []
        for index, (res_block, downsample) in enumerate(self.downs):
            x = res_block(x, time_embedding)
            if self.arm == "LDCT_CONTENT_MODULATION" and index < 2:
                condition = high if index == 0 else half
                x = self.variant_modules["encoder"][index](x, condition)
            skips.append(x)
            x = downsample(x)
        x = self.mid_block(x, time_embedding)
        if self.arm == "BOTTLENECK_CONTEXT":
            x = self.variant_modules["context"](x)
        elif self.arm == "PARTIAL_NAF_REFINEMENT":
            x = self.variant_modules["naf"][0](x)
        for index, (res_block, upsample) in enumerate(self.ups):
            skip = skips.pop()
            if self.arm == "GATED_SKIP":
                skip = self.variant_modules["skip"][3-index](skip)
            x = torch.cat((x, skip), dim=1)
            x = res_block(x, time_embedding)
            if self.arm == "LDCT_CONTENT_MODULATION" and index >= 2:
                condition = half if index == 2 else high
                x = self.variant_modules["decoder"][index-2](x, condition)
            elif self.arm == "PARTIAL_NAF_REFINEMENT" and index < 2:
                x = self.variant_modules["naf"][index+1](x)
            x = upsample(x)
        x = torch.cat((x, root_skip), dim=1)
        x = self.final_res_block(x, time_embedding)
        return self.final_conv(x)

def build_candidate(arm, seed):
    """Build a candidate with identical shared tensors to the seeded baseline."""
    assert arm in ARMS
    seed_everything(seed)
    diffusion = build_baseline()
    seed_everything(seed)
    diffusion.model.unet0 = VariantUnet(arm)
    return diffusion
