"""Single-backbone residual/noise RDDM; frozen baseline source is never edited."""
import torch
from torch import nn
from torch.nn import functional as F
from src.rddm_rleb import Unet, UnetRes, ResidualDiffusion, build_baseline

class JointUnet(Unet):
    def forward(self, x, time, x_self_cond=None):
        # This is the canonical Unet.forward through final_res_block, unchanged.
        if self.self_condition:
            x_self_cond = x_self_cond if x_self_cond is not None else torch.zeros_like(x)
            x = torch.cat((x_self_cond, x), dim=1)
        x = self.init_conv(x)
        r = x.clone()
        t = self.time_mlp(time)
        h = []
        for res_block, downsample in self.downs:
            x = res_block(x, t)
            h.append(x)
            x = downsample(x)
        x = self.mid_block(x, t)
        for res_block, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = res_block(x, t)
            x = upsample(x)
        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        return self.final_conv(x), self.noise_head(x)

class JointUnetRes(UnetRes):
    def forward(self, x, time, x_self_cond=None):
        assert self.num_unet == 1 and self.objective == 'pred_res_noise'
        # Two calls to the SAME U-Net object and parameters. Residual uses
        # alpha-time only; noise uses beta-time only.
        pred_res, _ = self.unet0(x, time[0], x_self_cond=x_self_cond)
        _, pred_noise = self.unet0(x, time[1], x_self_cond=x_self_cond)
        return pred_res, pred_noise

class JointResidualDiffusion(ResidualDiffusion):
    def set_loss_weights(self, lambda_noise, lambda_recon):
        self.lambda_noise = float(lambda_noise)
        self.lambda_recon = float(lambda_recon)
        assert self.lambda_noise >= 0 and self.lambda_recon >= 0

    def p_losses(self, imgs, t, noise=None):
        assert isinstance(imgs, list) and len(imgs) == 2
        x_start, x_input = imgs
        noise = torch.randn_like(x_start) if noise is None else noise
        x_res = x_input - x_start
        x_t = self.q_sample(x_start, x_res, t, noise=noise)
        x_in = torch.cat((x_t, x_input), dim=1)
        pred_res, pred_noise = self.model(
            x_in, [self.alphas_cumsum[t] * self.num_timesteps,
                   self.betas_cumsum[t] * self.num_timesteps])
        loss_res = F.mse_loss(pred_res, x_res)
        loss_noise = F.mse_loss(pred_noise, noise)
        x0_hat = self.predict_start_from_res_noise(x_t, t, pred_res, pred_noise)
        loss_recon = F.mse_loss(x0_hat, x_start)
        total = loss_res + self.lambda_noise * loss_noise + self.lambda_recon * loss_recon
        return [total, loss_res, loss_noise, loss_recon]

def build_joint(lambda_noise=0.0, lambda_recon=0.0):
    # Build ALL canonical parameters first: their RNG order and values stay exact.
    diffusion = build_baseline()
    assert sum(p.numel() for p in diffusion.parameters()) == 13130945
    diffusion.model.unet0.__class__ = JointUnet
    diffusion.model.__class__ = JointUnetRes
    diffusion.__class__ = JointResidualDiffusion
    diffusion.model.unet0.noise_head = nn.Conv2d(64, 1, 1)
    diffusion.model.objective = 'pred_res_noise'
    diffusion.model.test_res_or_noise = 'res_noise'
    diffusion.objective = 'pred_res_noise'
    diffusion.test_res_or_noise = 'res_noise'
    diffusion.set_loss_weights(lambda_noise, lambda_recon)
    assert sum(p.numel() for p in diffusion.parameters()) == 13131010
    assert diffusion.is_ddim_sampling and diffusion.sampling_timesteps == 2
    return diffusion
