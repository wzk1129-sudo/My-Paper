"""Fixed ProSAC-style three-term MSE for both M0 and M1.

The equal 1:1:1 weights are a reconstruction choice: the paper does not
publish the numerical M0/M1 loss weights.
"""
import torch.nn.functional as F


def compute_loss111(diffusion, x_t, t, pred_res, pred_noise,
                    residual, noise, x_start):
    x0_hat = diffusion.predict_start_from_res_noise(
        x_t, t, pred_res, pred_noise)
    loss_res = F.mse_loss(pred_res.float(), residual.float())
    loss_noise = F.mse_loss(pred_noise.float(), noise.float())
    loss_recon = F.mse_loss(x0_hat.float(), x_start.float())
    return {
        "loss_res": loss_res,
        "loss_noise": loss_noise,
        "loss_recon": loss_recon,
        "loss_total": loss_res + loss_noise + loss_recon,
        "x0_hat": x0_hat,
    }
