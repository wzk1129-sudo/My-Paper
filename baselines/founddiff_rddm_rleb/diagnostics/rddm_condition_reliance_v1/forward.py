"""Exact raw residual path extracted from Bottleneck V1's validated timestep probe.

The archived Bottleneck diagnose.py has this path inline. The new probe checks
all 1152 diagonal control rows against archived TIMESTEP_PROBE.csv before
performing any condition ablation.
"""
import torch


def q_state(model, x_start, x_input, t, noise_seed):
    """Match Bottleneck V1 q_sample and fixed CUDA noise realization."""
    x_res = x_input - x_start
    tt = torch.tensor([int(t)], device=x_start.device, dtype=torch.long)
    torch.manual_seed(int(noise_seed))
    if x_start.device.type == "cuda":
        torch.cuda.manual_seed(int(noise_seed))
    noise = torch.randn_like(x_start)
    xt = model.q_sample(x_start, x_res, tt, noise)
    return xt, x_res


def raw_pred_res(model, xt, ldct_condition, feed_t):
    """Match formal p_losses time inputs and Bottleneck V1 direct model call."""
    tt = torch.tensor([int(feed_t)], device=xt.device, dtype=torch.long)
    time_input = model.alphas_cumsum[tt] * model.num_timesteps
    beta_input = model.betas_cumsum[tt] * model.num_timesteps
    return model.model(torch.cat((xt, ldct_condition), dim=1),
                       [time_input, beta_input], None)[0]
