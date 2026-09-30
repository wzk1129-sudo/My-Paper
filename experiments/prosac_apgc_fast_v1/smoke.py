"""No-optimizer unit and physical-batch smoke for matched M0/M1."""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.optim import Adam

import compat  # noqa: F401; import-only shims for unused official Trainer
HERE = Path(__file__).resolve().parent
FROZEN = HERE.parent / "7_RDDM_CT_Backbone_v1"
sys.path.insert(0, str(FROZEN))
from rddm_ct import (PairedCTDataset, RDDMCTBackbone, RDDMCTDiffusion,
                     load_config, seed_everything, zero_one_to_diffusion)
from apgc_model import APGCBackbone
from loss111 import compute_loss111
from train_matched import TrainableEMA


def backbone_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if name.startswith("apgc."):
            continue
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def unit_diffusion(diffusion):
    device = next(diffusion.parameters()).device
    ndct = torch.rand(4, 1, 16, 16, device=device) * 2 - 1
    ldct = torch.rand_like(ndct) * 2 - 1
    eps = torch.randn_like(ndct)
    residual = ldct - ndct
    t = torch.tensor([0, 10, 500, 999], device=device)
    a = diffusion.alphas_cumsum[t].reshape(-1, 1, 1, 1)
    b = diffusion.betas_cumsum[t].reshape(-1, 1, 1, 1)
    actual = diffusion.q_sample(ndct, residual, t, noise=eps)
    error = float((actual - (ndct + a * residual + b * eps)).abs().max())
    a999 = float(diffusion.alphas_cumsum[999])
    b999 = float(diffusion.betas_cumsum[999])
    terminal = float((actual[-1] - (ldct[-1] + eps[-1])).abs().max())
    assert error < 2e-6 and terminal < 2e-6
    assert abs(a999 - 1) < 2e-6 and abs(b999 - 1) < 2e-6
    return {"q_sample_max_abs_error": error, "A999": a999, "B999": b999,
            "sampler_terminal_max_abs_error": terminal}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("M0_LOSS111", "M1_APGC_LOSS111"), required=True)
    args = parser.parse_args()
    config = load_config(HERE / ("CONFIG_" + args.arm + ".json"))
    assert config["data"]["train_patients"] == [
        "L067", "L096", "L109", "L143", "L192", "L286", "L291", "L310"]
    assert config["data"]["validation_patients"] == ["L333"]
    assert set(config["data"]["test_patients"]).isdisjoint(
        set(config["data"]["train_patients"] + config["data"]["validation_patients"]))
    assert int(config["training"]["batch_size"]) == 4
    assert not (HERE / "runs" / args.arm).exists()
    data = PairedCTDataset(Path(config["data"]["root"]), ["L067"], config["data"])
    batch = [data[i] for i in range(4)]
    contexts = torch.stack([item["ldct_context"] for item in batch])
    targets = torch.stack([item["ndct_target"] for item in batch])
    assert contexts.shape == (4, 3, 512, 512) and targets.shape == (4, 1, 512, 512)
    assert all(item["patient_id"] == "L067" for item in batch)
    assert bool(torch.isfinite(contexts).all() and torch.isfinite(targets).all())
    assert 0 <= float(contexts.min()) <= float(contexts.max()) <= 1
    assert 0 <= float(targets.min()) <= float(targets.max()) <= 1
    seed_everything(int(config["seed"]))
    canonical = RDDMCTBackbone(config["model"])
    canonical_hash = backbone_hash(canonical)
    if args.arm == "M0_LOSS111":
        model = canonical
    else:
        del canonical
        seed_everything(int(config["seed"]))
        model = APGCBackbone(config["model"])
        assert backbone_hash(model) == canonical_hash
        assert not any(parameter.requires_grad for parameter in model.apgc.med.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    device = torch.device("cuda:0")
    model = model.to(device)
    diffusion = RDDMCTDiffusion(model, config["diffusion"], 512).to(device)
    units = unit_diffusion(diffusion)
    contexts = contexts.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    assert float(zero_one_to_diffusion(contexts).min()) >= -1.00001
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    shared_params = [p for name, p in model.named_parameters()
                     if p.requires_grad and not name.startswith(
                         ("apgc.", "residual_head.", "noise_head."))]
    optimizer = Adam(trainable_params, lr=float(config["training"]["learning_rate"]))
    ema = TrainableEMA(model, config["training"]["ema_decay"])
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model.train()
    start = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        ndct = zero_one_to_diffusion(targets)
        context = zero_one_to_diffusion(contexts)
        center = context[:, 1:2]
        t = torch.randint(0, 1000, (4,), device=device)
        eps = torch.randn_like(ndct)
        residual = center - ndct
        x_t = diffusion.q_sample(ndct, residual, t, noise=eps)
        pred_res, pred_noise = model(x_t, context, t)
        assert pred_res.shape == pred_noise.shape == targets.shape
        components = compute_loss111(
            diffusion, x_t, t, pred_res, pred_noise, residual, eps, ndct)
        loss = components["loss_total"]
    assert bool(torch.isfinite(pred_res).all() and torch.isfinite(pred_noise).all())
    assert bool(torch.isfinite(components["x0_hat"]).all())
    assert all(bool(torch.isfinite(components[k])) for k in
               ("loss_res", "loss_noise", "loss_recon", "loss_total"))
    component_grad_norms = {}
    for key in ("loss_res", "loss_noise", "loss_recon"):
        grads = torch.autograd.grad(
            components[key], shared_params, retain_graph=True, allow_unused=False)
        assert all(bool(torch.isfinite(g).all()) for g in grads)
        component_grad_norms[key] = float(torch.sqrt(
            sum(g.detach().float().square().sum() for g in grads)))
        del grads
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None and bool(torch.isfinite(g).all()) for g in gradients)
    optimizer.step()
    ema.update(model)
    model.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        sample = diffusion.sample_ct(
            contexts[:1], generator=torch.Generator(device="cuda").manual_seed(20260829))
    assert sample.shape == (1, 1, 512, 512)
    assert bool(torch.isfinite(sample).all())
    torch.cuda.synchronize()
    steady_allocated_gib = torch.cuda.memory_allocated(device) / 2**30
    peak_reserved_gib = torch.cuda.max_memory_reserved(device) / 2**30
    output = {
        "status": "PASS", "arm": args.arm, "optimizer_steps": 1,
        "patient_ids_opened": ["L067"], "validation_images_opened": 0,
        "forbidden_test_opens": 0, "external_memory_opens": 0,
        "model_init_backbone_sha256": canonical_hash,
        "total_parameters": total, "trainable_parameters": trainable,
        "physical_batch": 4, "context_shape": list(contexts.shape),
        "target_shape": list(targets.shape),
        "sample_shape": list(sample.shape),
        "sample_min": float(sample.min()),
        "sample_max": float(sample.max()),
        "sampler_steps": int(diffusion.sampling_timesteps),
        "loss_res": float(components["loss_res"].detach()),
        "loss_noise": float(components["loss_noise"].detach()),
        "loss_recon": float(components["loss_recon"].detach()),
        "loss_total": float(loss.detach()),
        "shared_backbone_component_grad_norms": component_grad_norms,
        "grad_norm": float(torch.linalg.vector_norm(torch.stack(
            [g.detach().float().norm() for g in gradients]))),
        "elapsed_seconds": elapsed,
        "peak_vram_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "steady_with_adam_ema_gib": steady_allocated_gib,
        "peak_reserved_gib": peak_reserved_gib,
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__,
                        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device)},
        "unit": units,
    }
    if args.arm == "M1_APGC_LOSS111":
        output["alpha_grad_norm"] = float(model.apgc.alpha.grad.float().norm())
        output["gamma_grad_norm"] = float(model.apgc.gamma.grad.float().norm())
        output["aefm"] = model.apgc.last_activity
        assert output["alpha_grad_norm"] > 0 and math.isfinite(output["alpha_grad_norm"])
        assert output["gamma_grad_norm"] > 0 and math.isfinite(output["gamma_grad_norm"])
        assert output["aefm"]["aefm_rms"] > 0 and math.isfinite(output["aefm"]["aefm_rms"])
    output_path = HERE / ("SMOKE_" + args.arm + ".json")
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: output[key] for key in ("status", "arm", "loss_total",
          "peak_vram_gib", "elapsed_seconds", "trainable_parameters")}), flush=True)


if __name__ == "__main__":
    main()
