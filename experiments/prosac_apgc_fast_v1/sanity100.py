"""Independent 100-step loss111 sanity; no checkpoint feeds formal training."""
import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam

import compat  # noqa: F401
HERE = Path(__file__).resolve().parent
FROZEN = HERE.parent / "7_RDDM_CT_Backbone_v1"
sys.path.insert(0, str(FROZEN))
from rddm_ct import (PairedCTDataset, RDDMCTBackbone, RDDMCTDiffusion,
                     load_config, seed_everything, zero_one_to_diffusion)
from apgc_model import APGCBackbone
from loss111 import compute_loss111
from train_matched import TrainableEMA, make_batch


def shared_gradient_norm(loss, shared):
    gradients = torch.autograd.grad(
        loss, shared, retain_graph=True, allow_unused=False)
    if not all(bool(torch.isfinite(g).all()) for g in gradients):
        raise RuntimeError("Non-finite component gradient")
    value = float(torch.sqrt(
        sum(g.detach().float().square().sum() for g in gradients)))
    del gradients
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("M0_LOSS111", "M1_APGC_LOSS111"), required=True)
    args = parser.parse_args()
    config = load_config(HERE / ("CONFIG_" + args.arm + ".json"))
    output = HERE / "sanity100" / args.arm
    if output.exists():
        raise RuntimeError("Refusing to overwrite sanity output")
    output.mkdir(parents=True)
    seed = int(config["seed"])
    seed_everything(seed)
    cls = RDDMCTBackbone if args.arm == "M0_LOSS111" else APGCBackbone
    model = cls(config["model"]).cuda()
    diffusion = RDDMCTDiffusion(model, config["diffusion"], 512).cuda()
    shared = [p for name, p in model.named_parameters()
              if p.requires_grad and not name.startswith(
                  ("apgc.", "residual_head.", "noise_head."))]
    optimizer = Adam((p for p in model.parameters() if p.requires_grad),
                     lr=float(config["training"]["learning_rate"]))
    ema = TrainableEMA(model, config["training"]["ema_decay"])
    dataset = PairedCTDataset(Path(config["data"]["root"]),
                              config["data"]["train_patients"], config["data"])
    chain = bytes(32)
    rows = []
    torch.cuda.reset_peak_memory_stats()
    with (output / "sanity100.jsonl").open("w") as log:
        for step in range(1, 101):
            started = time.perf_counter()
            context_cpu, target_cpu, identity = make_batch(dataset, step, seed, 4)
            context = zero_one_to_diffusion(context_cpu.cuda())
            ndct = zero_one_to_diffusion(target_cpu.cuda())
            residual = context[:, 1:2] - ndct
            noise_seed = seed + 2000003 * step
            g = torch.Generator(device="cuda").manual_seed(noise_seed)
            t = torch.randint(0, 1000, (4,), device="cuda", generator=g)
            noise = torch.randn(ndct.shape, device="cuda", dtype=ndct.dtype, generator=g)
            chain = hashlib.sha256(
                chain + json.dumps({"batch": identity, "noise_seed": noise_seed,
                                    "t": t.tolist()}, sort_keys=True).encode()).digest()
            x_t = diffusion.q_sample(ndct, residual, t, noise=noise)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred_res, pred_noise = model(x_t, context, t)
                components = compute_loss111(
                    diffusion, x_t, t, pred_res, pred_noise, residual, noise, ndct)
            names = ("loss_res", "loss_noise", "loss_recon", "loss_total")
            if not all(bool(torch.isfinite(components[key])) for key in names):
                raise RuntimeError("Non-finite loss at step {}".format(step))
            component_norms = {
                key + "_shared_grad_norm": shared_gradient_norm(components[key], shared)
                for key in ("loss_res", "loss_noise", "loss_recon")}
            components["loss_total"].backward()
            gradients = [p.grad for p in model.parameters() if p.requires_grad]
            if not all(g is not None and bool(torch.isfinite(g).all()) for g in gradients):
                raise RuntimeError("Non-finite gradient at step {}".format(step))
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ema.update(model)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            row = {"step": step,
                   **{key: float(components[key].detach()) for key in names},
                   **component_norms,
                   "gradient_norm_before_clip": float(grad_norm),
                   "seconds_per_step": time.perf_counter() - started,
                   "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                   "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                   "train_stream_sha256": chain.hex()}
            if args.arm == "M1_APGC_LOSS111":
                row.update({
                    "alpha_mean": float(model.apgc.alpha.detach().mean()),
                    "gamma_mean": float(model.apgc.gamma.detach().mean()),
                    "aefm_mean": float(model.apgc.last_activity["aefm_mean"]),
                    "aefm_std": float(model.apgc.last_activity["aefm_std"]),
                })
            if not all(math.isfinite(float(row[key])) for key in
                       names + tuple(component_norms) +
                       ("gradient_norm_before_clip", "seconds_per_step")):
                raise RuntimeError("Non-finite sanity record")
            log.write(json.dumps(row, sort_keys=True) + "\n")
            log.flush()
            rows.append(row)
    tail = rows[50:]
    summary = {
        "status": "PASS", "arm": args.arm, "optimizer_steps": 100,
        "loss_weights": [1., 1., 1.],
        "median_last50": {key: statistics.median(row[key] for row in tail)
                          for key in names},
        "median_last50_shared_gradient_norm": {
            key: statistics.median(row[key] for row in tail)
            for key in component_norms},
        "seconds_per_step_median": statistics.median(
            row["seconds_per_step"] for row in rows),
        "peak_allocated_gib": max(row["peak_allocated_gib"] for row in rows),
        "peak_reserved_gib": max(row["peak_reserved_gib"] for row in rows),
        "train_stream_sha256": chain.hex(),
        "forbidden_test_opens": 0, "external_memory_opens": 0,
        "checkpoint_saved": False,
    }
    (output / "SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
