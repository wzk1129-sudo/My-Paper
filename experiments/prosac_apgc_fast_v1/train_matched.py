"""Matched 8k M0/M1 runner. This file is prepared only; launch requires user confirmation."""
import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam

import compat  # noqa: F401, import-only shims for unused official Trainer
HERE = Path(__file__).resolve().parent
FROZEN = HERE.parent / "7_RDDM_CT_Backbone_v1"
sys.path.insert(0, str(FROZEN))
from evaluation import slice_metrics
from rddm_ct import (PairedCTDataset, RDDMCTBackbone, RDDMCTDiffusion,
                     load_config, seed_everything, zero_one_to_diffusion)
from apgc_model import APGCBackbone
from loss111 import compute_loss111


class TrainableEMA:
    def __init__(self, model, beta):
        self.beta = float(beta)
        self.shadow = {name: p.detach().clone() for name, p in model.named_parameters()
                       if p.requires_grad}

    @torch.no_grad()
    def update(self, model):
        for name, p in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].lerp_(p.detach(), 1. - self.beta)

    @torch.no_grad()
    def swap_in(self, model):
        saved = {}
        for name, p in model.named_parameters():
            if name in self.shadow:
                saved[name] = p.detach().cpu().clone()
                p.copy_(self.shadow[name])
        return saved

    @staticmethod
    @torch.no_grad()
    def restore(model, saved):
        for name, p in model.named_parameters():
            if name in saved:
                p.copy_(saved[name].to(device=p.device))


def edge_metrics(prediction, target, config):
    lo, hi = float(config["data"]["hu_min"]), float(config["data"]["hu_max"])
    # Edge region = top quartile of NDCT Sobel magnitude. Definition is frozen
    # and identical for both arms; paper does not specify this metric.
    target_hu = (target.float() * (hi - lo) + lo).clamp(lo, hi)
    pred_hu = (prediction.float() * (hi - lo) + lo).clamp(lo, hi)
    sx = target_hu[..., :, 2:] - target_hu[..., :, :-2]
    sy = target_hu[..., 2:, :] - target_hu[..., :-2, :]
    gx = F.pad(sx, (1, 1, 0, 0))
    gy = F.pad(sy, (0, 0, 1, 1))
    magnitude = torch.sqrt(gx.square() + gy.square())
    threshold = torch.quantile(magnitude.flatten(), .75)
    mask = magnitude >= threshold
    error = (pred_hu - target_hu)[mask]
    return float(error.square().mean().sqrt()), float(error.abs().mean())


@torch.no_grad()
def validate(model, diffusion, config, dataset, indices, step):
    model.eval()
    rows = []
    started = time.perf_counter()
    for position in indices:
        item = dataset[position]
        context = item["ldct_context"].unsqueeze(0).cuda()
        target = item["ndct_target"].unsqueeze(0)
        generator = torch.Generator(device="cuda").manual_seed(
            int(config["evaluation"]["validation_seed"]) + int(position))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = diffusion.sample_ct(context, generator=generator)
        metrics = slice_metrics(prediction[0], target[0], config)
        edge_rmse, edge_mae = edge_metrics(prediction[0].cpu(), target[0], config)
        rows.append({"step": step, "position": position,
                     "patient_id": "L333", "slice_index": int(item["slice_index"]),
                     "edge_rmse_hu": edge_rmse, "edge_mae_hu": edge_mae,
                     **{name: metrics[name] for name in
                        ("psnr", "ssim", "rmse_hu", "mae_hu")}})
    keys = ("psnr", "ssim", "rmse_hu", "mae_hu", "edge_rmse_hu", "edge_mae_hu")
    summary = {"step": step, "n_slices": len(rows),
               "seconds": time.perf_counter() - started,
               **{key: sum(row[key] for row in rows) / len(rows) for key in keys}}
    model.train()
    return summary, rows


def make_batch(dataset, step, seed, batch_size):
    chooser = random.Random(seed + 1000003 * step)
    positions = [chooser.randrange(len(dataset)) for _ in range(batch_size)]
    transforms = [(chooser.random() < .5, chooser.randrange(4)) for _ in positions]
    loaded = [dataset[position] for position in positions]
    contexts, targets = [], []
    for item, (flip, rotations) in zip(loaded, transforms):
        context = item["ldct_context"]
        target = item["ndct_target"]
        if flip:
            context, target = context.flip(-1), target.flip(-1)
        if rotations:
            context = torch.rot90(context, rotations, (-2, -1))
            target = torch.rot90(target, rotations, (-2, -1))
        contexts.append(context)
        targets.append(target)
    identity = {"step": step, "positions": positions,
                "patient_ids": [x["patient_id"] for x in loaded],
                "slice_indices": [int(x["slice_index"]) for x in loaded],
                "transforms": transforms}
    return torch.stack(contexts), torch.stack(targets), identity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("M0_LOSS111", "M1_APGC_LOSS111"), required=True)
    parser.add_argument("--max-step", type=int, default=8000)
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()
    if args.max_step not in (4000, 8000):
        raise ValueError("Only frozen 4k gate or 8k terminal is allowed")
    config_path = HERE / ("CONFIG_" + args.arm + ".json")
    config = load_config(config_path)
    assert int(config["training"]["batch_size"]) == 4
    output = Path(config["training"]["output_dir"])
    if args.resume is None:
        if output.exists():
            raise RuntimeError("Refusing to overwrite existing arm output: " + str(output))
        output.mkdir(parents=True)
        (output / "CONFIG.json").write_text(config_path.read_text())
    else:
        if not output.is_dir() or (output / "CONFIG.json").read_text() != config_path.read_text():
            raise RuntimeError("Resume config/output mismatch")
        if args.resume.resolve().parent != output.resolve():
            raise RuntimeError("Resume checkpoint must belong to this arm output")
    seed = int(config["seed"])
    seed_everything(seed)
    model_class = RDDMCTBackbone if args.arm == "M0_LOSS111" else APGCBackbone
    model = model_class(config["model"]).cuda()
    diffusion = RDDMCTDiffusion(model, config["diffusion"], 512).cuda()
    optimizer = Adam((p for p in model.parameters() if p.requires_grad),
                     lr=float(config["training"]["learning_rate"]))
    ema = TrainableEMA(model, config["training"]["ema_decay"])
    train = PairedCTDataset(Path(config["data"]["root"]),
                            config["data"]["train_patients"], config["data"])
    validation = PairedCTDataset(Path(config["data"]["root"]), ["L333"], config["data"])
    positions = [int(x) for x in (HERE / "L333_fast64.txt").read_text().split()]
    assert len(positions) == len(set(positions)) == 64
    assert min(positions) >= 0 and max(positions) < len(validation)
    stream_state = bytes(32)
    history = []
    start_step = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
        if unexpected or set(missing) - {key for key in model.state_dict() if key.startswith("apgc.med.")}:
            raise RuntimeError("Unexpected checkpoint model keys")
        optimizer.load_state_dict(checkpoint["optimizer"])
        ema.shadow = {k: v.cuda() for k, v in checkpoint["ema_trainable"].items()}
        stream_state = bytes.fromhex(checkpoint["train_stream_sha256"])
        start_step = int(checkpoint["step"])
        torch.set_rng_state(checkpoint["torch_rng"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        if (output / "validation_history.json").is_file():
            history = json.loads((output / "validation_history.json").read_text())
    if start_step >= args.max_step:
        raise RuntimeError("Resume checkpoint is already at or beyond max step")
    torch.cuda.reset_peak_memory_stats()
    log_mode = "a" if args.resume is not None else "w"
    with (output / "train.jsonl").open(log_mode) as log:
        for step in range(start_step + 1, args.max_step + 1):
            started = time.perf_counter()
            context_cpu, target_cpu, identity = make_batch(
                train, step, seed, int(config["training"]["batch_size"]))
            context = context_cpu.cuda()
            target = target_cpu.cuda()
            diffusion_context = zero_one_to_diffusion(context)
            ndct = zero_one_to_diffusion(target)
            residual = diffusion_context[:, 1:2] - ndct
            noise_seed = seed + 2000003 * step
            g = torch.Generator(device="cuda").manual_seed(noise_seed)
            t = torch.randint(0, 1000, (target.shape[0],), device="cuda", generator=g)
            noise = torch.randn(ndct.shape, device="cuda", dtype=ndct.dtype, generator=g)
            stream_entry = {"batch": identity, "noise_seed": noise_seed,
                            "t": t.tolist()}
            stream_state = hashlib.sha256(
                stream_state + json.dumps(stream_entry, sort_keys=True).encode()).digest()
            x_t = diffusion.q_sample(ndct, residual, t, noise=noise)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred_res, pred_noise = model(x_t, diffusion_context, t)
                components = compute_loss111(
                    diffusion, x_t, t, pred_res, pred_noise,
                    residual, noise, ndct)
                loss = components["loss_total"]
            if not all(bool(torch.isfinite(components[key])) for key in
                       ("loss_res", "loss_noise", "loss_recon", "loss_total")):
                raise RuntimeError("Non-finite loss at step " + str(step))
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.requires_grad]
            if not all(g is not None and bool(torch.isfinite(g).all()) for g in gradients):
                raise RuntimeError("Non-finite or missing gradient at step " + str(step))
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ema.update(model)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            line = {"step": step,
                    "loss_res": float(components["loss_res"].detach()),
                    "loss_noise": float(components["loss_noise"].detach()),
                    "loss_recon": float(components["loss_recon"].detach()),
                    "loss_total": float(loss.detach()),
                    "gradient_norm_before_clip": float(grad_norm),
                    "seconds_per_step": time.perf_counter() - started,
                    "vram_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "train_stream_sha256": stream_state.hex()}
            if args.arm == "M1_APGC_LOSS111":
                line["alpha_mean"] = float(model.apgc.alpha.detach().mean())
                line["alpha_std"] = float(model.apgc.alpha.detach().std())
                line["gamma_mean"] = float(model.apgc.gamma.detach().mean())
                line["gamma_std"] = float(model.apgc.gamma.detach().std())
                line["aefm_mean"] = float(model.apgc.last_activity["aefm_mean"])
                line["aefm_std"] = float(model.apgc.last_activity["aefm_std"])
            log.write(json.dumps(line, sort_keys=True) + "\n")
            log.flush()
            if step in config["training"]["validation_steps"]:
                saved = ema.swap_in(model)
                summary, rows = validate(model, diffusion, config, validation, positions, step)
                TrainableEMA.restore(model, saved)
                history.append(summary)
                with (output / ("validation_" + str(step) + ".csv")).open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                (output / "validation_history.json").write_text(
                    json.dumps(history, indent=2, sort_keys=True) + "\n")
                torch.save({"model": {k: v for k, v in model.state_dict().items()
                              if not k.startswith("apgc.med.")},
                            "ema_trainable": ema.shadow,
                            "optimizer": optimizer.state_dict(), "step": step,
                            "train_stream_sha256": stream_state.hex(),
                            "torch_rng": torch.get_rng_state(),
                            "cuda_rng": torch.cuda.get_rng_state_all(),
                            "config": config}, output / ("step_" + str(step) + ".pt"))


if __name__ == "__main__":
    main()
