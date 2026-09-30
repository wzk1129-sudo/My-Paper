from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from evaluation import evaluate_dataset
from rddm_ct import (
    JsonlLogger,
    OFFICIAL_SOURCE,
    PairedCTDataset,
    atomic_torch_save,
    build_system,
    config_sha256,
    git_state,
    load_config,
    seed_everything,
    sha256_file,
    write_json,
)


def checkpoint_payload(model, optimizer, scaler, step, config_path, best_psnr):
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "step": int(step),
        "best_validation_psnr": float(best_psnr),
        "config_sha256": config_sha256(config_path),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for formal RDDM-CT training")
    device = torch.device("cuda:0")
    seed_everything(int(config["seed"]))
    training = config["training"]
    output_dir = Path(training["output_dir"]).resolve()
    if args.resume is None and output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError("Refusing to overwrite non-empty from-scratch output: {}".format(output_dir))
    if args.resume is not None and not args.resume.resolve().is_file():
        raise FileNotFoundError(str(args.resume.resolve()))
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(output_dir / "train.jsonl")

    expected_official_hash = str(config["diffusion"]["official_source_sha256"])
    actual_official_hash = sha256_file(OFFICIAL_SOURCE)
    if actual_official_hash != expected_official_hash:
        raise RuntimeError("Official diffusion source hash mismatch")

    model, diffusion, parameter_count = build_system(config, device)
    optimizer = Adam(model.parameters(), lr=float(training["learning_rate"]))
    amp_enabled = bool(training["amp"])
    amp_init_scale = float(training.get("amp_init_scale", 65536.0))
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled, init_scale=amp_init_scale)
    step = 0
    best_psnr = float("-inf")
    if args.resume is not None:
        checkpoint = torch.load(str(args.resume.resolve()), map_location=device)
        if checkpoint.get("config_sha256") != config_sha256(config_path):
            raise RuntimeError("Resume checkpoint config hash mismatch")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        step = int(checkpoint["step"])
        best_psnr = float(checkpoint["best_validation_psnr"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])

    data_config = config["data"]
    train_dataset = PairedCTDataset(data_config["root"], data_config["train_patients"], data_config)
    validation_dataset = PairedCTDataset(data_config["root"], data_config["validation_patients"], data_config)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(int(config["seed"]))
    loader = DataLoader(
        train_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        num_workers=int(training["num_workers"]),
        pin_memory=True,
        drop_last=True,
        generator=loader_generator,
    )
    manifest = {
        "name": config["name"],
        "started_at_unix": time.time(),
        "command": " ".join([sys.executable] + sys.argv),
        "cwd": os.getcwd(),
        "git": git_state(),
        "config_path": str(config_path),
        "config_sha256": config_sha256(config_path),
        "official_source": str(OFFICIAL_SOURCE),
        "official_source_sha256": actual_official_hash,
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0),
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
        },
        "tmux_session": os.environ.get("RDDM_CT_TMUX_SESSION"),
        "log_path": os.environ.get("RDDM_CT_LOG_PATH"),
        "output_dir": str(output_dir),
        "resume": str(args.resume.resolve()) if args.resume is not None else None,
        "parameter_count": parameter_count,
        "train_dataset": train_dataset.audit(),
        "validation_dataset": validation_dataset.audit(),
        "training": training,
    }
    write_json(output_dir / "run_manifest.json", manifest)
    logger.log(event="start", step=step, parameter_count=parameter_count, batch_size=int(training["batch_size"]), amp=amp_enabled)

    total_steps = int(training["total_steps"])
    log_every = int(training["log_every"])
    validation_every = int(training["validation_every"])
    checkpoint_every = int(training["checkpoint_every"])
    iterator = iter(loader)
    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    while step < total_steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        context = batch["ldct_context"].to(device, non_blocking=True)
        target = batch["ndct_target"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            losses = diffusion.training_losses(target, context)
        if not torch.isfinite(losses["loss"]):
            raise FloatingPointError("Non-finite training loss at step {}".format(step + 1))
        scale_before = float(scaler.get_scale())
        scaler.scale(losses["loss"]).backward()
        scaler.unscale_(optimizer)
        scaler_state = scaler._per_optimizer_states[id(optimizer)]
        grad_found_inf = sum(float(value.item()) for value in scaler_state["found_inf_per_device"].values())
        scaler.step(optimizer)
        scaler.update()
        scale_after = float(scaler.get_scale())
        optimizer_step_skipped = grad_found_inf > 0.0
        torch.cuda.synchronize(device)
        step += 1
        elapsed = time.perf_counter() - started
        if step == 1 or step % log_every == 0:
            logger.log(
                event="train",
                step=step,
                loss=float(losses["loss"].detach()),
                loss_residual=float(losses["loss_residual"].detach()),
                loss_noise=float(losses["loss_noise"].detach()),
                learning_rate=float(optimizer.param_groups[0]["lr"]),
                iteration_seconds=elapsed,
                peak_vram_mib=torch.cuda.max_memory_allocated(device) / (1024.0 ** 2),
                seed=int(config["seed"]),
                patient_ids=[str(value) for value in batch["patient_id"]],
                slice_ids=[int(value) for value in batch["slice_index"].tolist()],
                timesteps=[int(value) for value in losses["t"].detach().cpu().tolist()],
                scaler_scale_before=scale_before,
                scaler_scale_after=scale_after,
                grad_found_inf=grad_found_inf,
                optimizer_step_skipped=optimizer_step_skipped,
            )
        if step % validation_every == 0:
            validation_dir = output_dir / "validation" / "step_{:06d}".format(step)
            result = evaluate_dataset(
                diffusion,
                validation_dataset,
                config,
                device,
                output_dir=validation_dir,
                stride=int(training["validation_slice_stride"]),
                save_images=True,
            )
            psnr = float(result["metrics"]["overall_patient_mean"]["psnr"])
            logger.log(event="validation", step=step, **result["metrics"]["overall_patient_mean"])
            model.train()
            payload = checkpoint_payload(model, optimizer, scaler, step, config_path, max(best_psnr, psnr))
            atomic_torch_save(payload, output_dir / "last.pt")
            if psnr > best_psnr:
                best_psnr = psnr
                payload["best_validation_psnr"] = best_psnr
                atomic_torch_save(payload, output_dir / "best_psnr.pt")
        elif step % checkpoint_every == 0:
            atomic_torch_save(checkpoint_payload(model, optimizer, scaler, step, config_path, best_psnr), output_dir / "last.pt")
    final_payload = checkpoint_payload(model, optimizer, scaler, step, config_path, best_psnr)
    atomic_torch_save(final_payload, output_dir / "last.pt")
    atomic_torch_save(final_payload, output_dir / "final.pt")
    logger.log(event="complete", step=step, best_validation_psnr=best_psnr)


if __name__ == "__main__":
    main()
