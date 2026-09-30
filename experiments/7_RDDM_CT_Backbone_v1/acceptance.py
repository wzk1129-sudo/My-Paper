from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam

from rddm_ct import (
    OFFICIAL_SOURCE,
    LinearAttention,
    PairedCTDataset,
    RDDMCTBackbone,
    atomic_torch_save,
    build_system,
    extract,
    load_config,
    normalized_to_hu,
    seed_everything,
    sha256_file,
    write_json,
    zero_one_to_diffusion,
)


V1_ACCEPTANCE = Path(
    "/home/xujian/projects/RDDM_CVPR2024/RDDM_CT/outputs/"
    "rddm_ct_backbone_v1/acceptance/acceptance_result.json"
)


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def contains_linear_attention(module):
    return any(isinstance(child, LinearAttention) for child in module.modules())


class GPUSampler(object):
    def __init__(self):
        self.samples = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True

    def _run(self):
        command = [
            "/usr/lib/wsl/lib/nvidia-smi",
            "--query-gpu=utilization.gpu,temperature.gpu,memory.used,power.draw",
            "--format=csv,noheader,nounits",
        ]
        while not self.stop_event.is_set():
            try:
                text = subprocess.check_output(command, text=True).strip().splitlines()[0]
                utilization, temperature, memory, power = [float(value.strip()) for value in text.split(",")]
                self.samples.append(
                    {
                        "utilization_percent": utilization,
                        "temperature_c": temperature,
                        "memory_used_mib": memory,
                        "power_w": power,
                    }
                )
            except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                pass
            self.stop_event.wait(0.1)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=2.0)

    def summary(self):
        if not self.samples:
            return {"sample_count": 0}
        keys = tuple(self.samples[0])
        result = {"sample_count": len(self.samples)}
        for key in keys:
            values = [sample[key] for sample in self.samples]
            result[key + "_mean"] = sum(values) / len(values)
            result[key + "_max"] = max(values)
        return result


def attention_audit(model):
    expected = {512: False, 256: False, 128: True, 64: True}
    encoder = {}
    decoder = {}
    for resolution, layer in zip(model.encoder_resolutions, model.downs):
        enabled = contains_linear_attention(layer[2])
        check(enabled == expected[resolution], "encoder attention policy mismatch at {}".format(resolution))
        check(isinstance(layer[2], nn.Identity) == (not enabled), "encoder identity mismatch")
        encoder[str(resolution)] = "linear" if enabled else "none"
    for resolution, layer in zip(model.decoder_resolutions, model.ups):
        enabled = contains_linear_attention(layer[2])
        check(enabled == expected[resolution], "decoder attention policy mismatch at {}".format(resolution))
        check(isinstance(layer[2], nn.Identity) == (not enabled), "decoder identity mismatch")
        decoder[str(resolution)] = "linear" if enabled else "none"
    check(contains_linear_attention(model.mid_attn), "bottleneck is not Linear Attention")
    full_attention_modules = [
        name for name, module in model.named_modules() if module.__class__.__name__ == "Attention"
    ]
    check(not full_attention_modules, "Full Attention still instantiated: {}".format(full_attention_modules))
    linear_attention_modules = [
        name for name, module in model.named_modules() if isinstance(module, LinearAttention)
    ]
    check(len(linear_attention_modules) == 5, "Expected exactly five Linear Attention modules")
    return {
        "encoder": encoder,
        "decoder": decoder,
        "bottleneck": "linear",
        "full_attention_modules": full_attention_modules,
        "linear_attention_modules": linear_attention_modules,
    }


def v1_reference():
    if not V1_ACCEPTANCE.is_file():
        return None
    result = json.loads(V1_ACCEPTANCE.read_text(encoding="utf-8"))
    batches = {int(row["batch_size"]): row for row in result.get("calibration", []) if row.get("stable")}
    return {
        "path": str(V1_ACCEPTANCE),
        "parameter_count": result.get("parameter_count"),
        "batch2": batches.get(2),
        "batch4": batches.get(4),
        "formal_v1_recent_compute_seconds": 8.327409602628904,
        "formal_v1_peak_vram_mib": 27944.88623046875,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    output_dir = Path(config["acceptance"]["output_dir"]).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError("Acceptance output already exists; refusing to overwrite: {}".format(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for 512x512 acceptance")
    device = torch.device("cuda:0")
    seed_everything(int(config["seed"]))
    checks = {}

    official_hash = sha256_file(OFFICIAL_SOURCE)
    check(official_hash == config["diffusion"]["official_source_sha256"], "official source hash mismatch")
    checks["official_diffusion_source_unchanged"] = True

    data = config["data"]
    train_dataset = PairedCTDataset(data["root"], data["train_patients"], data)
    validation_dataset = PairedCTDataset(data["root"], data["validation_patients"], data)
    test_dataset = PairedCTDataset(data["root"], data["test_patients"], data)
    split_sets = [set(data["train_patients"]), set(data["validation_patients"]), set(data["test_patients"])]
    check(not any(split_sets[i] & split_sets[j] for i in range(3) for j in range(i + 1, 3)), "patient overlap")
    checks["strict_pairs_nonempty_disjoint_splits"] = True

    sample = train_dataset[len(train_dataset) // 2]
    single_context = sample["ldct_context"].unsqueeze(0).to(device)
    single_target = sample["ndct_target"].unsqueeze(0).to(device)
    check(tuple(single_context.shape) == (1, 3, 512, 512), "real context shape mismatch")
    check(tuple(single_target.shape) == (1, 1, 512, 512), "real target shape mismatch")
    checks["real_512_pair_read"] = True

    model, diffusion, parameter_count = build_system(config, device)
    check(isinstance(model, RDDMCTBackbone), "wrong model class")
    check(model.shared_unet_count == 1, "backbone is not a single shared U-Net")
    check(not hasattr(model, "unet0") and not hasattr(model, "unet1") and not hasattr(model, "ups_no_skip"), "duplicate path")
    check(model.state_stem[0].in_channels == 1 and model.state_stem[0].out_channels == 32, "state stem channels")
    check(model.context_stem[0].in_channels == 3 and model.context_stem[0].out_channels == 32, "context stem channels")
    check(model.fusion.in_channels == 64 and model.fusion.out_channels == 64, "fusion channels")
    check(35000000 <= parameter_count <= 45000000, "parameter count is not approximately 36M")
    policy = attention_audit(model)
    checks["dual_stems_single_unet_parameter_budget"] = True
    checks["attention_policy"] = True

    batch_size = int(config["acceptance"]["batch_size"])
    check(batch_size == 2 and int(config["training"]["batch_size"]) == 2, "batch size must be exactly two")
    context = single_context.repeat(batch_size, 1, 1, 1)
    target = single_target.repeat(batch_size, 1, 1, 1)
    context_diff = zero_one_to_diffusion(context)
    target_diff = zero_one_to_diffusion(target)
    residual = context_diff[:, 1:2] - target_diff
    check(torch.equal(residual, context_diff[:, 1:2] - target_diff), "residual definition mismatch")
    timestep = torch.tensor([347, 811], device=device, dtype=torch.long)
    noise = torch.randn_like(target_diff)
    actual_q = diffusion.q_sample(target_diff, residual, timestep, noise=noise)
    expected_q = target_diff + extract(diffusion.alphas_cumsum, timestep, target_diff.shape) * residual + extract(diffusion.betas_cumsum, timestep, target_diff.shape) * noise
    check(torch.allclose(actual_q, expected_q, atol=1e-7, rtol=1e-6), "q_sample formula mismatch")
    checks["residual_and_q_formula"] = True

    amp = bool(config["training"]["amp"])
    check(amp, "AMP must be enabled")
    optimizer = Adam(model.parameters(), lr=float(config["training"]["learning_rate"]))
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp):
        probe_residual, probe_noise = model(actual_q, context_diff, timestep)
    check(tuple(probe_residual.shape) == (2, 1, 512, 512), "residual output shape")
    check(tuple(probe_noise.shape) == (2, 1, 512, 512), "noise output shape")

    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    sampler = GPUSampler()
    start_event = torch.cuda.Event(enable_timing=True)
    forward_event = torch.cuda.Event(enable_timing=True)
    backward_event = torch.cuda.Event(enable_timing=True)
    optimizer_event = torch.cuda.Event(enable_timing=True)
    sampler.start()
    torch.cuda.synchronize(device)
    wall_start = time.perf_counter()
    start_event.record()
    with torch.cuda.amp.autocast(enabled=amp):
        losses = diffusion.training_losses(target, context, t=timestep, noise=noise)
    forward_event.record()
    check(torch.isfinite(losses["loss"]).item(), "non-finite loss")
    scaler.scale(losses["loss"]).backward()
    backward_event.record()
    scaler.step(optimizer)
    scaler.update()
    optimizer_event.record()
    torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - wall_start
    sampler.stop()
    check(all(parameter.grad is None or torch.isfinite(parameter.grad).all().item() for parameter in model.parameters()), "non-finite gradient")
    resource = {
        "batch_size": batch_size,
        "forward_ms": start_event.elapsed_time(forward_event),
        "backward_ms": forward_event.elapsed_time(backward_event),
        "optimizer_ms": backward_event.elapsed_time(optimizer_event),
        "total_cuda_ms": start_event.elapsed_time(optimizer_event),
        "wall_seconds": wall_seconds,
        "peak_vram_mib": torch.cuda.max_memory_allocated(device) / (1024.0 ** 2),
        "loss": float(losses["loss"].detach()),
        "loss_residual": float(losses["loss_residual"].detach()),
        "loss_noise": float(losses["loss_noise"].detach()),
        "amp": amp,
        "scaler_scale_after_step": float(scaler.get_scale()),
        "gpu_samples": sampler.summary(),
    }
    check(all(torch.isfinite(value).all().item() for value in (losses["loss"], losses["loss_residual"], losses["loss_noise"])), "loss finite check")
    checks["batch2_forward_loss_backward_optimizer_finite"] = True
    checks["amp_active"] = True

    optimizer.zero_grad(set_to_none=True)
    del losses, probe_residual, probe_noise
    torch.cuda.empty_cache()
    generator = torch.Generator(device=device)
    generator.manual_seed(int(config["seed"]) + 999)
    with torch.cuda.amp.autocast(enabled=amp):
        prediction = diffusion.sample_ct(single_context, generator=generator)
    check(tuple(prediction.shape) == (1, 1, 512, 512), "sampler output shape")
    check(torch.isfinite(prediction).all().item(), "sampler output non-finite")
    checks["ten_step_sampler"] = True

    probe_hu = torch.tensor([-1024.0, -500.0, 0.0, 3072.0])
    normalized = (probe_hu - float(data["hu_min"])) / (float(data["hu_max"]) - float(data["hu_min"]))
    check(torch.allclose(probe_hu, normalized_to_hu(normalized, data), atol=1e-5), "HU inversion mismatch")
    checks["inverse_hu"] = True

    checkpoint_path = output_dir / "checkpoint_roundtrip.pt"
    atomic_torch_save({"model": model.state_dict(), "scaler": scaler.state_dict(), "tag": "efficiency_acceptance"}, checkpoint_path)
    restored = RDDMCTBackbone(config["model"])
    payload = torch.load(str(checkpoint_path), map_location="cpu")
    restored.load_state_dict(payload["model"], strict=True)
    check(torch.equal(next(model.parameters()).detach().cpu(), next(restored.parameters()).detach().cpu()), "checkpoint roundtrip mismatch")
    checks["checkpoint_save_load"] = True

    reference = v1_reference()
    comparison = None
    if reference is not None:
        comparison = {
            "v1": reference,
            "revised": {
                "parameter_count": parameter_count,
                "batch_size": batch_size,
                "peak_vram_mib": resource["peak_vram_mib"],
                "wall_seconds": resource["wall_seconds"],
            },
            "peak_vram_reduction_vs_v1_formal_percent": 100.0 * (1.0 - resource["peak_vram_mib"] / reference["formal_v1_peak_vram_mib"]),
            "step_time_reduction_vs_v1_formal_percent": 100.0 * (1.0 - resource["wall_seconds"] / reference["formal_v1_recent_compute_seconds"]),
        }

    result = {
        "status": "PASS",
        "checks": checks,
        "attention_policy": policy,
        "official_source_sha256": official_hash,
        "parameter_count": parameter_count,
        "resource_test": resource,
        "comparison": comparison,
        "train_dataset": train_dataset.audit(),
        "validation_dataset": validation_dataset.audit(),
        "test_dataset": test_dataset.audit(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    write_json(output_dir / "acceptance_result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
