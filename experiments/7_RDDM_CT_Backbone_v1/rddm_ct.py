from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset


EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
OFFICIAL_EXPERIMENT_ROOT = PROJECT_ROOT / "experiments/2_Image_Restoration_deraing_raindrop_noise1"
OFFICIAL_SOURCE = OFFICIAL_EXPERIMENT_ROOT / "src/residual_denoising_diffusion_pytorch.py"
if str(OFFICIAL_EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_EXPERIMENT_ROOT))

from src.residual_denoising_diffusion_pytorch import (  # noqa: E402
    Downsample,
    LinearAttention,
    ModelResPrediction,
    PreNorm,
    Residual,
    ResidualDiffusion,
    ResnetBlock,
    SinusoidalPosEmb,
    Upsample,
    extract,
)


class SafeWeightStandardizedConv2d(nn.Conv2d):
    """Weight-standardized convolution with an FP32-only numerical path."""

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            x32 = x.float()
            weight32 = self.weight.float()
            bias32 = self.bias.float() if self.bias is not None else None
            mean = weight32.mean(dim=(1, 2, 3), keepdim=True)
            var = weight32.var(dim=(1, 2, 3), unbiased=False, keepdim=True)
            normalized_weight = (weight32 - mean) * (var + 1e-5).rsqrt()
            return F.conv2d(x32, normalized_weight, bias32, self.stride, self.padding, self.dilation, self.groups)


class SafeFP32Head(nn.Conv2d):
    """Prediction head whose input, parameters, and convolution run in FP32."""

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            x32 = x.float()
            weight32 = self.weight.float()
            bias32 = self.bias.float() if self.bias is not None else None
            return F.conv2d(x32, weight32, bias32, self.stride, self.padding, self.dilation, self.groups)


class SafeBlock(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = SafeWeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = self.proj(x.float())
            x = self.norm(x)
            if scale_shift is not None:
                scale, shift = scale_shift
                x = x * (scale.float() + 1.0) + shift.float()
            return self.act(x)


class SafeResnetBlock(nn.Module):
    """Checkpoint-compatible ResnetBlock with FP32 internals."""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2)) if time_emb_dim is not None else None
        self.block1 = SafeBlock(dim, dim_out, groups=groups)
        self.block2 = SafeBlock(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        input_dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            x32 = x.float()
            scale_shift = None
            if self.mlp is not None and time_emb is not None:
                values = self.mlp(time_emb.float()).reshape(time_emb.shape[0], -1, 1, 1)
                scale_shift = values.chunk(2, dim=1)
            h = self.block1(x32, scale_shift=scale_shift)
            h = self.block2(h)
            output = h + self.res_conv(x32)
        return output.to(input_dtype)


FILENAME_RE = re.compile(r"^(?P<patient>L\d{3})_(?P<view>target|\d+)_(?P<slice>\d+)_img\.npy$")


def load_config(path: Path) -> Dict[str, object]:
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {"data", "model", "diffusion", "training", "evaluation", "acceptance"}
    missing = required - set(config)
    if missing:
        raise KeyError("Missing config sections: {}".format(sorted(missing)))
    assert_patient_splits(config["data"])
    return config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_sha256(config_path: Path) -> str:
    return sha256_file(Path(config_path).resolve())


def assert_patient_splits(data_config: Mapping[str, object]) -> None:
    groups = [
        set(data_config["train_patients"]),
        set(data_config["validation_patients"]),
        set(data_config["test_patients"]),
    ]
    for i in range(len(groups)):
        if not groups[i]:
            raise RuntimeError("Patient split {} is empty".format(i))
        for j in range(i + 1, len(groups)):
            overlap = groups[i] & groups[j]
            if overlap:
                raise RuntimeError("Patient split overlap: {}".format(sorted(overlap)))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def normalize_stored_ct(array: np.ndarray, data_config: Mapping[str, object]) -> np.ndarray:
    hu = array.astype(np.float32) + float(data_config["stored_value_offset_hu"])
    hu_min = float(data_config["hu_min"])
    hu_max = float(data_config["hu_max"])
    hu = np.clip(hu, hu_min, hu_max)
    return ((hu - hu_min) / (hu_max - hu_min)).astype(np.float32)


def normalized_to_hu(value, data_config: Mapping[str, object]):
    hu_min = float(data_config["hu_min"])
    hu_max = float(data_config["hu_max"])
    return value * (hu_max - hu_min) + hu_min


def zero_one_to_diffusion(value: torch.Tensor) -> torch.Tensor:
    return value * 2.0 - 1.0


def diffusion_to_zero_one(value: torch.Tensor) -> torch.Tensor:
    return (value + 1.0) * 0.5


@dataclass(frozen=True)
class CTSliceRecord:
    patient_id: str
    slice_index: int
    ldct_path: str
    ndct_path: str
    context_paths: Tuple[str, str, str]
    context_indices: Tuple[int, int, int]


class PairedCTDataset(Dataset):
    def __init__(self, root: Path, patients: Sequence[str], data_config: Mapping[str, object]):
        self.root = Path(root).resolve()
        self.patients = tuple(str(value) for value in patients)
        self.data_config = dict(data_config)
        self.dose_label = str(data_config["dose_label"])
        if not self.root.is_dir():
            raise FileNotFoundError(str(self.root))
        if not self.patients:
            raise RuntimeError("Dataset patient list is empty")
        self.records: List[CTSliceRecord] = []
        for patient in self.patients:
            self.records.extend(self._build_patient(patient))
        if not self.records:
            raise RuntimeError("Paired CT dataset is empty")

    def _view_index(self, patient: str, view: str) -> Dict[int, Path]:
        result: Dict[int, Path] = {}
        for path in sorted(self.root.glob("{}_{}_*_img.npy".format(patient, view))):
            match = FILENAME_RE.match(path.name)
            if match is None:
                raise RuntimeError("Unexpected CT filename: {}".format(path))
            index = int(match.group("slice"))
            if match.group("patient") != patient or match.group("view") != view:
                raise RuntimeError("Parsed pair metadata mismatch: {}".format(path))
            if index in result:
                raise RuntimeError("Duplicate slice {} {} {}".format(patient, view, index))
            result[index] = path.resolve()
        return result

    def _build_patient(self, patient: str) -> List[CTSliceRecord]:
        ldct = self._view_index(patient, self.dose_label)
        ndct = self._view_index(patient, "target")
        if not ldct or not ndct:
            raise RuntimeError("Missing LDCT/NDCT files for {}".format(patient))
        if set(ldct) != set(ndct):
            raise RuntimeError("LDCT/NDCT slice mismatch for {}".format(patient))
        indices = sorted(ldct)
        if indices != list(range(indices[0], indices[-1] + 1)):
            raise RuntimeError("Non-consecutive slice indices for {}".format(patient))
        records = []
        for position, center in enumerate(indices):
            context_indices = (
                indices[max(0, position - 1)],
                center,
                indices[min(len(indices) - 1, position + 1)],
            )
            records.append(
                CTSliceRecord(
                    patient_id=patient,
                    slice_index=center,
                    ldct_path=str(ldct[center]),
                    ndct_path=str(ndct[center]),
                    context_paths=tuple(str(ldct[index]) for index in context_indices),
                    context_indices=context_indices,
                )
            )
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, object]:
        record = self.records[int(index)]
        context = np.stack(
            [normalize_stored_ct(np.load(path), self.data_config) for path in record.context_paths],
            axis=0,
        )
        target = normalize_stored_ct(np.load(record.ndct_path), self.data_config)[None]
        expected = int(self.data_config["image_size"])
        if context.shape != (3, expected, expected) or target.shape != (1, expected, expected):
            raise RuntimeError("Unexpected CT shape for {}:{}".format(record.patient_id, record.slice_index))
        return {
            "ldct_context": torch.from_numpy(context).float(),
            "ndct_target": torch.from_numpy(target).float(),
            "patient_id": record.patient_id,
            "slice_index": record.slice_index,
        }

    def audit(self) -> Dict[str, object]:
        counts = {patient: 0 for patient in self.patients}
        for record in self.records:
            counts[record.patient_id] += 1
        return {
            "root": str(self.root),
            "patients": list(self.patients),
            "slices_per_patient": counts,
            "total_slices": len(self.records),
            "paired_slices_exact": True,
            "boundary_policy": "nearest_slice_replication",
        }


class RDDMCTBackbone(nn.Module):
    def __init__(self, model_config: Mapping[str, object]):
        super().__init__()
        dim = int(model_config["base_dim"])
        dim_mults = tuple(int(value) for value in model_config["dim_mults"])
        state_channels = int(model_config["state_stem_channels"])
        context_channels = int(model_config["context_stem_channels"])
        fused_channels = int(model_config["fused_channels"])
        groups = int(model_config["resnet_groups"])
        image_size = int(model_config["image_size"])
        attention_resolutions = tuple(int(value) for value in model_config["attention_resolutions"])
        attention_resolution_set = set(attention_resolutions)
        if (state_channels, context_channels, fused_channels) != (32, 32, 64):
            raise ValueError("RDDM-CT v1 fixes stem channels at 32/32/64")
        if dim != 64 or dim_mults != (1, 2, 4, 8):
            raise ValueError("RDDM-CT v1 fixes U-Net dimensions at 64*(1,2,4,8)")
        if image_size != 512 or attention_resolution_set != {64, 128}:
            raise ValueError("Efficiency revision fixes attention resolutions at 128 and 64")

        self.channels = 1
        self.out_dim = 1
        self.self_condition = False
        self.random_or_learned_sinusoidal_cond = False
        self.shared_unet_count = 1
        self.image_size = image_size
        self.attention_resolutions = attention_resolutions
        self.bottleneck_resolution = image_size // (2 ** (len(dim_mults) - 1))
        self.state_stem = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.SiLU())
        self.context_stem = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.SiLU())
        self.fusion = nn.Conv2d(64, 64, 3, padding=1)

        dims = [64] + [dim * multiplier for multiplier in dim_mults]
        in_out = list(zip(dims[:-1], dims[1:]))
        block = partial(ResnetBlock, groups=groups)
        decoder_block = partial(SafeResnetBlock, groups=groups)
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim), nn.Linear(dim, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim)
        )
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        self.encoder_resolutions = []
        self.decoder_resolutions = []

        def resolution_attention(channels, resolution):
            if resolution in attention_resolution_set:
                return Residual(PreNorm(channels, LinearAttention(channels)))
            return nn.Identity()

        resolution = image_size
        for level, (dim_in, dim_out) in enumerate(in_out):
            last = level >= len(in_out) - 1
            self.encoder_resolutions.append(resolution)
            self.downs.append(nn.ModuleList([
                block(dim_in, dim_in, time_emb_dim=time_dim),
                block(dim_in, dim_in, time_emb_dim=time_dim),
                resolution_attention(dim_in, resolution),
                Downsample(dim_in, dim_out) if not last else nn.Conv2d(dim_in, dim_out, 3, padding=1),
            ]))
            if not last:
                resolution //= 2
        mid_dim = dims[-1]
        self.mid_block1 = block(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = block(mid_dim, mid_dim, time_emb_dim=time_dim)
        for level, (dim_in, dim_out) in enumerate(reversed(in_out)):
            last = level == len(in_out) - 1
            self.decoder_resolutions.append(resolution)
            self.ups.append(nn.ModuleList([
                decoder_block(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                decoder_block(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                resolution_attention(dim_out, resolution),
                Upsample(dim_out, dim_in) if not last else nn.Conv2d(dim_out, dim_in, 3, padding=1),
            ]))
            if not last:
                resolution *= 2
        self.final_res_block = block(dim * 2, dim, time_emb_dim=time_dim)
        self.residual_head = SafeFP32Head(dim, 1, 1)
        self.noise_head = SafeFP32Head(dim, 1, 1)

    def forward(self, x_t: torch.Tensor, ldct_context: torch.Tensor, t: torch.Tensor):
        if x_t.ndim != 4 or x_t.shape[1] != 1:
            raise ValueError("x_t must have shape [B,1,H,W]")
        if ldct_context.ndim != 4 or ldct_context.shape[1] != 3:
            raise ValueError("ldct_context must have shape [B,3,H,W]")
        if x_t.shape[0] != ldct_context.shape[0] or x_t.shape[-2:] != ldct_context.shape[-2:]:
            raise ValueError("State and context batch/spatial shapes must match")
        state_feature = self.state_stem(x_t)
        context_feature = self.context_stem(ldct_context)
        x = self.fusion(torch.cat((state_feature, context_feature), dim=1))
        root = x.clone()
        time_embedding = self.time_mlp(t)
        skips = []
        for block1, block2, attention, downsample in self.downs:
            x = block1(x, time_embedding)
            skips.append(x)
            x = block2(x, time_embedding)
            x = attention(x)
            skips.append(x)
            x = downsample(x)
        x = self.mid_block1(x, time_embedding)
        x = self.mid_attn(x)
        x = self.mid_block2(x, time_embedding)
        for block1, block2, attention, upsample in self.ups:
            x = torch.cat((x, skips.pop()), dim=1)
            x = block1(x, time_embedding)
            x = torch.cat((x, skips.pop()), dim=1)
            x = block2(x, time_embedding)
            x = attention(x)
            x = upsample(x)
        x = self.final_res_block(torch.cat((x, root), dim=1), time_embedding)
        return self.residual_head(x), self.noise_head(x)


class RDDMCTDiffusion(ResidualDiffusion):
    """CT interface around the unchanged official coefficient and q-sample implementation."""

    def __init__(self, model: RDDMCTBackbone, diffusion_config: Mapping[str, object], image_size: int):
        if str(diffusion_config["objective"]) != "pred_res_noise":
            raise ValueError("RDDM-CT v1 fixes objective=pred_res_noise")
        super().__init__(
            model,
            image_size=int(image_size),
            timesteps=int(diffusion_config["timesteps"]),
            sampling_timesteps=int(diffusion_config["sampling_timesteps"]),
            loss_type=str(diffusion_config["loss_type"]),
            objective="pred_res_noise",
            ddim_sampling_eta=float(diffusion_config["ddim_eta"]),
            condition=True,
            sum_scale=float(diffusion_config["sum_scale"]),
            input_condition=False,
        )
        if self.ddim_sampling_eta != 0.0:
            raise ValueError("RDDM-CT v1 requires deterministic eta=0 sampling")

    def ct_model_predictions(self, context: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor, clip=True):
        pred_res, pred_noise = self.model(x_t, context, t)
        if clip:
            pred_res = pred_res.clamp(-1.0, 1.0)
        x_start = self.predict_start_from_res_noise(x_t, t, pred_res, pred_noise)
        if clip:
            x_start = x_start.clamp(-1.0, 1.0)
        return ModelResPrediction(pred_res, pred_noise, x_start)

    def training_losses(
        self,
        ndct_zero_one: torch.Tensor,
        context_zero_one: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        ndct = zero_one_to_diffusion(ndct_zero_one)
        context = zero_one_to_diffusion(context_zero_one)
        ldct_center = context[:, 1:2]
        batch = ndct.shape[0]
        if t is None:
            t = torch.randint(0, self.num_timesteps, (batch,), device=ndct.device).long()
        if noise is None:
            noise = torch.randn_like(ndct)
        residual = ldct_center - ndct
        x_t = self.q_sample(ndct, residual, t, noise=noise)
        pred_residual, pred_noise = self.model(x_t, context, t)
        loss_residual = F.l1_loss(pred_residual.float(), residual.float())
        loss_noise = F.l1_loss(pred_noise.float(), noise.float())
        return {
            "loss": loss_residual + loss_noise,
            "loss_residual": loss_residual,
            "loss_noise": loss_noise,
            "residual": residual,
            "noise": noise,
            "x_t": x_t,
            "t": t,
        }

    @torch.no_grad()
    def sample_ct(self, context_zero_one: torch.Tensor, generator=None) -> torch.Tensor:
        context = zero_one_to_diffusion(context_zero_one)
        center = context[:, 1:2]
        batch, _, height, width = center.shape
        shape = (batch, 1, height, width)
        initial_noise = torch.randn(shape, device=center.device, dtype=center.dtype, generator=generator)
        image = center + math.sqrt(self.sum_scale) * initial_noise
        total = self.num_timesteps
        sampling = self.sampling_timesteps
        times = torch.linspace(-1, total - 1, steps=sampling + 1)
        time_pairs = list(zip(reversed(times.int().tolist()[:-1]), reversed(times.int().tolist()[1:])))
        # The preceding compact construction is replaced below by the official ordering for clarity.
        times = list(reversed(torch.linspace(-1, total - 1, steps=sampling + 1).int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        for current, next_time in time_pairs:
            t = torch.full((batch,), current, device=center.device, dtype=torch.long)
            predictions = self.ct_model_predictions(context, image, t)
            if next_time < 0:
                image = predictions.pred_x_start
                continue
            alpha_cumsum = self.alphas_cumsum[current]
            alpha_cumsum_next = self.alphas_cumsum[next_time]
            alpha = alpha_cumsum - alpha_cumsum_next
            betas2_cumsum = self.betas2_cumsum[current]
            betas2_cumsum_next = self.betas2_cumsum[next_time]
            sigma2 = self.ddim_sampling_eta * (
                (betas2_cumsum - betas2_cumsum_next) * betas2_cumsum_next / betas2_cumsum
            )
            image = (
                image
                - alpha * predictions.pred_res
                - (self.betas_cumsum[current] - (betas2_cumsum_next - sigma2).sqrt()) * predictions.pred_noise
            )
        return diffusion_to_zero_one(image).clamp(0.0, 1.0)


def build_system(config: Mapping[str, object], device: torch.device):
    model = RDDMCTBackbone(config["model"]).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters > int(config["model"]["max_parameters"]):
        raise RuntimeError("Parameter limit exceeded: {}".format(parameters))
    diffusion = RDDMCTDiffusion(model, config["diffusion"], int(config["data"]["image_size"])).to(device)
    return model, diffusion, parameters


def git_state() -> Dict[str, object]:
    def run(*args):
        return subprocess.check_output(["git"] + list(args), cwd=str(PROJECT_ROOT), text=True).strip()
    status = run("status", "--short")
    return {
        "branch": run("branch", "--show-current"),
        "commit": run("rev-parse", "HEAD"),
        "status_short": status.splitlines(),
        "dirty": bool(status),
    }


def atomic_torch_save(value: object, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, str(temporary))
    os.replace(str(temporary), str(path))


def write_json(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


class JsonlLogger:
    def __init__(self, path: Path):
        self.path = Path(path)

    def log(self, **values: object) -> None:
        record = {"time_unix": time.time(), **values}
        line = json.dumps(record, sort_keys=True)
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
