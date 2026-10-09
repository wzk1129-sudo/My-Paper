"""Saved-checkpoint, fixed Fast64 paired analysis after MLSF reaches 8000 steps."""
import csv
import hashlib
import json
import math
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
V1 = ROOT.parent / "rddm_official_2p5d_meddino_condition_v1"
AUDIT = ROOT.parent / "meddino_feature_audit_v1"
for path in (str(AUDIT), str(V1), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)
from mlsf_model import build_model as build_mlsf_model, parameter_hash
from model import build_model as build_v1_model, load_state_without_med
from context_data import ContextPairs, ContextDataGuard
from fast_dev.protocol import CONFIG, check_config
from mayo2016_smallset_v1.frozen_protocol import slice_metrics
from mayo2016_smallset_v1.runtime import METRICS, seed_everything
from run_audit import masks, pixel_regions, region_errors

RUN = ROOT / "runs/MLSF_RDDM_8K_20261009"
V1_RUN = V1 / "runs/OFFICIAL_RDDM_2P5D_MEDDINO_CONDITION_retry1"
METRIC_NAMES = ("psnr", "ssim", "rmse_hu", "mae_hu")
REGION_NAMES = ("foreground", "soft", "high_gradient")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    assert not path.exists(), path
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    assert rows and not path.exists(), path
    with path.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_model(builder, checkpoint):
    model = builder()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["optimizer_step"] == saved["ema_updates"] == 8000
    load_state_without_med(model, saved["ema_state_dict"])
    del saved
    return model.to("cuda:0").eval()


@torch.no_grad()
def predict(model, prev, center, next_, seed):
    unet = model.model.unet0
    unet.set_context(prev, center, next_)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    started = time.perf_counter()
    output = model.sample([center], last=True)[-1]
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    unet.clear_context()
    assert output.shape == (1, 1, 512, 512) and torch.isfinite(output).all()
    return output, seconds


def bootstrap(values):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(20260829)
    draw = rng.integers(0, len(values), size=(10000, len(values)))
    sample_means = values[draw].mean(axis=1)
    return [float(np.quantile(sample_means, .025)), float(np.quantile(sample_means, .975))]


def main():
    assert not (ROOT / "REPORT.md").exists()
    final = json.loads((RUN / "FINAL_STATE.json").read_text())
    assert final["optimizer_step"] == final["ema_updates"] == 8000
    reference = json.loads((ROOT / "V1_STREAM_REFERENCE.json").read_text())["steps"]
    assert final["train_stream_sha256"] == reference["8000"]["train_stream_sha256"]
    train_access = json.loads((RUN / "DATA_ACCESS_AUDIT.json").read_text())
    assert all(train_access[k] == 0 for k in ("L506_opens", "external_memory_opens", "neighbor_target_opens", "blocked_attempts"))
    candidate_checkpoint = RUN / "checkpoints/step_8000.pt"
    v1_checkpoint = V1_RUN / "checkpoints/step_8000.pt"
    assert candidate_checkpoint.is_file() and v1_checkpoint.is_file()
    checkpoint_hashes = {"mlsf_step8000": sha(candidate_checkpoint), "v1_step8000": sha(v1_checkpoint)}
    cfg = json.loads(CONFIG.read_text())
    check_config(cfg)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed_everything(20260829)
    saved_v1 = json.loads((V1_RUN / "validation_step_8000.json").read_text())
    saved_mlsf = json.loads((RUN / "validation_step_8000.json").read_text())
    assert len(saved_v1) == len(saved_mlsf) == 64
    with ContextDataGuard(cfg, allow_training=False) as guard:
        data = ContextPairs(cfg, "validation")
        data.bind_guard(guard)
        v1 = load_model(lambda: build_v1_model(anatomy=True), v1_checkpoint)
        mlsf = load_model(build_mlsf_model, candidate_checkpoint)
        med_before = parameter_hash(mlsf.model.unet0.med.state_dict())
        rows = []
        time_v1 = 0.0
        time_mlsf = 0.0
        with torch.random.fork_rng(devices=[0]):
            for position in range(len(data)):
                prev, center, next_, target, meta = data.load(position)
                tensors = [torch.from_numpy(x).unsqueeze(0).to("cuda:0") for x in (prev, center, next_, target)]
                prev, center, next_, target = tensors
                seed = 20260829 + int(meta["slice_index"])
                pred_v1, seconds_v1 = predict(v1, prev, center, next_, seed)
                pred_mlsf, seconds_mlsf = predict(mlsf, prev, center, next_, seed)
                time_v1 += seconds_v1
                time_mlsf += seconds_mlsf
                v1_metrics = slice_metrics(pred_v1, target, cfg)
                mlsf_metrics = slice_metrics(pred_mlsf, target, cfg)
                for saved, metrics in ((saved_v1[position], v1_metrics), (saved_mlsf[position], mlsf_metrics)):
                    assert saved["slice_index"] == meta["slice_index"]
                    assert all(abs(saved[k] - metrics[k]) < 1e-8 for k in METRIC_NAMES), (position, saved, metrics)
                _, center_hu, foreground, soft = masks(center)
                region = pixel_regions(center_hu, foreground, soft)
                v1_region = region_errors(pred_v1, target, cfg, region)
                mlsf_region = region_errors(pred_mlsf, target, cfg, region)
                row = {"patient_id": "L333", "slice_index": int(meta["slice_index"]), "sampling_seed": seed}
                for name in METRIC_NAMES:
                    row["v1_" + name] = v1_metrics[name]
                    row["mlsf_" + name] = mlsf_metrics[name]
                    row["delta_" + name] = mlsf_metrics[name] - v1_metrics[name]
                for name in (*REGION_NAMES,):
                    for metric in ("rmse_hu", "mae_hu"):
                        key = name + "_" + metric
                        row["v1_" + key] = v1_region[key]
                        row["mlsf_" + key] = mlsf_region[key]
                        row["delta_" + key] = mlsf_region[key] - v1_region[key]
                rows.append(row)
                if position % 8 == 7:
                    print("PAIRED_FAST64", position + 1, flush=True)
        assert parameter_hash(mlsf.model.unet0.med.state_dict()) == med_before
        access = guard.report()
        assert all(access[k] == 0 for k in ("L506_opens", "external_memory_opens", "neighbor_target_opens", "blocked_attempts"))
    write_csv(ROOT / "PER_SLICE_DELTA.csv", rows)
    candidate_validation = list(csv.DictReader((RUN / "validation_metrics.csv").open()))
    v1_history = list(csv.DictReader((V1 / "VALIDATION_HISTORY.csv").open()))
    assert [int(r["step"]) for r in candidate_validation] == [0, 1000, 2000, 4000, 6000, 8000]
    comparisons = []
    for base, current in zip(v1_history, candidate_validation):
        step = int(base["step"])
        assert int(current["step"]) == step
        item = {"step": step}
        for name in METRIC_NAMES:
            base_value = float(base["candidate_" + name])
            current_value = float(current[name])
            item["v1_" + name] = base_value
            item["mlsf_" + name] = current_value
            item["delta_" + name] = current_value - base_value
        item["train_stream_sha256"] = current["train_stream_sha256"]
        if step > 0:
            assert item["train_stream_sha256"] == reference[str(step)]["train_stream_sha256"]
        comparisons.append(item)
    write_csv(ROOT / "VALIDATION_COMPARISON.csv", comparisons)
    with (RUN / "train.jsonl").open() as f:
        training = [json.loads(line) for line in f]
    assert len(training) == 8000 and [r["optimizer_step"] for r in training] == list(range(1, 8001))
    train_rows = [{key: r.get(key) for key in ("optimizer_step", "train_loss", "gradient_norm_before_clip", "attention_gradient_norm", "parameters_finite", "seconds_per_step", "peak_vram_bytes", "train_stream_sha256")} for r in training]
    write_csv(ROOT / "TRAINING_HISTORY.csv", train_rows)
    summary = {"slices": 64, "v1_checkpoint_sha256": checkpoint_hashes["v1_step8000"],
               "mlsf_checkpoint_sha256": checkpoint_hashes["mlsf_step8000"],
               "improved_psnr_count": int(sum(r["delta_psnr"] > 0 for r in rows)),
               "improved_psnr_fraction": float(np.mean([r["delta_psnr"] > 0 for r in rows])),
               "inference_seconds_per_slice_v1": time_v1 / 64,
               "inference_seconds_per_slice_mlsf": time_mlsf / 64,
               "training_mean_seconds_per_step": float(np.mean([r["seconds_per_step"] for r in training])),
               "training_peak_vram_bytes": final["peak_vram_bytes"]}
    for name in METRIC_NAMES:
        values = [r["delta_" + name] for r in rows]
        summary["delta_" + name + "_mean"] = float(np.mean(values))
        summary["delta_" + name + "_ci95"] = bootstrap(values)
    for region in REGION_NAMES:
        for metric in ("rmse_hu", "mae_hu"):
            key = region + "_" + metric
            for prefix in ("v1", "mlsf", "delta"):
                summary[prefix + "_" + key] = float(np.mean([r[prefix + "_" + key] for r in rows]))
    write_json(ROOT / "PAIRED_SUMMARY.json", summary)
    write_json(ROOT / "DATA_ACCESS_AUDIT.json", {"training": train_access, "postprocess": access})
    model_audit = json.loads((ROOT / "MODEL_AUDIT.json").read_text())
    final_row = comparisons[-1]
    gate = (final_row["delta_psnr"] >= .03 and final_row["delta_ssim"] >= 0 and summary["improved_psnr_fraction"] >= .60)
    write_json(ROOT / "FULL_L333_GATE.json", {"eligible": bool(gate), "delta_psnr": final_row["delta_psnr"], "delta_ssim": final_row["delta_ssim"], "improved_fraction": summary["improved_psnr_fraction"]})
    lines = ["# MLSF-RDDM matched 8k result", "", "## Model and audit", "",
             "MLSF keeps the Official RDDM residual diffusion, native ResnetBlocks, nine native Attention sites, 2.5D stem, frozen MedDINO L2/L5/L8/L11 Adapter/Fusion, and nine V1 FiLM sites. Added LocalDetailEncoder, one aligned 3x3 L8/L11 cross-attention at 128x128, and zero-initialized gamma/beta corrections inside E1/E2/D1/D2 FiLM. No extra loss or training arm.", "",
             f"Shared V1 state tensors exact: {model_audit['common_state_tensors_exact_equal']}; added trainable parameters: {model_audit['added_trainable_parameters']:,}; original native Attention sites: {model_audit['native_attention_count']}.",
             "Shapes: L256 [B,32,256,256], L128/F128 [B,64,128,128], M32 [B,64,32,32], F256 [B,32,256,256]. Each 128x128 query attends to exactly 9 clamped 32x32 MedDINO tokens; no global attention matrix.",
             "The four corrected FiLM sites compute the original gamma/beta and the spatial delta in one FiLM formula before native Attention. Other five sites use V1 unchanged.", "",
             "Step 0 Fast64 PSNR/SSIM/RMSE/MAE matched V1 exactly; first-batch train stream and all checkpoint stream hashes matched V1. The real batch=2 two-step readiness passed with frozen MedDINO and finite native/new gradients. L506 and external-memory opens were zero.", "",
             "## EMA Fast64 trajectory", "",
             "| Step | V1 PSNR | MLSF PSNR | Delta PSNR | V1 SSIM | MLSF SSIM | Delta SSIM | V1 RMSE HU | MLSF RMSE HU | V1 MAE HU | MLSF MAE HU |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in comparisons:
        lines.append(f"| {r['step']} | {r['v1_psnr']:.6f} | {r['mlsf_psnr']:.6f} | {r['delta_psnr']:+.6f} | {r['v1_ssim']:.6f} | {r['mlsf_ssim']:.6f} | {r['delta_ssim']:+.6f} | {r['v1_rmse_hu']:.4f} | {r['mlsf_rmse_hu']:.4f} | {r['v1_mae_hu']:.4f} | {r['mlsf_mae_hu']:.4f} |")
    lines += ["", "## Paired 8k findings", "",
              f"PSNR improved on {summary['improved_psnr_count']}/64 slices ({summary['improved_psnr_fraction']:.1%}). Mean paired delta PSNR = {summary['delta_psnr_mean']:+.6f} dB; fixed-seed 10,000-replicate paired-slice bootstrap 95% CI = [{summary['delta_psnr_ci95'][0]:+.6f}, {summary['delta_psnr_ci95'][1]:+.6f}] dB. This is within-patient slice uncertainty, not cross-patient significance.",
              f"Soft-tissue RMSE HU: V1 {summary['v1_soft_rmse_hu']:.4f}, MLSF {summary['mlsf_soft_rmse_hu']:.4f}; high-gradient RMSE HU: V1 {summary['v1_high_gradient_rmse_hu']:.4f}, MLSF {summary['mlsf_high_gradient_rmse_hu']:.4f}. Masks exactly reuse the previous MedDINO feature audit: center LDCT HU foreground > -500, soft [-160,240] within foreground, and Gaussian sigma 0.8 + Scharr top 10% within foreground.",
              f"Peak training allocation {summary['training_peak_vram_bytes']/2**30:.2f} GiB; mean training step {summary['training_mean_seconds_per_step']:.3f} s; sampling only inference per slice V1 {summary['inference_seconds_per_slice_v1']:.3f} s, MLSF {summary['inference_seconds_per_slice_mlsf']:.3f} s.",
              "", "## Decision and limits", "",
              f"Full-L333 gate eligible: **{gate}** (requires delta PSNR >= +0.03 dB, >=60% slices improved, and nondecreasing SSIM).",
              "The conclusion is limited to one L333 patient's fixed Fast64 slices. More frequency energy or module activity alone would not establish real anatomical improvement. No L506 or external-memory data were accessed.", ""]
    assert not (ROOT / "REPORT.md").exists()
    (ROOT / "REPORT.md").write_text("\n".join(lines))
    print("POSTPROCESS_PASS", final_row["delta_psnr"], gate, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        (ROOT / "POSTPROCESS_FAILURE.txt").write_text(traceback.format_exc())
        raise
