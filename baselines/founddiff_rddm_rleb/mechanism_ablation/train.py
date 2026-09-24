"""Two-arm smoke and isolated 12k training. No baseline source is edited."""
import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
import torch
from fast_dev.data import DataAccessGuard
from fast_dev.protocol import FAST_INDICES
from mayo2016_smallset_v1.runtime import atomic_save, sha256, write_json, ROOT
from .protocol import ARMS, HERE, STEPS, experiment_config, initialization_audit
from .runtime import MatchedState, direct_predict
from .evaluate import validate_arm

def source_manifest():
    paths = [ROOT/"src/rddm_rleb.py", ROOT/"configs/rddm_fast_dev.json",
             ROOT/"configs/fast_dev_protocol.json", ROOT/"splits/L333_fast64.txt",
             ROOT/"splits/fast_train_2patients.txt",
             ROOT/"mayo2016_smallset_v1/data.py",
             ROOT/"mayo2016_smallset_v1/frozen_protocol.py",
             ROOT/"fast_dev/data.py", ROOT/"fast_dev/evaluate.py",
             ROOT/"fast_dev/protocol.py"]
    paths += sorted((ROOT/"mechanism_ablation").glob("*.py"))
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths}

def _append_csv(path, row):
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())

def smoke(arm, device):
    cfg, base = experiment_config()
    state = None
    guard = DataAccessGuard(base, "FAST", allow_training=True, allow_memory=False)
    observed = {"q_sample_calls": 0, "timesteps": []}
    with guard:
        state = MatchedState(arm, device)
        if arm == ARMS[0]:
            original = state.model.q_sample
            def checked_q_sample(x_start, x_res, t, noise=None):
                observed["q_sample_calls"] += 1
                observed["timesteps"].extend(t.detach().cpu().tolist())
                assert noise is not None and torch.isfinite(noise).all()
                return original(x_start, x_res, t, noise=noise)
            state.model.q_sample = checked_q_sample
        else:
            assert not hasattr(state.model, "q_sample")
            ldct, _, _ = state.train_data.load(0)
            x = torch.from_numpy(ldct).unsqueeze(0).to(state.device)
            state.model.eval()
            with torch.no_grad():
                one = direct_predict(state.model, x)
                two = direct_predict(state.model, x)
                assert torch.equal(one, two), "Direct inference is not deterministic"
        records = [state.update() for _ in range(2)]
        assert all(math.isfinite(r["train_loss"]) and math.isfinite(r["gradient_norm_pre_clip"]) for r in records)
        assert all(torch.isfinite(p).all() for p in state.model.parameters())
    if arm == ARMS[0]:
        assert observed["q_sample_calls"] == 2
        assert len(observed["timesteps"]) == 4
        assert all(0 <= t < 1000 for t in observed["timesteps"])
        assert len(set(observed["timesteps"])) > 1
    access = guard.report()
    assert access["training_patients_opened"] == ["L067", "L143"] or set(access["training_patients_opened"]) <= {"L067", "L143"}
    assert access["test_opens"] == access["external_memory_opens"] == access["blocked_attempts"] == 0
    result = {"arm": arm, "status": "SMOKE_PASS", "optimizer_steps": 2,
              "losses": [r["train_loss"] for r in records],
              "gradient_norms_pre_clip": [r["gradient_norm_pre_clip"] for r in records],
              "parameter_count": sum(p.numel() for p in state.model.parameters()),
              "all_trainable_parameter_gradients_present": True,
              "all_gradients_finite": True, "all_parameters_finite": True,
              "output_shape": [2,1,512,512],
              "initial_hash": state.initial_hash,
              "fresh_restart_required": True,
              "control_q_sample_calls": observed["q_sample_calls"],
              "control_sampled_timesteps": observed["timesteps"],
              "regression_direct_deterministic": arm == ARMS[1],
              "regression_no_q_sample_or_diffusion": arm == ARMS[1],
              "data_access": access}
    write_json(HERE / ("SMOKE_" + arm + ".json"), result)
    print(json.dumps(result), flush=True)
    return result

def train(arm, device):
    cfg, base = experiment_config()
    for a in ARMS:
        smoke_path = HERE / ("SMOKE_" + a + ".json")
        assert smoke_path.is_file() and json.loads(smoke_path.read_text())["status"] == "SMOKE_PASS"
    output = HERE / "runs" / arm
    output.mkdir(parents=True, exist_ok=False)
    (output/"checkpoints").mkdir()
    write_json(output/"CONFIG.json", cfg)
    write_json(output/"source_manifest.json", source_manifest())
    guard = DataAccessGuard(base, "FAST", allow_training=True, allow_memory=False)
    state = None
    manifest = {"arm": arm, "recovery": [], "matched": []}
    history = []
    try:
        with guard:
            state = MatchedState(arm, device)
            assert state.initial_hash == json.loads((HERE/"INITIALIZATION_AUDIT.json").read_text())["control_unetres_sha256"]
            with (output/"train.jsonl").open("x", buffering=1) as log:
                while state.step < 12000:
                    row = state.update()
                    log.write(json.dumps(row, allow_nan=False)+"\n")
                    if state.step % 100 == 0:
                        print("TRAIN", arm, state.step, row["train_loss"], row["gradient_norm_pre_clip"], flush=True)
                        write_json(output/"progress.json",
                                   {"status":"RUNNING","step":state.step,"stop":12000,
                                    "train_stream_sha256":state.stream_sha256})
                    if state.step in STEPS:
                        summary, _ = validate_arm(arm, state.ema.model, state.validation_data,
                                                  base, state.device,
                                                  output/("validation_step_%d.json"%state.step))
                        assert summary["slices"] == 64 and summary["weights"] == "ema"
                        metric = {"step": state.step, **summary, "train_loss":row["train_loss"],
                                  "gradient_norm_pre_clip":row["gradient_norm_pre_clip"],
                                  "learning_rate":row["learning_rate"],
                                  "peak_vram_bytes":state.peak_vram,
                                  "seconds_per_step":state.elapsed_train_seconds/state.step,
                                  "train_stream_sha256":state.stream_sha256}
                        assert all(math.isfinite(metric[k]) for k in
                                   ("psnr","ssim","rmse_hu","mae_hu","validation_loss"))
                        history.append(metric)
                        _append_csv(output/"validation_history.csv", metric)
                        print("VALIDATION", arm, json.dumps(metric), flush=True)
                    if state.step % 1000 == 0:
                        latest = output/"checkpoints"/"latest.pt"
                        atomic_save(latest, state.checkpoint())
                        manifest["recovery"].append({"step":state.step,"path":str(latest),
                                                     "sha256":sha256(latest)})
                        if state.step in STEPS:
                            matched = output/"checkpoints"/("step_%d.pt"%state.step)
                            assert not matched.exists()
                            os.link(latest, matched)
                            manifest["matched"].append({"step":state.step,"path":str(matched),
                                                        "sha256":sha256(matched),"weights":"ema_in_payload"})
                        write_json(output/"checkpoint_manifest.json", manifest)
        access = guard.report()
        assert access["training_patients_opened"] == ["L067","L143"]
        assert access["validation_indices_opened"] == list(FAST_INDICES)
        assert access["test_opens"] == access["external_memory_opens"] == access["blocked_attempts"] == 0
        assert [r["step"] for r in history] == list(STEPS)
        summary = {"status":"COMPLETE","arm":arm,"optimizer_steps":state.step,
                   "initial_hash":state.initial_hash,"train_stream_sha256":state.stream_sha256,
                   "training_seconds":state.elapsed_train_seconds,
                   "seconds_per_step":state.elapsed_train_seconds/state.step,
                   "peak_vram_bytes":state.peak_vram,
                   "gradient_min":state.gradient_min,"gradient_max":state.gradient_max,
                   "nonfinite_count":state.nonfinite_count,"history":history,
                   "data_access":access,"source_manifest":source_manifest(),
                   "git_commit_at_run":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()}
        write_json(output/"final_summary.json",summary)
        write_json(output/"data_access.json",access)
        write_json(output/"progress.json",{"status":"COMPLETE","step":12000,"stop":12000})
        print("ARM_COMPLETE",arm,flush=True)
    except BaseException as exc:
        write_json(output/"data_access.json",guard.report())
        write_json(output/"progress.json",{"status":"FAILED","step":state.step if state else 0,
                                           "error":repr(exc)})
        raise

def main():
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("prepare","smoke","train"))
    p.add_argument("--arm",choices=ARMS)
    p.add_argument("--device",default="cuda:0")
    a=p.parse_args()
    if a.mode=="prepare":
        print(json.dumps(initialization_audit()),flush=True)
        return
    if not a.arm:
        p.error("--arm required")
    if a.mode=="smoke":
        smoke(a.arm,a.device)
    else:
        train(a.arm,a.device)

if __name__=="__main__":
    main()
