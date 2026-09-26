"""Serial single-GPU queue. D3 OOM is recorded and does not block D4."""
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch

from mayo2016_smallset_v1.runtime import ROOT, write_json, sha256
from .protocol import HERE, ARMS, BASE_COMMIT, config, verify
from .structure import OFFICIAL_COMMIT, WEIGHT_SHA256
from .report import build_report


def queue_event(message):
    with (HERE / "queue.log").open("a", buffering=1) as out:
        out.write(message + "\n")
    print(message, flush=True)


def run_command(arm, mode):
    path = HERE / "logs"
    path.mkdir(exist_ok=True)
    log = path / (arm + "_" + mode + ".log")
    cmd = [sys.executable, "-m", "meddino_four_arm_v1.train", mode,
           "--arm", arm, "--device", "cuda:0"]
    queue_event("START " + arm + " " + mode + " " + " ".join(cmd))
    with log.open("x", buffering=1) as out:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
    queue_event("END " + arm + " " + mode + " exit=" + str(proc.returncode))
    return proc.returncode, log


def is_oom(path):
    text = Path(path).read_text(errors="replace")[-10000:]
    return "OutOfMemoryError" in text or "CUDA out of memory" in text


def main():
    assert torch.cuda.is_available()
    assert len(ARMS) == 4
    verify()
    if HERE.exists():
        raise FileExistsError("Refusing to overwrite experiment output: " + str(HERE))
    HERE.mkdir(parents=True)
    configs = {arm: config(arm) for arm in ARMS}
    write_json(HERE / "EXPERIMENT_CONFIG.json", configs)
    write_json(HERE / "OFFICIAL_MEDDINO_AUDIT.json", {
        "official_source_commit": OFFICIAL_COMMIT,
        "checkpoint_sha256": WEIGHT_SHA256,
        "extraction": "get_intermediate_layers(n=[2,5,8,11],reshape=True,norm=True)",
        "checkpoint_role": "teacher", "strict": True,
        "removed_keys": ["ibot", "dino_head"], "stripped_prefix": "backbone.",
    })
    status = {
        "status": "RUNNING", "base_commit": BASE_COMMIT,
        "branch": "experiment/rddm-meddino-four-arm-v1",
        "started_unix": time.time(), "arms": {},
        "order": list(ARMS), "single_gpu_serial": True,
    }
    write_json(HERE / "queue_status.json", status)
    try:
        for arm in ARMS:
            status["arms"][arm] = {"status": "SMOKE_RUNNING"}
            write_json(HERE / "queue_status.json", status)
            code, log = run_command(arm, "smoke")
            if code:
                if arm == "D3_MULTI_1024" and is_oom(log):
                    status["arms"][arm] = {
                        "status": "OOM", "phase": "smoke",
                        "error_log": str(log), "protocol_change": False,
                    }
                    write_json(HERE / "queue_status.json", status)
                    queue_event("D3_OOM_CONTINUE_D4 smoke")
                    torch.cuda.empty_cache()
                    continue
                raise RuntimeError(arm + " smoke failed; see " + str(log))
            status["arms"][arm] = {"status": "TRAINING"}
            write_json(HERE / "queue_status.json", status)
            code, log = run_command(arm, "train")
            if code:
                if arm == "D3_MULTI_1024" and is_oom(log):
                    status["arms"][arm] = {
                        "status": "OOM", "phase": "train",
                        "error_log": str(log), "protocol_change": False,
                    }
                    write_json(HERE / "queue_status.json", status)
                    queue_event("D3_OOM_CONTINUE_D4 train")
                    torch.cuda.empty_cache()
                    continue
                raise RuntimeError(arm + " train failed; see " + str(log))
            summary = json.loads((HERE / "runs" / arm / "final_summary.json").read_text())
            assert summary["status"] == "COMPLETE" and summary["optimizer_steps"] == 6000
            status["arms"][arm] = {"status": "COMPLETE", "optimizer_steps": 6000}
            write_json(HERE / "queue_status.json", status)
        build_report()
        status["status"] = "COMPLETE_WITH_D3_OOM" if status["arms"]["D3_MULTI_1024"]["status"] == "OOM" else "COMPLETE"
        status["finished_unix"] = time.time()
        write_json(HERE / "queue_status.json", status)
        queue_event("FOUR_ARM_QUEUE " + status["status"])
    except BaseException as exc:
        status["status"] = "FAILED"
        status["error"] = repr(exc)
        status["traceback"] = traceback.format_exc()
        write_json(HERE / "queue_status.json", status)
        queue_event("FOUR_ARM_QUEUE FAILED " + repr(exc))
        raise


if __name__ == "__main__":
    main()
