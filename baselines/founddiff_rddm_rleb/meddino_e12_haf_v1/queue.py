"""Serial E12 training then inference-only Full-L333, with terminal status."""
import json
import subprocess
import sys
import time
import traceback

from mayo2016_smallset_v1.runtime import write_json
from .protocol import HERE, ARM, PARENT_COMMIT, verify


def main():
    verify()
    smoke = json.loads((HERE / "SMOKE_RESULT.json").read_text())
    preflight = json.loads((HERE / "CHECKPOINT_PREFLIGHT.json").read_text())
    assert smoke["status"] == preflight["status"] == "PASS"
    assert smoke["steps"] == 20
    assert not (HERE / "checkpoints/step_6000.pt").exists()
    status = {"status": "RUNNING", "arm": ARM,
              "parent_commit": PARENT_COMMIT,
              "started_unix": time.time(), "phases": {}}
    write_json(HERE / "QUEUE_STATUS.json", status)
    try:
        for phase, command in (
            ("TRAIN_6K", [sys.executable, "-m", "meddino_e12_haf_v1.train", "train"]),
            ("FULL_L333", [sys.executable, "-m", "meddino_e12_haf_v1.full"]),
        ):
            status["phase"] = phase
            status["phases"][phase] = {"status": "RUNNING", "command": command}
            write_json(HERE / "QUEUE_STATUS.json", status)
            with (HERE / (phase + ".log")).open("x", buffering=1) as log:
                result = subprocess.run(command, cwd=HERE.parents[1],
                                        stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"{phase} exited {result.returncode}")
            status["phases"][phase]["status"] = "COMPLETE"
            write_json(HERE / "QUEUE_STATUS.json", status)
        final = json.loads((HERE / "FINAL_AUDIT.json").read_text())
        assert final["status"] == "PASS"
        status["status"] = "COMPLETE"
        status["finished_unix"] = time.time()
        write_json(HERE / "QUEUE_STATUS.json", status)
        print("E12_QUEUE_COMPLETE", flush=True)
    except BaseException as exc:
        status["status"] = "FAILED"
        status["error"] = repr(exc)
        status["traceback"] = traceback.format_exc()
        status["finished_unix"] = time.time()
        write_json(HERE / "QUEUE_STATUS.json", status)
        print("E12_QUEUE_FAILED", repr(exc), flush=True)
        raise


if __name__ == "__main__":
    main()

