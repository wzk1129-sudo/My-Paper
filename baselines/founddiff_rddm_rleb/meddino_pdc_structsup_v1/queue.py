"""Sequential remote tmux queue: independent Fast6k, then Full-L333."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from mayo2016_smallset_v1.runtime import write_json
from .protocol import HERE, ARM

ROOT = Path(__file__).resolve().parent.parent


def run(label, module, logfile):
    with (HERE / logfile).open("x", buffering=1) as handle:
        command = [sys.executable, "-u", "-m", module]
        started = time.time()
        print(label, "START", command, flush=True)
        code = subprocess.call(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        print(label, "EXIT", code, "SECONDS", time.time() - started, flush=True)
        if code != 0:
            raise RuntimeError(label + " exit code " + str(code))


def main():
    assert not (HERE / "QUEUE_STATUS_RETRY.json").exists()
    status = {"arm": ARM, "state": "RUNNING", "stage": "TRAIN_FAST6K",
              "started_unix": time.time(), "pid": os.getpid()}
    write_json(HERE / "QUEUE_STATUS_RETRY.json", status)
    try:
        run("TRAIN_FAST6K", "meddino_pdc_structsup_v1.train", "TRAIN_6K_RETRY.log")
        status["stage"] = "FULL_L333"
        write_json(HERE / "QUEUE_STATUS_RETRY.json", status)
        run("FULL_L333", "meddino_pdc_structsup_v1.full", "FULL_L333_RETRY.log")
        audit = json.loads((HERE / "FINAL_AUDIT.json").read_text())
        assert audit["status"] == "PASS"
        status.update({"state": "COMPLETE", "stage": "DONE",
                       "finished_unix": time.time()})
    except Exception as exc:
        status.update({"state": "FAILED", "error": repr(exc),
                       "finished_unix": time.time()})
        write_json(HERE / "QUEUE_STATUS_RETRY.json", status)
        raise
    write_json(HERE / "QUEUE_STATUS_RETRY.json", status)


if __name__ == "__main__":
    main()
