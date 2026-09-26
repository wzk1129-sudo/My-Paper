"""D4-only controlled retry. Never rerun or mutate D1, D2, or D3."""
import json
import subprocess
import sys
import time
import traceback

from mayo2016_smallset_v1.runtime import write_json
from .protocol import HERE, verify
from .report import build_report


def main():
    verify()
    status_path = HERE / "D4_RETRY_STATUS.json"
    status = json.loads(status_path.read_text())
    assert status["status"] == "SMOKE_PASS"
    assert status["from_step"] == 2000 and status["to_step"] == 6000
    smoke = json.loads((HERE / "smoke/D4_MULTI_FT_512_BF16_RETRY/SMOKE_RESULT.json").read_text())
    assert smoke["status"] == "PASS" and smoke["steps"] == 20
    assert smoke["fork_audit"]["fork_step"] == 2000
    assert smoke["fork_audit"]["output_max_abs_difference"] <= 1.e-6
    assert smoke["meddino_nonzero_gradient_steps"] > 0
    assert not (HERE / "runs/D4_MULTI_FT_512").exists()
    for arm in ("D1_MULTI_512", "D2_MULTI_ADD_512", "D3_MULTI_1024"):
        s = json.loads((HERE / "runs" / arm / "final_summary.json").read_text())
        assert s["status"] == "COMPLETE" and s["optimizer_steps"] == 6000
    status["status"] = "TRAINING"
    status["started_unix"] = time.time()
    write_json(status_path, status)
    command = [sys.executable, "-m", "meddino_four_arm_v1.train", "train",
               "--arm", "D4_MULTI_FT_512", "--device", "cuda:0"]
    log = HERE / "logs/D4_MULTI_FT_512_BF16_RETRY_train.log"
    try:
        with log.open("x", buffering=1) as f:
            completed = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError("D4 formal continuation failed; see " + str(log))
        final = json.loads((HERE / "runs/D4_MULTI_FT_512/final_summary.json").read_text())
        assert final["status"] == "COMPLETE" and final["optimizer_steps"] == 6000
        assert final["nonfinite_count"] == 0
        status["status"] = "REPORTING"
        write_json(status_path, status)
        status["status"] = "COMPLETE"
        write_json(status_path, status)
        build_report()
        status["finished_unix"] = time.time()
        write_json(status_path, status)
        print("D4_RETRY_AND_FOUR_ARM_REPORT_COMPLETE", flush=True)
    except BaseException as exc:
        status["status"] = "FAILED"
        status["error"] = repr(exc)
        status["traceback"] = traceback.format_exc()
        write_json(status_path, status)
        raise


if __name__ == "__main__":
    main()
