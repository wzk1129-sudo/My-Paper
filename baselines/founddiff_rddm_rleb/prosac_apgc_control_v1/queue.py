"""One GPU, sequential C3 then C4; no retry and no formal training."""
import json
import traceback
from mayo2016_smallset_v1.runtime import write_json
from .protocol import HERE, ARMS, verify
from .train import train, gate
from .report import build_report


def main():
    verify()
    for arm in ARMS:
        smoke = json.loads((HERE / "smoke" / arm / "SMOKE_RESULT.json").read_text())
        assert smoke["status"] == "PASS" and smoke["steps"] == 2
    try:
        for arm in ARMS:
            write_json(HERE / "queue_status.json", {
                "status": "RUNNING_FIRST_STAGE", "arm": arm})
            train(arm, 512, "cuda:0", False)
        decisions = gate()
        for arm in ARMS:
            if decisions[arm]["continue_to_6k"]:
                write_json(HERE / "queue_status.json", {
                    "status": "RUNNING_PROMOTION", "arm": arm,
                    "decisions": decisions})
                train(arm, 512, "cuda:0", True)
        build_report()
        write_json(HERE / "queue_status.json", {
            "status": "COMPLETE",
            "first_stage_completed": list(ARMS),
            "promoted": [arm for arm in ARMS if decisions[arm]["continue_to_6k"]],
            "decisions": decisions})
        print("COMPLETE", flush=True)
    except BaseException as exc:
        write_json(HERE / "queue_status.json", {
            "status": "FAILED", "error": repr(exc)})
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

