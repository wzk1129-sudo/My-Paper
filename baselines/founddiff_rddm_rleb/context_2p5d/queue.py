"""Serial GPU queue: A fully terminates before B starts."""
import json
import time
import traceback
from mayo2016_smallset_v1.runtime import write_json
from .model import ARMS
from .protocol import HERE, config, source_manifest
from .train import train, audited_smoke
from .report import build


def main():
    cfg,_=config()
    manifest=source_manifest()
    for arm in ARMS:
        smoke,changed=audited_smoke(arm)
        print("SMOKE_PROVENANCE",arm,changed,flush=True)
    status={
      "status":"RUNNING","started_unix":time.time(),
      "current_arm":None,"arms":{arm:"PENDING" for arm in ARMS},
      "source_manifest":manifest,
      "execution":"serial A then B; one GPU process",
      "maximum_optimizer_steps_per_arm":12000,
    }
    write_json(HERE/"queue_status.json",status)
    for arm in ARMS:
        status["current_arm"]=arm;status["arms"][arm]="RUNNING"
        write_json(HERE/"queue_status.json",status)
        try:
            train(arm,"cuda:0")
            summary=json.loads((HERE/"runs"/arm/"final_summary.json").read_text())
            status["arms"][arm]=summary["status"]
            assert summary["status"] in ("COMPLETE","EARLY_STOP_PERFORMANCE")
        except BaseException as exc:
            status["arms"][arm]="FAILED";status["status"]="FAILED"
            status["error"]=repr(exc);status["traceback"]=traceback.format_exc()
            status["finished_unix"]=time.time()
            write_json(HERE/"queue_status.json",status)
            raise
        write_json(HERE/"queue_status.json",status)
    build()
    status["status"]="COMPLETE";status["current_arm"]=None
    status["finished_unix"]=time.time()
    write_json(HERE/"queue_status.json",status)
    print("QUEUE_COMPLETE",flush=True)

if __name__=="__main__":main()
