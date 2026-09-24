"""Serial, no-retry GPU queue for the two pre-registered arms."""
import json
import os
import subprocess
import sys
import time
from mayo2016_smallset_v1.runtime import write_json
from .protocol import ARMS, HERE, experiment_config

def main():
    cfg,_=experiment_config()
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG")==":4096:8"
    for arm in ARMS:
        record=json.loads((HERE/("SMOKE_"+arm+".json")).read_text())
        assert record["status"]=="SMOKE_PASS" and record["optimizer_steps"]==2
    status={"state":"RUNNING","order":list(ARMS),"current":None,"completed":[],
            "started_at_unix":time.time(),"max_optimizer_steps_per_arm":12000,
            "automatic_retry":False,"automatic_extension":False}
    write_json(HERE/"queue_status.json",status)
    try:
        for arm in ARMS:
            status["current"]=arm
            write_json(HERE/"queue_status.json",status)
            subprocess.run([sys.executable,"-m","mechanism_ablation.train","train",
                            "--arm",arm,"--device","cuda:0"],check=True)
            status["completed"].append(arm)
            write_json(HERE/"queue_status.json",status)
        status["current"]="FULL_L333_AND_REPORT"
        write_json(HERE/"queue_status.json",status)
        subprocess.run([sys.executable,"-m","mechanism_ablation.report"],check=True)
        status.update(state="COMPLETE",current=None,ended_at_unix=time.time())
        write_json(HERE/"queue_status.json",status)
    except BaseException as exc:
        status.update(state="FAILED",error=repr(exc),ended_at_unix=time.time())
        write_json(HERE/"queue_status.json",status)
        raise

if __name__=="__main__":
    main()
