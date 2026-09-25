"""Serial GPU queue. Stop on first failure; never restart or extend."""
import json
import subprocess
import sys
import traceback
from pathlib import Path
import torch
from mayo2016_smallset_v1.runtime import ROOT, sha256, write_json
from .protocol import HERE, DIRECT_CK, DIRECT_SHA, experiment_config
from .train import train

def main():
    assert json.loads((HERE/"runs/C0_DIRECT/final_summary.json").read_text())["status"]=="COMPLETE"
    for arm in ("C1_CAPACITY","C2_EDGE"):
        assert json.loads((HERE/"smoke_verified"/arm/"SMOKE_RESULT.json").read_text())["status"]=="SMOKE_PASS"
    files=("anatomy_controlnet_v1/model.py","anatomy_controlnet_v1/structure.py",
           "anatomy_controlnet_v1/runtime.py","anatomy_controlnet_v1/train.py",
           "anatomy_controlnet_v1/evaluate.py","anatomy_controlnet_v1/protocol.py",
           "context_2p5d/model.py","context_2p5d/data.py",
           "src/rddm_rleb.py","splits/L333_fast64.txt",
           "configs/rddm_fast_dev.json","configs/fast_dev_protocol.json")
    source={name:sha256(ROOT/name) for name in files}
    write_json(HERE/"EXPERIMENT_CONFIG.json",{
        "arms":["C0_DIRECT","C1_CAPACITY","C2_EDGE"],
        "C0":experiment_config("C0_DIRECT"),
        "C1":experiment_config("C1_CAPACITY"),
        "C2":experiment_config("C2_EDGE"),
        "medical_arms":"MEDICAL_ENCODER_BLOCKED"})
    write_json(HERE/"PROVENANCE.json",{
        "parent_commit":"058ed8455ca386966b9f82ba7e12cee2116cedc4",
        "branch":subprocess.check_output(["git","branch","--show-current"],cwd=str(ROOT),text=True).strip(),
        "git_head":subprocess.check_output(["git","rev-parse","HEAD"],cwd=str(ROOT),text=True).strip(),
        "source_sha256":source,"checkpoint_sha256":sha256(DIRECT_CK),
        "python":sys.version,"torch":torch.__version__,"cuda":torch.version.cuda,
        "gpu":torch.cuda.get_device_name(0),"seed":20260829})
    assert sha256(DIRECT_CK)==DIRECT_SHA
    for arm in ("C1_CAPACITY","C2_EDGE"):
        write_json(HERE/"queue_status.json",{"status":"RUNNING","arm":arm,"completed":[],
                                             "next":"C2_EDGE" if arm=="C1_CAPACITY" else None})
        try:
            train(arm,"cuda:0")
        except BaseException as exc:
            write_json(HERE/"queue_status.json",{"status":"FAILED","arm":arm,"error":repr(exc)})
            traceback.print_exc()
            raise
    write_json(HERE/"queue_status.json",{"status":"TRAINING_COMPLETE",
        "completed":["C0_DIRECT","C1_CAPACITY","C2_EDGE"],"medical_arms":"BLOCKED"})
    print("TRAINING_COMPLETE",flush=True)
if __name__=="__main__":main()
