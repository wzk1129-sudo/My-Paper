"""Non-destructive continuation after original Control stopped at step 2800."""
import csv
import json
import time
import traceback
from pathlib import Path
import torch
from context_2p5d.data import ContextPairs
from context_2p5d.protocol import config as context_config
from knowledge_prior_v1.stream import make_manifest,digest
from knowledge_prior_v1.core import DIRECT_CK,DIRECT_SHA,MED_CK,MED_SHA,sha
from mayo2016_smallset_v1.runtime import ROOT,seed_everything,write_json
from fulltoken_ca_v1 import queue as q

OLD=ROOT/"experiments/meddino_fulltoken_ca_v1"
NEW=ROOT/"experiments/meddino_fulltoken_ca_v1_recovery_20261001"
CONTROL_CHECKPOINT_SHA="e619d97c445152e2ebb7a103e8e173cb3ea36f352fa4646c280348df390f4b3f"

def main():
    assert NEW.is_dir() and not (NEW/"QUEUE_STATUS.json").exists()
    assert not (NEW/q.ARMS[1]).exists()
    assert not (NEW/q.ARMS[0]/"step_6000.pt").exists()
    assert sha(OLD/q.ARMS[0]/"step_2000.pt")==CONTROL_CHECKPOINT_SHA
    assert sha(DIRECT_CK)==DIRECT_SHA and sha(MED_CK)==MED_SHA
    assert sha(ROOT/"src/rddm_rleb.py")==q.FORMAL_SOURCE_SHA
    readiness=json.loads((OLD/"READINESS_AUDIT.json").read_text())
    assert readiness["status"]=="PASS" and readiness["step0_max_abs_output_diff"]==0
    assert readiness["data_access"]["L506_opens"]==0
    q.OUT=NEW
    seed_everything(20260829);torch.set_num_threads(2)
    cfg,base=context_config()
    train=ContextPairs(base,"train")
    assert len(train)==1145
    manifest=make_manifest(train,steps=8000)
    manifest_sha=digest(manifest)
    assert digest(manifest[:6000])==q.MANIFEST_6K_SHA
    old_ck=torch.load(OLD/q.ARMS[0]/"step_2000.pt",map_location="cpu",weights_only=False)
    assert old_ck["step"]==old_ck["ema_updates"]==2000
    assert old_ck["manifest_sha256"]==manifest_sha and old_ck["nonfinite_count"]==0
    del old_ck
    old_rows=list(csv.DictReader(open(NEW/q.ARMS[0]/"validation_history.csv",newline="")))
    assert [int(x["step"]) for x in old_rows]==[0,1000,2000]
    for row in old_rows:
        step=int(row["step"])
        assert row["train_stream_sha256"]==digest(manifest[:step])
    write_json(NEW/"RECOVERY_SOURCE.json",{
        "original_output":str(OLD),
        "recovery_output":str(NEW),
        "original_control_last_observed_step":2800,
        "last_intact_control_checkpoint_step":2000,
        "control_checkpoint_sha256":CONTROL_CHECKPOINT_SHA,
        "original_failure_cause":"unknown; original error log was overwritten by a second launch",
        "second_launch_failure":"FileExistsError on existing DIRECT_CONTINUE directory before any optimizer step",
        "original_results_untouched":True,
        "control_first_2000_metrics_carried_exactly":True,
        "control_steps_2001_to_2800_reexecuted_from_step_2000_in_new_output":True,
        "old_first_2000_file_open_audit":"not persisted because original process disappeared",
        "guarded_dataset_only_L067_L143":True,
        "old_checkpoint_manifest_sha256":manifest_sha})
    cfg_record=json.loads((OLD/"CONFIG.json").read_text())
    cfg_record["recovery"]={"control_resume_step":2000,
        "control_checkpoint_sha256":CONTROL_CHECKPOINT_SHA,
        "candidate_start_step":0,"original_root":str(OLD)}
    write_json(NEW/"CONFIG.json",cfg_record)
    write_json(NEW/"PROVENANCE.json",{
        "parent_commit":"51c608b9d2148102611c3e5717fa0f79bdf1b2ae",
        "branch":"experiment/rddm-meddino-fulltoken-ca-v1",
        "formal_source_sha256":q.FORMAL_SOURCE_SHA,
        "direct_checkpoint_sha256":DIRECT_SHA,
        "meddino_checkpoint_sha256":MED_SHA,
        "control_resume_checkpoint_sha256":CONTROL_CHECKPOINT_SHA,
        "model_source_sha256":sha(ROOT/"fulltoken_ca_v1/model.py"),
        "queue_source_sha256":sha(ROOT/"fulltoken_ca_v1/queue.py"),
        "recovery_source_sha256":sha(ROOT/"fulltoken_ca_v1/recovery.py"),
        "train_manifest_sha256_6k":q.MANIFEST_6K_SHA,
        "train_manifest_sha256_8k":manifest_sha})
    q.status(state="RUNNING",arm=q.ARMS[0],step=2000,target=6000,
             recovery_from_step=2000)
    control=q.train_arm(q.ARMS[0],cfg,base,manifest,manifest_sha,6000,start_step=2000)
    candidate=q.train_arm(q.ARMS[1],cfg,base,manifest,manifest_sha,6000)
    bystep={int(r["step"]):r for r in candidate}
    d4=float(bystep[4000]["delta_psnr"])
    d6=float(bystep[6000]["delta_psnr"])
    full=None
    if d6<=0:
        decision="STOP_DELTA_6K_NONPOSITIVE"
    elif d4>0 and d6>=.02 and d6>=d4-.005:
        decision="EXTEND_TO_8K"
        q.status(state="RUNNING",arm=q.ARMS[0],step=6000,target=8000)
        control=q.train_arm(q.ARMS[0],cfg,base,manifest,manifest_sha,8000,start_step=6000)
        candidate=q.train_arm(q.ARMS[1],cfg,base,manifest,manifest_sha,8000,start_step=6000)
        d8=float(candidate[-1]["delta_psnr"])
        fraction=float(candidate[-1]["positive_fraction"])
        if d8>=.03 and fraction>=.6:
            decision="FULL_L333_PROMOTED"
            c=q.full_validate(q.ARMS[0],cfg,base,8000)
            m=q.full_validate(q.ARMS[1],cfg,base,8000)
            full={"step":8000,**{"control_"+k:c[k] for k in q.FIELDS},
                  **{"candidate_"+k:m[k] for k in q.FIELDS},
                  **{"delta_"+k:m[k]-c[k] for k in q.FIELDS},
                  "slices":610,"weights":"ema"}
            write_json(NEW/"FULL_L333_COMPARISON.json",full)
        else:decision="STOP_8K_PROMOTION_NOT_MET"
    else:decision="STOP_6K_EXTENSION_NOT_MET"
    audits={}
    for arm in q.ARMS:
        checks=[]
        for filename in ("DATA_ACCESS_AUDIT.json","DATA_ACCESS_AUDIT_RESUME.json"):
            path=NEW/arm/filename
            if path.exists():checks.append(json.loads(path.read_text()))
        assert checks
        assert all(all(x[k]==0 for k in ("L506_opens","external_memory_opens",
                                         "neighbor_target_opens","blocked_attempts")) for x in checks)
        assert json.loads((NEW/arm/"final_summary.json").read_text())["nonfinite_count"]==0
        audits[arm]=checks
    q.final_report(control,candidate,decision,full)
    write_json(NEW/"FINAL_AUDIT.json",{
        "status":"PASS","control_last_step":int(control[-1]["step"]),
        "candidate_last_step":int(candidate[-1]["step"]),
        "control_resume_checkpoint_sha256":CONTROL_CHECKPOINT_SHA,
        "matched_stream_sha256_6k":q.MANIFEST_6K_SHA,
        "same_direct_initialization":True,"readiness_pass":True,
        "recovery_phase_data_access":audits,
        "first_2000_control_file_open_audit":"not persisted; guarded dataset and stream verified",
        "formal_source_sha256":q.FORMAL_SOURCE_SHA,"decision":decision})
    with open(NEW/"REPORT.md","a") as f:
        f.write("\nRecovery note: Control resumed from an intact step-2000 checkpoint into a new output root. ")
        f.write("The original partial run remains unchanged. First-2000 file-open audit was not persisted; ")
        f.write("the guarded data loader and train manifest were verified.\n")
    q.status(state="COMPLETE",decision=decision,
             control_step=int(control[-1]["step"]),candidate_step=int(candidate[-1]["step"]))
    print(json.dumps({"state":"COMPLETE","decision":decision,
        "delta_psnr":float(candidate[-1]["psnr"])-float(control[-1]["psnr"])}))

if __name__=="__main__":
    try:main()
    except Exception as exc:
        NEW.mkdir(parents=True,exist_ok=True)
        q.OUT=NEW
        q.status(state="FAILED",error=repr(exc),traceback=traceback.format_exc())
        traceback.print_exc()
        raise
