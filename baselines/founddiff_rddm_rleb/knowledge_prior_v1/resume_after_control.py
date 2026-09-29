"""Resume only A, teacher, B after the preserved Control 6k and BF16 audit fix."""
import time,traceback
import torch
from context_2p5d.protocol import config as context_config
from mayo2016_smallset_v1.runtime import write_json
from .engine import HERE,ARMS,NODES,train_arm,read_json,paired
from .stream import digest
from .queue import preflight,teacher_train,calibrate,load_teacher,gate,full_eval,report

def main():
    cfg,base=context_config()
    original=read_json(HERE/"QUEUE_STATUS.json")
    assert original["status"]=="FAILED" and original["phase"]=="A_MEDDINO_BCA"
    assert "unsupported ScalarType BFloat16" in original["error"]
    preflight(cfg,base)
    assert read_json(HERE/"A_INIT_RETRY_AUDIT.json")["status"]=="PASS"
    h=(HERE/"TRAIN_STREAM_SHA256.txt").read_text().strip()
    manifest=read_json(HERE/"TRAIN_STREAM_MANIFEST.json")
    assert len(manifest)==6000 and digest(manifest)==h
    control=read_json(HERE/"CONTROL_CONTINUE/final_summary.json")
    assert control["status"]=="COMPLETE" and control["steps"]==6000 and control["manifest_sha256"]==h
    assert [r["step"] for r in control["history"]]==list(NODES)
    assert control["history"][-1]["train_stream_sha256"]==h
    assert control["access"]["L506_opens"]==control["access"]["external_memory_opens"]==control["access"]["neighbor_target_opens"]==control["access"]["blocked_attempts"]==0
    assert (HERE/"CONTROL_CONTINUE/step_6000.pt").is_file()
    for arm in (ARMS[1],ARMS[2]):
        assert not (HERE/arm).exists(),arm
    assert not (HERE/"B_TEACHER_PRETRAIN").exists()
    assert not (HERE/"diagnostics").exists()
    status={"status":"RUNNING","phase":"A_MEDDINO_BCA","started_at":time.time(),
        "preserved_original_queue_status":"FAILED","preserved_control_6000":True,
        "repair":"dtype-safe exact state hash for BFloat16 model audit"}
    assert not (HERE/"QUEUE_STATUS_RETRY.json").exists()
    write_json(HERE/"QUEUE_STATUS_RETRY.json",status)
    try:
        a=train_arm(ARMS[1],cfg,base,manifest,h,control["history"])
        status["phase"]="B_TEACHER_PRETRAIN";write_json(HERE/"QUEUE_STATUS_RETRY.json",status)
        teacher_audit=teacher_train(base)
        status["phase"]="B_LAMBDA_CALIBRATION";write_json(HERE/"QUEUE_STATUS_RETRY.json",status)
        lam=calibrate(base,manifest)
        status["phase"]="B_NDCT_DETAIL_PRIOR";write_json(HERE/"QUEUE_STATUS_RETRY.json",status)
        teacher=load_teacher("cuda:0")
        b=train_arm(ARMS[2],cfg,base,manifest,h,control["history"],teacher,lam)
        del teacher;torch.cuda.empty_cache()
        gates={"A":gate(a,control,ARMS[1]),"B":gate(b,control,ARMS[2])}
        full={}
        if gates["A"]["promote"] or gates["B"]["promote"]:
            status["phase"]="FULL_L333";write_json(HERE/"QUEUE_STATUS_RETRY.json",status)
            full["control"]=full_eval(ARMS[0],cfg,base)
            ctrlrows=read_json(HERE/"full_l333"/f"{ARMS[0]}_per_slice.json")
            for key,arm in (("A",ARMS[1]),("B",ARMS[2])):
                if gates[key]["promote"]:
                    full[key]=full_eval(arm,cfg,base)
                    rows=read_json(HERE/"full_l333"/f"{arm}_per_slice.json")
                    full[key]["paired"]=paired(rows,ctrlrows)
                    write_json(HERE/"full_l333"/f"{arm}_paired_summary.json",full[key]["paired"])
        report(control,a,b,gates,full)
        status.update(status="COMPLETE",phase="COMPLETE",finished_at=time.time())
        write_json(HERE/"QUEUE_STATUS_RETRY.json",status)
    except BaseException as exc:
        status.update(status="FAILED",error=repr(exc),traceback=traceback.format_exc(),finished_at=time.time())
        write_json(HERE/"QUEUE_STATUS_RETRY.json",status)
        raise
if __name__=="__main__":main()
