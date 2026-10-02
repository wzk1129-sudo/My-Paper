"""Recover only the unstarted candidate; preserve completed Control and failed logs."""
import datetime,json,subprocess,traceback
from zoneinfo import ZoneInfo
from provenance import ROOT,write_json
from train import ARMS
from postprocess import run_paths
def main():
    assert json.loads((ROOT/"READINESS.json").read_text())["status"]=="PASS"
    audit=json.loads((ROOT/"HASH_FIX_AUDIT.json").read_text());assert audit["status"]=="PASS"
    paths=run_paths()
    control=json.loads((paths[ARMS[0]]/"FINAL_STATE.json").read_text())
    assert control["optimizer_step"]==control["ema_updates"]==8000
    failed=ROOT/"runs"/ARMS[1]
    assert not (failed/"train.jsonl").exists() and not list((failed/"checkpoints").glob("*.pt"))
    candidate=paths[ARMS[1]];assert not candidate.exists()
    write_json(ROOT/"QUEUE_STATUS.json",{"status":"RUNNING","phase":ARMS[1],"recovery":True,"at":datetime.datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()})
    with (ROOT/"logs"/(ARMS[1]+"_retry1.log")).open("x") as f:
        subprocess.run(["/home/xujian/miniconda3/envs/flexict/bin/python","-u","train.py","--arm",ARMS[1],"--output",str(candidate)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    write_json(ROOT/"QUEUE_STATUS.json",{"status":"RUNNING","phase":"postprocess","recovery":True})
    with (ROOT/"logs/postprocess_retry1.log").open("x") as f:
        subprocess.run(["/home/xujian/miniconda3/envs/flexict/bin/python","-u","postprocess.py"],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    summary=json.loads((ROOT/"FINAL_SUMMARY.json").read_text());assert summary["status"]=="COMPLETE"
    with (ROOT/"REPORT.md").open("a") as report:
        report.write("\n## 运行恢复审计\n\nControl 已完成后，Candidate 在训练 step 0 前的 MedDINO state hash 审计中因 BFloat16→NumPy 不支持而退出。修复仅将哈希数据读取改为 contiguous uint8 byte view；train main AST、所有 model/readiness/diffusion 依赖不变，float32 shared-parameter hash 与已完成 Control 的初始化完全一致。HASH_FIX_AUDIT.json 记录 dtype 兼容、实际 MedDINO hash、RNG 不变及初始参数匹配证明。Candidate 在新的 retry1 目录从相同父 EMA 和 fresh Adam 开始；Control 结果与失败目录/log 均保留。RUN_PATHS.json 明确给出实际两臂路径。最终 matched stream 审计仍要求全部 8000 steps 一致。\n")
    write_json(ROOT/"QUEUE_STATUS.json",{"status":"COMPLETE","recovery":True,"at":datetime.datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()})
    names=("REPORT.md","FINAL_SUMMARY.json","VALIDATION_HISTORY.csv","PER_SLICE_DELTA.csv","DATA_ACCESS_AUDIT.json","CONDITION_ACTIVITY.csv","TRAINING_DIAGNOSTICS.csv","QUEUE_STATUS.json","RUN_MANIFEST.json","RECOVERY_RUN_MANIFEST.json")
    files=[ROOT/n for n in names]+list(ROOT.glob("FULL_L333_*.json"))+list(ROOT.glob("DIAGNOSTIC_*.json"))
    assert all(p.exists() and p.suffix in (".md",".json",".csv") for p in files)
    repo=ROOT.parents[1]
    subprocess.run(["git","add",*[str(p.relative_to(repo)) for p in files]],cwd=repo,check=True)
    subprocess.run(["git","diff","--cached","--check"],cwd=repo,check=True)
    subprocess.run(["git","diff","--cached","--stat"],cwd=repo,check=True)
    subprocess.run(["git","diff","--cached","--",str((ROOT/"REPORT.md").relative_to(repo))],cwd=repo,check=True)
    subprocess.run(["git","commit","-m","Report official RDDM 2.5D MedDINO condition matched 8k results"],cwd=repo,check=True)
    sha=subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip()
    write_json(ROOT/"runs/COMPLETION.json",{"commit":sha,"summary":summary})
    print("QUEUE_COMPLETE_COMMIT "+sha,flush=True)
if __name__=="__main__":
    try:main()
    except Exception:
        write_json(ROOT/"QUEUE_STATUS.json",{"status":"FAILED","recovery":True,"traceback":traceback.format_exc(),"at":datetime.datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()})
        raise
