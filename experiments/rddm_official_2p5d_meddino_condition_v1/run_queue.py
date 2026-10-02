"""Sequential queue, bounded to this experiment, with audited result-only final commit."""
import datetime,json,subprocess,traceback
from pathlib import Path
from zoneinfo import ZoneInfo
from provenance import ROOT,write_json
from train import ARMS
def main():
    ready=json.loads((ROOT/"READINESS.json").read_text());assert ready["status"]=="PASS"
    for a in ARMS:assert not (ROOT/"runs"/a).exists()
    for a in ARMS:
        write_json(ROOT/"QUEUE_STATUS.json",{"status":"RUNNING","phase":a,"at":datetime.datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()})
        with (ROOT/"logs"/(a+".log")).open("x") as f:
            subprocess.run(["/home/xujian/miniconda3/envs/flexict/bin/python","-u","train.py","--arm",a,"--output",str(ROOT/"runs"/a)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    write_json(ROOT/"QUEUE_STATUS.json",{"status":"RUNNING","phase":"postprocess"})
    with (ROOT/"logs/postprocess.log").open("x") as f:
        subprocess.run(["/home/xujian/miniconda3/envs/flexict/bin/python","-u","postprocess.py"],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    summary=json.loads((ROOT/"FINAL_SUMMARY.json").read_text());assert summary["status"]=="COMPLETE"
    write_json(ROOT/"QUEUE_STATUS.json",{"status":"COMPLETE","at":datetime.datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()})
    files=[ROOT/n for n in ("REPORT.md","FINAL_SUMMARY.json","VALIDATION_HISTORY.csv","PER_SLICE_DELTA.csv","DATA_ACCESS_AUDIT.json","CONDITION_ACTIVITY.csv","TRAINING_DIAGNOSTICS.csv","QUEUE_STATUS.json","RUN_MANIFEST.json")]
    files+=list(ROOT.glob("FULL_L333_*.json"))+list(ROOT.glob("DIAGNOSTIC_*.json"))
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
        write_json(ROOT/"QUEUE_STATUS.json",{"status":"FAILED","traceback":traceback.format_exc(),"at":datetime.datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()})
        raise
