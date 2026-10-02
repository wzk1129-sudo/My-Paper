"""Finish verified existing 8k runs; rerun only matched Fast64 diagnostics and report."""
import ast,datetime,json,subprocess,traceback
from zoneinfo import ZoneInfo
from provenance import ROOT,write_json
from train import ARMS
from postprocess import run_paths

def main():
    audit=json.loads((ROOT/"POSTPROCESS_THREAD_AUDIT.json").read_text());assert audit["status"]=="PASS"
    for path in run_paths().values():
        final=json.loads((path/"FINAL_STATE.json").read_text())
        assert final["optimizer_step"]==final["ema_updates"]==8000 and final["parameters_finite"]
    write_json(ROOT/"QUEUE_STATUS.json",{"status":"RUNNING","phase":"postprocess","recovery":True,"training_completed":True,
        "at":datetime.datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()})
    with (ROOT/"logs/postprocess_threads2.log").open("x") as f:
        subprocess.run(["/home/xujian/miniconda3/envs/flexict/bin/python","-u","postprocess.py"],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    summary=json.loads((ROOT/"FINAL_SUMMARY.json").read_text());assert summary["status"]=="COMPLETE"
    source=ast.parse((ROOT/"recover_candidate_queue.py").read_text())
    old_note=[ast.literal_eval(n.args[0]) for n in ast.walk(source) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
        and isinstance(n.func.value,ast.Name) and n.func.value.id=="report" and n.func.attr=="write"][0]
    note="\n## 后处理数值一致性恢复\n\n两臂 8000 步训练和 EMA checkpoint 均已完成。首轮后处理硬编码 cpu_threads=4，与训练配置 cpu_threads=2 不一致，导致 CPU float32 metric mean 归约顺序产生 2.46e-8 dB 的平均 PSNR 差异，触发 1e-8 断言。POSTPROCESS_THREAD_AUDIT.json 使用同一不变预测图，分别精确复现两组逐 slice 指标，并确认重新加载的 frozen MedDINO hash 与训练初始化一致。postprocess.py 改为读取相同配置的 cpu_threads；保留原断言、评价函数、训练结果及 checkpoint。三组 Fast64 诊断在 2 threads 下重新执行；未重跑训练。旧日志和首次后处理输出保存在 logs/postprocess_retry1.log 与忽略目录 runs/postprocess_attempt1_metadata_20261002。\n"
    normal=summary["inference_diagnostics"]["NORMAL"]
    assert all(abs(normal[k]-summary["candidate_8k"][k])<=1e-8 for k in ("psnr","ssim","rmse_hu","mae_hu"))
    summary["postprocess_cpu_threads"]=2;summary["checkpoint_reload_metrics_match_training"]=True
    summary["postprocess_thread_audit"]="POSTPROCESS_THREAD_AUDIT.json"
    write_json(ROOT/"FINAL_SUMMARY.json",summary)
    with (ROOT/"REPORT.md").open("a") as f:
        f.write(old_note+note)
        diag=summary["inference_diagnostics"]
        f.write("\n8k NORMAL − CONDITION_OFF ΔPSNR = %.6f dB；NORMAL − SHUFFLED_MEDDINO ΔPSNR = %.6f dB。64/64 slices 的正向增益仍只有 %.6f dB 平均值，未达到预定 Full-L333 门槛。关闭条件后的下降说明已训练 Candidate 使用了该条件路径；shuffle 差异很小，不足以证明强 patient-specific anatomy 利用。\n"%(normal["psnr"]-diag["CONDITION_OFF"]["psnr"],normal["psnr"]-diag["SHUFFLED_MEDDINO"]["psnr"],summary["delta_8k"]["psnr"]))
    write_json(ROOT/"QUEUE_STATUS.json",{"status":"COMPLETE","recovery":True,"training_completed":True,
        "at":datetime.datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()})
    names=("REPORT.md","FINAL_SUMMARY.json","VALIDATION_HISTORY.csv","PER_SLICE_DELTA.csv","DATA_ACCESS_AUDIT.json",
        "CONDITION_ACTIVITY.csv","TRAINING_DIAGNOSTICS.csv","QUEUE_STATUS.json","RUN_MANIFEST.json","RECOVERY_RUN_MANIFEST.json","POSTPROCESS_RUN_MANIFEST.json")
    files=[ROOT/n for n in names]+list(ROOT.glob("FULL_L333_*.json"))+list(ROOT.glob("DIAGNOSTIC_*.json"))
    assert all(p.is_file() and p.suffix in (".md",".json",".csv") for p in files)
    repo=ROOT.parents[1]
    subprocess.run(["git","add",*[str(p.relative_to(repo)) for p in files]],cwd=repo,check=True)
    subprocess.run(["git","diff","--cached","--check"],cwd=repo,check=True)
    subprocess.run(["git","diff","--cached","--stat"],cwd=repo,check=True)
    subprocess.run(["git","diff","--cached","--",str((ROOT/"REPORT.md").relative_to(repo))],cwd=repo,check=True)
    subprocess.run(["git","commit","-m","Report matched official RDDM 2.5D MedDINO 8k results and diagnostics"],cwd=repo,check=True)
    sha=subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip()
    write_json(ROOT/"runs/COMPLETION.json",{"commit":sha,"summary":summary})
    print("QUEUE_COMPLETE_COMMIT "+sha,flush=True)
if __name__=="__main__":
    try:main()
    except Exception:
        write_json(ROOT/"QUEUE_STATUS.json",{"status":"FAILED","phase":"postprocess","training_completed":True,
            "traceback":traceback.format_exc(),"at":datetime.datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()})
        raise
