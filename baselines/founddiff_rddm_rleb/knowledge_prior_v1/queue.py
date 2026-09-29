"""Serial Control -> MedDINO BCA -> disjoint NDCT teacher -> detail-prior queue."""
import argparse,csv,json,math,os,platform,subprocess,time,traceback
from pathlib import Path
import numpy as np
import torch
from context_2p5d.data import ContextPairs,ContextBatchStream,ContextDataGuard
from context_2p5d.evaluate import validate
from context_2p5d.protocol import config as context_config
from mayo2016_smallset_v1.runtime import ROOT,seed_everything,write_json,atomic_save,sha256
from .core import load_arm,matched_loss,DIRECT_CK,DIRECT_SHA,MED_CK,MED_SHA,MED_ROOT,MED_COMMIT
from .stream import assert_batch,timestep_noise
from .teacher import Teacher,DonorPatches,DonorGuard,teacher_loss,corrupt,DONORS
from .engine import HERE,NODES,ARMS,FIELDS,prepare_manifest,train_arm,read_json,csv_write,paired

def provenance():
    def git(*a):return subprocess.check_output(["git",*a],cwd=ROOT,text=True).strip()
    return {"branch":git("branch","--show-current"),"parent_commit":git("rev-parse","HEAD"),
       "python":platform.python_version(),"torch":torch.__version__,"cuda":torch.version.cuda,
       "gpu":torch.cuda.get_device_name(0),"CUBLAS_WORKSPACE_CONFIG":os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
       "direct_checkpoint_sha256":sha256(DIRECT_CK),"meddino_checkpoint_sha256":sha256(MED_CK),
       "meddino_official_commit":git("-C",str(MED_ROOT),"rev-parse","HEAD"),
       "formal_source_sha256":sha256(ROOT/"src/rddm_rleb.py"),
       "fast64_split_sha256":sha256(ROOT/"splits/L333_fast64.txt")}
def preflight(cfg,base):
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG")==":4096:8"
    assert sha256(DIRECT_CK)==DIRECT_SHA and sha256(MED_CK)==MED_SHA
    assert subprocess.check_output(["git","-C",str(MED_ROOT),"rev-parse","HEAD"],text=True).strip()==MED_COMMIT
    assert sha256(ROOT/"src/rddm_rleb.py")=="e8ed93223fbdf4a0e2521f73c7a4f213561f6a191a1a7b2a5f95934b85ebf574"
    assert sha256(ROOT/"splits/L333_fast64.txt")=="0d3170fd2462c7b19d1cc0f187b12e03bac2c9b15babca7c1d881568adde6895"
    parity=read_json(HERE/"STEP0_PARITY.json")
    assert parity["pass"] and abs(parity["psnr"]-40.76762730415604)<=.005
    assert parity["access"]["L506_opens"]==parity["access"]["external_memory_opens"]==parity["access"]["blocked_attempts"]==0
    return parity
def teacher_train(base):
    output=HERE/"B_TEACHER_PRETRAIN";output.mkdir(exist_ok=False)
    seed_everything(20260929);torch.set_num_threads(2);device=torch.device("cuda:0")
    teacher=Teacher().to(device);opt=torch.optim.AdamW(teacher.parameters(),lr=1e-4,betas=(.9,.99),weight_decay=1e-4)
    data=DonorPatches(base);logs=[];seconds=0;nonfinite=0
    torch.cuda.reset_peak_memory_stats(device)
    with DonorGuard(base["data"]["root"]) as guard:
        for step in range(1,8001):
            started=time.perf_counter()
            clean,meta=data.batch(8);clean=clean.to(device)
            corrupted=corrupt(clean)
            teacher.train();opt.zero_grad(set_to_none=True)
            pred=teacher(corrupted)
            assert pred.shape==clean.shape and bool(torch.isfinite(pred).all())
            total,img,bands=teacher_loss(pred,clean)
            if not bool(torch.isfinite(total)):nonfinite+=1;raise FloatingPointError("teacher loss")
            total.backward()
            norm=float(torch.nn.utils.clip_grad_norm_(teacher.parameters(),1.0))
            if not math.isfinite(norm):nonfinite+=1;raise FloatingPointError("teacher gradient")
            opt.step();torch.cuda.synchronize(device);seconds+=time.perf_counter()-started
            if step==1 or step%100==0:
                log={"step":step,"L_img":float(img.detach()),"L_D1":float(bands[0].detach()),
                    "L_D2":float(bands[1].detach()),"L_D3":float(bands[2].detach()),
                    "L_teacher":float(total.detach()),"gradient_norm_pre_clip":norm,
                    "output_std":float(pred.detach().std()),"seconds_per_step":seconds/step,
                    "peak_vram_bytes":torch.cuda.max_memory_allocated(device)}
                logs.append(log);csv_write(output/"TEACHER_TRAIN_HISTORY.csv",logs)
                write_json(output/"progress.json",log)
            if step%2000==0:
                if float(pred.detach().std())<.005:raise RuntimeError("teacher collapsed to near-constant output")
                atomic_save(output/f"step_{step}.pt",{"step":step,"model_state_dict":teacher.state_dict(),"optimizer_state_dict":opt.state_dict()})
        access=guard.report()
        write_json(output/"DATA_ACCESS_AUDIT.json",access)
    assert nonfinite==0 and access["L506_opens"]==access["donor_ldct_opens"]==access["blocked_attempts"]==0
    audit={"status":"PASS","steps":8000,"nonfinite_count":nonfinite,"access":access,"final":logs[-1],
           "train_seconds":seconds,"peak_vram_bytes":torch.cuda.max_memory_allocated(device)}
    write_json(output/"TEACHER_AUDIT.json",audit)
    return audit
def load_teacher(device):
    ck=torch.load(HERE/"B_TEACHER_PRETRAIN/step_8000.pt",map_location="cpu",weights_only=False)
    assert ck["step"]==8000
    teacher=Teacher();teacher.load_state_dict(ck["model_state_dict"],strict=True)
    return teacher.to(device).eval().requires_grad_(False)
def calibrate(base,manifest):
    output=HERE/"diagnostics";output.mkdir(exist_ok=False)
    device=torch.device("cuda:0");seed_everything(20260829)
    model,_=load_arm(ARMS[2],device);teacher=load_teacher(device)
    data=ContextPairs(base,"train");stream=ContextBatchStream(data,20260829)
    res=[];pri=[];terms_record=[]
    with ContextDataGuard(base,"FAST") as guard:
        data.bind_guard(guard)
        with torch.no_grad():
            for step in range(100):
                want=manifest[step]
                prev,center,next_,target,meta=stream.next(2);assert_batch(meta,want)
                prev,center,next_,target=(x.to(device) for x in (prev,center,next_,target))
                t,noise=timestep_noise(want,target.shape,device)
                lres,pred,xin,xgt=matched_loss(model,prev,center,next_,target,t,noise)
                lp,terms=teacher.prior(xin-pred,xgt)
                assert bool(torch.isfinite(lres)) and bool(torch.isfinite(lp))
                res.append(float(lres));pri.append(float(lp));terms_record.append([float(t) for t in terms])
        access=guard.report()
    mres=float(np.median(res[50:]));mprior=float(np.median(pri[50:]))
    lam=.1*mres/(mprior+1e-12)
    audit={"batches":100,"optimizer_steps":0,"median_last50_L_res":mres,"median_last50_L_prior":mprior,
        "lambda_prior":lam,"weighted_prior_fraction_vs_res":lam*mprior/mres,
        "L_res":res,"L_prior":pri,"layer_terms":terms_record,"access":access}
    write_json(output/"B_LAMBDA_CALIBRATION.json",audit)
    assert 1e-4<=lam<=10 and access["L506_opens"]==access["external_memory_opens"]==access["blocked_attempts"]==0
    del model,teacher;torch.cuda.empty_cache()
    return lam
def full_eval(arm,cfg,base):
    device=torch.device("cuda:0")
    ck=torch.load(HERE/arm/"step_6000.pt",map_location="cpu",weights_only=False)
    assert ck["step"]==6000
    model,_=load_arm(arm,device);model.load_state_dict(ck["ema_state_dict"],strict=True)
    out=HERE/"full_l333";out.mkdir(exist_ok=True)
    with ContextDataGuard(base,"FULL",allow_training=False) as guard:
        ds=ContextPairs(base,"validation","FULL");ds.bind_guard(guard)
        rows_path=out/f"{arm}_per_slice.json"
        result=validate(model,ds,cfg,device,rows_path)
        access=guard.report()
    assert result["slices"]==610 and access["L506_opens"]==access["external_memory_opens"]==access["blocked_attempts"]==0
    write_json(out/f"{arm}_summary.json",{**result,"access":access})
    del model;torch.cuda.empty_cache()
    return result
def gate(summary,control,arm):
    if summary["steps"]!=6000:return {"promote":False,"reason":"STOP_AT_4K"}
    final=summary["history"][-1];ref=control["history"][-1]
    ok=final["delta_vs_control"]>=.03 and final["positive_slice_fraction"]>=.65 and final["ssim"]>=ref["ssim"]-.00005
    return {"promote":ok,"reason":"PROMOTE_FULL_L333" if ok else "FAST_ONLY","delta":final["delta_vs_control"],"positive_slice_fraction":final["positive_slice_fraction"]}
def report(control,a,b,gates,full):
    rows=[]
    for summary in (control,a,b):
        for r in summary["history"]:
            rows.append({"arm":summary["arm"],"step":r["step"],"psnr":r["psnr"],
                "delta_vs_control":r.get("delta_vs_control",0),"ssim":r["ssim"],"rmse_hu":r["rmse_hu"],
                "mae_hu":r["mae_hu"],"positive_slice_fraction":r.get("positive_slice_fraction"),
                "seconds_per_step":summary["seconds_per_step"],"peak_vram_bytes":summary["peak_vram_bytes"]})
    csv_write(HERE/"FINAL_COMPARISON.csv",rows)
    summary={"status":"COMPLETE","control":control,"A":a,"B":b,"gates":gates,"full_l333":full,
             "stream_sha256":(HERE/"TRAIN_STREAM_SHA256.txt").read_text().strip()}
    write_json(HERE/"FINAL_SUMMARY.json",summary)
    lines=["# Knowledge-Prior Fast Screen V1","","Matched flexict continuation from Direct Concat 12k EMA; no A+B experiment.","",
        "| Arm | Step | PSNR | Δ vs Control | SSIM | RMSE HU | MAE HU |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:lines.append(f"| {r['arm']} | {r['step']} | {r['psnr']:.6f} | {r['delta_vs_control']:+.6f} | {r['ssim']:.6f} | {r['rmse_hu']:.4f} | {r['mae_hu']:.4f} |")
    lines+=["","## Gates","",f"A: {gates['A']}","",f"B: {gates['B']}","","## Full L333","",json.dumps(full,indent=2),""]
    lines+=["## Questions answered from observed data","",
        f"Control continuation gain: {control['history'][-1]['psnr']-control['history'][0]['psnr']:+.6f} dB.",
        f"A 6k increment: {a['history'][-1].get('delta_vs_control',float('nan')):+.6f} dB; gate {gates['A']['reason']}.",
        f"B 6k increment: {b['history'][-1].get('delta_vs_control',float('nan')):+.6f} dB; gate {gates['B']['reason']}.",
        "B prior-loss trend is in B_NDCT_DETAIL_PRIOR/train_history.csv.",
        "A and B stability is limited to the matched 0/1k/2k/4k/6k nodes and any promoted Full-L333 results.",
        "Neither prior is a paper result at this stage; no L506 access.",
        "A+B combination is a candidate next experiment only if both have positive matched gains. It was not run.",""]
    (HERE/"REPORT.md").write_text("\n".join(lines))
def main():
    cfg,base=context_config()
    HERE.mkdir(parents=True,exist_ok=True)
    status={"phase":"PREFLIGHT","status":"RUNNING","started_at":time.time()}
    write_json(HERE/"QUEUE_STATUS.json",status)
    try:
        parity=preflight(cfg,base)
        prov=provenance();write_json(HERE/"ENVIRONMENT_AUDIT.json",prov)
        manifest_cfg,manifest_base,manifest,h=prepare_manifest()
        experiment={"name":"Knowledge-Prior Fast Screen V1","branch":"experiment/rddm-knowledge-prior-serial-v1",
            "starting_checkpoint_sha256":DIRECT_SHA,"seed":20260829,"train_patients":["L067","L143"],
            "validation":"L333-fast64","teacher_donor_patients":list(DONORS),"max_adaptation_steps":6000,
            "nodes":list(NODES),"batch_size":2,"grad_accumulation":1,"optimizer":"Adam",
            "lr":2e-4,"betas":[.9,.99],"weight_decay":1e-9,"ema":.995,"amp":False,"clip":1.,
            "diffusion":{"timesteps":1000,"sampling_timesteps":2,"sum_scale":.01,"objective":"pred_res"},
            "meddino":{"sha256":MED_SHA,"official_commit":MED_COMMIT,"layers":[2,5,8,11],"input_size":512,
                "bca_stages":[128,64],"windows":[8,16],"heads":4,"dim_head":32,"alpha_init":.1,"output_projection_zero_init":True},
            "teacher":{"donors":list(DONORS),"NDCT_only":True,"steps":8000,"batch_size":8,"crop":256,
                "loss":"L1 image + .5 weighted Gaussian detail bands"},"train_stream_sha256":h}
        write_json(HERE/"EXPERIMENT_CONFIG.json",experiment)
        status["phase"]="CONTROL_CONTINUE";write_json(HERE/"QUEUE_STATUS.json",status)
        control=train_arm(ARMS[0],cfg,base,manifest,h)
        assert abs(control["history"][0]["psnr"]-parity["psnr"])<=1e-5
        status["phase"]="A_MEDDINO_BCA";write_json(HERE/"QUEUE_STATUS.json",status)
        a=train_arm(ARMS[1],cfg,base,manifest,h,control["history"])
        status["phase"]="B_TEACHER_PRETRAIN";write_json(HERE/"QUEUE_STATUS.json",status)
        teacher_audit=teacher_train(base)
        status["phase"]="B_LAMBDA_CALIBRATION";write_json(HERE/"QUEUE_STATUS.json",status)
        lam=calibrate(base,manifest)
        status["phase"]="B_NDCT_DETAIL_PRIOR";write_json(HERE/"QUEUE_STATUS.json",status)
        teacher=load_teacher("cuda:0")
        b=train_arm(ARMS[2],cfg,base,manifest,h,control["history"],teacher,lam)
        del teacher;torch.cuda.empty_cache()
        gates={"A":gate(a,control,ARMS[1]),"B":gate(b,control,ARMS[2])}
        full={}
        if gates["A"]["promote"] or gates["B"]["promote"]:
            status["phase"]="FULL_L333";write_json(HERE/"QUEUE_STATUS.json",status)
            full["control"]=full_eval(ARMS[0],cfg,base)
            ctrlrows=read_json(HERE/"full_l333"/f"{ARMS[0]}_per_slice.json")
            for key,arm in (("A",ARMS[1]),("B",ARMS[2])):
                if gates[key]["promote"]:
                    full[key]=full_eval(arm,cfg,base)
                    rows=read_json(HERE/"full_l333"/f"{arm}_per_slice.json")
                    full[key]["paired"]=paired(rows,ctrlrows)
                    write_json(HERE/"full_l333"/f"{arm}_paired_summary.json",full[key]["paired"])
        report(control,a,b,gates,full)
        status.update(phase="COMPLETE",status="COMPLETE",finished_at=time.time())
        write_json(HERE/"QUEUE_STATUS.json",status)
    except BaseException as exc:
        status.update(status="FAILED",error=repr(exc),traceback=traceback.format_exc(),finished_at=time.time())
        write_json(HERE/"QUEUE_STATUS.json",status)
        raise
if __name__=="__main__":main()
