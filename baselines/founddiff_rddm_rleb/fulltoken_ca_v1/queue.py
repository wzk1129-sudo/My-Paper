"""Serial matched continuation from the frozen Direct 12k EMA checkpoint."""
import copy
import csv
import json
import math
import os
import random
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from context_2p5d.data import ContextPairs, ContextBatchStream, ContextDataGuard
from context_2p5d.evaluate import validate
from context_2p5d.protocol import config as context_config
from knowledge_prior_v1.core import load_direct, matched_loss, DIRECT_CK, DIRECT_SHA, MED_CK, MED_SHA, MED_COMMIT, sha
from knowledge_prior_v1.stream import make_manifest, digest, assert_batch, timestep_noise
from mayo2016_smallset_v1.runtime import ROOT, seed_everything, write_json, atomic_save
from fulltoken_ca_v1.model import load_candidate

OUT = ROOT / "experiments/meddino_fulltoken_ca_v1"
ARMS = ("DIRECT_CONTINUE", "MEDDINO_FULLTOKEN_CA_128_64")
NODES = (0,1000,2000,4000,6000,8000)
FIELDS = ("psnr","ssim","rmse_hu","mae_hu")
MANIFEST_6K_SHA = "ca176a324db9ce0547f0a5ad863108028e2b49dde2a554fc952cd059c3ca4768"
FORMAL_SOURCE_SHA = "e8ed93223fbdf4a0e2521f73c7a4f213561f6a191a1a7b2a5f95934b85ebf574"
QUERY_CHUNK = 0

def save_csv(path,rows):
    if not rows:return
    with open(path,"w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)

def read_json(path):
    return json.loads(Path(path).read_text())

def status(**kwargs):
    write_json(OUT/"QUEUE_STATUS.json",{"updated_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),**kwargs})

class EMA:
    def __init__(self,model):
        self.model=copy.deepcopy(model).eval().requires_grad_(False)
        self.names=tuple(n for n,p in model.named_parameters() if p.requires_grad)
        self.updates=0
    @torch.no_grad()
    def update(self,model):
        src=dict(model.named_parameters());dst=dict(self.model.named_parameters())
        for n in self.names:dst[n].lerp_(src[n],.005)
        for a,b in zip(self.model.buffers(),model.buffers()):a.copy_(b)
        self.updates+=1

def init_model(arm,device):
    seed_everything(20260829)
    model=load_direct(device) if arm==ARMS[0] else load_candidate(device,QUERY_CHUNK)
    ema=EMA(model)
    optimizer=torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
        lr=2e-4,betas=(.9,.99),weight_decay=1e-9)
    seed_everything(20260829)
    assert not optimizer.state and ema.updates==0
    return model,ema,optimizer

def training_state(model,ema,optimizer,stream,step,manifest_sha,seconds,last_grad,nonfinite):
    return {
        "model_state_dict":model.state_dict(),"ema_state_dict":ema.model.state_dict(),
        "optimizer_state_dict":optimizer.state_dict(),"step":step,
        "stream_state":stream.state_dict(),"manifest_sha256":manifest_sha,
        "ema_updates":ema.updates,"training_seconds":seconds,
        "last_gradient_norm":last_grad,"nonfinite_count":nonfinite,
        "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all(),
        "numpy_rng":np.random.get_state(),"python_rng":random.getstate(),
    }

def compare_rows(candidate_path,control_path):
    own=read_json(candidate_path);ref=read_json(control_path)
    assert len(own)==len(ref)==64
    assert [(r["patient_id"],r["slice_index"]) for r in own]==[(r["patient_id"],r["slice_index"]) for r in ref]
    positive=sum(a["psnr"]>b["psnr"] for a,b in zip(own,ref))
    return positive,positive/64.0

def train_arm(arm,cfg,base,manifest,manifest_sha,end_step,start_step=0):
    assert arm in ARMS and end_step in (6000,8000) and start_step in (0,2000,6000)
    device=torch.device("cuda:0")
    output=OUT/arm
    if start_step==0:output.mkdir(exist_ok=False)
    else:assert output.is_dir()
    train=ContextPairs(base,"train")
    val=ContextPairs(base,"validation","FAST")
    assert len(train)==1145 and len(val)==64
    stream=ContextBatchStream(train,20260829)
    model,ema,opt=init_model(arm,device)
    params=[p for p in model.parameters() if p.requires_grad]
    u=model.model.unet0
    history=[] if start_step==0 else list(csv.DictReader(open(output/"validation_history.csv",newline="")))
    seconds=0.0;last_grad=None;last_ca_grad=None;nonfinite=0
    if start_step:
        ck=torch.load(output/f"step_{start_step}.pt",map_location="cpu",weights_only=False)
        assert ck["step"]==start_step and ck["manifest_sha256"]==manifest_sha
        model.load_state_dict(ck["model_state_dict"],strict=True)
        ema.model.load_state_dict(ck["ema_state_dict"],strict=True)
        opt.load_state_dict(ck["optimizer_state_dict"])
        stream.load_state_dict(ck["stream_state"])
        assert ck["ema_updates"]==start_step
        ema.updates=start_step;seconds=ck["training_seconds"]
        last_grad=ck["last_gradient_norm"];nonfinite=ck["nonfinite_count"]
        torch.set_rng_state(ck["torch_rng"]);torch.cuda.set_rng_state_all(ck["cuda_rng"])
        np.random.set_state(ck["numpy_rng"]);random.setstate(ck["python_rng"])
        del ck
    else:
        write_json(output/"INITIALIZATION_AUDIT.json",{
            "arm":arm,"direct_12k_checkpoint_sha256":DIRECT_SHA,
            "main_parameter_count":sum(p.numel() for p in model.model.unet0.parameters() if p.requires_grad),
            "all_trainable_parameter_count":sum(p.numel() for p in params),
            "total_parameter_count":sum(p.numel() for p in model.parameters()),
            "fresh_optimizer":not opt.state,"fresh_ema":ema.updates==0,
            "meddino_frozen_eval":arm==ARMS[0] or (all(not p.requires_grad for p in u.med.parameters()) and not u.med.training),
        })
    torch.cuda.reset_peak_memory_stats(device)
    guard=ContextDataGuard(base,"FAST")
    try:
        with guard:
            train.bind_guard(guard);val.bind_guard(guard)
            for step in range(start_step,end_step+1):
                if step in NODES and (step==0 or step>start_step):
                    row_path=output/f"fast64_step_{step}.json"
                    if arm==ARMS[1]:
                        ema.model.model.unet0.ca128.reset_activity()
                        ema.model.model.unet0.ca64.reset_activity()
                    result=validate(ema.model,val,cfg,device,row_path)
                    row={"step":step,**{k:result[k] for k in FIELDS},
                        "validation_loss":result["validation_loss"],"slices":64,"weights":"ema",
                        "train_stream_sha256":digest(manifest[:step]),
                        "validation_seconds":result["validation_seconds"],
                        "seconds_per_train_step":seconds/max(1,step),
                        "peak_allocated_bytes":torch.cuda.max_memory_allocated(device),
                        "peak_reserved_bytes":torch.cuda.max_memory_reserved(device),
                        "last_gradient_norm":last_grad,"last_ca_gradient_norm":last_ca_grad,
                        "train_loss":None if step==0 else last_loss}
                    if arm==ARMS[1]:
                        eu=ema.model.model.unet0
                        eu.ca128.capture=False;eu.ca64.capture=False
                        row.update(gate_128=float(eu.ca128.gate.detach()),gate_64=float(eu.ca64.gate.detach()))
                        for name,mod in (("ca128",eu.ca128),("ca64",eu.ca64)):
                            for key,value in mod.activity_summary().items():
                                row[name+"_"+key]=value
                        ctrl=next(r for r in csv.DictReader(open(OUT/ARMS[0]/"validation_history.csv",newline=""))
                                  if int(r["step"])==step)
                        for key in FIELDS:row["delta_"+key]=row[key]-float(ctrl[key])
                        count,frac=compare_rows(row_path,OUT/ARMS[0]/f"fast64_step_{step}.json")
                        row["positive_slices"]=count;row["positive_fraction"]=frac
                    history.append(row)
                    save_csv(output/"validation_history.csv",history)
                    write_json(output/"progress.json",{"arm":arm,"step":step,"metrics":row,"nonfinite_count":nonfinite})
                    status(state="RUNNING",arm=arm,step=step,target=end_step)
                    if step:
                        atomic_save(output/f"step_{step}.pt",training_state(
                            model,ema,opt,stream,step,manifest_sha,seconds,last_grad,nonfinite))
                    if step==end_step:break
                if step==end_step:break
                want=manifest[step]
                prev,center,next_,target,meta=stream.next(2)
                assert_batch(meta,want)
                assert prev.shape==center.shape==next_.shape==target.shape==(2,1,512,512)
                prev,center,next_,target=(x.to(device) for x in (prev,center,next_,target))
                t,noise=timestep_noise(want,target.shape,device)
                started=time.perf_counter()
                model.train();opt.zero_grad(set_to_none=True)
                loss,pred,_,_=matched_loss(model,prev,center,next_,target,t,noise)
                if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(pred).all()):
                    nonfinite+=1;raise FloatingPointError("nonfinite loss or output")
                loss.backward()
                if arm==ARMS[1]:
                    ca_params=list(u.ca128.parameters())+list(u.ca64.parameters())
                    last_ca_grad=math.sqrt(sum(float(p.grad.detach().float().square().sum()) for p in ca_params if p.grad is not None))
                    if not math.isfinite(last_ca_grad):raise FloatingPointError("nonfinite CA gradient")
                    assert all(p.grad is None for p in u.med.parameters())
                last_grad=float(torch.nn.utils.clip_grad_norm_(params,1.0))
                if not math.isfinite(last_grad):
                    nonfinite+=1;raise FloatingPointError("nonfinite gradient")
                opt.step();ema.update(model)
                torch.cuda.synchronize(device)
                seconds+=time.perf_counter()-started
                last_loss=float(loss.detach())
                if (step+1)%100==0:
                    with open(output/"train.jsonl","a") as f:
                        f.write(json.dumps({"step":step+1,"loss":last_loss,"preclip_gradient_norm":last_grad,
                            "ca_gradient_norm":last_ca_grad,"seconds_per_step":seconds/(step+1)})+"\n")
                    write_json(output/"progress.json",{"arm":arm,"step":step+1,"train_loss":last_loss,
                        "preclip_gradient_norm":last_grad,"nonfinite_count":nonfinite})
                    status(state="RUNNING",arm=arm,step=step+1,target=end_step)
    finally:
        access=guard.report()
        write_json(output/("DATA_ACCESS_AUDIT.json" if not start_step else "DATA_ACCESS_AUDIT_RESUME.json"),access)
    assert nonfinite==0 and ema.updates==end_step
    assert all(access[key]==0 for key in ("L506_opens","external_memory_opens","neighbor_target_opens","blocked_attempts"))
    summary={"arm":arm,"step":end_step,"ema_updates":ema.updates,"nonfinite_count":nonfinite,
             "seconds_training":seconds,"seconds_per_step":seconds/end_step,
             "peak_allocated_bytes":torch.cuda.max_memory_allocated(device),
             "peak_reserved_bytes":torch.cuda.max_memory_reserved(device),
             "final_metrics":history[-1],"data_access":access}
    write_json(output/"final_summary.json",summary)
    del model,ema,opt
    torch.cuda.empty_cache()
    return history

def full_validate(arm,cfg,base,step):
    device=torch.device("cuda:0")
    model=load_direct(device) if arm==ARMS[0] else load_candidate(device,QUERY_CHUNK)
    ck=torch.load(OUT/arm/f"step_{step}.pt",map_location="cpu",weights_only=False)
    assert ck["step"]==step and ck["ema_updates"]==step
    model.load_state_dict(ck["ema_state_dict"],strict=True)
    del ck
    model.eval()
    val=ContextPairs(base,"validation","FULL")
    assert len(val)==610
    guard=ContextDataGuard(base,"FULL",allow_training=False)
    with guard:
        val.bind_guard(guard)
        result=validate(model,val,cfg,device,OUT/f"FULL_L333_{arm}_ROWS.json")
    access=guard.report()
    assert all(access[k]==0 for k in ("L506_opens","external_memory_opens","neighbor_target_opens","blocked_attempts"))
    assert result["slices"]==610 and result["weights"]=="ema"
    write_json(OUT/f"FULL_L333_{arm}_SUMMARY.json",{**result,"data_access":access,"step":step})
    del model
    torch.cuda.empty_cache()
    return result

def final_report(control,candidate,decision,full=None):
    ctrl={int(r["step"]):r for r in control}
    cand={int(r["step"]):r for r in candidate}
    common=sorted(set(ctrl)&set(cand))
    rows=[]
    for step in common:
        row={"step":step}
        for field in FIELDS:
            row["control_"+field]=float(ctrl[step][field])
            row["candidate_"+field]=float(cand[step][field])
            row["delta_"+field]=row["candidate_"+field]-row["control_"+field]
        row["positive_slices"]=int(cand[step]["positive_slices"])
        row["gate_128"]=float(cand[step]["gate_128"])
        row["gate_64"]=float(cand[step]["gate_64"])
        rows.append(row)
    save_csv(OUT/"MATCHED_COMPARISON.csv",rows)
    summary={"status":"COMPLETE","decision":decision,"matched_nodes":rows,
             "full_l333":full,"direct_checkpoint_sha256":DIRECT_SHA,
             "meddino_checkpoint_sha256":MED_SHA,"meddino_source_commit":MED_COMMIT,
             "formal_source_sha256":FORMAL_SOURCE_SHA,
             "baseline_6k_train_manifest_sha256":MANIFEST_6K_SHA,
             "weights":"ema","validation":"L333-fast64",
             "l506_access":0,"external_memory_access":0}
    write_json(OUT/"FINAL_SUMMARY.json",summary)
    lines=["# MedDINO full-token global cross-attention matched continuation","",
        "Direct 12k EMA is the common initialization. Control and candidate use fresh matched Adam/EMA.",
        "The only candidate feature is frozen final-layer MedDINO [1024,768] dense patch tokens.",
        "Independent global CA at encoder post-downsample 128 and 64; zero-initialized scalar gates.",
        "No Sobel, Laplacian, Wavelet, SALF, HAF, PDC, ControlNet, structure loss or external memory.",
        "","| Extra step | Control PSNR | Candidate PSNR | Delta PSNR | Positive /64 |",
        "|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['step']} | {row['control_psnr']:.6f} | {row['candidate_psnr']:.6f} | {row['delta_psnr']:+.6f} | {row['positive_slices']} |")
    lines.extend(["",f"Decision: {decision}",""])
    if full:
        lines.append(f"Full L333 610-slice PSNR delta: {full['delta_psnr']:+.6f} dB.")
    lines.append("L506, external memory and neighbor NDCT were not opened by the guarded data pipeline.")
    (OUT/"REPORT.md").write_text("\n".join(lines)+"\n")
    return summary

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    readiness=read_json(OUT/"READINESS_AUDIT.json")
    assert readiness["status"]=="PASS" and readiness["query_chunk"]==QUERY_CHUNK
    assert readiness["step0_max_abs_output_diff"]<=1e-6
    assert readiness["data_access"]["L506_opens"]==0
    assert sha(DIRECT_CK)==DIRECT_SHA and sha(MED_CK)==MED_SHA
    assert sha(ROOT/"src/rddm_rleb.py")==FORMAL_SOURCE_SHA
    seed_everything(20260829);torch.set_num_threads(2)
    cfg,base=context_config()
    train=ContextPairs(base,"train")
    manifest=make_manifest(train,steps=8000)
    manifest_sha=digest(manifest)
    assert digest(manifest[:6000])==MANIFEST_6K_SHA
    write_json(OUT/"TRAIN_STREAM_MANIFEST.json",manifest)
    config={"name":"MedDINO_FullToken_CA_128_64_V1",
        "branch":"experiment/rddm-meddino-fulltoken-ca-v1",
        "parent_commit":"51c608b9d2148102611c3e5717fa0f79bdf1b2ae",
        "environment":"flexict","seed":20260829,
        "direct_checkpoint_sha256":DIRECT_SHA,"meddino_checkpoint_sha256":MED_SHA,
        "meddino_source_commit":MED_COMMIT,
        "train_patients":["L067","L143"],"validation":"L333-fast64, fixed 64 indices",
        "batch_size":2,"grad_accumulation":1,"optimizer":"Adam",
        "lr":2e-4,"betas":[.9,.99],"weight_decay":1e-9,
        "ema":.995,"amp":False,"grad_clip_max_norm":1.0,
        "pred_objective":"pred_res","loss":"L2",
        "timesteps":1000,"sampling_timesteps":2,"sum_scale":.01,
        "additional_train_steps_initial":6000,"validation_nodes_initial":[0,1000,2000,4000,6000],
        "conditional_extension_step":8000,"full_l333_only_after_8k_gate":True,
        "attention":{"stages":[128,64],"meddino_layer":11,"patch_tokens":1024,
            "meddino_width":768,"query_channels":[128,256],"hidden_dim":256,
            "heads":4,"dim_head":64,"global_keys_per_query":1024,
            "query_chunk":QUERY_CHUNK,"gate_initial":0.0},
        "frozen_protocol":base,"train_manifest_sha256_6k":MANIFEST_6K_SHA,
        "train_manifest_sha256_8k":manifest_sha}
    write_json(OUT/"CONFIG.json",config)
    write_json(OUT/"PROVENANCE.json",{
        "direct_checkpoint_sha256":DIRECT_SHA,"meddino_checkpoint_sha256":MED_SHA,
        "meddino_source_commit":MED_COMMIT,"formal_source_sha256":FORMAL_SOURCE_SHA,
        "model_source_sha256":sha(ROOT/"fulltoken_ca_v1/model.py"),
        "queue_source_sha256":sha(ROOT/"fulltoken_ca_v1/queue.py"),
        "split_sha256":sha(ROOT/"splits/L333_fast64.txt"),
        "evaluator_sha256":sha(ROOT/"mayo2016_smallset_v1/frozen_protocol.py"),
        "pytorch":torch.__version__,"cuda":torch.version.cuda,
        "gpu":torch.cuda.get_device_name(0),
        "train_manifest_sha256_6k":MANIFEST_6K_SHA,
        "train_manifest_sha256_8k":manifest_sha})
    status(state="RUNNING",arm=ARMS[0],step=0,target=6000)
    control=train_arm(ARMS[0],cfg,base,manifest,manifest_sha,6000)
    candidate=train_arm(ARMS[1],cfg,base,manifest,manifest_sha,6000)
    bystep={int(r["step"]):r for r in candidate}
    d4=float(bystep[4000]["delta_psnr"])
    d6=float(bystep[6000]["delta_psnr"])
    full=None
    if d6<=0:
        decision="STOP_DELTA_6K_NONPOSITIVE"
    elif d4>0 and d6>=.02 and d6>=d4-.005:
        decision="EXTEND_TO_8K"
        status(state="RUNNING",arm=ARMS[0],step=6000,target=8000)
        control=train_arm(ARMS[0],cfg,base,manifest,manifest_sha,8000,start_step=6000)
        candidate=train_arm(ARMS[1],cfg,base,manifest,manifest_sha,8000,start_step=6000)
        d8=float(candidate[-1]["delta_psnr"])
        consistency=float(candidate[-1]["positive_fraction"])
        if d8>=.03 and consistency>=.6:
            decision="FULL_L333_PROMOTED"
            c=full_validate(ARMS[0],cfg,base,8000)
            m=full_validate(ARMS[1],cfg,base,8000)
            full={"step":8000,**{"control_"+k:c[k] for k in FIELDS},
                  **{"candidate_"+k:m[k] for k in FIELDS},
                  **{"delta_"+k:m[k]-c[k] for k in FIELDS},
                  "slices":610,"weights":"ema"}
            write_json(OUT/"FULL_L333_COMPARISON.json",full)
        else:decision="STOP_8K_PROMOTION_NOT_MET"
    else:
        decision="STOP_6K_EXTENSION_NOT_MET"
    for arm in ARMS:
        checks=[read_json(OUT/arm/"DATA_ACCESS_AUDIT.json")]
        resumed=OUT/arm/"DATA_ACCESS_AUDIT_RESUME.json"
        if resumed.exists():checks.append(read_json(resumed))
        assert all(all(x[k]==0 for k in ("L506_opens","external_memory_opens","neighbor_target_opens","blocked_attempts")) for x in checks)
        assert read_json(OUT/arm/"final_summary.json")["nonfinite_count"]==0
    summary=final_report(control,candidate,decision,full)
    write_json(OUT/"FINAL_AUDIT.json",{
        "status":"PASS","control_last_step":int(control[-1]["step"]),
        "candidate_last_step":int(candidate[-1]["step"]),
        "matched_stream_sha256_6k":MANIFEST_6K_SHA,
        "same_direct_initialization":True,"readiness_pass":True,
        "l506_opens":0,"external_memory_opens":0,"neighbor_target_opens":0,
        "formal_source_sha256":FORMAL_SOURCE_SHA,"decision":decision})
    status(state="COMPLETE",decision=decision,
           control_step=int(control[-1]["step"]),candidate_step=int(candidate[-1]["step"]))
    print(json.dumps({"status":"COMPLETE","decision":decision,
        "control_psnr":float(control[-1]["psnr"]),"candidate_psnr":float(candidate[-1]["psnr"])}))

if __name__=="__main__":
    try:main()
    except Exception as exc:
        OUT.mkdir(parents=True,exist_ok=True)
        status(state="FAILED",error=repr(exc),traceback=traceback.format_exc())
        traceback.print_exc()
        raise
