"""Matched RDDM adaptation engine; no optimizer state reused from Direct 12k."""
import copy,csv,hashlib,json,math,time
from pathlib import Path
import numpy as np
import torch
from context_2p5d.data import ContextPairs,ContextBatchStream,ContextDataGuard
from context_2p5d.evaluate import validate
from context_2p5d.protocol import config as context_config
from context_2p5d.model import state_hash
from mayo2016_smallset_v1.runtime import ROOT,seed_everything,write_json,atomic_save
from .core import load_arm,matched_loss,DIRECT_SHA
from .hashing import exact_state_hash
from .stream import make_manifest,digest,assert_batch,timestep_noise
from .teacher import Teacher

HERE=ROOT/"experiments/knowledge_prior_serial_v1"
NODES=(0,1000,2000,4000,6000)
ARMS=("CONTROL_CONTINUE","A_MEDDINO_BCA","B_NDCT_DETAIL_PRIOR")
FIELDS=("psnr","ssim","rmse_hu","mae_hu")
def csv_write(path,rows):
    if not rows:return
    with open(path,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def read_json(path):return json.loads(Path(path).read_text())
def prepare_manifest():
    cfg,base=context_config()
    data=ContextPairs(base,"train")
    assert len(data)==1145
    manifest=make_manifest(data)
    h=digest(manifest)
    write_json(HERE/"TRAIN_STREAM_MANIFEST.json",manifest)
    (HERE/"TRAIN_STREAM_SHA256.txt").write_text(h+"\n")
    return cfg,base,manifest,h
class TrainableEMA:
    def __init__(self,model,decay=.995):
        self.model=copy.deepcopy(model).eval().requires_grad_(False)
        self.names=tuple(n for n,p in model.named_parameters() if p.requires_grad)
        self.decay=decay;self.updates=0
    @torch.no_grad()
    def update(self,model):
        src=dict(model.named_parameters());dst=dict(self.model.named_parameters())
        for n in self.names:dst[n].lerp_(src[n],1-self.decay)
        for a,b in zip(self.model.buffers(),model.buffers()):a.copy_(b)
        self.updates+=1
def input_batch(stream,want,device):
    prev,center,next_,target,meta=stream.next(2)
    assert_batch(meta,want)
    assert prev.shape==center.shape==next_.shape==target.shape==(2,1,512,512)
    for x in (prev,center,next_,target):
        assert x.dtype==torch.float32 and bool(torch.isfinite(x).all())
        assert 0<=float(x.min())<=float(x.max())<=1
    return (prev.to(device),center.to(device),next_.to(device),target.to(device),meta)
def init_arm(arm,device):
    model,main_hash=load_arm(arm,device)
    ema=TrainableEMA(model)
    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.Adam(params,lr=2e-4,betas=(.9,.99),weight_decay=1e-9)
    audit={"arm":arm,"starting_direct_ema_sha256":DIRECT_SHA,"starting_main_hash":main_hash,
           "starting_model_state_hash":exact_state_hash(model.model.unet0.state_dict()),
           "starting_ema_equal_model":all(torch.equal(a,b) for a,b in zip(model.state_dict().values(),ema.model.state_dict().values())),
           "fresh_adam_state_count":len(opt.state),"trainable_parameters":sum(p.numel() for p in params),
           "total_parameters":sum(p.numel() for p in model.parameters())}
    if arm==ARMS[1]:
        u=model.model.unet0
        audit.update(meddino_frozen=all(not p.requires_grad for p in u.structure.med.parameters()),
            meddino_eval=not u.structure.med.training,
            bca128_out_zero=bool(torch.count_nonzero(u.bca128.out.weight)==0),
            bca64_out_zero=bool(torch.count_nonzero(u.bca64.out.weight)==0),
            alpha128=float(u.bca128.alpha),alpha64=float(u.bca64.alpha))
        assert audit["meddino_frozen"] and audit["bca128_out_zero"] and audit["bca64_out_zero"]
    return model,ema,opt,audit
def aggregate_rows(path):
    rows=read_json(path)
    return {k:float(np.mean([r[k] for r in rows])) for k in FIELDS},rows
def paired(focus,control,seed=20260929):
    assert len(focus)==len(control) and all(a["slice_index"]==b["slice_index"] for a,b in zip(focus,control))
    diff=np.array([a["psnr"]-b["psnr"] for a,b in zip(focus,control)],dtype=float)
    rng=np.random.default_rng(seed)
    boot=np.mean(diff[rng.integers(len(diff),size=(10000,len(diff)))],axis=1)
    return {"mean_delta_psnr":float(diff.mean()),"positive_slice_fraction":float(np.mean(diff>0)),
        "bootstrap_95_ci":[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],"slices":len(diff)}
def train_arm(arm,cfg,base,manifest,manifest_sha,control=None,teacher_ck=None,lambda_prior=None):
    assert arm in ARMS
    device=torch.device("cuda:0")
    seed_everything(20260829);torch.set_num_threads(2)
    model,ema,opt,init=init_arm(arm,device)
    output=HERE/arm;output.mkdir(exist_ok=False)
    write_json(output/"INITIALIZATION_AUDIT.json",init)
    train=ContextPairs(base,"train");val=ContextPairs(base,"validation","FAST")
    stream=ContextBatchStream(train,20260829)
    history=[];logs=[];nonfinite=0;grad_min=None;grad_max=None;seconds=0
    torch.cuda.reset_peak_memory_stats(device)
    max_step=6000
    with ContextDataGuard(base,"FAST") as guard:
        train.bind_guard(guard);val.bind_guard(guard)
        for step in range(0,max_step+1):
            if step in NODES:
                row_path=output/f"fast64_step_{step}.json"
                result=validate(ema.model,val,cfg,device,row_path)
                row={"step":step,**{k:result[k] for k in FIELDS},"validation_loss":result["validation_loss"],"slices":64,
                     "weights":"ema","train_stream_sha256":digest(manifest[:step]),"seconds_per_step":seconds/max(1,step),
                     "peak_vram_bytes":torch.cuda.max_memory_allocated(device)}
                if control is not None:
                    ctrl=next(r for r in control if r["step"]==step)
                    row["delta_vs_control"]=row["psnr"]-ctrl["psnr"]
                    own=read_json(row_path);ref=read_json(HERE/"CONTROL_CONTINUE"/f"fast64_step_{step}.json")
                    row["positive_slice_fraction"]=paired(own,ref)["positive_slice_fraction"]
                history.append(row)
                csv_write(output/"validation_history.csv",history)
                write_json(output/"progress.json",{"arm":arm,"step":step,"metrics":row,"nonfinite_count":nonfinite})
                if step:
                    atomic_save(output/f"step_{step}.pt",{"model_state_dict":model.state_dict(),"ema_state_dict":ema.model.state_dict(),
                        "optimizer_state_dict":opt.state_dict(),"step":step,"stream_state":stream.state_dict(),
                        "manifest_sha256":manifest_sha,"ema_updates":ema.updates})
                if control is not None and step==4000:
                    d2=next(r for r in history if r["step"]==2000)["delta_vs_control"]
                    d4=row["delta_vs_control"]
                    if d2<=0 and d4<=0: max_step=4000
                    elif d4<.01:max_step=4000
                if step==max_step:break
            started=time.perf_counter()
            want=manifest[step]
            prev,center,next_,target,meta=input_batch(stream,want,device)
            t,noise=timestep_noise(want,target.shape,device)
            model.train();opt.zero_grad(set_to_none=True)
            loss_res,pred,x_input,x_gt=matched_loss(model,prev,center,next_,target,t,noise)
            if arm==ARMS[2]:
                assert teacher_ck is not None and lambda_prior is not None
                prior,terms=teacher_ck.prior(x_input-pred,x_gt)
                loss=loss_res+lambda_prior*prior
            else:prior=None;terms=None;loss=loss_res
            if not bool(torch.isfinite(loss)):nonfinite+=1;raise FloatingPointError("nonfinite loss")
            loss.backward()
            params=[p for p in model.parameters() if p.requires_grad]
            norm=float(torch.nn.utils.clip_grad_norm_(params,1.0))
            if not math.isfinite(norm):nonfinite+=1;raise FloatingPointError("nonfinite gradient")
            grad_min=norm if grad_min is None else min(grad_min,norm)
            grad_max=norm if grad_max is None else max(grad_max,norm)
            opt.step();ema.update(model)
            torch.cuda.synchronize(device);elapsed=time.perf_counter()-started;seconds+=elapsed
            if step==0 or (step+1)%100==0:
                log={"step":step+1,"loss_total":float(loss.detach()),"loss_res":float(loss_res.detach()),
                    "loss_prior":None if prior is None else float(prior.detach()),"gradient_norm_pre_clip":norm,
                    "seconds_per_step":elapsed,"peak_vram_bytes":torch.cuda.max_memory_allocated(device),"learning_rate":2e-4}
                logs.append(log);csv_write(output/"train_history.csv",logs)
        write_json(output/"DATA_ACCESS_AUDIT.json",guard.report())
    assert nonfinite==0 and ema.updates==max_step
    final={"arm":arm,"status":"COMPLETE","steps":max_step,"history":history,"gradient_norm_min":grad_min,
        "gradient_norm_max":grad_max,"nonfinite_count":nonfinite,"train_seconds":seconds,
        "seconds_per_step":seconds/max_step,"peak_vram_bytes":torch.cuda.max_memory_allocated(device),
        "manifest_sha256":manifest_sha,"access":guard.report()}
    write_json(output/"final_summary.json",final)
    del model,ema,opt
    torch.cuda.empty_cache()
    return final
