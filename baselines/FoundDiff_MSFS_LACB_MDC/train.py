"""Independent seed10 FP32 8k training; original RDDM and frozen metrics."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import copy,json,csv,hashlib,time,datetime,traceback,random
from pathlib import Path
import torch,numpy as np
from model import build,HERE,without_med
from src.DADiff import set_seed
from mayo_data import Pairs,Stream,VALID,CFG,access_report
from frozen_metrics import slice_metrics
from med_loader import sha,MED_SHA
def setup():
    torch.set_num_threads(CFG["training"]["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;set_seed(CFG["seed"])
def write_json(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n")
def rng():return {"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state(),"cuda":torch.cuda.get_rng_state_all()}
def save(path,obj):
    with path.open("xb") as f:torch.save(obj,f)
def csv_write(path,rows):
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
class EMA:
    def __init__(self,m):
        med=m.model.unet0.med
        self.model=copy.deepcopy(m,{id(med):med}).eval().requires_grad_(False);self.updates=0
    @torch.no_grad()
    def update(self,m):
        dst=dict(self.model.named_parameters());decay=CFG["ema"]["decay"]
        for n,p in m.named_parameters():
            if p.requires_grad:dst[n].mul_(decay).add_(p,alpha=1-decay)
        dst=dict(self.model.named_buffers())
        for n,b in m.named_buffers():
            if not n.startswith("model.unet0.med."):dst[n].copy_(b)
        self.updates+=1
def sources():
    ps=list(HERE.glob("*.py"))+list((HERE/"src").glob("*.py"))+[HERE/"config.json",HERE/"L333_fast64.txt",HERE/"reference/ACL_layers.py"]
    return {str(p.relative_to(HERE)):sha(p) for p in sorted(ps)}
@torch.no_grad()
def validate(m,data,step):
    m.eval();u=m.model.unet0;rows=[];soft=copy.deepcopy(CFG);soft["evaluation"]["hu_window"]=[-160,240]
    with torch.random.fork_rng(devices=[0]):
        for pos in range(len(data)):
            x,y,meta=data.load(pos)
            context=torch.from_numpy(x[None]).cuda();target=torch.from_numpy(y[None]).cuda()
            seed=CFG["evaluation"]["validation_seed"]+meta["slice_index"]
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            u.prepare(context);pred=m.sample([context[:,1:2],context],last=True)[-1];u.clear()
            assert pred.shape==(1,1,512,512) and torch.isfinite(pred).all()
            main=slice_metrics(pred,target,CFG);aux=slice_metrics(pred,target,soft)
            rows.append({**meta,"seed":seed,**{k:main[k] for k in ("psnr","ssim","rmse_hu","mae_hu")},
                         **{"soft_"+k:aux[k] for k in ("psnr","ssim","rmse_hu")}})
    assert [r["slice_index"] for r in rows]==list(VALID)
    path=HERE/f"validation_step_{step}.csv";assert not path.exists();csv_write(path,rows)
    keys=("psnr","ssim","rmse_hu","mae_hu","soft_psnr","soft_ssim","soft_rmse_hu")
    return {"step":step,**{k:float(np.mean([r[k] for r in rows])) for k in keys},"slices":64}
def main():
    pre=json.loads((HERE/"PREFLIGHT.json").read_text());assert pre["status"]=="PASS"
    assert pre["model_sha256"]==sha(HERE/"model.py")
    setup();data=Pairs("train");val=Pairs("validation");stream=Stream(data)
    m=build().cuda().train();u=m.model.unet0
    params=[p for p in m.parameters() if p.requires_grad];tcfg=CFG["training"]
    opt=torch.optim.Adam(params,lr=tcfg["learning_rate"],betas=tuple(tcfg["betas"]),weight_decay=tcfg["weight_decay"])
    ema=EMA(m);hashes=sources();history=[];stream_digest=hashlib.sha256().hexdigest()
    def checkpoint(step):
        save(HERE/"checkpoints"/f"step_{step}.pt",{"step":step,"ema_updates":ema.updates,
           "model_state_dict":without_med(m),"ema_state_dict":without_med(ema.model),"optimizer":opt.state_dict(),
           "rng_state":rng(),"stream":stream.state_dict(),"history":history,"source_hashes":hashes,
           "config":CFG,"med_sha256":MED_SHA,"train_stream_sha256":stream_digest})
    assert not (HERE/"train.jsonl").exists();checkpoint(0)
    write_json(HERE/"INITIAL_STATE.json",{"fresh_random_seed":CFG["seed"],"no_diffusion_checkpoint_loaded":True,
         "parameters_trainable":sum(p.numel() for p in params),"parameters_total":sum(p.numel() for p in m.parameters()),
         "source_hashes":hashes,"train_manifest":data.manifest_hash,"validation_manifest":val.manifest_hash})
    torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    with (HERE/"train.jsonl").open("x",buffering=1) as log:
        for step in range(1,tcfg["max_optimizer_steps"]+1):
            tick=time.perf_counter();opt.zero_grad(set_to_none=True);value=0.;metas=[]
            chain=hashlib.sha256(bytes.fromhex(stream_digest));accum=tcfg["gradient_accumulation"]
            for _ in range(accum):
                context,target,meta=stream.next();context=context.cuda();target=target.cuda()
                u.prepare(context)
                t=torch.randint(0,m.num_timesteps,(context.shape[0],),device="cuda").long();noise=torch.randn_like(target)
                chain.update(json.dumps(meta,sort_keys=True).encode());chain.update(t.cpu().numpy().tobytes());chain.update(noise.cpu().numpy().tobytes())
                loss=m.p_losses([2*target-1,2*context[:,1:2]-1,2*context-1],t,noise=noise)[0]
                assert torch.isfinite(loss)
                (loss/accum).backward();value+=float(loss.detach())/accum;metas+=meta;u.clear()
            stream_digest=chain.hexdigest()
            grad=float(torch.nn.utils.clip_grad_norm_(params,tcfg["gradient_clip"],error_if_nonfinite=True))
            opt.step();ema.update(m);torch.cuda.synchronize()
            if step==1 or step%100==0:
                assert not u.med.training and all(p.grad is None for p in u.med.parameters())
            row={"step":step,"loss":value,"gradient_norm":grad,"seconds":time.perf_counter()-tick,
                 "peak_allocated":torch.cuda.max_memory_allocated(),"samples":metas,"train_stream_sha256":stream_digest}
            log.write(json.dumps(row,allow_nan=False)+"\n")
            if step==1 or step%100==0:print("TRAIN",json.dumps(row),flush=True)
            if step in tcfg["matched_checkpoint_steps"]:
                result=validate(ema.model,val,step);history.append(result)
                csv_write(HERE/"validation_summary.csv",history);checkpoint(step)
                write_json(HERE/"DATA_ACCESS.json",access_report());print("VALIDATION",json.dumps(result),flush=True)
    assert sources()==hashes
    write_json(HERE/"COMPLETE.json",{"step":step,"ema_updates":ema.updates,"history":history,
      "seconds":time.perf_counter()-started,"peak_allocated":torch.cuda.max_memory_allocated(),
      "peak_reserved":torch.cuda.max_memory_reserved(),"parameters_trainable":sum(p.numel() for p in params),
      "parameters_total":sum(p.numel() for p in m.parameters()),"source_hashes":hashes,"train_stream_sha256":stream_digest})
    print("COMPLETE",step,flush=True)
if __name__=="__main__":
    try:main()
    except BaseException:
        write_json(HERE/("FAILURE_"+datetime.datetime.now().strftime("%Y%m%dT%H%M%S")+".json"),{"traceback":traceback.format_exc()})
        raise
