"""Independent official FoundDiff Lite baseline. No pretrained weights or external services."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import argparse,copy,csv,json,random,time,sys,hashlib,subprocess,traceback
from pathlib import Path
import numpy as np
import torch
from src.DADiff import UnetRes,ResidualDiffusion,TimeOnlyDACB,set_seed
from mayo_data import CFG,OUT,ROOT,Pairs,Stream,VALID,access_report
from frozen_metrics import slice_metrics

def write_json(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n")
def build():
    m=UnetRes(dim=64,dim_mults=(1,2,4,8),channels=1,num_unet=1,condition=True,
        input_condition=False,objective="pred_res",test_res_or_noise="res")
    return ResidualDiffusion(m,image_size=512,timesteps=1000,sampling_timesteps=2,
        objective="pred_res",loss_type="l2",condition=True,sum_scale=.01,test_res_or_noise="res")
def setup():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    set_seed(10)
def rng():
    return {"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state(),"cuda":torch.cuda.get_rng_state_all()}
def restore(s):
    random.setstate(s["python"]);np.random.set_state(s["numpy"]);torch.set_rng_state(s["torch"]);torch.cuda.set_rng_state_all(s["cuda"])
def sources():
    files=list(ROOT.glob("*.py"))+list((ROOT/"src").glob("*.py"))+[ROOT/"config.json",ROOT/"L333_fast64.txt"]
    result={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    pair=ROOT.parents[1]
    shared=list((pair/"vendor").rglob("*.py"))+[pair/"paired_preflight.py",pair/"finalize_pair.py"]
    result.update({"../../"+str(p.relative_to(pair)):hashlib.sha256(p.read_bytes()).hexdigest() for p in shared})
    return result
def save(path,obj):
    assert not path.exists(),str(path)
    # Exclusive output; interrupted files are never silently reused.
    with path.open("xb") as f:torch.save(obj,f)
class EMA:
    def __init__(self,m):self.model=copy.deepcopy(m).eval().requires_grad_(False);self.updates=0
    @torch.no_grad()
    def update(self,m):
        for a,b in zip(self.model.parameters(),m.parameters()):
            if b.requires_grad:a.mul_(.995).add_(b,alpha=.005)
        for a,b in zip(self.model.buffers(),m.buffers()):a.copy_(b)
        self.updates+=1
def csv_write(path,rows):
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
@torch.no_grad()
def validate(m,data,step):
    m.eval();rows=[];soft=copy.deepcopy(CFG);soft["evaluation"]["hu_window"]=[-160,240]
    with torch.random.fork_rng(devices=[0]):
        for pos in range(len(data)):
            x,y,meta=data.load(pos)
            context=torch.from_numpy(x[None]).cuda();target=torch.from_numpy(y[None]).cuda()
            seed=CFG["evaluation"]["validation_seed"]+meta["slice_index"]
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            pred=m.sample([context[:,1:2],context],last=True)[-1]
            assert pred.shape==(1,1,512,512) and torch.isfinite(pred).all()
            main=slice_metrics(pred,target,CFG);aux=slice_metrics(pred,target,soft)
            row={**meta,"seed":seed,**{k:main[k] for k in ("psnr","ssim","rmse_hu","mae_hu")},
                 **{"soft_"+k:aux[k] for k in ("psnr","ssim","rmse_hu")}}
            rows.append(row)
    assert [r["slice_index"] for r in rows]==list(VALID)
    csv_write(OUT/f"validation_step_{step}.csv",rows)
    keys=("psnr","ssim","rmse_hu","mae_hu","soft_psnr","soft_ssim","soft_rmse_hu")
    return {"step":step,**{k:float(np.mean([r[k] for r in rows])) for k in keys},"slices":64}
def report(history,params,peak,reserved):
    from reporting import write_report
    write_report(history,params,peak,reserved)
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--resume",type=Path);args=ap.parse_args()
    assert json.loads((OUT/"PREFLIGHT.json").read_text())["status"]=="PASS"
    setup();train=Pairs("train");val=Pairs("validation");stream=Stream(train)
    m=build().cuda()
    if CFG["variant"]=="medcnn":
        init=torch.load(Path(CFG["paired_control_dir"])/"checkpoints/step_0.pt",map_location="cpu",weights_only=False)
        shared=init["model_state_dict"]
        result=m.load_state_dict(shared,strict=False)
        assert not result.unexpected_keys and all((".structure." in k or ".adapters." in k) for k in result.missing_keys)
        assert all(torch.equal(m.state_dict()[k].cpu(),v) for k,v in shared.items())
        del init,shared
    opt=torch.optim.Adam((p for p in m.parameters() if p.requires_grad),lr=2e-4,betas=(.9,.99))
    ema=EMA(m);history=[];step=0;params=sum(p.numel() for p in m.parameters())
    set_seed(10)  # Same diffusion timestep/noise sequence after architecture-specific initialization.
    state=sources()
    pre=json.loads((OUT/"PREFLIGHT.json").read_text())
    assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in pre["source_hashes"].items())
    def checkpoint(step):
        save(OUT/"checkpoints"/f"step_{step}.pt",{"step":step,"model_state_dict":m.state_dict(),
          "ema_state_dict":ema.model.state_dict(),"optimizer":opt.state_dict(),"ema_updates":ema.updates,
          "rng_state":rng(),"stream":stream.state_dict(),"history":history,"source_hashes":state,"config":CFG})
    if args.resume:
        ck=torch.load(args.resume,map_location="cpu",weights_only=False)
        assert ck["source_hashes"]==state and ck["config"]==CFG
        m.load_state_dict(ck["model_state_dict"]);ema.model.load_state_dict(ck["ema_state_dict"])
        opt.load_state_dict(ck["optimizer"]);ema.updates=ck["ema_updates"]
        stream.load_state_dict(ck["stream"]);restore(ck["rng_state"]);history=ck["history"];step=ck["step"];del ck
    else:
        assert not (OUT/"train.jsonl").exists()
        checkpoint(0)
    torch.cuda.reset_peak_memory_stats();started=time.perf_counter();m.train()
    with (OUT/"train.jsonl").open("a",buffering=1) as log:
        while step<8000:
            tick=time.perf_counter();opt.zero_grad(set_to_none=True);loss_value=0.;samples=[];noise_states=[]
            for _ in range(2):
                context,target,meta=stream.next();context=context.cuda();target=target.cuda()
                noise_states.append(hashlib.sha256(torch.cuda.get_rng_state().numpy().tobytes()).hexdigest())
                loss=m([target,context[:,1:2],context])[0]
                assert torch.isfinite(loss)
                (loss/2).backward();loss_value+=float(loss.detach())/2;samples+=meta
            grad=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True))
            opt.step();ema.update(m);step+=1;torch.cuda.synchronize()
            row={"step":step,"loss":loss_value,"gradient_norm":grad,"seconds":time.perf_counter()-tick,
                 "peak_allocated":torch.cuda.max_memory_allocated(),"samples":samples,"diffusion_rng_states":noise_states}
            log.write(json.dumps(row)+"\n")
            if step==1 or step%100==0:print("TRAIN",json.dumps(row),flush=True)
            if step in (2000,4000,6000,8000):
                result=validate(ema.model,val,step);history.append(result);csv_write(OUT/"validation_summary.csv",history)
                checkpoint(step);write_json(OUT/"DATA_ACCESS.json",access_report())
                report(history,params,torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved())
                print("VALIDATION",json.dumps(result),flush=True)
    write_json(OUT/"COMPLETE.json",{"step":step,"history":history,"seconds":time.perf_counter()-started,
       "peak_allocated":torch.cuda.max_memory_allocated(),"peak_reserved":torch.cuda.max_memory_reserved(),"source_hashes":state})
    print("COMPLETE 8000",flush=True)
if __name__=="__main__":
    try:main()
    except Exception:
        write_json(OUT/"FAILURE.json",{"traceback":traceback.format_exc()})
        raise
