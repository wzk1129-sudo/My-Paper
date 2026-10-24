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
    return {n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ("src/DADiff.py","train.py","mayo_data.py","frozen_metrics.py","preflight.py")}
def save(path,obj):
    assert not path.exists(),str(path)
    # Exclusive output; interrupted files are never silently reused.
    with path.open("xb") as f:torch.save(obj,f)
class EMA:
    def __init__(self,m):self.model=copy.deepcopy(m).eval().requires_grad_(False);self.updates=0
    @torch.no_grad()
    def update(self,m):
        for a,b in zip(self.model.parameters(),m.parameters()):a.mul_(.995).add_(b,alpha=.005)
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
    best=max(history,key=lambda r:r["psnr"])
    text=f"""# FoundDiff-Lite-TimeDACB-2p5D
Official commit: {CFG['official_commit']}.
Completed optimizer steps: {history[-1]['step']}. Fresh random initialization, seed=10.
Trainable parameters: {params:,}. Peak allocated: {peak/2**30:.3f} GiB; peak reserved: {reserved/2**30:.3f} GiB.

## Architecture
Input [I_t, previous LDCT, center LDCT, next LDCT], 4 channels; stem 7x7 conv to 64.
Encoder (TimeOnlyDACB -> original single-convolution ResnetBlock -> downsample):
E0 64@512 ->64@256; E1 64@256 ->128@128; E2 128@128 ->256@64;
E3 256@64 ->512@64 (official final 3x3 convolution, no spatial downsampling).
Bottleneck: original ResnetBlock 512@64 -> TimeOnlyDACB 512@64.
Decoder (concat skip -> original ResnetBlock -> TimeOnlyDACB -> upsample):
D0 (512+256)->512@64 ->256@128;
D1 (256+128)->256@128 ->128@256;
D2 (128+64)->128@256 ->64@512;
D3 (64+64)->64@512 ->64@512 (official final 3x3 convolution).
Stem skip concat -> original ResnetBlock 128->64 -> original 1x1 head 64->1.
Ten original ResnetBlocks; nine TimeOnlyDACBs. Commented second conv/time modulation remain commented.
time_mlp: official sinusoid 64 -> Linear 256 -> GELU -> Linear 256.

## Removed and retained
Removed CLIP/DA-CLIP loading, dose/anatomy embeddings, prompt/text/prompt MLP,
CSSM/SS2D/Mamba and all attention modules. Removed unused LPIPS instantiation.
src/DADiff.py: TimeOnlyDACB implements channel-last affine-free LayerNorm eps=1e-6:
shift, scale, gate = Linear(SiLU(time_mlp(t))); x_out=x+gate*(LN(x)*(1+scale)+shift).
Its final Linear is zero initialized. No convolution, attention or external conditions within DACB.
ResidualDiffusion.condition=True, channels=out_dim=1. Residual is center LDCT minus center NDCT.
Official q_sample, linear-beta convert_to_ddim=True schedule, residual MSE and two-step sampling retained.

## EMA validation (arithmetic mean of 64 per-slice metrics)
|step|PSNR dB|SSIM|RMSE HU|MAE HU|soft PSNR|soft SSIM|soft RMSE HU|
|---:|---:|---:|---:|---:|---:|---:|---:|
"""
    for r in history:text+="|"+"|".join(str(r[k]) if k=="step" else f"{r[k]:.8f}" for k in ("step","psnr","ssim","rmse_hu","mae_hu","soft_psnr","soft_ssim","soft_rmse_hu"))+"|\n"
    text+=f"""
## Protocol and reference comparisons
Main window [-1000,1000] HU, PSNR range 2000, SSIM win_size=7. Auxiliary window [-160,240] HU,
range 400. frozen_metrics.py is byte-identical to historical metric implementation.
Historical Fast64 slice list and order and per-slice validation seed=20260829+index retained.
Eight train patients: L067 L096 L109 L143 L192 L286 L291 L310. No L506. No augmentation.
Stored pixel offset -1024 HU; explicit requested training clip [-1000,1000], then model [-1,1].
Historical referenced model used [-1024,3072] normalization and only L067/L143 training, so not a matched experiment.
MLSF+Pyramid@8k supplied reference: 40.956836033 dB. Current step {history[-1]['step']} delta: {history[-1]['psnr']-40.956836033:.8f} dB.
Models, losses, initialization, train patients, normalization, effective batch and seed differ; no isolated attribution.
Historical simplified FoundDiff_RDDM_RLEB Control (2D, no DACB) recorded:
2k 38.4787, 4k 40.1454, 6k 40.3872, 8k 40.4846 dB (rounded report values).
Its 8k SSIM=.9449, RMSE=19.0149 HU, MAE=12.4597 HU.
Exact metrics/source are preserved in historical_reference/. These are contextual references, not matched ablations.
Historical training used two patients, different normalization, seed, effective batch, and paired flip/rotation;
the current baseline also preserves upstream FoundDiff schedule/sampler rather than historical CT alterations.
Adam lr=2e-4, betas=(.9,.99), weight_decay=0 (official optimizer); batch=2, accumulation=2;
EMA=.995 every optimizer step, initialized from step0 (historical matched EMA convention).
amp=False; no scheduler; clip grad norm=1. Peak memory includes training/validation.

## Reproduction
Best checkpoint: {OUT}/checkpoints/step_{best['step']}.pt (ema_state_dict).
Step0 with model, optimizer, RNG and stream state: checkpoints/step_0.pt.
Full command: cd {ROOT} && CUDA_VISIBLE_DEVICES=0 /home/xujian/miniconda3/bin/conda run --no-capture-output -n flexict python -u train.py
Requires preflight.py PASS. Existing outputs are protected; resume explicitly with --resume PATH.
Config: config.json; training: train.jsonl; launch: RUN_MANIFEST.json; per-slice: validation_step_*.csv.
"""
    (OUT/"REPORT.md").write_text(text)
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--resume",type=Path);args=ap.parse_args()
    assert json.loads((OUT/"PREFLIGHT.json").read_text())["status"]=="PASS"
    setup();train=Pairs("train");val=Pairs("validation");stream=Stream(train)
    m=build().cuda();opt=torch.optim.Adam(m.parameters(),lr=2e-4,betas=(.9,.99))
    ema=EMA(m);history=[];step=0;params=sum(p.numel() for p in m.parameters())
    state=sources()
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
            tick=time.perf_counter();opt.zero_grad(set_to_none=True);loss_value=0.;samples=[]
            for _ in range(2):
                context,target,meta=stream.next();context=context.cuda();target=target.cuda()
                loss=m([target,context[:,1:2],context])[0]
                assert torch.isfinite(loss)
                (loss/2).backward();loss_value+=float(loss.detach())/2;samples+=meta
            grad=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True))
            opt.step();ema.update(m);step+=1;torch.cuda.synchronize()
            row={"step":step,"loss":loss_value,"gradient_norm":grad,"seconds":time.perf_counter()-tick,
                 "peak_allocated":torch.cuda.max_memory_allocated(),"samples":samples}
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
