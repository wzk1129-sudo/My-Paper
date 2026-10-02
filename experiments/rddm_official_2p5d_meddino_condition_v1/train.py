"""Two independent fresh-Adam continuations with a common deterministic effective-batch stream."""
import argparse,copy,csv,gc,hashlib,json,math,time
from pathlib import Path
import torch
from model import build_model,native_attention_grad,state_without_med,load_state_without_med
from provenance import ROOT,write_json,source_manifest
from med_loader import sha
from fast_dev.protocol import CONFIG,check_config
from context_data import ContextPairs,ContextBatchStream,ContextDataGuard
from mayo2016_smallset_v1.runtime import seed_everything,rng_state,restore_rng,atomic_save
from evaluate import validate
ARMS=("OFFICIAL_RDDM_2P5D","OFFICIAL_RDDM_2P5D_MEDDINO_CONDITION")
STEPS=(0,1000,2000,4000,6000,8000)
def tensor_state_hash(state):
    h=hashlib.sha256()
    for n,t in sorted(state.items()):
        h.update(n.encode());h.update(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
def rng_digest():
    h=hashlib.sha256(torch.get_rng_state().numpy().tobytes())
    for t in torch.cuda.get_rng_state_all():h.update(t.cpu().numpy().tobytes())
    return h.hexdigest()
def finite_model(model):
    return all(bool(torch.isfinite(p).all()) for p in model.parameters() if p.requires_grad)
class MatchedEMA:
    def __init__(self,model):
        self.model=copy.deepcopy(model).eval().requires_grad_(False)
        if model.model.unet0.anatomy:self.model.model.unet0.med=model.model.unet0.med
        self.updates=0
    @torch.no_grad()
    def update(self,model):
        target=dict(self.model.named_parameters())
        for n,p in model.named_parameters():
            if p.requires_grad:target[n].mul_(.995).add_(p,alpha=.005)
        buffers=dict(self.model.named_buffers())
        for n,b in model.named_buffers():buffers[n].copy_(b)
        self.updates+=1
def save_history(output,history):
    with (output/"validation_metrics.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(history[0]),lineterminator="\n");w.writeheader();w.writerows(history)
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--arm",choices=ARMS,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--resume",type=Path)
    args=ap.parse_args()
    ready=json.loads((ROOT/"READINESS.json").read_text());assert ready["status"]=="PASS"
    batch_cfg=json.loads((ROOT/"BATCH_CONFIG.json").read_text())
    batch,accum=batch_cfg["batch"],batch_cfg["accum"];assert batch*accum==2 and (batch,accum) in ((2,1),(1,2))
    provenance=json.loads((ROOT/"SOURCE_PROVENANCE.json").read_text())
    for n in ("model.py","med_loader.py","readiness.py","context_data.py","src/official_rddm_ct.py"):
        assert sha(ROOT/n)==provenance["new_source_hashes"][n],("changed readiness dependency",n)
    cfg=json.loads(CONFIG.read_text());check_config(cfg)
    torch.set_num_threads(cfg["training"]["cpu_threads"])
    seed_everything(20260829);device=torch.device("cuda:0")
    anatomy=args.arm==ARMS[1]
    output=args.output.resolve()
    if args.resume:assert output.is_dir() and args.resume.is_file()
    else:output.mkdir(parents=True,exist_ok=False);(output/"checkpoints").mkdir()
    with ContextDataGuard(cfg) as guard:
        train=ContextPairs(cfg,"train");validation=ContextPairs(cfg,"validation")
        train.bind_guard(guard);validation.bind_guard(guard)
        stream=ContextBatchStream(train,20260829)
        initial_rng=rng_state()
        model=build_model(anatomy).to(device).train()
        params=[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.Adam(params,lr=2e-4,betas=(.9,.99),weight_decay=1e-9)
        ema=MatchedEMA(model)
        restore_rng(initial_rng)
        initial_shared={n:t for n,t in state_without_med(model).items() if not any(x in n for x in (".layer_adapters.",".fusion.",".conditions."))}
        shared_sha=tensor_state_hash(initial_shared);del initial_shared
        med_hash=tensor_state_hash(model.model.unet0.med.state_dict()) if anatomy else None
        history=[];step=0;stream_digest=hashlib.sha256().hexdigest();last={}
        if args.resume:
            ck=torch.load(args.resume,map_location="cpu",weights_only=False)
            assert ck["arm"]==args.arm and ck["batch_config"]==batch_cfg
            load_state_without_med(model,ck["model_state_dict"]);load_state_without_med(ema.model,ck["ema_state_dict"])
            optimizer.load_state_dict(ck["optimizer_state_dict"]);ema.updates=ck["ema_updates"];step=ck["optimizer_step"]
            stream.load_state_dict(ck["stream_state"]);restore_rng(ck["rng_state"]);history=ck["history"]
            # Resume stream chaining uses the previous digest as the next immutable prefix.
            stream_digest=ck["train_stream_sha256"]
            del ck;gc.collect()
        else:
            write_json(output/"INITIAL_STATE.json",{"arm":args.arm,"shared_initial_parameters_sha256":shared_sha,
                "initial_rng_sha256":rng_digest(),"train_manifest_sha256":train.manifest_hash,
                "validation_manifest_sha256":validation.manifest_hash,"batch_config":batch_cfg,
                "trainable_parameters":sum(p.numel() for p in params),"total_parameters":sum(p.numel() for p in model.parameters()),
                "meddino_initial_weights_sha256":med_hash,"fresh_optimizer":True,"ema_updates":0})
        hashes=source_manifest()
        torch.cuda.reset_peak_memory_stats()
        def evaluate_step():
            before=rng_digest()
            row,activity=validate(ema.model,validation,cfg,device,output/("validation_step_%d.json"%step))
            assert before==rng_digest()
            row={"arm":args.arm,"step":step,**row,"train_stream_sha256":stream_digest,
                "attention_gradient_norm":last.get("attention_gradient_norm",0),"peak_vram_bytes":torch.cuda.max_memory_allocated()}
            history.append(row);save_history(output,history)
            if anatomy:write_json(output/("condition_activity_step_%d.json"%step),activity)
            write_json(output/"DATA_ACCESS_AUDIT.json",guard.report())
            print("VALIDATION "+json.dumps(row),flush=True)
        if not args.resume:evaluate_step()
        with (output/"train.jsonl").open("a" if args.resume else "x",buffering=1) as log:
            while step<8000:
                started=time.perf_counter();pre=rng_digest() if step==0 else None
                stream_hash=hashlib.sha256(bytes.fromhex(stream_digest))
                optimizer.zero_grad(set_to_none=True);loss_sum=0.;samples=[];tensors_audit=[];noise_audit=[];micro_noise=[]
                for micro in range(accum):
                    prev,center,next_,target,meta=stream.next(batch)
                    samples.extend(meta)
                    for x in (prev,center,next_,target):stream_hash.update(x.numpy().tobytes())
                    prev,center,next_,target=[x.to(device) for x in (prev,center,next_,target)]
                    t=torch.randint(0,model.num_timesteps,(batch,),device=device).long()
                    noise=torch.randn_like(target)
                    times=t.detach().cpu().tolist();noise_bytes=noise.detach().cpu().numpy().tobytes()
                    tensors_audit.extend(times);noise_audit.append(hashlib.sha256(noise_bytes).hexdigest());micro_noise.append(noise_bytes)
                    stream_hash.update(json.dumps(meta,sort_keys=True,separators=(",",":")).encode())
                    stream_hash.update(t.detach().cpu().numpy().tobytes());stream_hash.update(noise_bytes)
                    u=model.model.unet0;u.set_context(prev,center,next_)
                    loss=model.p_losses([2*target-1,2*center-1],t,noise=noise)[0]
                    assert torch.isfinite(loss)
                    loss_sum+=float(loss.detach())/accum
                    (loss/accum).backward();u.clear_context()
                stream_digest=stream_hash.hexdigest()
                attn=native_attention_grad(model)
                native_norm=math.sqrt(sum(v*v for v in attn.values()))
                if anatomy:assert all(not p.requires_grad and p.grad is None for p in model.model.unet0.med.parameters())
                grad=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True))
                optimizer.step();ema.update(model);step+=1
                finite=None
                if step==1 or step%100==0:
                    finite=finite_model(model);assert finite
                torch.cuda.synchronize()
                last={"arm":args.arm,"optimizer_step":step,"train_loss":loss_sum,"gradient_norm_before_clip":grad,
                    "attention_gradient_norm":native_norm,"parameters_finite":finite,"seconds_per_step":time.perf_counter()-started,
                    "peak_vram_bytes":torch.cuda.max_memory_allocated(),"samples":samples,
                    "timesteps":tensors_audit if step==1 else None,"noise_sha256":hashlib.sha256(b"".join(micro_noise)).hexdigest() if step==1 else None,
                    "microbatch_noise_sha256":noise_audit if step==1 else None,
                    "pre_step_rng_sha256":pre,"post_step_rng_sha256":rng_digest() if step==1 else None,
                    "train_stream_sha256":stream_digest}
                log.write(json.dumps(last,allow_nan=False)+"\n")
                if step==1 or step%100==0:print("TRAIN "+json.dumps({k:last[k] for k in ("optimizer_step","train_loss","attention_gradient_norm","seconds_per_step","peak_vram_bytes")}),flush=True)
                if step in STEPS:
                    evaluate_step()
                    path=output/"checkpoints"/("step_%d.pt"%step)
                    assert not path.exists()
                    atomic_save(path,{"arm":args.arm,"optimizer_step":step,"ema_updates":ema.updates,
                        "model_state_dict":state_without_med(model),"ema_state_dict":state_without_med(ema.model),
                        "optimizer_state_dict":optimizer.state_dict(),"rng_state":rng_state(),"stream_state":stream.state_dict(),
                        "history":history,"train_stream_sha256":stream_digest,"batch_config":batch_cfg,"source_hashes":hashes,
                        "diffusion":cfg["diffusion"],"parent_checkpoint_sha256":provenance["parent_checkpoint_sha256"]})
        report=guard.report()
        assert report["L506_opens"]==report["external_memory_opens"]==report["neighbor_target_opens"]==report["blocked_attempts"]==0
        if anatomy:
            assert tensor_state_hash(model.model.unet0.med.state_dict())==med_hash
            assert all(p.grad is None and not p.requires_grad for p in model.model.unet0.med.parameters())
        write_json(output/"DATA_ACCESS_AUDIT.json",report)
        write_json(output/"FINAL_STATE.json",{"arm":args.arm,"optimizer_step":step,"ema_updates":ema.updates,
            "batch_config":batch_cfg,"shared_initial_parameters_sha256":shared_sha,"train_stream_sha256":stream_digest,
            "parameters_finite":finite_model(model),"meddino_weights_unchanged":True if anatomy else None,
            "peak_vram_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_bytes":torch.cuda.max_memory_reserved(),
            "source_hashes":hashes})
        print("COMPLETE "+args.arm,flush=True)
if __name__=="__main__":main()
