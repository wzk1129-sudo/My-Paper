"""Paired real-data preflight. No training queue may start without this PASS."""
import sys,importlib.util,importlib,hashlib,json,ast,copy,platform,gc,time
from pathlib import Path
import torch,numpy as np
ROOT=Path(__file__).resolve().parent
A=ROOT/"experiments/Fast2P_TimeOnly_Control_8k";B=ROOT/"experiments/Fast2P_TimeOnly_MedCNN_8k"
sys.path.insert(0,str(A))
from mayo_data import Pairs,Stream,VALID,access_report
def package(name,path):
    spec=importlib.util.spec_from_file_location(name,path/"__init__.py",submodule_search_locations=[str(path)])
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    return importlib.import_module(name+".DADiff")
ac=package("control_src",A/"src");bc=package("candidate_src",B/"src")
def build(module):
    u=module.UnetRes(dim=64,dim_mults=(1,2,4,8),channels=1,num_unet=1,condition=True,input_condition=False,objective="pred_res",test_res_or_noise="res")
    return module.ResidualDiffusion(u,image_size=512,timesteps=1000,sampling_timesteps=2,objective="pred_res",loss_type="l2",condition=True,sum_scale=.01,test_res_or_noise="res")
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False))
def cls(s,n):return next(x for x in ast.parse(s).body if isinstance(x,ast.ClassDef) and x.name==n)
def grad(module):
    gs=[p.grad for p in module.parameters() if p.requires_grad and p.grad is not None]
    assert gs and all(torch.isfinite(g).all() for g in gs)
    return sum(float(g.abs().sum()) for g in gs)
def main():
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=(ROOT/"reference/baseline_src_DADiff.py").read_text()
    assert (A/"src/DADiff.py").read_text()==base
    for name in ["TimeOnlyDACB","Block","ResnetBlock","ResidualDiffusion"]:
        assert ast.dump(cls(base,name),include_attributes=False)==ast.dump(cls((B/"src/DADiff.py").read_text(),name),include_attributes=False)
    for name in ["mayo_data.py","frozen_metrics.py","L333_fast64.txt"]:
        assert (A/name).read_bytes()==(B/name).read_bytes()
    assert (A/"frozen_metrics.py").read_bytes()==(ROOT/"reference/baseline_frozen_metrics.py").read_bytes()
    assert len(Pairs("train"))==1145 and len(VALID)==64
    data=Pairs("train")
    for pos in [0,559,560,1144]:
        x,_,meta=data.load(pos)
        if meta["slice_index"]==0:assert np.array_equal(x[0],x[1])
        else:assert np.array_equal(x[2],x[1])
    ac.set_seed(10);a=build(ac).cuda()
    ac.set_seed(10);b=build(bc).cuda()
    shared=a.state_dict()
    rr=b.load_state_dict(shared,strict=False)
    assert not rr.unexpected_keys and all(".structure." in k or ".adapters." in k for k in rr.missing_keys)
    assert all(torch.equal(b.state_dict()[k],v) for k,v in shared.items())
    assert sum(isinstance(v,ac.TimeOnlyDACB) for v in a.modules())==9
    assert sum(isinstance(v,bc.TimeOnlyDACB) for v in b.modules())==9
    assert sum(isinstance(v,bc.ResnetBlock) for v in b.modules())==10
    assert len(b.model.unet0.adapters)==9 and len({id(v) for v in b.model.unet0.adapters.values()})==9
    assert not any(any(k in type(v).__name__.lower() for k in ["ss2d","mamba","transposed","linearattention","clip"]) for v in b.modules())
    assert not any("attention" in type(v).__name__.lower() for v in a.modules())
    med=b.model.unet0.structure.med
    assert all(not p.requires_grad for p in med.parameters());b.train()
    assert not med.training and not med.net.training
    frozen_before=med.net.patch_embed.proj.weight.detach().clone()
    context,target,metas=Stream(data).next();context=context.cuda();target=target.cuda()
    x=torch.cat((torch.zeros_like(target),2*context-1),1);t=torch.tensor([10.,900.],device="cuda")
    shapes={};calls={};hooks=[]
    def shape_hook(name):
        def hook(mod,args,out):
            calls[name]=calls.get(name,0)+1
            if isinstance(out,dict):shapes[name]={k:list(v.shape) for k,v in out.items()}
            elif isinstance(out,(tuple,list)):shapes[name]=[list(v.shape) for v in out]
            else:
                assert torch.isfinite(out).all()
                shapes[name]=list(out.shape)
                if name.startswith("adapter"):assert args[0].shape==out.shape
        return hook
    st=b.model.unet0.structure
    for name,mod in [("med",med),("M32",st.med_fuse),("structure",st),*[(f"cnn{i}",v) for i,v in enumerate(st.cnn)],*[(f"adapter_{n}",v) for n,v in b.model.unet0.adapters.items()]]:
        hooks.append(mod.register_forward_hook(shape_hook(name)))
    a.eval();b.eval()
    with torch.no_grad():
        pa=a.model.unet0(x,t);pb=b.model.unet0(x,t)
        assert pa.shape==pb.shape==(2,1,512,512)
        maxdiff=float((pa-pb).abs().max());assert torch.equal(pa,pb),maxdiff
    assert all(calls.get("adapter_"+n,0)>0 for n in b.model.unet0.adapters)
    assert shapes["med"]==[[2,768,32,32]]*4 and shapes["M32"]==[2,128,32,32]
    assert [shapes[f"cnn{i}"] for i in range(4)]==[[2,32,512,512],[2,32,256,256],[2,64,128,128],[2,128,64,64]]
    assert shapes["structure"]=={"F64":[2,128,64,64],"F128":[2,64,128,128],"F256":[2,32,256,256],"F512":[2,32,512,512]}
    # Compare actual sampled timesteps and training noise at the same RNG state.
    captures={}
    for label,model in [("A",a),("B",b)]:
        original=model.q_sample
        def wrapped(x_start,x_res,t,noise=None,_label=label,_original=original):
            captures[_label]={"t":t.detach().cpu().tolist(),"noise_sha":hashlib.sha256(noise.detach().cpu().numpy().tobytes()).hexdigest()}
            return _original(x_start,x_res,t,noise)
        model.q_sample=wrapped
        ac.set_seed(1010)
        with torch.no_grad():loss=model([target,context[:,1:2],context])[0];assert torch.isfinite(loss)
        model.q_sample=original
    assert captures["A"]==captures["B"]
    # All trainable branches must update after initially-zero adapters open.
    a.train();b.train();peaks={};gradient_rows=[];timings={}
    for label,model in [("A",a),("B",b)]:
        opt=torch.optim.Adam((p for p in model.parameters() if p.requires_grad),lr=2e-4,betas=(.9,.99))
        tracked={}
        if label=="B":
            for name,v in st.named_parameters():
                if v.requires_grad:tracked["structure."+name]=v
            for name,v in b.model.unet0.adapters.named_parameters():tracked["adapters."+name]=v
        before={n:p.detach().cpu().clone() for n,p in tracked.items()}
        torch.cuda.reset_peak_memory_stats()
        timings[label]=[]
        for update in range(2):
            torch.cuda.synchronize();tick=time.perf_counter()
            ac.set_seed(2000+update);opt.zero_grad(set_to_none=True)
            loss=model([target,context[:,1:2],context])[0];assert torch.isfinite(loss);loss.backward()
            if label=="B":
                row={"update":update+1,"adapters":{n:grad(v) for n,v in b.model.unet0.adapters.items()},"cnn":grad(st.cnn),"projections":grad(st.projections),"fuse64":grad(st.fuse64),"fuse128":grad(st.fuse128),"fuse256":grad(st.fuse256),"fuse512":grad(st.fuse512)}
                if update:assert all(v>0 for k,v in row.items() if k not in ["update","adapters"]) and all(v>0 for v in row["adapters"].values())
                assert all(p.grad is None for p in med.parameters())
                gradient_rows.append(row)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
            torch.cuda.synchronize();timings[label].append(time.perf_counter()-tick)
            print("UPDATE",label,update+1,float(loss.detach()),"SECONDS",timings[label][-1],flush=True)
        peaks[label]=torch.cuda.max_memory_allocated()
        if label=="B":
            diffs={n:float((p.detach().cpu()-before[n]).abs().max()) for n,p in tracked.items()}
            groups=[f"structure.cnn.{i}." for i in range(4)]+[f"structure.projections.{i}." for i in range(4)]+["structure.med_fuse.","structure.fuse64.","structure.fuse128.","structure.fuse256.","structure.fuse512."]+["adapters."+n+"." for n in b.model.unet0.adapters]
            assert all(any(v>0 for n,v in diffs.items() if n.startswith(prefix)) for prefix in groups)
        del opt,before
        gc.collect();torch.cuda.empty_cache()
    assert torch.equal(frozen_before,med.net.patch_embed.proj.weight)
    for h in hooks:h.remove()
    for label,model in [("A",a),("B",b)]:
        model.eval();seen=[]
        def stem(mod,args):assert torch.equal(args[0][:,1:],2*context[:1]-1);seen.append(1)
        h=model.model.unet0.init_conv.register_forward_pre_hook(stem)
        with torch.no_grad():pred=model.sample([context[:1,1:2],context[:1]],last=True)[-1]
        h.remove();assert len(seen)==2 and pred.shape==(1,1,512,512) and torch.isfinite(pred).all()
    assert max(peaks.values())<24*2**30 and access_report()["L506_opens"]==0
    result={"status":"PASS","initial_prediction_max_diff":maxdiff,"same_sampled_training_timestep_noise":captures,"shapes":shapes,"calls":calls,"gradient_checks":gradient_rows,"parameter_update_max_diffs":diffs,"peak_allocated":peaks,"update_seconds":timings,"meddino":med.load_confirmation,"meddino_frozen_unchanged":True,"data_access":access_report(),
       "parameters":{"A":{"total":sum(p.numel() for p in a.parameters()),"trainable":sum(p.numel() for p in a.parameters() if p.requires_grad)},"B":{"total":sum(p.numel() for p in b.parameters()),"trainable":sum(p.numel() for p in b.parameters() if p.requires_grad)}}}
    write(ROOT/"PAIRED_PREFLIGHT.json",result)
    for out in [A,B]:
        files=list(out.glob("*.py"))+list((out/"src").glob("*.py"))+[out/"config.json",out/"L333_fast64.txt"]
        hashes={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        write(out/"PREFLIGHT.json",{"status":"PASS","paired_report":str(ROOT/"PAIRED_PREFLIGHT.json"),"source_hashes":hashes})
    print("PAIRED_PREFLIGHT_PASS",result["parameters"],peaks,flush=True)
if __name__=="__main__":main()
