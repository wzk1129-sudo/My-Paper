"""Architecture, ACL numeric equivalence, real batch2 gradients and 2-step sampling."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import ast,copy,json,hashlib,platform
import torch,numpy as np
from torch import nn
from model import build,HERE,LACB,MDC,FAM,LinearAttention
from train import setup,write_json
from mayo_data import Pairs,Stream,access_report,CFG,OUT
from src.DADiff import ResnetBlock,TimeOnlyDACB
from med_loader import sha,MED_CK,MED_SHA
def grad(module):
    gs=[p.grad for p in module.parameters() if p.requires_grad and p.grad is not None]
    assert all(torch.isfinite(g).all() for g in gs)
    return sum(float(g.abs().sum()) for g in gs)
def main():
    setup();assert OUT==HERE and sha(MED_CK)==MED_SHA
    # Extract only the reference class, with no timm/SSM imports or installation.
    tree=ast.parse((HERE/"reference/ACL_layers.py").read_text())
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=="LinearAttention")
    env={"torch":torch,"nn":nn}
    exec(compile(ast.Module(body=[cls],type_ignores=[]),"<pinned ACL LinearAttention>","exec"),env)
    ref=env["LinearAttention"](64,4).cuda();ours=LinearAttention(64,4).cuda()
    ours.load_state_dict(ref.state_dict())
    a=torch.randn(2,63,64,device="cuda",requires_grad=True);b=a.detach().clone().requires_grad_()
    ra=ref(a,(2,64,7,9));rb=ours(b,(7,9));la_diff=float((ra-rb).detach().abs().max())
    assert la_diff==0
    ra.square().mean().backward();rb.square().mean().backward()
    assert torch.equal(a.grad,b.grad)
    unit=LACB(64).cuda();x=torch.randn(2,64,7,9,device="cuda");c=torch.randn(2,256,device="cuda")
    assert torch.equal(unit(x,c),x)
    del ref,ours,a,b,ra,rb,unit,x,c
    setup();m=build().cuda().train();u=m.model.unet0
    lacbs=[(n,v) for n,v in u.named_modules() if isinstance(v,LACB)]
    assert len(lacbs)==9 and sum(isinstance(v,MDC) for v in u.modules())==6
    assert not any(isinstance(v,TimeOnlyDACB) for v in u.modules())
    assert sum(isinstance(v,ResnetBlock) for v in u.modules())==10
    assert set(u.mdcs)=={"E0","E1","E2","D1","D2","D3"}
    assert not any(isinstance(v,nn.BatchNorm2d) for v in u.modules())
    assert not any(any(z in type(v).__name__.lower() for z in ("mamba","ss2d","clip","selectivescan")) for v in u.modules())
    assert m.channels==1 and m.condition and u.init_conv.in_channels==4
    data=Pairs("train");stream=Stream(data);context,target,_=stream.next();context=context.cuda();target=target.cuda()
    shapes={};handles=[]
    def hook(name):
        def f(mod,args,out):
            assert args[0].shape==out.shape
            if isinstance(mod,FAM):assert args[0].shape==args[1].shape
            shapes[name]=list(out.shape)
        return f
    for n,v in u.named_modules():
        if isinstance(v,(LACB,MDC,FAM)):handles.append(v.register_forward_hook(hook(n)))
    stem_calls=[]
    def stem(mod,args):
        assert torch.equal(args[0][:,1:],2*context-1)
        stem_calls.append(True)
    handles.append(u.init_conv.register_forward_pre_hook(stem))
    params=[p for p in m.parameters() if p.requires_grad];opt=torch.optim.Adam(params,lr=2e-4,betas=(.9,.99))
    tracked={"msfs_cnn":u.msfs.cnn[0][0].weight,"lacb_gate":u.downs[0][1].modulation[-1].weight,
             "lacb_qk":u.downs[0][1].attn.qk.weight,"mdc_depthwise":u.mdcs["E0"].conv3[0].weight}
    initial={k:v.detach().cpu().clone() for k,v in tracked.items()}
    torch.cuda.reset_peak_memory_stats();gradients=[]
    for update in range(2):
        opt.zero_grad(set_to_none=True);u.prepare(context)
        loss=m([target,context[:,1:2],context])[0];assert torch.isfinite(loss);loss.backward();u.clear()
        gs={"MSFS":grad(u.msfs),"global_projection":grad(u.global_projection),"joint":grad(u.joint),
            **{"FAM"+str(i):grad(v) for i,v in enumerate(u.fams)},
            **{"MDC_"+n:grad(v) for n,v in u.mdcs.items()},
            **{"LACB_"+n:grad(v) for n,v in lacbs},
            **{"LA_qk_"+n:grad(v.attn.qk) for n,v in lacbs}}
        assert all(v>0 for k,v in gs.items() if not k.startswith(("LA_qk_","global","joint")))
        if update==1:assert all(v>0 for v in gs.values()),gs
        assert not u.med.training and all(not p.requires_grad and p.grad is None for p in u.med.parameters())
        gradients.append(gs)
        torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step()
        print("UPDATE",update+1,float(loss.detach()),gs,flush=True)
    updates={k:float((v.detach().cpu()-initial[k]).abs().max()) for k,v in tracked.items()}
    assert all(v>0 for v in updates.values()),updates
    for h in handles:h.remove()
    expected=[]
    for i in range(4):
        expected+=["E"+str(i)+":"+n for n in (["FAM","ResBlock","LACB","MDC","Skip"] if i<3 else ["FAM","ResBlock","LACB","Skip"])]
    expected+=["MID:ResBlock","MID:LACB"]
    for i in range(4):expected+=["D"+str(i)+":"+n for n in (["ResBlock","LACB"] if i==0 else ["ResBlock","LACB","MDC"])]
    assert u.events==expected
    expected_shapes=[[2,64,512,512],[2,64,256,256],[2,128,128,128],[2,256,64,64]]
    assert u.msfs.last_shapes["cnn"]==u.msfs.last_shapes["structure"]==expected_shapes
    topology=copy.deepcopy(shapes);msfs_shapes=copy.deepcopy(u.msfs.last_shapes)
    ca_shapes=[copy.deepcopy(f.last_shapes) for f in u.msfs.fusions]
    assert all(s["q"]==s["k"]==s["v"]==[2,4,1024,32] for s in ca_shapes)
    m.eval()
    with torch.no_grad():
        cc=context[:1];u.prepare(cc);sp,ea=u._condition
        xx=torch.cat((torch.zeros_like(target[:1]),2*cc-1),1)
        a=u(xx,torch.tensor([10.],device="cuda"));b=u(xx,torch.tensor([900.],device="cuda"))
        time_diff=float((a-b).abs().max());assert time_diff>0
        u._condition=(sp,ea+.5)
        d=u(xx,torch.tensor([10.],device="cuda"))
        global_diff=float((a-d).abs().max());assert global_diff>0
        u.clear()
        count=[]
        def sample_hook(mod,args):
            assert torch.equal(args[0][:,1:],2*cc-1);count.append(1)
        handle=u.init_conv.register_forward_pre_hook(sample_hook)
        u.prepare(cc);sample=m.sample([cc[:,1:2],cc],last=True)[-1];u.clear();handle.remove()
        assert sample.shape==(1,1,512,512) and torch.isfinite(sample).all() and len(count)==2
    peak=torch.cuda.max_memory_allocated();assert peak<24*2**30
    result={"status":"PASS","model_sha256":sha(HERE/"model.py"),"acl_linear_forward_max_diff":la_diff,
      "acl_linear_backward_equal":True,"lacb_init_identity":True,"lacb_count":9,"mdc_count":6,
      "events":expected,"shapes":topology,"msfs_shapes":msfs_shapes,"msfs_attention_shapes":ca_shapes,
      "gradients":gradients,"parameter_update_max_diffs":updates,"time_prediction_max_diff":time_diff,"global_prediction_max_diff":global_diff,
      "med_frozen":True,"med_sha256":MED_SHA,"sample_calls":len(count),"sample_shape":list(sample.shape),
      "peak_allocated":peak,"peak_reserved":torch.cuda.max_memory_reserved(),
      "parameters_trainable":sum(p.numel() for p in params),"parameters_total":sum(p.numel() for p in m.parameters()),
      "environment":{"python":platform.python_version(),"torch":torch.__version__,"cuda":torch.version.cuda,"numpy":np.__version__,"gpu":torch.cuda.get_device_name(0)},
      "data_access":access_report()}
    write_json(HERE/"PREFLIGHT.json",result);print("PREFLIGHT_PASS",peak,flush=True)
if __name__=="__main__":main()
