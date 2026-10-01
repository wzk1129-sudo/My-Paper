"""Synthetic B=2 full-resolution topology and gradient readiness audit."""
import json
from pathlib import Path
import torch
from src.official_rddm_ct import build_baseline, ResnetBlock, LinearAttention, Attention, Residual, PreNorm
from src.rddm_rleb import build_baseline as build_control

def count(m):
    return sum(p.numel() for p in m.parameters())

def main():
    torch.set_num_threads(2)
    device=torch.device('cuda:0')
    torch.cuda.set_device(device)
    official=build_baseline().to(device)
    control=build_control()
    u=official.model.unet0
    c=count(control);p=count(official)
    assert c==13130945
    assert len(u.downs)==len(u.ups)==4
    for stage in list(u.downs)+list(u.ups):
        assert len(stage)==4
        assert isinstance(stage[0],ResnetBlock) and isinstance(stage[1],ResnetBlock)
        assert isinstance(stage[2],Residual) and isinstance(stage[2].fn,PreNorm)
        assert isinstance(stage[2].fn.fn,LinearAttention)
    assert isinstance(u.mid_block1,ResnetBlock) and isinstance(u.mid_block2,ResnetBlock)
    assert isinstance(u.mid_attn,Residual) and isinstance(u.mid_attn.fn,PreNorm)
    assert isinstance(u.mid_attn.fn.fn,Attention)
    assert isinstance(u.final_res_block,ResnetBlock) and isinstance(u.final_conv,torch.nn.Conv2d)
    assert u.final_conv.kernel_size==(1,1)
    shapes={};calls={}
    handles=[]
    for name,module in u.named_modules():
        if isinstance(module,(ResnetBlock,LinearAttention,Attention)):
            def hook(mod,inp,out,key=name):
                calls[key]=calls.get(key,0)+1
                shapes[key]=list(out.shape)
            handles.append(module.register_forward_hook(hook))
    torch.manual_seed(20260829);torch.cuda.manual_seed_all(20260829)
    torch.cuda.reset_peak_memory_stats(device)
    x=torch.randn(2,2,512,512,device=device)
    t=torch.tensor([41,731],device=device)
    y=u(x,t)
    assert y.shape==(2,1,512,512) and torch.isfinite(y).all()
    loss=y.square().mean();loss.backward()
    assert all(v==1 for v in calls.values())
    shapes_b2={k:list(v) for k,v in shapes.items()}
    calls_b2=dict(calls)
    assert len([m for m in u.modules() if isinstance(m,LinearAttention)])==8
    assert len([m for m in u.modules() if isinstance(m,Attention)])==1
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in u.parameters())
    attention={}
    for name,module in u.named_modules():
        if isinstance(module,(LinearAttention,Attention)):
            norms=[float(p.grad.norm()) for p in module.parameters()]
            assert norms and all(torch.isfinite(torch.tensor(norms))) and sum(norms)>0
            attention[name]=sum(norms)
    del y,loss
    torch.cuda.synchronize(device)
    peak=int(torch.cuda.max_memory_allocated(device))
    u.eval()
    with torch.no_grad():
        inp=torch.zeros(1,2,512,512,device=device)
        tm=torch.tensor([371],device=device)
        a=u(inp,tm)
        b=u(inp,tm+1)
        changed=inp.clone();changed[:,1]=0.1
        d=u(changed,tm)
        timestep_delta=float((b-a).abs().mean())
        condition_delta=float((d-a).abs().mean())
    assert timestep_delta>0 and condition_delta>0
    for h in handles:h.remove()
    report={'official_commit':'524caea34cedb3733aa5bf6c27dc1cd6e0469216',
      'candidate_parameters':p,'control_parameters':c,'parameter_delta':p-c,
      'encoder_stages':len(u.downs),'decoder_stages':len(u.ups),
      'linear_attention_count':8,'full_attention_count':1,
      'feature_shapes':shapes_b2,'forward_calls':calls_b2,'attention_grad_norm':attention,
      'forward_b2_1x512_pass':True,'backward_finite':True,
      'timestep_sensitivity':timestep_delta,'condition_sensitivity':condition_delta,
      'peak_vram_bytes':peak,'L506_access_count':0,'synthetic_only':True}
    Path('readiness.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('feature_shapes','forward_calls','attention_grad_norm')}),flush=True)
if __name__=='__main__':main()
