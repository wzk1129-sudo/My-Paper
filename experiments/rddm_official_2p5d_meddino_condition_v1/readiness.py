"""Hard readiness gate, including permitted B=1/accum=2 fallback only for CUDA OOM."""
import argparse,gc,json,traceback,datetime
from pathlib import Path
import torch
from torch.nn import functional as F
from model import build_model,load_parent,native_attention_grad,STAGES,CHANNELS
from src.official_rddm_ct import LinearAttention,Attention,ResnetBlock
from provenance import ROOT,audit_sources,write_json
from mayo2016_smallset_v1.runtime import seed_everything
from context_data import ContextPairs,ContextBatchStream,ContextDataGuard

def finite_nonzero(module):
    grads=[p.grad for p in module.parameters()]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
    norm=float(torch.stack([g.norm() for g in grads]).norm())
    assert norm>0
    return norm
def clean():
    gc.collect();torch.cuda.empty_cache()
def trial(batch,cfg):
    seed_everything(20260829)
    device=torch.device("cuda:0")
    with ContextDataGuard(cfg) as guard:
        data=ContextPairs(cfg,"train");data.bind_guard(guard)
        stream=ContextBatchStream(data,20260829)
        prev,center,next_,target,metadata=stream.next(batch)
        access=guard.report()
    assert access["L506_opens"]==access["external_memory_opens"]==access["neighbor_target_opens"]==access["blocked_attempts"]==0
    prev,center,next_,target=[x.to(device) for x in (prev,center,next_,target)]
    t=torch.tensor([102,842][:batch],device=device,dtype=torch.long)
    noise=torch.randn_like(target)
    old=load_parent().to(device).eval()
    xt=old.q_sample(2*target-1,2*(center-target),t,noise=noise)
    inp=torch.cat((xt,2*center-1),1)
    time=old.alphas_cumsum[t]*old.num_timesteps
    with torch.no_grad():reference=old.model.unet0(inp,time).cpu()
    del old;clean()
    control=build_model(False).to(device).eval()
    u=control.model.unet0
    u.set_context(prev,center,next_)
    assert torch.count_nonzero(u.init_conv.weight[:,[1,3]])==0
    with torch.no_grad():control_output=u(inp,time).cpu()
    error_2d=float((reference-control_output).abs().max())
    assert error_2d<=1e-6,("2D vs 2.5D",error_2d)
    shared={k:v.cpu().clone() for k,v in u.state_dict().items()}
    del control,u;clean()
    candidate=build_model(True).to(device).train()
    u=candidate.model.unet0
    assert all(torch.equal(v,u.state_dict()[k].cpu()) for k,v in shared.items())
    del shared
    assert all(not p.requires_grad for p in u.med.parameters()) and not u.med.training
    native_calls={};condition_calls={s:0 for s in STAGES};pointer_checks={s:0 for s in STAGES};last={};shapes={}
    handles=[]
    for n,m in u.named_modules():
        if isinstance(m,(LinearAttention,Attention)):
            native_calls[n]=0
            def entered(mod,args,out,key=n):
                native_calls[key]+=1
            handles.append(m.register_forward_hook(entered))
    wrappers=[x[2] for x in u.downs]+[u.mid_attn]+[x[2] for x in u.ups]
    for s,w in zip(STAGES,wrappers):
        block=u.conditions[s];block.record=True
        def conditioned(mod,args,out,stage=s):
            condition_calls[stage]+=1;last[stage]=out.data_ptr();shapes[stage]=list(out.shape)
        def before_attention(mod,args,stage=s):
            assert last[stage]==args[0].data_ptr();pointer_checks[stage]+=1
        handles.append(block.register_forward_hook(conditioned))
        handles.append(w.register_forward_pre_hook(before_attention))
    u.set_context(prev,center,next_)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():candidate_output=u(inp,time).cpu()
    error_pair=float((candidate_output-control_output).abs().max())
    assert error_pair<=1e-6,("Control vs Candidate",error_pair)
    assert all(v==1 for v in native_calls.values()) and len(native_calls)==9
    assert sum(isinstance(m,LinearAttention) for m in u.modules())==8
    assert sum(isinstance(m,Attention) for m in u.modules())==1
    assert sum(isinstance(m,ResnetBlock) for m in u.modules())==19
    assert all(condition_calls[s]==pointer_checks[s]==1 for s in STAGES)
    zero_activity={s:u.conditions[s].rows[0] for s in STAGES}
    assert all(r["delta_feature_rms"]==r["gamma_rms"]==r["beta_rms"]==0 for r in zero_activity.values())
    assert u.feature_shapes==[[batch,768,32,32]]*4 and u.anatomy_shape==[batch,128,32,32]
    for block in u.conditions.values():block.record=False
    optimizer=torch.optim.Adam([p for p in candidate.parameters() if p.requires_grad],lr=2e-4,betas=(.9,.99),weight_decay=1e-9)
    loss=candidate.p_losses([2*target-1,2*center-1],t,noise=noise)[0]
    assert torch.isfinite(loss)
    loss.backward()
    attention_first=native_attention_grad(candidate)
    projections={s:finite_nonzero(u.conditions[s].projection) for s in STAGES}
    upstream_first={n:float(p.grad.norm()) if p.grad is not None else None for n,p in u.named_parameters()
        if n.startswith(("layer_adapters.","fusion."))}
    assert all(v==0 for v in upstream_first.values())
    assert all(p.grad is None and not p.requires_grad for p in u.med.parameters())
    torch.nn.utils.clip_grad_norm_([p for p in candidate.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    loss2=candidate.p_losses([2*target-1,2*center-1],t,noise=noise)[0]
    assert torch.isfinite(loss2);loss2.backward()
    attention_second=native_attention_grad(candidate)
    upstream_second={**{"layer_"+str(i):finite_nonzero(m) for i,m in enumerate(u.layer_adapters)},
        "fusion":finite_nonzero(u.fusion),**{s:finite_nonzero(u.conditions[s].adapter) for s in STAGES}}
    assert all(torch.isfinite(p).all() for p in candidate.parameters())
    assert all(p.grad is None and not p.requires_grad for p in u.med.parameters()) and not u.med.training
    peak=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved()
    assert peak<24*1024**3
    for h in handles:h.remove()
    result={"status":"PASS","batch":batch,"accum":2//batch,"effective_batch":2,
        "data_access_audit":access,"first_samples":metadata,"timestep":t.cpu().tolist(),
        "feature_shapes":u.feature_shapes,"fused_anatomy_shape":u.anatomy_shape,
        "condition_shapes":shapes,"native_attention_forward_calls":native_calls,
        "condition_forward_calls":condition_calls,"attention_receives_condition_output":pointer_checks,
        "condition_step0":zero_activity,"two_dimensional_vs_two_point_five_max_abs_diff":error_2d,
        "control_vs_candidate_step0_max_abs_diff":error_pair,
        "first_backward_projection_gradient_norms":projections,"first_backward_anatomy_gradient_norms":upstream_first,
        "second_backward_anatomy_gradient_norms":upstream_second,"attention_gradient_norms_first":attention_first,
        "attention_gradient_norms_second":attention_second,"meddino_frozen":True,"meddino_grad_none":True,
        "peak_allocated_bytes":peak,"peak_reserved_bytes":reserved,
        "parameters_total":sum(p.numel() for p in candidate.parameters()),
        "parameters_trainable":sum(p.numel() for p in candidate.parameters() if p.requires_grad),
        "loss_first":float(loss.detach()),"loss_second":float(loss2.detach())}
    print("READINESS_PASS "+json.dumps({"batch":batch,"peak_gib":peak/1024**3,"init_diff":error_pair}),flush=True)
    return result

def main():
    torch.set_num_threads(4)
    cfg,source=audit_sources()
    fallback=None
    try:result=trial(2,cfg)
    except torch.cuda.OutOfMemoryError:
        fallback={"batch2":"CUDA_OOM","traceback":traceback.format_exc()}
        write_json(ROOT/"READINESS_B2_OOM.json",fallback)
        clean()
        result=trial(1,cfg)
    result.update({"official_source_commit":source["official_rddm_commit"],
        "meddino_source_commit":source["meddino_commit"],"meddino_checkpoint_sha256":source["meddino_checkpoint_sha256"],
        "fallback":fallback,"checks_passed":20})
    write_json(ROOT/"READINESS.json",result)
    write_json(ROOT/"BATCH_CONFIG.json",{"batch":result["batch"],"accum":result["accum"],"effective_batch":2})
if __name__=="__main__":
    try:main()
    except Exception:
        write_json(ROOT/("READINESS_FAILURE_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+".json"),{"status":"FAIL","traceback":traceback.format_exc()})
        raise
