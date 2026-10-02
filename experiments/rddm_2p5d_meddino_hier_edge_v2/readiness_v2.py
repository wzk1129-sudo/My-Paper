"""Mandatory V2 readiness: topology, spatial sources, fixed edges, real batch, zero init, finite backward."""
import bootstrap
import ast,gc,inspect,json,math,os,sys,time,traceback,hashlib
from pathlib import Path
import torch,numpy as np
from torch import nn
import model as baseline
from model_v2 import build_v2,FixedEdges,SpatialFiLM,HierarchyFusion,EdgeEncoder,STAGES,MAP,statistics
from context_data import ContextPairs,ContextBatchStream,ContextDataGuard
from fast_dev.protocol import CONFIG,check_config
from mayo2016_smallset_v1.runtime import seed_everything
from train import tensor_state_hash
from src.official_rddm_ct import LinearAttention,Attention,ResnetBlock
from med_loader import sha,MED_CK,MED_SHA
from bootstrap import ROOT,BASE
def write(name,obj):(ROOT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def clean():gc.collect();torch.cuda.empty_cache()
def core_ast(cls):
    t=ast.parse(inspect.getsource(cls))
    body=next(n for n in t.body[0].body if isinstance(n,ast.FunctionDef) and n.name=="forward").body
    i=next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=="init_conv")
    return ast.dump(ast.Module(body=body[i:],type_ignores=[]),include_attributes=False)
def grad_norm(module):
    ps=[p.grad for p in module.parameters() if p.requires_grad]
    assert ps and all(g is not None and bool(torch.isfinite(g).all()) for g in ps)
    return math.sqrt(sum(float(g.square().sum()) for g in ps))
def main():
    from model_v2 import HierEdgeUnet
    assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
    cfg=json.loads(CONFIG.read_text());check_config(cfg)
    v2=json.loads((ROOT/'CONFIG.json').read_text());bp=json.loads((ROOT/'BASELINE_PROVENANCE.json').read_text())
    torch.set_num_threads(2);seed_everything(20260829);device=torch.device('cuda:0')
    for name,h in bp['baseline_source_sha256'].items():assert sha(BASE/name)==h
    assert sha(baseline.PARENT_CHECKPOINT)==bp['parent_checkpoint_sha256']
    assert sha(MED_CK)==MED_SHA
    assert core_ast(HierEdgeUnet)==core_ast(baseline.ContextUnet)
    # No condition feature pooling or flattening; normalization/diagnostic reductions are scalar statistics only.
    condition_text='\n'.join(inspect.getsource(c) for c in [HierarchyFusion,EdgeEncoder,SpatialFiLM])
    forbidden=['adaptive_avg_pool','AdaptiveAvgPool','avg_pool','global_pool','.flatten(','reshape(-1']
    assert not any(s in condition_text for s in forbidden)
    environment={'python':sys.version,'executable':sys.executable,'torch':torch.__version__,'cuda':torch.version.cuda,'numpy':np.__version__,'gpu':torch.cuda.get_device_name(0),'cudnn':torch.backends.cudnn.version(),'cudnn_deterministic':torch.backends.cudnn.deterministic,'cudnn_benchmark':torch.backends.cudnn.benchmark,'tf32':False,'amp':False,'cublas_workspace_config':os.environ['CUBLAS_WORKSPACE_CONFIG']}
    old=json.load(open(BASE/'ENVIRONMENT.json'))
    for key in ['python','executable','torch','cuda','numpy','gpu']:assert environment[key]==old[key],(key,environment[key],old[key])
    write('ENVIRONMENT.json',environment)
    with ContextDataGuard(cfg) as guard:
        data=ContextPairs(cfg,'train');data.bind_guard(guard)
        prev,center,next_,target,meta=ContextBatchStream(data,20260829).next(2)
        val=ContextPairs(cfg,'validation');val.bind_guard(guard)
        diagnostic=val.load(32)
        access=guard.report()
    assert meta[0]['patient_id'] in ['L067','L143'] and meta[1]['patient_id'] in ['L067','L143']
    assert diagnostic[-1]['patient_id']=='L333' and diagnostic[-1]['slice_index']==309
    assert access['L506_opens']==access['external_memory_opens']==access['neighbor_target_opens']==access['blocked_attempts']==0
    prev,center,next_,target=[x.to(device) for x in [prev,center,next_,target]]
    t=torch.tensor([102,842],device=device);noise=torch.randn_like(target)
    control=baseline.build_model(False).to(device).eval();u=control.model.unet0
    u.set_context(prev,center,next_)
    xt=control.q_sample(2*target-1,2*(center-target),t,noise=noise)
    inp=torch.cat((xt,2*center-1),dim=1);times=control.alphas_cumsum[t]*control.num_timesteps
    with torch.no_grad():ref=u(inp,times).cpu()
    shared=tensor_state_hash(baseline.state_without_med(control));u.clear_context()
    del control,u;clean()
    model=build_v2().to(device).train();u=model.model.unet0
    params=[p for p in model.parameters() if p.requires_grad]
    med_before=tensor_state_hash(u.med.state_dict())
    assert med_before==bp['baseline_initial_state']['meddino_initial_weights_sha256']
    core={k:v for k,v in baseline.state_without_med(model).items() if not any(x in k for x in ['.layer_adapters.','.fusion.','.conditions.'])}
    core_hash=tensor_state_hash(core);del core
    assert core_hash==bp['baseline_initial_state']['shared_initial_parameters_sha256']
    assert sum(isinstance(m,ResnetBlock) for m in u.modules())==19
    assert sum(isinstance(m,LinearAttention) for m in u.modules())==8
    assert sum(isinstance(m,Attention) for m in u.modules())==1
    kernels={k:{'shape':list(v.shape),'requires_grad':v.requires_grad,'sum':float(v.sum()),'sha256':hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest()} for k,v in u.fusion.edge.fixed.named_buffers()}
    assert len(kernels)==6 and all(not v['requires_grad'] for v in kernels.values())
    assert all(p.grad is None and not p.requires_grad for p in u.med.parameters()) and not u.med.training
    for s,b in u.conditions.items():
        assert b.norm.num_groups==8 and b.norm.affine is False
        assert torch.count_nonzero(b.projection.weight)==torch.count_nonzero(b.projection.bias)==0
        b.record=True;b.rows=[]
    u.fusion.record=True;u.set_context(prev,center,next_)
    shapes={};calls={s:0 for s in STAGES};last={};pointer_checks={s:0 for s in STAGES};hooks=[]
    wrappers=[v[2] for v in u.downs]+[u.mid_attn]+[v[2] for v in u.ups]
    for s,w in zip(STAGES,wrappers):
        def conditioned(m,args,out,stage=s):
            shapes[stage]={'feature':list(args[0].shape),'source':list(args[1].shape),'resized_source':m.last_source_shape}
            calls[stage]+=1;last[stage]=out.data_ptr()
        def before(m,args,stage=s):assert args[0].data_ptr()==last[stage];pointer_checks[stage]+=1
        hooks.append(u.conditions[s].register_forward_hook(conditioned));hooks.append(w.register_forward_pre_hook(before))
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():result=u(inp,times).cpu()
    difference=float((result-ref).abs().max());assert difference<=1e-6
    expected=[2,128,32,32]
    assert all(v==expected for k,v in u.fusion.latest_shapes.items())
    numerics=u.fusion.latest_stats.copy()
    shape_checks=u.fusion.latest_shapes.copy();full_edge_shape=u.fusion.edge.full_shape.copy()
    assert all(r['finite'] and r['max_abs']<1000 for r in numerics.values())
    assert u.fusion.edge.full_shape==[2,128,512,512]
    assert all(calls[s]==pointer_checks[s]==1 for s in STAGES)
    zero={s:b.rows[0] for s,b in u.conditions.items()}
    assert all(r['delta_feature_rms']==r['gamma_rms']==r['beta_rms']==0 for r in zero.values())
    for h in hooks:h.remove()
    for b in u.conditions.values():b.record=False;b.rows=[]
    u.fusion.record=False
    # Step-0 visualization, only diagnostic L333 center 309.
    u.eval();dv=[torch.from_numpy(x).unsqueeze(0).to(device) for x in diagnostic[:4]]
    with torch.no_grad():
        raw=u.fusion.edge.fixed.raw(dv[1]);u.fusion.record=True;u.set_context(*dv[:3])
        u(torch.cat((dv[1]*2-1,dv[1]*2-1),dim=1),torch.tensor([1.],device=device))
        heat=u.fusion.edge.preview_heatmap[0,0].numpy()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    arrays=[dv[1][0,0].cpu().numpy()*4096-1024]+[x[0,0].cpu().numpy() for x in raw]+[heat]
    titles=['L333 Fast64 center 309, LDCT HU','Gaussian 0.8 + Scharr','Gaussian 1.6 + Scharr','Gaussian 3.2 + Scharr','Gaussian 0.8 + Laplacian (signed)','Fused edge E: channel mean']
    fig,axes=plt.subplots(2,3,figsize=(15,10))
    for i,(a,ax,title) in enumerate(zip(arrays,axes.flat,titles)):
        if i==0:ax.imshow(a,cmap='gray',vmin=-160,vmax=240)
        elif i==4:
            lim=float(np.quantile(np.abs(a),.99));ax.imshow(a,cmap='coolwarm',vmin=-lim,vmax=lim)
        else:ax.imshow(a,cmap='magma',vmin=float(np.quantile(a,.01)),vmax=float(np.quantile(a,.99)))
        ax.set_title(title);ax.axis('off')
    fig.tight_layout();fig.savefig(ROOT/'visualizations/L333_fast64_309_edges.png',dpi=120);plt.close(fig)
    u.clear_context()
    u.train();u.fusion.record=False;u.set_context(prev,center,next_)
    optimizer=torch.optim.Adam(params,lr=2e-4,betas=(.9,.99),weight_decay=1e-9)
    loss=model.p_losses([2*target-1,2*center-1],t,noise=noise)[0];assert torch.isfinite(loss)
    loss.backward()
    first_projection={s:grad_norm(b.projection) for s,b in u.conditions.items()}
    assert all(v>0 for v in first_projection.values())
    assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in params)
    attention_first=baseline.native_attention_grad(model)
    assert all(p.grad is None and not p.requires_grad for p in u.med.parameters())
    assert all(b.grad is None and not b.requires_grad for b in u.fusion.edge.fixed.buffers())
    total_grad=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True))
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    loss2=model.p_losses([2*target-1,2*center-1],t,noise=noise)[0];assert torch.isfinite(loss2);loss2.backward()
    second={**{'layer_'+str(i):grad_norm(m) for i,m in enumerate(u.layer_adapters)},
            'local':grad_norm(u.fusion.local),'semantic':grad_norm(u.fusion.semantic),
            **{'edge_branch_'+str(i):grad_norm(m) for i,m in enumerate(u.fusion.edge.branches)},
            'edge_fusion':grad_norm(u.fusion.edge.fuse),'edge_local':grad_norm(u.fusion.edge.project_local),
            'edge_semantic':grad_norm(u.fusion.edge.project_semantic)}
    assert all(v>0 for v in second.values())
    assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in params)
    assert all(p.grad is None and not p.requires_grad for p in u.med.parameters()) and not u.med.training
    assert all(b.grad is None and not b.requires_grad for b in u.fusion.edge.fixed.buffers())
    assert tensor_state_hash(u.med.state_dict())==med_before
    attention_second=baseline.native_attention_grad(model)
    peak=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved()
    u.clear_context();optimizer.zero_grad(set_to_none=True)
    result={'status':'PASS','batch':2,'accum':1,'shape_test':shape_checks,'edge_full_shape':full_edge_shape,
            'core_forward_ast_identical':True,'official_topology':{'ResnetBlock':19,'LinearAttention':8,'Attention':1},'stage_map':MAP,
            'stage_shapes':shapes,'condition_calls':calls,'native_attention_receives_condition_output':pointer_checks,
            'spatial_preservation':True,'fixed_kernels':kernels,'zero_init':zero,'step0_max_abs_difference':difference,
            'numerical_statistics':numerics,'loss_first':float(loss.detach()),'loss_second':float(loss2.detach()),'grad_norm_first':total_grad,
            'projection_gradient_norm_first':first_projection,'feature_gradient_norm_second':second,'native_attention_gradient_norm_first':attention_first,
            'native_attention_gradient_norm_second':attention_second,'meddino_hash_before':med_before,'meddino_hash_after':tensor_state_hash(u.med.state_dict()),
            'meddino_frozen':True,'meddino_grad_none':True,'fixed_kernel_grad_none':True,'parameters_finite':all(bool(torch.isfinite(p).all()) for p in params),
            'data_access':access,'shared_initial_parameters_sha256':core_hash,'baseline_shared_initial_parameters_sha256':bp['baseline_initial_state']['shared_initial_parameters_sha256'],
            'peak_allocated_bytes':peak,'peak_reserved_bytes':reserved,'parameters_trainable':sum(p.numel() for p in params),'parameters_total':sum(p.numel() for p in model.parameters()),
            'diagnostic_patient':'L333','diagnostic_slice':309,'visualization':'visualizations/L333_fast64_309_edges.png',
            'visual_review_status':'PENDING','nonfinite_count':0}
    write('READINESS.json',result);print('READINESS_PASS '+json.dumps({k:result[k] for k in ['status','step0_max_abs_difference','peak_allocated_bytes','parameters_trainable']}),flush=True)
if __name__=='__main__':
    try:main()
    except Exception as e:
        write('READINESS_FAILURE_'+time.strftime('%Y%m%d_%H%M%S')+'.json',{'status':'FAIL','exception':type(e).__name__,'traceback':traceback.format_exc()})
        raise
