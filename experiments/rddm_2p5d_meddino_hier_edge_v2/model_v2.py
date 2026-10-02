"""Hierarchical spatial MedDINO sources, fixed center-LDCT edges and unchanged residual FiLM."""
import bootstrap
import gc,math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
import model as baseline
from model import STAGES,CHANNELS,GLOBAL_STAGES,ContextUnet
MAP={"E0":"S_local","E1":"S_local","E2":"S_mid","E3":"S_semantic","MID":"S_semantic","D0":"S_semantic","D1":"S_semantic","D2":"S_local","D3":"S_local"}
def statistics(x):
    v=x.detach().float()
    return {"shape":list(v.shape),"mean":float(v.mean()),"std":float(v.std(unbiased=False)),"max_abs":float(v.abs().max()),"rms":float(v.square().mean().sqrt()),"finite":bool(torch.isfinite(v).all())}
class FixedEdges(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmas=(.8,1.6,3.2);self.radii=tuple(math.ceil(3*s) for s in self.sigmas)
        for i,(sigma,radius) in enumerate(zip(self.sigmas,self.radii)):
            axis=torch.arange(-radius,radius+1,dtype=torch.float32)
            k=torch.exp(-axis.square()/(2*sigma*sigma));k=k/k.sum()
            self.register_buffer("gaussian_"+str(i),k.reshape(1,1,1,-1))
        sx=torch.tensor([[-3,0,3],[-10,0,10],[-3,0,3]],dtype=torch.float32)/32
        self.register_buffer("scharr_x",sx.reshape(1,1,3,3))
        self.register_buffer("scharr_y",sx.T.reshape(1,1,3,3))
        self.register_buffer("laplacian",torch.tensor([[0,1,0],[1,-4,1],[0,1,0]],dtype=torch.float32).reshape(1,1,3,3))
    def blur(self,x,index):
        r=self.radii[index];k=getattr(self,"gaussian_"+str(index))
        x=F.conv2d(F.pad(x,(r,r,0,0),mode="reflect"),k)
        return F.conv2d(F.pad(x,(0,0,r,r),mode="reflect"),k.transpose(-1,-2))
    @torch.no_grad()
    def raw(self,center):
        assert center.shape[1:]==(1,512,512)
        results=[]
        for i in range(3):
            smooth=self.blur(center,i)
            padded=F.pad(smooth,(1,1,1,1),mode="reflect")
            gx=F.conv2d(padded,self.scharr_x);gy=F.conv2d(padded,self.scharr_y)
            results.append(torch.sqrt(gx.square()+gy.square()))
            if i==0:lap=F.conv2d(padded,self.laplacian)
        return tuple(results)+(lap,)
    @torch.no_grad()
    def forward(self,center):
        raw=self.raw(center)
        normalized=tuple(x/(x.square().mean(dim=(-2,-1),keepdim=True).sqrt()+1e-6) for x in raw)
        assert all(bool(torch.isfinite(x).all()) for x in normalized)
        return normalized
class EdgeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fixed=FixedEdges()
        self.branches=nn.ModuleList([nn.Sequential(nn.Conv2d(1,32,3,padding=1),nn.GroupNorm(8,32),nn.SiLU()) for _ in range(4)])
        self.fuse=nn.Sequential(nn.Conv2d(128,128,3,padding=1),nn.GroupNorm(8,128),nn.SiLU())
        self.project_local=nn.Sequential(nn.Conv2d(128,128,1),nn.GroupNorm(8,128))
        self.project_semantic=nn.Sequential(nn.Conv2d(128,128,1),nn.GroupNorm(8,128))
        self.record=False;self.full_shape=None;self.preview_heatmap=None
    def _features(self,*maps):
        e=self.fuse(torch.cat([b(x) for b,x in zip(self.branches,maps)],dim=1))
        self.full_shape=list(e.shape)
        if self.record:self.preview_heatmap=e.detach().mean(dim=1,keepdim=True).cpu()
        small=F.interpolate(e,size=(32,32),mode="bilinear",align_corners=False)
        return self.project_local(small),self.project_semantic(small)
    def forward(self,*maps):
        if self.training and torch.is_grad_enabled():
            return checkpoint(self._features,*maps,use_reentrant=False,preserve_rng_state=True)
        return self._features(*maps)
class HierarchyFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.local=nn.Sequential(nn.Conv2d(128,128,1),nn.SiLU())
        self.semantic=nn.Sequential(nn.Conv2d(128,128,1),nn.SiLU())
        self.edge=EdgeEncoder()
        self.raw_edges=None;self.record=False;self.latest_stats={};self.latest_shapes={}
        self.lambda_local=.1;self.lambda_semantic=.1
    def forward(self,projected):
        assert projected.shape[1:]==(256,32,32) and self.raw_edges is not None
        al=self.local(projected[:,:128]);ase=self.semantic(projected[:,128:])
        self.edge.record=self.record
        el,ese=self.edge(*self.raw_edges)
        sl=.1*(al*el);ss=.1*(ase*ese);sm=.5*sl+.5*ss
        fields={"A_local":al,"A_semantic":ase,"E_local":el,"E_semantic":ese,"S_local":sl,"S_semantic":ss,"S_mid":sm}
        self.latest_shapes={k:list(x.shape) for k,x in fields.items()}
        if self.record:self.latest_stats={k:statistics(x) for k,x in fields.items()}
        return {"S_local":sl,"S_semantic":ss,"S_mid":sm}
class SpatialFiLM(nn.Module):
    """Same existing adapter/projection shapes; Linear adapters applied at each spatial position."""
    def __init__(self,original):
        super().__init__()
        self.norm=original.norm
        self.adapter=original.adapter
        self.projection=original.projection
        self.linear_adapter=original.global_mode
        self.record=False;self.rows=[];self.force_off=False
        self.last_source_shape=None
    def _affine(self,x,s):
        source=F.interpolate(s,size=x.shape[-2:],mode="bilinear",align_corners=False)
        self.last_source_shape=list(source.shape)
        if self.linear_adapter:
            affine=self.projection(self.adapter(source.permute(0,2,3,1))).permute(0,3,1,2)
        else:affine=self.projection(self.adapter(source))
        gamma,beta=affine.chunk(2,dim=1)
        delta=gamma*self.norm(x)+beta
        if self.force_off:delta=torch.zeros_like(delta)
        if self.record:
            base=baseline.rms(x);d=baseline.rms(delta)
            self.rows.append({"delta_feature_rms":d,"base_feature_rms":base,"delta_base_ratio":d/max(base,1e-30),"gamma_rms":baseline.rms(gamma),"beta_rms":baseline.rms(beta)})
        return x+delta
    def forward(self,x,s):
        if self.training and torch.is_grad_enabled():
            return checkpoint(self._affine,x,s,use_reentrant=False,preserve_rng_state=True)
        return self._affine(x,s)
class HierEdgeUnet(ContextUnet):
    def __init__(self):
        super().__init__(anatomy=True)
        self.conditions=nn.ModuleDict({s:SpatialFiLM(self.conditions[s]) for s in STAGES})
        self.fusion=HierarchyFusion()
    def set_context(self,prev,center,next_,patches=None):
        super().set_context(prev,center,next_,patches)
        self.fusion.raw_edges=self.fusion.edge.fixed(center.detach())
    def clear_context(self):
        super().clear_context()
        self.fusion.raw_edges=None
    def modulate(self,stage,x,a):return self.conditions[stage](x,a[MAP[stage]])
    def forward(self,x,time,x_self_cond=None):
        assert not self.self_condition and self._context is not None and x.shape[1]==2
        prev,next_=self._context
        x=torch.cat((x[:,:1],prev,x[:,1:2],next_),dim=1)
        a=None
        if self.anatomy:
            assert self._patches is not None
            a=self.fusion(torch.cat([m(p) for m,p in zip(self.layer_adapters,self._patches)],dim=1))
            self.anatomy_shape={k:list(v.shape) for k,v in a.items()}
        x=self.init_conv(x)
        r=x.clone()
        t=self.time_mlp(time)
        h=[]
        for i,(block1,block2,attn,downsample) in enumerate(self.downs):
            x=block1(x,t);h.append(x)
            x=block2(x,t)
            x=self.modulate("E"+str(i),x,a)
            x=attn(x);h.append(x)
            x=downsample(x)
        x=self.mid_block1(x,t)
        x=self.modulate("MID",x,a)
        x=self.mid_attn(x)
        x=self.mid_block2(x,t)
        for i,(block1,block2,attn,upsample) in enumerate(self.ups):
            x=torch.cat((x,h.pop()),dim=1);x=block1(x,t)
            x=torch.cat((x,h.pop()),dim=1);x=block2(x,t)
            x=self.modulate("D"+str(i),x,a)
            x=attn(x);x=upsample(x)
        assert not h
        x=torch.cat((x,r),dim=1)
        return self.final_conv(self.final_res_block(x,t))
def build_v2():
    result=baseline.load_parent()
    old=result.model.unet0.state_dict()
    u=HierEdgeUnet()
    stem=old["init_conv.weight"]
    new=torch.zeros((64,4,7,7),dtype=stem.dtype)
    new[:,0]=stem[:,0];new[:,2]=stem[:,1];old["init_conv.weight"]=new
    missing=u.load_state_dict(old,strict=False)
    assert not missing.unexpected_keys
    assert all(k.startswith(("med.","layer_adapters.","fusion.","conditions.")) for k in missing.missing_keys)
    result.model.unet0=u
    assert result.objective=="pred_res" and result.sampling_timesteps==2 and result.sum_scale==.01
    gc.collect()
    return result
