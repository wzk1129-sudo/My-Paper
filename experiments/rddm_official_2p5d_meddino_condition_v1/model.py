"""Official topology with LDCT-only 2.5D stem and pre-native-attention anatomy modulation."""
import gc
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from src.official_rddm_ct import Unet, build_baseline, LinearAttention, Attention
from med_loader import official_med, MEAN_HU, STD_HU
PARENT_CHECKPOINT=Path("/home/xujian/projects/CoreDiff_official_rddm_ct_adapt_v1/experiments/rddm_official_ct_adapt_v1/runs/candidate_8k_20261001/checkpoints/step_8000.pt")
STAGES=("E0","E1","E2","E3","MID","D0","D1","D2","D3")
CHANNELS=(64,64,128,256,512,512,256,128,64)
GLOBAL_STAGES={"E0","E1","D2","D3"}

def rms(x):
    return float(x.detach().float().square().mean().sqrt())

class AnatomyCondition(nn.Module):
    def __init__(self,c,global_mode):
        super().__init__()
        self.global_mode=global_mode
        self.norm=nn.GroupNorm(8,c,affine=False)
        if global_mode:
            self.adapter=nn.Sequential(nn.Linear(128,256),nn.SiLU())
            self.projection=nn.Linear(256,2*c)
        else:
            self.adapter=nn.Sequential(nn.Conv2d(128,min(c,128),3,padding=1),nn.SiLU())
            self.projection=nn.Conv2d(min(c,128),2*c,1)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        self.record=False
        self.rows=[]
        self.force_off=False
    def forward(self,x,a):
        if self.global_mode:
            source=F.adaptive_avg_pool2d(a,1).flatten(1)
            affine=self.projection(self.adapter(source))[:,:,None,None]
        else:
            source=F.interpolate(a,size=x.shape[-2:],mode="bilinear",align_corners=False)
            affine=self.projection(self.adapter(source))
        gamma,beta=affine.chunk(2,dim=1)
        delta=gamma*self.norm(x)+beta
        if self.force_off:delta=torch.zeros_like(delta)
        if self.record:
            base=rms(x);d=rms(delta)
            self.rows.append({"delta_feature_rms":d,"base_feature_rms":base,
                "delta_base_ratio":d/max(base,1e-30),"gamma_rms":rms(gamma),"beta_rms":rms(beta)})
        return x+delta

class ContextUnet(Unet):
    def __init__(self,anatomy=False):
        super().__init__(dim=64,dim_mults=(1,2,4,8),channels=1,condition=True)
        self.init_conv=nn.Conv2d(4,64,7,padding=3)
        self.anatomy=anatomy
        self._context=None
        self._patches=None
        if anatomy:
            self.med=official_med()
            self.layer_adapters=nn.ModuleList([nn.Conv2d(768,64,1) for _ in range(4)])
            self.fusion=nn.Sequential(nn.Conv2d(256,128,1),nn.SiLU())
            self.conditions=nn.ModuleDict({s:AnatomyCondition(c,s in GLOBAL_STAGES) for s,c in zip(STAGES,CHANNELS)})
        self.feature_shapes=None
        self.anatomy_shape=None
    def train(self,mode=True):
        super().train(mode)
        if self.anatomy:self.med.eval().requires_grad_(False)
        return self
    @torch.no_grad()
    def extract(self,center):
        assert self.anatomy and center.shape[1:]==(1,512,512)
        self.med.eval()
        center_neg=2*center-1
        hu=((center_neg+1)*2048-1024).clamp(-1000,1000)
        med_input=((hu-MEAN_HU)/STD_HU).repeat(1,3,1,1)
        tokens=self.med.get_intermediate_layers(med_input,n=[2,5,8,11],norm=True,
            return_class_token=False,return_extra_tokens=False)
        assert len(tokens)==4 and all(t.shape==(center.shape[0],1024,768) for t in tokens)
        maps=tuple(t.transpose(1,2).reshape(center.shape[0],768,32,32).contiguous().detach() for t in tokens)
        self.feature_shapes=[list(t.shape) for t in maps]
        return maps
    def set_context(self,prev,center,next_,patches=None):
        assert prev.shape==center.shape==next_.shape and center.shape[1:]==(1,512,512)
        self._context=(2*prev-1,2*next_-1)
        if self.anatomy:self._patches=self.extract(center) if patches is None else tuple(t.detach() for t in patches)
    def clear_context(self):
        self._context=None
        self._patches=None
    def modulate(self,stage,x,a):
        return self.conditions[stage](x,a) if self.anatomy else x
    def forward(self,x,time,x_self_cond=None):
        assert not self.self_condition and self._context is not None and x.shape[1]==2
        prev,next_=self._context
        x=torch.cat((x[:,:1],prev,x[:,1:2],next_),dim=1)
        a=None
        if self.anatomy:
            assert self._patches is not None
            a=self.fusion(torch.cat([m(p) for m,p in zip(self.layer_adapters,self._patches)],dim=1))
            self.anatomy_shape=list(a.shape)
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

def load_parent():
    model=build_baseline()
    ck=torch.load(PARENT_CHECKPOINT,map_location="cpu",weights_only=False)
    assert ck["optimizer_step"]==ck["ema_updates"]==8000
    model.load_state_dict(ck["ema_state_dict"],strict=True)
    assert model.objective=="pred_res" and model.sampling_timesteps==2 and model.sum_scale==.01
    del ck
    gc.collect()
    return model

def build_model(anatomy=False):
    model=load_parent()
    old=model.model.unet0.state_dict()
    unet=ContextUnet(anatomy)
    stem=old["init_conv.weight"]
    new=torch.zeros((64,4,7,7),dtype=stem.dtype)
    new[:,0]=stem[:,0];new[:,2]=stem[:,1]
    old["init_conv.weight"]=new
    result=unet.load_state_dict(old,strict=False)
    assert not result.unexpected_keys
    assert all(k.startswith(("med.","layer_adapters.","fusion.","conditions.")) for k in result.missing_keys)
    if not anatomy:assert not result.missing_keys
    model.model.unet0=unet
    return model

def native_attention_grad(model):
    result={}
    for n,m in model.model.unet0.named_modules():
        if isinstance(m,(LinearAttention,Attention)):
            grads=[p.grad for p in m.parameters()]
            assert all(g is not None and torch.isfinite(g).all() for g in grads),n
            norm=float(torch.stack([g.norm() for g in grads]).norm())
            assert norm>0,n
            result[n]=norm
    assert len(result)==9
    return result

def state_without_med(model):
    return {n:t.detach().cpu() for n,t in model.state_dict().items() if ".med." not in n}

def load_state_without_med(model,state):
    result=model.load_state_dict(state,strict=False)
    assert not result.unexpected_keys
    assert all(".med." in n for n in result.missing_keys)
