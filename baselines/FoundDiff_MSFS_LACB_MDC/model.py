"""FoundDiff-MSFS-LACB-MDC. No changes to src/DADiff.py RDDM or ResnetBlock."""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from src.DADiff import Unet,UnetRes,ResidualDiffusion
from med_loader import official_med,MEAN_HU,STD_HU
HERE=Path(__file__).resolve().parent
def cp(fn,*args):
    return checkpoint(fn,*args,use_reentrant=False,preserve_rng_state=False) if torch.is_grad_enabled() else fn(*args)
class LinearAttention(nn.Module):
    """ACLNet 905f6d0 LinearAttention, identical kernel/normalization/LePE."""
    def __init__(self,dim,num_heads=4):
        super().__init__();self.dim=dim;self.num_heads=num_heads
        self.qk=nn.Linear(dim,dim*2,bias=True);self.elu=nn.ELU()
        self.lepe=nn.Conv2d(dim,dim,3,padding=1,groups=dim)
    def forward(self,x,shape):
        b,n,c=x.shape;h,w=shape
        assert h*w==n
        qk=self.qk(x).reshape(b,n,2,c).permute(2,0,1,3)
        q,k,v=qk[0],qk[1],x
        q=self.elu(q)+1.;k=self.elu(k)+1.
        q=q.reshape(b,n,self.num_heads,c//self.num_heads).permute(0,2,1,3)
        k=k.reshape(b,n,self.num_heads,c//self.num_heads).permute(0,2,1,3)
        v=v.reshape(b,n,self.num_heads,c//self.num_heads).permute(0,2,1,3)
        z=1/(q@k.mean(dim=-2,keepdim=True).transpose(-2,-1)+1e-6)
        kv=(k.transpose(-2,-1)*(n**-.5))@(v*(n**-.5))
        out=(q@kv*z).transpose(1,2).reshape(b,n,c)
        image=v.transpose(1,2).reshape(b,h,w,c).permute(0,3,1,2)
        return out+self.lepe(image).permute(0,2,3,1).reshape(b,n,c)
class LACB(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.cpe1=nn.Conv2d(c,c,3,padding=1,groups=c)
        self.norm1=nn.LayerNorm(c,elementwise_affine=False,eps=1e-6)
        self.modulation=nn.Sequential(nn.SiLU(),nn.Linear(256,3*c))
        nn.init.zeros_(self.modulation[-1].weight);nn.init.zeros_(self.modulation[-1].bias)
        self.in_proj=nn.Linear(c,c);self.act_proj=nn.Linear(c,c)
        self.dwc=nn.Conv2d(c,c,3,padding=1,groups=c);self.act=nn.SiLU()
        self.attn=LinearAttention(c,4);self.out_proj=nn.Linear(c,c)
        self.cpe2=nn.Conv2d(c,c,3,padding=1,groups=c);self.norm2=nn.LayerNorm(c)
        self.ffn=nn.Sequential(nn.Linear(c,2*c),nn.GELU(),nn.Linear(2*c,c))
    def forward(self,x,condition):
        b,c,h,w=x.shape
        shift,scale,gate=self.modulation(condition).chunk(3,-1)
        tokens=(x+self.cpe1(x)).flatten(2).transpose(1,2)
        shortcut=tokens
        normalized=self.norm1(tokens)*(1+scale[:,None,:])+shift[:,None,:]
        multiplier=self.act(self.act_proj(normalized))
        a=self.in_proj(normalized).reshape(b,h,w,c).permute(0,3,1,2)
        a=self.act(self.dwc(a)).flatten(2).transpose(1,2)
        a=self.attn(a,(h,w))
        tokens=shortcut+self.out_proj(a*multiplier)
        image=tokens.transpose(1,2).reshape(b,c,h,w)
        tokens=tokens+self.cpe2(image).flatten(2).transpose(1,2)
        tokens=tokens+self.ffn(self.norm2(tokens))
        complete=tokens.transpose(1,2).reshape(b,c,h,w)
        return x+gate[:,:,None,None]*(complete-x)
class MDC(nn.Module):
    def __init__(self,c):
        super().__init__()
        def branch(k):
            return nn.Sequential(nn.Conv2d(c,c,k,padding=k-1,dilation=2,groups=c,bias=False),
                                 nn.GroupNorm(8,c),nn.SiLU())
        self.conv3=branch(3);self.conv5=branch(5);self.mix=nn.Conv2d(2*c,c,1,bias=False)
    def forward(self,x):return x+self.mix(torch.cat((self.conv3(x),self.conv5(x)),1))
class FAM(nn.Module):
    def __init__(self,c):
        super().__init__();self.fuse=nn.Conv2d(2*c,c,3,padding=1)
    def forward(self,x,s):
        assert x.shape==s.shape
        return self.fuse(torch.cat((x,s),1))
class ScaleFusion(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.query=nn.Conv2d(c,128,1)
        self.key=nn.Conv2d(768,128,1);self.value=nn.Conv2d(768,128,1)
        self.fuse=nn.Sequential(nn.Conv2d(c+128,c,3,padding=1),nn.GroupNorm(8,c),nn.SiLU())
        self.last_shapes={}
    def forward(self,cnn,med):
        b=cnn.shape[0]
        q=self.query(F.adaptive_avg_pool2d(cnn,(32,32))).flatten(2).transpose(1,2).reshape(b,1024,4,32).transpose(1,2)
        k=self.key(med).flatten(2).transpose(1,2).reshape(b,1024,4,32).transpose(1,2)
        v=self.value(med).flatten(2).transpose(1,2).reshape(b,1024,4,32).transpose(1,2)
        a=F.scaled_dot_product_attention(q,k,v,dropout_p=0.,is_causal=False)
        a=a.transpose(1,2).reshape(b,1024,128).transpose(1,2).reshape(b,128,32,32)
        a=F.interpolate(a,size=cnn.shape[-2:],mode="bilinear",align_corners=False)
        self.last_shapes={"q":list(q.shape),"k":list(k.shape),"v":list(v.shape)}
        return self.fuse(torch.cat((cnn,a),1))
class MSFS(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn=nn.ModuleList([nn.Sequential(nn.Conv2d(a,c,3,stride=s,padding=1),
            nn.GroupNorm(8,c),nn.SiLU()) for a,c,s in ((1,64,1),(64,64,2),(64,128,2),(128,256,2))])
        self.fusions=nn.ModuleList([ScaleFusion(c) for c in (64,64,128,256)])
        self.last_shapes={}
    def forward(self,center,*med):
        features=[];cnn=[];x=center
        for stage,fuse,t in zip(self.cnn,self.fusions,med):
            x=stage(x);cnn.append(list(x.shape));features.append(fuse(x,t))
        self.last_shapes={"cnn":cnn,"structure":[list(x.shape) for x in features]}
        return tuple(features)
class ConditionUnet(Unet):
    def attach(self):
        for c,group in zip((64,64,128,256),self.downs):group[1]=LACB(c)
        self.mid_attn=LACB(512)
        for c,group in zip((512,256,128,64),self.ups):group[1]=LACB(c)
        self.msfs=MSFS()
        self.fams=nn.ModuleList([FAM(c) for c in (64,64,128,256)])
        self.mdcs=nn.ModuleDict({name:MDC(c) for name,c in (("E0",64),("E1",64),("E2",128),("D1",256),("D2",128),("D3",64))})
        self.global_projection=nn.Linear(768,256)
        self.joint=nn.Sequential(nn.Linear(512,256),nn.SiLU(),nn.Linear(256,256))
        # Pretrained frozen weights only; their constructor must not alter the train RNG.
        with torch.random.fork_rng(devices=[]):self.med=official_med()
        self._condition=None;self.events=[];self.last_shapes={}
    def train(self,mode=True):
        super().train(mode);self.med.eval();return self
    @torch.no_grad()
    def extract(self,center_model):
        assert not self.med.training
        x=((center_model*1000.-MEAN_HU)/STD_HU).repeat(1,3,1,1)
        ts=self.med.get_intermediate_layers(x,n=[2,5,8,11],norm=True,return_class_token=False,return_extra_tokens=False)
        assert len(ts)==4 and all(t.shape==(x.shape[0],1024,768) for t in ts)
        return tuple(t.transpose(1,2).reshape(x.shape[0],768,32,32).contiguous().detach() for t in ts)
    def prepare_model(self,center_model):
        assert self._condition is None and center_model.shape[1]==1
        ts=self.extract(center_model)
        spatial=cp(self.msfs,center_model,*ts)
        ea=self.global_projection(ts[-1].mean((-2,-1)))
        self._condition=(spatial,ea)
        self.last_shapes={"med":[list(t.shape) for t in ts],"global":list(ea.shape)}
    def prepare(self,context_zero_one):self.prepare_model(2*context_zero_one[:,1:2]-1)
    def clear(self):self._condition=None
    def forward(self,x,time,x_self_cond=None):
        assert x.shape[1]==4 and not self.self_condition
        own=self._condition is None
        if own:self.prepare_model(x[:,2:3])
        spatial,ea=self._condition
        condition=self.joint(torch.cat((self.time_mlp(time),ea),-1))
        x=self.init_conv(x);stem=x.clone();skips=[];self.events=[]
        for i,(block,lacb,down) in enumerate(self.downs):
            name="E"+str(i)
            x=cp(self.fams[i],x,spatial[i]);self.events.append(name+":FAM")
            x=block(x);self.events.append(name+":ResBlock")
            x=cp(lacb,x,condition);self.events.append(name+":LACB")
            if name in self.mdcs:x=cp(self.mdcs[name],x);self.events.append(name+":MDC")
            skips.append(x);self.events.append(name+":Skip")
            x=down(x)
        x=self.mid_block(x);self.events.append("MID:ResBlock")
        x=cp(self.mid_attn,x,condition);self.events.append("MID:LACB")
        for i,(block,lacb,up) in enumerate(self.ups):
            name="D"+str(i);x=torch.cat((x,skips.pop()),1)
            x=block(x);self.events.append(name+":ResBlock")
            x=cp(lacb,x,condition);self.events.append(name+":LACB")
            if name in self.mdcs:x=cp(self.mdcs[name],x);self.events.append(name+":MDC")
            x=up(x)
        out=self.final_conv(self.final_res_block(torch.cat((x,stem),1)))
        if own:self.clear()
        return out
def build():
    m=UnetRes(dim=64,dim_mults=(1,2,4,8),channels=1,num_unet=1,condition=True,
       input_condition=False,objective="pred_res",test_res_or_noise="res")
    m.unet0.__class__=ConditionUnet;m.unet0.attach()
    return ResidualDiffusion(m,image_size=512,timesteps=1000,sampling_timesteps=2,
       objective="pred_res",loss_type="l2",condition=True,sum_scale=.01,test_res_or_noise="res")
def without_med(m):return {n:v for n,v in m.state_dict().items() if not n.startswith("model.unet0.med.")}
