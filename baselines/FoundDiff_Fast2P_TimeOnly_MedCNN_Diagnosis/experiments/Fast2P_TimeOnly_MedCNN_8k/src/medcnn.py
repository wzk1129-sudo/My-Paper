"""Frozen official MedDINO + trainable 2.5D CNN and local structural conditioning."""
import sys,hashlib,json
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
PAIR=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(PAIR/"vendor"))
from dinov3.models.vision_transformer import vit_base
WEIGHT=PAIR/"pretrained/MedDINOv3_ViTB16_CT3M_model.pth"
EXPECTED_SHA="492b2b96c00e9d3b93ba3b07a8d387aa2196b5e04698378fcc23049b447966d2"
def checked(fn,*args):
    if torch.is_grad_enabled():return checkpoint(fn,*args,use_reentrant=False,preserve_rng_state=False)
    return fn(*args)
class FrozenMedDINO(nn.Module):
    def __init__(self):
        super().__init__()
        assert WEIGHT.is_file(),str(WEIGHT)
        assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest()==EXPECTED_SHA,"Wrong official MedDINO checkpoint"
        self.net=vit_base(drop_path_rate=.2,layerscale_init=1e-5,n_storage_tokens=4,qkv_bias=False,mask_k_bias=True,pos_embed_rope_dtype="fp32")
        ck=torch.load(WEIGHT,map_location="cpu",weights_only=True)
        state=ck["teacher"] if "teacher" in ck else ck
        state={k.removeprefix("backbone."):v for k,v in state.items() if "ibot" not in k and "dino_head" not in k}
        result=self.net.load_state_dict(state,strict=True)
        self.load_confirmation={"path":str(WEIGHT),"sha256":EXPECTED_SHA,"strict":True,"missing":result.missing_keys,"unexpected":result.unexpected_keys,"layers_zero_based":[2,5,8,11]}
        self.net.eval().requires_grad_(False)
    def train(self,mode=True):
        super().train(False);self.net.eval();return self
    @torch.no_grad()
    def forward(self,center):
        self.net.eval()
        # Input is backbone [-1,1]. Restore HU and follow the official MedDINO CT demo.
        hu=(center*1000).clamp(-1000,1000)
        x=((hu-65.1084213256836)/(178.01663208007812+1e-8)).repeat(1,3,1,1)
        outputs=self.net.get_intermediate_layers(x,n=[2,5,8,11],reshape=True,norm=True)
        assert len(outputs)==4 and all(v.shape==(center.shape[0],768,32,32) for v in outputs)
        return outputs
class ResidualCNNBlock(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.body=nn.Sequential(nn.Conv2d(c,c,3,padding=1),nn.GroupNorm(8,c),nn.SiLU(),nn.Conv2d(c,c,3,padding=1),nn.GroupNorm(8,c))
    def forward(self,x):return F.silu(x+self.body(x))
class CNNStage(nn.Module):
    def __init__(self,ci,co,stride):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(ci,co,3,stride=stride,padding=1),nn.GroupNorm(8,co),nn.SiLU(),ResidualCNNBlock(co),ResidualCNNBlock(co))
    def forward(self,x):return self.net(x)
class LocalCrossAttention(nn.Module):
    def __init__(self,channels,size):
        super().__init__();self.channels=channels;self.size=size
        self.q=nn.Linear(channels,128);self.k=nn.Linear(128,128);self.v=nn.Linear(128,128);self.out=nn.Linear(128,channels)
        axis=torch.floor((torch.arange(size)+.5)*32/size).long()
        yy,xx=torch.meshgrid(axis,axis,indexing="ij")
        offsets=torch.tensor([(dy,dx) for dy in [-1,0,1] for dx in [-1,0,1]])
        ys=yy.flatten()[:,None]+offsets[:,0];xs=xx.flatten()[:,None]+offsets[:,1]
        valid=(ys>=0)&(ys<32)&(xs>=0)&(xs<32)
        self.register_buffer("index",ys.clamp(0,31)*32+xs.clamp(0,31),persistent=False)
        self.register_buffer("valid",valid,persistent=False)
    def forward(self,cnn,m):
        b,c,h,w=cnn.shape;assert h==w==self.size and m.shape[2:]==(32,32)
        ct=cnn.flatten(2).transpose(1,2);mt=m.flatten(2).transpose(1,2);n=h*w
        q=self.q(ct).reshape(b,n,4,32)
        k=self.k(mt)[:,self.index].reshape(b,n,9,4,32).permute(0,1,3,2,4)
        v=self.v(mt)[:,self.index].reshape(b,n,9,4,32).permute(0,1,3,2,4)
        scores=(q.unsqueeze(-2)*k).sum(-1)*(32**-.5)
        scores=scores.masked_fill(~self.valid[None,:,None,:],float("-inf"))
        attended=(scores.softmax(-1).unsqueeze(-1)*v).sum(-2).reshape(b,n,128)
        return self.out(attended).transpose(1,2).reshape(b,c,h,w)
class LocalFusion(nn.Module):
    def __init__(self,c,size):
        super().__init__();self.attn=LocalCrossAttention(c,size);self.gate=nn.Conv2d(2*c,c,1)
    def forward(self,c,m):
        a=self.attn(c,m);return c+torch.sigmoid(self.gate(torch.cat((c,a),1)))*a
class HighFusion(nn.Module):
    def __init__(self,ci):
        super().__init__();self.up=nn.Conv2d(ci,32,1);self.gate=nn.Conv2d(64,32,3,padding=1);self.delta=nn.Conv2d(64,32,3,padding=1)
    def forward(self,c,f):
        u=self.up(F.interpolate(f,size=c.shape[-2:],mode="bilinear",align_corners=False))
        z=torch.cat((c,u),1);return c+torch.sigmoid(self.gate(z))*self.delta(z)
class StructuralPrior(nn.Module):
    def __init__(self):
        super().__init__();self.med=FrozenMedDINO()
        self.projections=nn.ModuleList([nn.Conv2d(768,32,1) for _ in range(4)])
        self.med_fuse=nn.Sequential(nn.Conv2d(128,128,1),nn.GroupNorm(8,128),nn.SiLU())
        self.cnn=nn.ModuleList([CNNStage(3,32,1),CNNStage(32,32,2),CNNStage(32,64,2),CNNStage(64,128,2)])
        self.fuse64=LocalFusion(128,64);self.fuse128=LocalFusion(64,128)
        self.fuse256=HighFusion(64);self.fuse512=HighFusion(32)
    def forward(self,context):
        assert context.shape[1:]==(3,512,512)
        med=self.med(context[:,1:2])
        m=self.med_fuse(torch.cat([p(x) for p,x in zip(self.projections,med)],1))
        cnn=[];x=context
        for stage in self.cnn:x=checked(stage,x);cnn.append(x)
        f64=checked(self.fuse64,cnn[3],m);f128=checked(self.fuse128,cnn[2],m)
        f256=checked(self.fuse256,cnn[1],f128);f512=checked(self.fuse512,cnn[0],f256)
        return {"F64":f64,"F128":f128,"F256":f256,"F512":f512}
class SpatialAdapter(nn.Module):
    def __init__(self,c,fc):
        super().__init__();self.delta=nn.Conv2d(fc,c,1)
        self.spatial_gate=nn.Conv2d(2,1,3,padding=1);self.time_gate=nn.Sequential(nn.SiLU(),nn.Linear(256,c))
        nn.init.zeros_(self.delta.weight);nn.init.zeros_(self.delta.bias)
    def forward(self,x,f,t):
        sg=torch.sigmoid(self.spatial_gate(torch.cat((x.mean(1,keepdim=True),f.mean(1,keepdim=True)),1)))
        tg=torch.sigmoid(self.time_gate(t))[:,:,None,None]
        return x+tg*sg*self.delta(f)
class FiLMAdapter(nn.Module):
    def __init__(self,c,fc):
        super().__init__();self.affine=nn.Conv2d(fc,2*c,1);self.norm=nn.GroupNorm(8,c)
        self.time_gate=nn.Sequential(nn.SiLU(),nn.Linear(256,c))
        nn.init.zeros_(self.affine.weight);nn.init.zeros_(self.affine.bias)
    def forward(self,x,f,t):
        gamma,beta=self.affine(f).chunk(2,1);tg=torch.sigmoid(self.time_gate(t))[:,:,None,None]
        return x+tg*(gamma*self.norm(x)+beta)
def make_adapters():
    return nn.ModuleDict({
      "E0":SpatialAdapter(64,32),"E1":SpatialAdapter(64,32),"E2":FiLMAdapter(128,64),"E3":FiLMAdapter(256,128),
      "MID":FiLMAdapter(512,128),"D0":FiLMAdapter(512,128),"D1":FiLMAdapter(256,64),
      "D2":SpatialAdapter(128,32),"D3":SpatialAdapter(64,32)})
