"""Canonical Direct continuation, frozen MedDINO anatomy, and BCA."""
import copy
import hashlib
import sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from src.rddm_rleb import build_baseline, normalize_to_neg_one_to_one
from context_2p5d.model import DirectUnet, state_hash, neighbor_context
from mayo2016_smallset_v1.runtime import seed_everything

DIRECT_CK = Path("/home/xujian/projects/CoreDiff/experiments/rddm_2p5d_screen_v1_20260924/baselines/founddiff_rddm_rleb/experiments/rddm_2p5d_screen_v1/runs/2P5D_DIRECT_CONCAT/checkpoints/step_12000.pt")
DIRECT_SHA = "151b7e58d2ca0d936be0842434e3d17809b325f144d0d48e47c8565287bcdcdf"
MED_ROOT = Path("/home/xujian/projects/CoreDiff/experiments/rddm_prosac_add_fusion_v1_20260926/official_source/MedDINOv3")
MED_CK = Path("/home/xujian/models/ct_foundation/meddinov3/model.pth")
MED_SHA = "492b2b96c00e9d3b93ba3b07a8d387aa2196b5e04698378fcc23049b447966d2"
MED_COMMIT = "a2460c2547899283446e520b84523454f7e25608"
MEAN_HU = 65.1084213256836
STD_HU = 178.01663208007812

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1048576),b""): h.update(chunk)
    return h.hexdigest()

def load_direct(device="cpu"):
    assert sha(DIRECT_CK)==DIRECT_SHA
    seed_everything(20260829)
    model=build_baseline()
    model.model.unet0=DirectUnet()
    ck=torch.load(DIRECT_CK,map_location="cpu",weights_only=False)
    assert ck["arm"]=="2P5D_DIRECT_CONCAT" and ck["optimizer_step"]==12000 and ck["ema_updates"]==12000
    model.load_state_dict(ck["ema_state_dict"],strict=True)
    assert model.objective=="pred_res" and model.sampling_timesteps==2 and model.sum_scale==.01
    return model.to(device).eval()

def official_med():
    assert sha(MED_CK)==MED_SHA
    parent=MED_ROOT/"nnUNet/nnunetv2/training/nnUNetTrainer/dinov3"
    if str(parent) not in sys.path: sys.path.insert(0,str(parent))
    from dinov3.models.vision_transformer import vit_base
    med=vit_base(drop_path_rate=.2,layerscale_init=1e-5,n_storage_tokens=4,qkv_bias=False,mask_k_bias=True)
    ck=torch.load(MED_CK,map_location="cpu",weights_only=False)
    state={k.replace("backbone.",""):v for k,v in ck["teacher"].items() if "ibot" not in k and "dino_head" not in k}
    med.load_state_dict(state,strict=True)
    del ck,state
    return med.eval().requires_grad_(False)

class Reassembly(nn.Module):
    def __init__(self,cin,cout):
        super().__init__()
        self.reduce=nn.Conv2d(cin,cout,1)
        self.activate=nn.GELU()
        self.depthwise=nn.Conv2d(cout,cout,3,padding=1,groups=cout)
        self.zero=nn.Conv2d(cout,cout,1)
        nn.init.zeros_(self.zero.weight);nn.init.zeros_(self.zero.bias)
    def forward(self,x):
        return self.zero(self.depthwise(self.activate(self.reduce(F.interpolate(x,scale_factor=2,mode="bilinear",align_corners=False)))))

class HierarchicalSALF(nn.Module):
    """Previous SALF scoring and E12 64->128 reassembly; only active BCA scales retained."""
    def __init__(self):
        super().__init__()
        self.med=official_med()
        self.projections=nn.ModuleDict({"s128":nn.Conv2d(3072,64,1),"s64":nn.Conv2d(3072,128,1)})
        self.scorers=nn.ModuleDict({scale:nn.ModuleList([nn.Linear(768,1) for _ in range(4)]) for scale in ("s128","s64")})
        for group in self.scorers.values():
            for layer in group:
                nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)
        self.up64_128=Reassembly(128,64)
        self.last_activity={}
    def train(self,mode=True):
        super().train(mode);self.med.eval();return self
    def forward(self,center):
        assert center.shape[1:]==(1,512,512)
        with torch.no_grad():
            hu=((center.float()+1)*2048-1024).clamp(-1000,1000)
            med_input=((hu-MEAN_HU)/STD_HU).repeat(1,3,1,1)
            features=self.med.get_intermediate_layers(med_input,n=[2,5,8,11],reshape=True,norm=True)
        assert len(features)==4 and all(f.shape[1:]==(768,32,32) for f in features)
        descriptors=[f.mean((2,3)) for f in features]
        result={}
        for scale,channels,size in (("s128",64,128),("s64",128,64)):
            weights=torch.softmax(torch.cat([layer(d) for layer,d in zip(self.scorers[scale],descriptors)],dim=1),dim=1)
            gated=torch.cat([4*weights[:,i,None,None,None]*f for i,f in enumerate(features)],dim=1)
            result[scale]=F.interpolate(self.projections[scale](gated),size=(size,size),mode="bilinear",align_corners=False)
        residual=self.up64_128(result["s64"])
        result["s128"]=result["s128"]+residual
        self.last_activity={"s128_rms":float(result["s128"].detach().square().mean().sqrt()),"s64_rms":float(result["s64"].detach().square().mean().sqrt()),"hierarchy_rms":float(residual.detach().square().mean().sqrt())}
        return result

class WindowCrossAttention(nn.Module):
    def __init__(self,main_channels,structure_channels,window):
        super().__init__()
        self.window=window;self.heads=4;self.dim_head=32;self.inner=128
        self.q_norm=nn.LayerNorm(main_channels);self.kv_norm=nn.LayerNorm(structure_channels)
        self.q=nn.Linear(main_channels,128);self.k=nn.Linear(structure_channels,128);self.v=nn.Linear(structure_channels,128)
        self.out=nn.Linear(128,main_channels)
        nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
        self.alpha=nn.Parameter(torch.tensor(.1))
        self.last_activity={}
    def windows(self,x):
        b,c,h,w=x.shape;s=self.window
        assert h%s==w%s==0
        return x.view(b,c,h//s,s,w//s,s).permute(0,2,4,3,5,1).reshape(-1,s*s,c)
    def restore(self,x,b,h,w):
        s=self.window;c=x.shape[-1]
        return x.view(b,h//s,w//s,s,s,c).permute(0,5,1,3,2,4).reshape(b,c,h,w)
    def forward(self,main,structure):
        b,_,h,w=main.shape
        assert structure.shape[0]==b and structure.shape[-2:]==(h,w)
        q=self.q(self.q_norm(self.windows(main)))
        kv=self.kv_norm(self.windows(structure));k=self.k(kv);v=self.v(kv)
        n,t,_=q.shape
        def heads(x): return x.view(n,t,4,32).transpose(1,2)
        out=F.scaled_dot_product_attention(heads(q),heads(k),heads(v),dropout_p=0)
        residual=self.restore(self.out(out.transpose(1,2).reshape(n,t,128)),b,h,w)
        self.last_activity={"residual_rms":float(residual.detach().square().mean().sqrt()),"alpha":float(self.alpha.detach())}
        return main+self.alpha*residual

class BCAUnet(DirectUnet):
    def forward(self,x,time,x_self_cond=None):
        assert x.shape[1]==2
        prev,next_=self.neighbors(x);center=x[:,1:2]
        hints=self.structure(center)
        x=torch.cat((x[:,:1],prev,center,next_),dim=1)
        x=self.init_conv(x);r=x.clone();t=self.time_mlp(time);skips=[]
        for i,(block,down) in enumerate(self.downs):
            x=block(x,t);skips.append(x);x=down(x)
            if i==1:x=self.bca128(x,hints["s128"])
            elif i==2:x=self.bca64(x,hints["s64"])
        x=self.mid_block(x,t)
        for block,up in self.ups:
            x=torch.cat((x,skips.pop()),1);x=block(x,t);x=up(x)
        x=torch.cat((x,r),1);x=self.final_res_block(x,t)
        return self.final_conv(x)

def load_arm(arm,device):
    model=load_direct("cpu")
    base_hash=state_hash(model.model.unet0.state_dict())
    if arm=="A_MEDDINO_BCA":
        unet=model.model.unet0
        unet.__class__=BCAUnet
        unet.structure=HierarchicalSALF()
        unet.bca128=WindowCrossAttention(128,64,8)
        unet.bca64=WindowCrossAttention(256,128,16)
        unet.structure.med.requires_grad_(False)
        assert torch.count_nonzero(unet.bca128.out.weight)==torch.count_nonzero(unet.bca64.out.weight)==0
    model=model.to(device).eval()
    return model,base_hash

def matched_loss(model,prev,center,next_,target,t,noise):
    x_start=normalize_to_neg_one_to_one(target)
    x_input=normalize_to_neg_one_to_one(center)
    residual=x_input-x_start
    x_t=model.q_sample(x_start,residual,t,noise=noise)
    x_in=torch.cat((x_t,x_input),dim=1)
    with neighbor_context(model,prev,next_):
        pred=model.model(x_in,[model.alphas_cumsum[t]*model.num_timesteps,model.betas_cumsum[t]*model.num_timesteps])[0]
    return F.mse_loss(pred,residual),pred,x_input,x_start
