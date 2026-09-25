"""Frozen trained Direct U-Net plus copied trainable encoder and zero residual taps."""
import copy
import torch
from torch import nn
from context_2p5d.model import DirectUnet, state_hash
from .structure import ZeroStructureProvider, EdgeStructureProvider

MAIN_NAMES=("init_conv","time_mlp","downs","mid_block","ups","final_res_block","final_conv")
ENCODER_NAMES=("init_conv","time_mlp","downs","mid_block")

class ControlEncoder(nn.Module):
    def __init__(self, trained_main):
        super().__init__()
        for name in ENCODER_NAMES:
            setattr(self,name,copy.deepcopy(getattr(trained_main,name)))
        self.hint_projections=nn.ModuleDict({
            "s256":nn.Conv2d(32,64,1,bias=False),
            "s128":nn.Conv2d(64,128,1,bias=False),
            "s64":nn.Conv2d(128,256,1,bias=False)})
    def forward(self, inputs, time, hints):
        x=self.init_conv(inputs)
        t=self.time_mlp(time)
        outputs=[]
        for i,(block,down) in enumerate(self.downs):
            x=down(block(x,t))
            if i<3:
                key=("s256","s128","s64")[i]
                x=x+self.hint_projections[key](hints[key])
                outputs.append(x)
        x=self.mid_block(x,t)
        outputs.append(x)
        return outputs

class ControlledUnet(DirectUnet):
    def __init__(self, trained_main, arm):
        assert arm in ("C1_CAPACITY","C2_EDGE")
        super().__init__()
        for name in MAIN_NAMES:
            setattr(self,name,copy.deepcopy(getattr(trained_main,name)))
        self.control=ControlEncoder(trained_main)
        self.structure=(ZeroStructureProvider() if arm=="C1_CAPACITY" else EdgeStructureProvider())
        self.zero_convs=nn.ModuleList([nn.Conv2d(c,c,1) for c in (64,128,256,512)])
        for module in self.zero_convs:
            nn.init.zeros_(module.weight);nn.init.zeros_(module.bias)
        self.arm=arm
        self.collect_activity=False
        self.activity={}
        for name in MAIN_NAMES:
            getattr(self,name).requires_grad_(False)
    def main_state_hash(self):
        state={}
        for name in MAIN_NAMES:
            for key,value in getattr(self,name).state_dict().items():
                state[name+"."+key]=value
        return state_hash(state)
    def forward(self, x, time, x_self_cond=None):
        assert x.shape[1]==2 and not self.self_condition
        prev,next_=self.neighbors(x)
        inp=torch.cat((x[:,:1],prev,x[:,1:2],next_),1)
        hint=self.structure(inp[:,2:3])
        residuals=self.control(inp,time,hint)
        injections=[z(r) for z,r in zip(self.zero_convs,residuals)]
        main=self.init_conv(inp);r=main.clone();t=self.time_mlp(time);skips=[]
        main_features=[]
        for i,(block,down) in enumerate(self.downs):
            main=block(main,t);skips.append(main);main=down(main)
            if i<3:
                main_features.append(main)
                main=main+injections[i]
        main=self.mid_block(main,t);main_features.append(main)
        main=main+injections[3]
        if self.collect_activity:
            self.activity={
                "zero_weight_norms":[float(z.weight.detach().norm()) for z in self.zero_convs],
                "zero_output_rms":[float(v.detach().square().mean().sqrt()) for v in injections],
                "control_residual_rms":[float(v.detach().square().mean().sqrt()) for v in residuals],
                "structure_rms":[float(hint[k].detach().square().mean().sqrt()) for k in ("s256","s128","s64")],
                "main_feature_rms":[float(v.detach().square().mean().sqrt()) for v in main_features],
                "edge_feature_rms":None if self.arm=="C1_CAPACITY" else
                    [float(hint[k].detach().square().mean().sqrt()) for k in ("s256","s128","s64")]
            }
        for block,up in self.ups:
            main=torch.cat((main,skips.pop()),1);main=up(block(main,t))
        main=torch.cat((main,r),1)
        return self.final_conv(self.final_res_block(main,t))

def trainable_parameters(diffusion):
    return [p for p in diffusion.parameters() if p.requires_grad]

def parameter_counts(diffusion):
    total=sum(p.numel() for p in diffusion.parameters())
    trainable=sum(p.numel() for p in diffusion.parameters() if p.requires_grad)
    return {"total_parameters":total,"frozen_parameters":total-trainable,
            "trainable_parameters":trainable}
