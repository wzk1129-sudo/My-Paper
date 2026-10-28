import torch
from torch import nn
from .rddm_blocks import ResnetBlock, LinearAttention, Attention, Residual, PreNorm, Downsample, Upsample, SinusoidalPosEmb

class SingleResBlockUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.init_conv = nn.Conv2d(4,64,7,padding=3)
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(64),nn.Linear(64,256),nn.GELU(),nn.Linear(256,256))
        pairs = [(64,64),(64,128),(128,256),(256,512)]
        self.downs = nn.ModuleList([
            nn.ModuleList([ResnetBlock(a,a,time_emb_dim=256,groups=8),
                Residual(PreNorm(a,LinearAttention(a))),
                Downsample(a,b) if i<3 else nn.Conv2d(a,b,3,padding=1)])
            for i,(a,b) in enumerate(pairs)])
        self.mid_block = ResnetBlock(512,512,time_emb_dim=256,groups=8)
        self.mid_attn = Residual(PreNorm(512,Attention(512)))
        self.ups = nn.ModuleList([
            nn.ModuleList([ResnetBlock(a+b,b,time_emb_dim=256,groups=8),
                Residual(PreNorm(b,LinearAttention(b))),
                Upsample(b,a) if i<3 else nn.Conv2d(b,a,3,padding=1)])
            for i,(a,b) in enumerate(reversed(pairs))])
        self.final_res_block = ResnetBlock(128,64,time_emb_dim=256,groups=8)
        self.final_conv = nn.Conv2d(64,1,1)
        self.trace = None

    def forward(self,x,time):
        assert x.ndim==4 and x.shape[1]==4 and time.dtype==torch.float32
        trace={"input":list(x.shape),"encoder":[],"decoder":[]}
        x=self.init_conv(x); stem=x
        t=self.time_mlp(time); skips=[]
        for block,attn,down in self.downs:
            x=attn(block(x,t)); skips.append(x)
            item={"skip":list(x.shape)}
            x=down(x); item["out"]=list(x.shape); trace["encoder"].append(item)
        x=self.mid_attn(self.mid_block(x,t)); trace["bottleneck"]=list(x.shape)
        for block,attn,up in self.ups:
            x=torch.cat([x,skips.pop()],dim=1)
            item={"concat":list(x.shape)}
            x=up(attn(block(x,t))); item["out"]=list(x.shape); trace["decoder"].append(item)
        assert not skips
        x=self.final_conv(self.final_res_block(torch.cat([x,stem],dim=1),t))
        trace.update(output=list(x.shape),long_skips_saved=4,long_skips_consumed=4,stem_skips=1)
        self.trace=trace
        return x
