import torch
from torch import nn
from .single_resblock_unet import SingleResBlockUNet

class ResidualDiffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=SingleResBlockUNet()
        beta=torch.linspace(0.0001,0.02,1000,dtype=torch.float32)
        prod=torch.cumprod(1-beta,dim=0)
        self.register_buffer("beta",beta)
        self.register_buffer("A",1-prod.sqrt())
        self.register_buffer("B",(1-prod).sqrt())

    def loss(self,condition_u,target_u):
        condition=condition_u*2-1; target=target_u*2-1
        y=condition[:,1:2]; residual=y-target
        t=torch.randint(0,1000,(len(y),),device=y.device)
        noise=torch.randn_like(y)
        state=target+self.A[t,None,None,None]*residual+self.B[t,None,None,None]*noise
        pred=self.model(torch.cat([state,condition],1),self.A[t]*1000)
        return (pred-residual).square().mean()

    @torch.no_grad()
    def sample(self,condition_u,generator):
        c=condition_u*2-1; y=c[:,1:2]
        x=y+0.1*torch.randn(y.shape,device=y.device,dtype=y.dtype,generator=generator)
        pred=self.model(torch.cat([x,c],1),(self.A[999]*1000).expand(len(y))).clamp(-1,1)
        x=x-(self.A[999]-self.A[499])*pred
        pred=self.model(torch.cat([x,c],1),(self.A[499]*1000).expand(len(y))).clamp(-1,1)
        return (y-pred).clamp(-1,1)
