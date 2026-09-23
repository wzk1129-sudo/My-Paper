"""Four isolated changes on the frozen clean RDDM backbone."""
import torch
from torch import nn
from torch.nn import functional as F
from src.rddm_rleb import (build_baseline, ResidualDiffusion, Unet, ResnetBlock,
                           WeightStandardizedConv2d, extract)

class NoiseDiffusion(ResidualDiffusion):
    def __init__(self,*args,noise_scale=1.,**kwargs):
        super().__init__(*args,**kwargs)
        self.noise_scale=float(noise_scale)
    def q_sample(self,x_start,x_res,t,noise=None):
        if noise is None:noise=torch.randn_like(x_start)
        return x_start+extract(self.alphas_cumsum,t,x_start.shape)*x_res+self.noise_scale*extract(self.betas_cumsum,t,x_start.shape)*noise

class WindowDiffusion(ResidualDiffusion):
    def __init__(self,*args,lambda_window=0.,**kwargs):
        super().__init__(*args,**kwargs)
        self.lambda_window=float(lambda_window)
    def p_losses(self,imgs,t,noise=None):
        assert isinstance(imgs,list) and len(imgs)==2 and self.objective=='pred_res'
        x_start,x_input=imgs
        if noise is None:noise=torch.randn_like(x_start)
        x_res=x_input-x_start
        x=self.q_sample(x_start,x_res,t,noise=noise)
        pred_res=self.model(torch.cat((x,x_input),dim=1),
            [self.alphas_cumsum[t]*self.num_timesteps,self.betas_cumsum[t]*self.num_timesteps])[0]
        res_loss=F.mse_loss(pred_res,x_res)
        pred_x0=x_input-pred_res
        # Internal model domain [-1,1] -> stored HU protocol, then evaluation window.
        pred_hu=(pred_x0+1.)*2048.-1024.
        gt_hu=(x_start+1.)*2048.-1024.
        window_error=(pred_hu.clamp(-1000.,1000.)-gt_hu.clamp(-1000.,1000.))/2048.
        window_loss=window_error.square().mean()
        return [res_loss+self.lambda_window*window_loss]

class MultiScaleUnet(Unet):
    """Condition path is zero gated at initialization; four encoder RLEB injections."""
    def __init__(self):
        super().__init__(dim=64,dim_mults=(1,2,4,8),channels=1,condition=True,input_condition=False)
        widths=(64,64,128,256)
        self.cond_convs=nn.ModuleList([nn.Conv2d(1,64,3,padding=1),nn.Conv2d(64,64,3,stride=2,padding=1),
            nn.Conv2d(64,128,3,stride=2,padding=1),nn.Conv2d(128,256,3,stride=2,padding=1)])
        self.cond_projections=nn.ModuleList([nn.Conv2d(c,c,1) for c in widths])
        self.cond_gammas=nn.ParameterList([nn.Parameter(torch.zeros(())) for _ in widths])
    def forward(self,x,time,x_self_cond=None):
        assert x.shape[1]==2
        condition=x[:,1:2]
        pyramid=[]
        for conv in self.cond_convs:
            condition=F.silu(conv(condition));pyramid.append(condition)
        x=self.init_conv(x)
        r=x.clone();t=self.time_mlp(time);h=[]
        for level,(res_block,downsample) in enumerate(self.downs):
            addition=self.cond_gammas[level]*self.cond_projections[level](pyramid[level])
            assert addition.shape==x.shape
            x=res_block(x+addition,t)
            h.append(x);x=downsample(x)
        x=self.mid_block(x,t)
        for res_block,upsample in self.ups:
            x=torch.cat((x,h.pop()),dim=1);x=res_block(x,t);x=upsample(x)
        x=torch.cat((x,r),dim=1)
        return self.final_conv(self.final_res_block(x,t))

class TwoConvRLEB(ResnetBlock):
    def __init__(self,original):
        dim=original.block1.proj.in_channels;dim_out=original.block1.proj.out_channels
        time_dim=original.mlp[1].in_features if original.mlp is not None else None
        super().__init__(dim,dim_out,time_emb_dim=time_dim)
        self.mlp=original.mlp;self.block1=original.block1;self.res_conv=original.res_conv
        self.conv2=WeightStandardizedConv2d(dim_out,dim_out,3,padding=1)
        self.norm2=nn.GroupNorm(8,dim_out)
    def forward(self,x,time_emb=None):
        scale_shift=None
        if self.mlp is not None and time_emb is not None:
            scale,shift=self.mlp(time_emb)[:,:,None,None].chunk(2,dim=1)
            scale_shift=(scale,shift)
        h=self.block1(x,scale_shift=scale_shift)
        h=self.norm2(self.conv2(h))
        return h+self.res_conv(x)

def _replace_rleb(unet):
    count=0
    for blocks in list(unet.downs)+list(unet.ups):
        blocks[0]=TwoConvRLEB(blocks[0]);count+=1
    unet.mid_block=TwoConvRLEB(unet.mid_block);count+=1
    unet.final_res_block=TwoConvRLEB(unet.final_res_block);count+=1
    assert count==10

def build_variant(spec):
    kind=spec['kind']
    if kind in ('F1','F4'):
        from src.rddm_rleb import UnetRes
        backbone=UnetRes(dim=64,dim_mults=(1,2,4,8),channels=1,num_unet=1,
            condition=True,input_condition=False,objective='pred_res',test_res_or_noise='res')
        cls=NoiseDiffusion if kind=='F1' else WindowDiffusion
        parameter={'noise_scale':spec['value']} if kind=='F1' else {'lambda_window':spec['value']}
        return cls(backbone,image_size=512,timesteps=1000,sampling_timesteps=2,
            objective='pred_res',loss_type='l2',condition=True,sum_scale=.01,
            input_condition=False,input_condition_mask=False,test_res_or_noise='res',**parameter)
    model=build_baseline()
    if kind=='F2':
        old=model.model.unet0
        upgraded=MultiScaleUnet()
        missing,unexpected=upgraded.load_state_dict(old.state_dict(),strict=False)
        assert not unexpected and all(k.startswith(('cond_convs.','cond_projections.','cond_gammas.')) for k in missing)
        model.model.unet0=upgraded
    elif kind=='F3':_replace_rleb(model.model.unet0)
    else:raise ValueError(kind)
    return model
