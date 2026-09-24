"""2.5D context without changing the frozen residual diffusion."""
import hashlib
from contextlib import contextmanager
import torch
from torch import nn
from mayo2016_smallset_v1.runtime import seed_everything, rng_state, restore_rng
from src.rddm_rleb import Unet, ResnetBlock, build_baseline, normalize_to_neg_one_to_one

ARMS = ("2P5D_DIRECT_CONCAT", "2P5D_CONTEXT_BRANCH")
SOURCE_PIN = "e8ed93223fbdf4a0e2521f73c7a4f213561f6a191a1a7b2a5f95934b85ebf574"


class NeighborMixin:
    _neighbors = None
    def set_neighbors(self, prev, next_):
        assert self._neighbors is None and prev.shape == next_.shape
        self._neighbors = (prev, next_)
    def clear_neighbors(self):
        self._neighbors = None
    def neighbors(self, x):
        assert self._neighbors is not None
        prev, next_ = self._neighbors
        assert prev.shape == next_.shape == x[:, :1].shape
        return prev, next_


class DirectUnet(NeighborMixin, Unet):
    def __init__(self):
        super().__init__(dim=64, dim_mults=(1,2,4,8), channels=1, condition=True)
        self.init_conv = nn.Conv2d(4,64,7,padding=3)
    def forward(self, x, time, x_self_cond=None):
        assert x.shape[1] == 2
        prev, next_ = self.neighbors(x)
        return super().forward(torch.cat((x[:,:1],prev,x[:,1:2],next_),1),time,x_self_cond)


class BranchUnet(NeighborMixin, Unet):
    def __init__(self):
        super().__init__(dim=64, dim_mults=(1,2,4,8), channels=1, condition=True)
        self.context0=nn.Sequential(nn.Conv2d(2,32,3,padding=1),nn.SiLU())
        self.context1=nn.Sequential(nn.Conv2d(32,32,3,stride=2,padding=1),nn.SiLU())
        self.context2=nn.Sequential(nn.Conv2d(32,64,3,stride=2,padding=1),nn.SiLU())
        self.context3=nn.Sequential(nn.Conv2d(64,128,3,stride=2,padding=1),nn.SiLU())
        self.projections=nn.ModuleList((nn.Conv2d(32,64,1),nn.Conv2d(64,128,1),nn.Conv2d(128,256,1)))
        for p in self.projections: nn.init.zeros_(p.weight);nn.init.zeros_(p.bias)
    def forward(self,x,time,x_self_cond=None):
        assert x.shape[1]==2
        prev,next_=self.neighbors(x)
        c0=self.context0(torch.cat((prev,next_),1))
        c1=self.context1(c0);c2=self.context2(c1);c3=self.context3(c2)
        features=(c1,c2,c3)
        x=self.init_conv(x);r=x.clone();t=self.time_mlp(time);skips=[]
        for i,(block,down) in enumerate(self.downs):
            x=block(x,t);skips.append(x);x=down(x)
            if i<3:x=x+self.projections[i](features[i])
        x=self.mid_block(x,t)
        for block,up in self.ups:
            x=torch.cat((x,skips.pop()),1);x=block(x,t);x=up(x)
        x=torch.cat((x,r),1);x=self.final_res_block(x,t)
        return self.final_conv(x)


def state_hash(state):
    h=hashlib.sha256()
    for name,t in sorted(state.items()):
        h.update(name.encode());h.update(str(tuple(t.shape)).encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def construct(arm,device="cpu",seed=20260829):
    assert arm in ARMS
    seed_everything(seed)
    baseline=build_baseline()
    original=baseline.state_dict()
    original_hash=state_hash(original)
    canonical_rng=rng_state()
    model=build_baseline()
    model.model.unet0=DirectUnet() if arm==ARMS[0] else BranchUnet()
    observed=model.state_dict()
    with torch.no_grad():
        for name,t in original.items():
            if arm==ARMS[0] and name=="model.unet0.init_conv.weight":continue
            assert name in observed and observed[name].shape==t.shape
            observed[name].copy_(t)
        if arm==ARMS[0]:
            old=original["model.unet0.init_conv.weight"]
            w=observed["model.unet0.init_conv.weight"]
            w.zero_();w[:,0].copy_(old[:,0]);w[:,2].copy_(old[:,1])
            assert torch.equal(w[:,0],old[:,0]) and torch.equal(w[:,2],old[:,1])
            assert torch.count_nonzero(w[:,1])==torch.count_nonzero(w[:,3])==0
    shared=all(torch.equal(t,observed[k]) for k,t in original.items()
        if not (arm==ARMS[0] and k=="model.unet0.init_conv.weight"))
    assert shared
    new=sorted(set(observed)-set(original))
    assert (not new) if arm==ARMS[0] else bool(new)
    assert sum(isinstance(m,ResnetBlock) for m in model.modules())==10
    assert model.objective=="pred_res" and model.num_timesteps==1000
    assert model.sampling_timesteps==2 and model.sum_scale==0.01
    restore_rng(canonical_rng)
    device=torch.device(device)
    baseline=baseline.to(device).eval();model=model.to(device).eval()
    with torch.random.fork_rng(devices=[device.index or 0] if device.type=="cuda" else []),torch.no_grad():
        torch.manual_seed(41)
        x=torch.randn(1,2,64,64,device=device);p=torch.randn(1,1,64,64,device=device)
        n=torch.randn(1,1,64,64,device=device);t=torch.tensor([10.],device=device)
        ref=baseline.model.unet0(x,t)
        u=model.model.unet0;u.set_neighbors(p,n)
        try:out=u(x,t)
        finally:u.clear_neighbors()
        delta=float((out-ref).abs().max())
        assert out.shape==(1,1,64,64) and delta<=1e-6
    restore_rng(canonical_rng)
    audit={"arm":arm,"seed":seed,"canonical_state_sha256":original_hash,
      "shared_tensor_count":len(original)-(arm==ARMS[0]),"shared_bitwise_identical":True,
      "new_tensor_keys":new,"step_zero_max_abs_output_difference":delta,
      "step_zero_equivalent":True,"baseline_source_sha256":SOURCE_PIN,
      "parameter_count":sum(p.numel() for p in model.parameters()),
      "rleb_count":10,"canonical_training_rng_restored":True}
    if arm==ARMS[0]:
        w=model.model.unet0.init_conv.weight
        audit["mapped_init_conv_equal"]=True
        audit["initial_neighbor_channel_norms"]=[float(w[:,1].norm()),float(w[:,3].norm())]
    else:
        audit["initial_projection_norms"]=[float(p.weight.norm()) for p in model.model.unet0.projections]
        assert audit["initial_projection_norms"]==[0.,0.,0.]
    del baseline
    return model,audit


@contextmanager
def neighbor_context(diffusion,prev,next_):
    u=diffusion.model.unet0
    u.set_neighbors(normalize_to_neg_one_to_one(prev),normalize_to_neg_one_to_one(next_))
    try:yield
    finally:u.clear_neighbors()


def training_loss(diffusion,target,center,prev,next_):
    with neighbor_context(diffusion,prev,next_):
        return diffusion([target,center])[0]


def restored_slice(diffusion,center,prev,next_):
    with neighbor_context(diffusion,prev,next_):
        return diffusion.sample([center],last=True)[-1]


def parameter_usage(diffusion,arm):
    u=diffusion.model.unet0
    if arm==ARMS[0]:
        w=u.init_conv.weight
        return {"prev_stem_weight_norm":float(w[:,1].detach().norm()),
                "next_stem_weight_norm":float(w[:,3].detach().norm())}
    return {"projection_weight_norms":[float(p.weight.detach().norm()) for p in u.projections]}
