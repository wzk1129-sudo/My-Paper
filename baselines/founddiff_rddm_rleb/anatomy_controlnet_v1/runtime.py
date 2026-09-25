"""Trained 12k EMA Direct initialization and paired RNG restoration."""
import hashlib
import torch
from src.rddm_rleb import build_baseline
from context_2p5d.model import DirectUnet, state_hash
from mayo2016_smallset_v1.runtime import seed_everything, rng_state, restore_rng
from .model import ControlledUnet, ENCODER_NAMES, parameter_counts
from .protocol import DIRECT_CK, DIRECT_SHA, frozen_config

def substate(module, names):
    out={}
    for name in names:
        for key,tensor in getattr(module,name).state_dict().items():
            out[name+"."+key]=tensor
    return out

def rng_hash(state):
    h=hashlib.sha256()
    h.update(repr(state["python"]).encode())
    n=state["numpy"];h.update(n[0].encode());h.update(n[1].tobytes())
    h.update(repr(n[2:]).encode())
    h.update(state["torch"].cpu().numpy().tobytes())
    for x in state["cuda"]:h.update(x.cpu().numpy().tobytes())
    return h.hexdigest()

def load_direct(device="cpu"):
    frozen_config()
    seed_everything(20260829)
    model=build_baseline()
    model.model.unet0=DirectUnet()
    ck=torch.load(str(DIRECT_CK),map_location="cpu")
    assert ck["arm"]=="2P5D_DIRECT_CONCAT" and ck["optimizer_step"]==12000
    assert ck["config"]["data"]["train_patients"]==["L067","L143"]
    assert ck["ema_updates"]==12000
    model.load_state_dict(ck["ema_state_dict"],strict=True)
    del ck
    assert model.objective=="pred_res" and model.num_timesteps==1000
    assert model.sampling_timesteps==2 and model.sum_scale==.01
    return model.to(device).eval()

def construct(arm,device="cpu",check_step_zero=True):
    assert arm in ("C1_CAPACITY","C2_EDGE")
    base=load_direct("cpu")
    canonical=rng_state()
    direct=base.model.unet0
    main_hash=state_hash(direct.state_dict())
    copy_hash=state_hash(substate(direct,ENCODER_NAMES))
    controlled=ControlledUnet(direct,arm)
    assert controlled.main_state_hash()==main_hash
    assert state_hash(substate(controlled.control,ENCODER_NAMES))==copy_hash
    base.model.unet0=controlled
    restore_rng(canonical)
    initial_rng=rng_hash(rng_state())
    device=torch.device(device)
    base=base.to(device)
    difference=None
    if check_step_zero:
        reference=load_direct(device).eval()
        with torch.random.fork_rng(devices=[device.index or 0] if device.type=="cuda" else []),torch.no_grad():
            torch.manual_seed(109)
            if device.type=="cuda":torch.cuda.manual_seed(109)
            x=torch.randn(1,2,512,512,device=device)
            p=torch.randn(1,1,512,512,device=device)
            n=torch.randn(1,1,512,512,device=device)
            t=torch.tensor([10.],device=device)
            u0=reference.model.unet0;u1=base.model.unet0
            u0.set_neighbors(p,n);u1.set_neighbors(p,n)
            try:
                a=u0(x,t);b=u1(x,t)
            finally:
                u0.clear_neighbors();u1.clear_neighbors()
            difference=float((a-b).abs().max())
            assert a.shape==b.shape==(1,1,512,512)
            assert difference<=1e-6, difference
        del reference
    restore_rng(canonical)
    zeros=[{"weight_norm":float(z.weight.detach().norm()),
            "bias_norm":float(z.bias.detach().norm())} for z in controlled.zero_convs]
    assert all(v["weight_norm"]==v["bias_norm"]==0 for v in zeros)
    audit={"arm":arm,"checkpoint_sha256":DIRECT_SHA,"main_state_hash":main_hash,
           "control_copy_hash":copy_hash,"zero_conv_norms":zeros,
           "step_zero_max_abs_difference":difference,
           "step_zero_equivalent":difference is not None and difference<=1e-6,
           "canonical_training_rng_restored":True,
           "canonical_training_rng_sha256":initial_rng,
           **parameter_counts(base)}
    return base,audit

import copy
import json
import math
import time
from context_2p5d.data import ContextPairs, ContextBatchStream
from context_2p5d.model import training_loss
from mayo2016_smallset_v1.runtime import atomic_save
from .model import MAIN_NAMES, trainable_parameters, parameter_counts
from .protocol import experiment_config, frozen_config

class FrozenMainEMA:
    def __init__(self, model, decay=.995):
        self.model=copy.deepcopy(model).eval().requires_grad_(False)
        self.decay=decay
        self.updates=0
    @torch.no_grad()
    def update(self, model):
        source=dict(model.named_parameters())
        for name,p in self.model.named_parameters():
            q=source[name]
            if name.startswith("model.unet0.") and name.split(".")[2] in MAIN_NAMES:
                p.copy_(q)
            else:
                p.mul_(self.decay).add_(q,alpha=1.-self.decay)
        for name,b in self.model.named_buffers():
            b.copy_(dict(model.named_buffers())[name])
        self.updates+=1

class TrainState:
    def __init__(self,arm,device,guard):
        self.arm=arm;self.device=torch.device(device)
        self.cfg=experiment_config(arm)
        torch.set_num_threads(2)
        self.data=ContextPairs(frozen_config(),"train")
        self.validation=ContextPairs(frozen_config(),"validation","FAST")
        self.data.bind_guard(guard);self.validation.bind_guard(guard)
        assert len(self.data)==1145 and len(self.validation)==64
        self.stream=ContextBatchStream(self.data,self.cfg["seed"])
        self.model,self.init_audit=construct(arm,self.device)
        self.main_hash=self.init_audit["main_state_hash"]
        self.initial_rng_sha256=rng_hash(rng_state())
        assert self.initial_rng_sha256==self.init_audit["canonical_training_rng_sha256"]
        tr=self.cfg["training"]
        self.optimizer=torch.optim.Adam(trainable_parameters(self.model),lr=tr["learning_rate"],
            betas=tuple(tr["betas"]),weight_decay=tr["weight_decay"])
        self.ema=FrozenMainEMA(self.model,self.cfg["ema"]["decay"])
        assert self.ema.model.model.unet0.main_state_hash()==self.main_hash
        self.step=0;self.nonfinite_count=0;self.train_seconds=0.;self.peak_vram=0
        self.stream_digest=hashlib.sha256(b"").digest()
        self.last=None
        if self.device.type=="cuda":torch.cuda.reset_peak_memory_stats(self.device)
    def update(self):
        if self.device.type=="cuda":torch.cuda.synchronize(self.device)
        started=time.perf_counter()
        prev,center,next_,target,metadata=self.stream.next(2)
        assert prev.shape==center.shape==next_.shape==target.shape==(2,1,512,512)
        for tensor in (prev,center,next_,target):
            assert tensor.dtype==torch.float32 and bool(torch.isfinite(tensor).all())
            assert 0<=float(tensor.min())<=float(tensor.max())<=1
        for meta in metadata:
            assert meta["patient_id"] in ("L067","L143")
            record={k:meta[k] for k in ("patient_id","slice_index","horizontal_flip","rotation_degrees")}
            self.stream_digest=hashlib.sha256(self.stream_digest+
                json.dumps(record,sort_keys=True,separators=(",",":")).encode()).digest()
        prev=prev.to(self.device);center=center.to(self.device)
        next_=next_.to(self.device);target=target.to(self.device)
        pre_rng=rng_hash(rng_state()) if self.step in (0,499,999,1999,3999,5999) else None
        self.model.train();self.optimizer.zero_grad(set_to_none=True)
        loss=training_loss(self.model,target,center,prev,next_)
        if not bool(torch.isfinite(loss)):
            self.nonfinite_count+=1;raise FloatingPointError("nonfinite loss")
        loss.backward()
        for name in MAIN_NAMES:
            assert all(p.grad is None for p in getattr(self.model.model.unet0,name).parameters())
        grads=[p.grad for p in trainable_parameters(self.model) if p.grad is not None]
        if not grads or any(not bool(torch.isfinite(g).all()) for g in grads):
            self.nonfinite_count+=1;raise FloatingPointError("nonfinite or missing trainable gradients")
        norm=float(torch.nn.utils.clip_grad_norm_(trainable_parameters(self.model),1.))
        if not math.isfinite(norm):
            self.nonfinite_count+=1;raise FloatingPointError("nonfinite gradient norm")
        self.optimizer.step();self.ema.update(self.model);self.step+=1
        if any(not bool(torch.isfinite(p).all()) for p in trainable_parameters(self.model)):
            self.nonfinite_count+=1;raise FloatingPointError("nonfinite parameter")
        if self.device.type=="cuda":
            torch.cuda.synchronize(self.device)
            self.peak_vram=max(self.peak_vram,torch.cuda.max_memory_allocated(self.device))
            if self.peak_vram>22*(1024**3):raise MemoryError("peak VRAM exceeded 22 GiB")
        elapsed=time.perf_counter()-started;self.train_seconds+=elapsed
        self.last={"step":self.step,"train_loss":float(loss.detach()),
            "gradient_norm_pre_clip":norm,"learning_rate":self.optimizer.param_groups[0]["lr"],
            "seconds_per_step":elapsed,"peak_vram_bytes":self.peak_vram,
            "pre_step_rng_sha256":pre_rng,"stream_sha256":self.stream_digest.hex()}
        return self.last
    def checkpoint(self,path):
        atomic_save(path,{"arm":self.arm,"optimizer_step":self.step,
            "model_state_dict":self.model.state_dict(),
            "ema_state_dict":self.ema.model.state_dict(),
            "optimizer_state_dict":self.optimizer.state_dict(),
            "ema_updates":self.ema.updates,"rng":rng_state(),
            "stream":self.stream.state_dict(),"stream_sha256":self.stream_digest.hex(),
            "config":self.cfg,"main_hash":self.main_hash,
            "initial_rng_sha256":self.initial_rng_sha256,
            "train_seconds":self.train_seconds,"peak_vram":self.peak_vram,
            "nonfinite_count":self.nonfinite_count,"last":self.last})
