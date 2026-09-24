"""Independent 12k optimizer state for a single 2.5D arm."""
import hashlib
import json
import math
import time
import torch
from mayo2016_smallset_v1.runtime import EMA, rng_state, restore_rng
from .data import ContextPairs, ContextBatchStream
from .model import ARMS, construct, training_loss, parameter_usage
from .protocol import config, source_manifest


def rng_hash(state):
    h=hashlib.sha256()
    h.update(repr(state["python"]).encode())
    np_state=state["numpy"]
    h.update(np_state[0].encode());h.update(np_state[1].tobytes())
    h.update(repr(np_state[2:]).encode())
    h.update(state["torch"].cpu().numpy().tobytes())
    for t in state["cuda"]:h.update(t.cpu().numpy().tobytes())
    return h.hexdigest()


class ScreenState:
    def __init__(self,arm,device,guard):
        assert arm in ARMS
        self.cfg,self.base=config();self.arm=arm;self.device=torch.device(device)
        torch.set_num_threads(self.cfg["training"]["cpu_threads"])
        if self.device.type=="cuda":torch.cuda.set_device(self.device)
        self.train_data=ContextPairs(self.base,"train")
        self.validation_data=ContextPairs(self.base,"validation","FAST")
        self.train_data.bind_guard(guard);self.validation_data.bind_guard(guard)
        assert len(self.train_data)==1145 and len(self.validation_data)==64
        self.stream=ContextBatchStream(self.train_data,self.cfg["seed"])
        self.model,self.init_audit=construct(arm,self.device,self.cfg["seed"])
        self.training_rng=rng_state();self.initial_rng_sha256=rng_hash(self.training_rng)
        tr=self.cfg["training"]
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=tr["learning_rate"],
            betas=tuple(tr["betas"]),weight_decay=tr["weight_decay"])
        self.ema=EMA(self.model,self.cfg["ema"]["decay"])
        self.step=0;self.last=None;self.train_seconds=0.;self.peak_vram=0
        self.gradient_min=None;self.gradient_max=None;self.nonfinite_count=0
        self.stream_digest=hashlib.sha256(b"").digest()
        self.manifest=source_manifest()
        self.grad_audit={
            "first_step_projection_gradient_abs":[],
            "first_step_context_encoder_gradient_abs":None,
            "later_context_encoder_gradient_abs":0.,
        }
        if self.device.type=="cuda":torch.cuda.reset_peak_memory_stats(self.device)

    def update(self):
        if self.device.type=="cuda":torch.cuda.synchronize(self.device)
        started=time.perf_counter()
        prev,center,next_,target,metadata=self.stream.next(2)
        assert prev.shape==center.shape==next_.shape==target.shape==(2,1,512,512)
        for t in (prev,center,next_,target):
            assert t.dtype==torch.float32 and bool(torch.isfinite(t).all())
            assert 0<=float(t.min())<=float(t.max())<=1
        for meta in metadata:
            assert meta["patient_id"] in ("L067","L143")
            assert meta["prev_index"]==max(0,meta["slice_index"]-1)
            assert meta["next_index"]==min((560 if meta["patient_id"]=="L067" else 585)-1,meta["slice_index"]+1)
            record={k:meta[k] for k in ("patient_id","slice_index","horizontal_flip","rotation_degrees")}
            self.stream_digest=hashlib.sha256(
                self.stream_digest+json.dumps(record,sort_keys=True,separators=(",",":")).encode()
            ).digest()
        prev=prev.to(self.device);center=center.to(self.device)
        next_=next_.to(self.device);target=target.to(self.device)
        self.model.train();self.optimizer.zero_grad(set_to_none=True)
        loss=training_loss(self.model,target,center,prev,next_)
        if not bool(torch.isfinite(loss)):
            self.nonfinite_count+=1;raise FloatingPointError("nonfinite loss")
        loss.backward()
        missing=[n for n,p in self.model.named_parameters() if p.grad is None]
        # Decoder injection consumes C512/C256/C128; C64 is deliberately inactive.
        allowed={n for n in missing if self.arm==ARMS[2] and
                 n.startswith("model.unet0.context3.")}
        if set(missing)-allowed:
            raise RuntimeError("missing parameter gradients: "+repr(sorted(set(missing)-allowed)))
        u=self.model.model.unet0
        if self.step==0:
            self.grad_audit["first_step_projection_gradient_abs"]=[
                float(p.weight.grad.abs().sum()) for p in u.projections]
            self.grad_audit["first_step_context_encoder_gradient_abs"]=float(
                u.context0[0].weight.grad.abs().sum())
        else:
            self.grad_audit["later_context_encoder_gradient_abs"]+=float(
                u.context0[0].weight.grad.abs().sum())
        norm=float(torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.0))
        if not math.isfinite(norm):
            self.nonfinite_count+=1;raise FloatingPointError("nonfinite gradient")
        self.gradient_min=norm if self.gradient_min is None else min(self.gradient_min,norm)
        self.gradient_max=norm if self.gradient_max is None else max(self.gradient_max,norm)
        self.optimizer.step();self.ema.update(self.model);self.step+=1
        if any(not bool(torch.isfinite(p).all()) for p in self.model.parameters()):
            self.nonfinite_count+=1;raise FloatingPointError("nonfinite parameter")
        if self.device.type=="cuda":
            torch.cuda.synchronize(self.device)
            self.peak_vram=max(self.peak_vram,torch.cuda.max_memory_allocated(self.device))
        elapsed=time.perf_counter()-started;self.train_seconds+=elapsed
        self.last={
            "step":self.step,"train_loss":float(loss.detach()),"gradient_norm_pre_clip":norm,
            "learning_rate":self.optimizer.param_groups[0]["lr"],
            "seconds_per_step":elapsed,"peak_vram_bytes":self.peak_vram,"samples":metadata
        }
        return self.last

    def gradient_audit(self):
        result=dict(self.grad_audit)
        result["parameter_usage"]=parameter_usage(self.model,self.arm)
        u=self.model.model.unet0
        result["projection_optimizer_states_present"]=[
            bool(self.optimizer.state.get(p.weight)) for p in u.projections]
        result["context_encoder_optimizer_state_present"]=bool(
            self.optimizer.state.get(u.context0[0].weight))
        result["context_stage4_optimizer_state_present"]=bool(
            self.optimizer.state.get(u.context3[0].weight))
        result["context_stage4_expected_active"]=self.arm!=ARMS[2]
        return result

    def payload(self):
        return {
            "arm":self.arm,"optimizer_step":self.step,
            "model_state_dict":self.model.state_dict(),
            "ema_state_dict":self.ema.model.state_dict(),
            "optimizer_state_dict":self.optimizer.state_dict(),
            "ema_updates":self.ema.updates,"rng":rng_state(),
            "data_stream":self.stream.state_dict(),"stream_digest_hex":self.stream_digest.hex(),
            "config":self.cfg,"source_manifest":self.manifest,
            "initialization_audit":self.init_audit,"initial_rng_sha256":self.initial_rng_sha256,
            "train_seconds":self.train_seconds,"peak_vram_bytes":self.peak_vram,
            "gradient_min":self.gradient_min,"gradient_max":self.gradient_max,
            "nonfinite_count":self.nonfinite_count,"last":self.last,"grad_audit":self.grad_audit,
        }

    def resume(self,path):
        ck=torch.load(str(path),map_location="cpu")
        assert ck["arm"]==self.arm and ck["config"]==self.cfg and ck["source_manifest"]==self.manifest
        assert ck["optimizer_step"]<12000 and ck["initialization_audit"]==self.init_audit
        self.model.load_state_dict(ck["model_state_dict"],strict=True)
        self.ema.model.load_state_dict(ck["ema_state_dict"],strict=True)
        self.optimizer.load_state_dict(ck["optimizer_state_dict"])
        self.ema.updates=ck["ema_updates"];self.step=ck["optimizer_step"]
        self.stream.load_state_dict(ck["data_stream"])
        self.stream_digest=bytes.fromhex(ck["stream_digest_hex"])
        self.train_seconds=ck["train_seconds"];self.peak_vram=ck["peak_vram_bytes"]
        self.gradient_min=ck["gradient_min"];self.gradient_max=ck["gradient_max"]
        self.nonfinite_count=ck["nonfinite_count"];self.last=ck["last"]
        self.grad_audit=ck["grad_audit"]
        assert self.ema.updates==self.step
        restore_rng(ck["rng"])
