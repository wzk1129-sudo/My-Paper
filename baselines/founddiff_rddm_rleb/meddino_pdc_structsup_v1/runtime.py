"""Independent PDC side-supervised train state; parent diffusion and backbone unchanged."""
import hashlib
import json
import math
import time
import torch
from torch.nn import functional as F

from anatomy_controlnet_v1.model import MAIN_NAMES, parameter_counts, trainable_parameters
from context_2p5d.model import training_loss
from mayo2016_smallset_v1.runtime import rng_state, restore_rng, atomic_save
from meddino_pdc_gated_bca_v1.runtime import PDCTrainState
from prosac_add_fusion_v1.runtime import FrozenParameterEMA
from .model import SupervisedPDCFusionProvider
from .structure_target import CleanStructureTarget
from .protocol import ARM, config


class StructSupTrainState(PDCTrainState):
    def __init__(self, device, guard):
        super().__init__(device, guard)
        canonical_rng = rng_state()
        u = self.model.model.unet0
        u.structure.__class__ = SupervisedPDCFusionProvider
        u.structure.add_side_heads()
        u.structure.head128.to(self.device)
        u.structure.head64.to(self.device)
        restore_rng(canonical_rng)
        del self.optimizer, self.ema
        self.arm = ARM
        self.cfg = config()
        tr = self.cfg["training"]
        self.optimizer = torch.optim.Adam(
            trainable_parameters(self.model), lr=tr["learning_rate"],
            betas=tuple(tr["betas"]), weight_decay=tr["weight_decay"])
        self.ema = FrozenParameterEMA(self.model, self.cfg["ema"]["decay"])
        self.target_maker = CleanStructureTarget().to(self.device)
        self.lambda128 = self.cfg["structure_supervision"]["lambda128"]
        self.lambda64 = self.cfg["structure_supervision"]["lambda64"]
        self.lambda_img = self.cfg["structure_supervision"]["lambda_img"]
        heads = sum(p.numel() for p in u.structure.head128.parameters()) + \
                sum(p.numel() for p in u.structure.head64.parameters())
        self.init_audit.update({
            "parent_pdc_model_parameters": self.init_audit["parameter_counts"],
            "parameter_counts": parameter_counts(self.model),
            "added_side_head_parameters": heads,
            "no_pdc_6k_warm_start": True,
            "target_has_trainable_parameters": False,
        })
        self.grad_usage.update({"head128": 0, "head64": 0})
        self.last_aux = None
        self.loss_running = None

    def update(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        prev, center, next_, target, metadata = self.stream.next(2)
        assert prev.shape == center.shape == next_.shape == target.shape == (2,1,512,512)
        for v in (prev, center, next_, target):
            assert v.dtype == torch.float32 and bool(torch.isfinite(v).all())
            assert 0 <= float(v.min()) <= float(v.max()) <= 1
        for meta in metadata:
            assert meta["patient_id"] in ("L067", "L143")
            record = {k:meta[k] for k in ("patient_id","slice_index","horizontal_flip","rotation_degrees")}
            self.stream_digest = hashlib.sha256(
                self.stream_digest + json.dumps(record,sort_keys=True,separators=(",",":")).encode()
            ).digest()
        prev, center, next_, target = (v.to(self.device) for v in (prev,center,next_,target))
        self.model.train()
        u = self.model.model.unet0
        s = u.structure
        assert not s.anatomy.med.training
        self.optimizer.zero_grad(set_to_none=True)
        captured = []
        hook = self.model.model.register_forward_hook(
            lambda module, inputs, output: captured.append(output[0]))
        try:
            base = training_loss(self.model,target,center,prev,next_)
        finally:
            hook.remove()
        assert len(captured) == 1 and captured[0].shape == target.shape
        pred_clean = (2. * center - 1.) - captured[0]
        image_aux = F.l1_loss(pred_clean, 2. * target - 1.)
        preds = s.last_predictions
        targets = self.target_maker(target)
        assert preds["128"].shape == targets["128"].shape == (2,1,128,128)
        assert preds["64"].shape == targets["64"].shape == (2,1,64,64)
        assert all(bool(torch.isfinite(v).all()) for v in (*preds.values(),*targets.values()))
        l128 = F.l1_loss(preds["128"],targets["128"])
        l64 = F.l1_loss(preds["64"],targets["64"])
        total = base + self.lambda128*l128 + self.lambda64*l64 + self.lambda_img*image_aux
        if not all(bool(torch.isfinite(v)) for v in (base,l128,l64,image_aux,total)):
            self.nonfinite_count += 1
            raise FloatingPointError("nonfinite structure/base/total loss")
        total.backward()
        groups = {
            "pdc": list(s.shape.parameters()),
            "shape_proj128": list(s.shape_proj128.parameters()),
            "shape_proj64": list(s.shape_proj64.parameters()),
            "gate128": list(s.gate128.parameters()),
            "gate64": list(s.gate64.parameters()),
            "beta128": [s.beta128], "beta64": [s.beta64],
            "bca128": list(u.bca128.parameters()), "bca64": list(u.bca64.parameters()),
            "alpha128": [u.bca128.alpha], "alpha64": [u.bca64.alpha],
            "main": [p for name in MAIN_NAMES for p in getattr(u,name).parameters()],
            "anatomy": [p for p in s.anatomy.parameters() if p.requires_grad],
            "head128": list(s.head128.parameters()), "head64": list(s.head64.parameters()),
        }
        assert all(p.grad is None for p in s.anatomy.med.parameters())
        for name,group in groups.items():
            grads=[p.grad for p in group if p.grad is not None]
            if not grads or any(not bool(torch.isfinite(g).all()) for g in grads):
                self.nonfinite_count += 1
                raise FloatingPointError("nonfinite/missing "+name+" gradient")
            if any(bool(g.abs().max()>0) for g in grads):
                self.grad_usage[name] += 1
        params=trainable_parameters(self.model)
        norm=float(torch.nn.utils.clip_grad_norm_(params,1.0))
        if not math.isfinite(norm):
            self.nonfinite_count += 1
            raise FloatingPointError("nonfinite gradient norm")
        self.optimizer.step()
        self.ema.update(self.model)
        self.step += 1
        if any(not bool(torch.isfinite(p).all()) for p in params):
            self.nonfinite_count += 1
            raise FloatingPointError("nonfinite parameter")

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            self.peak_vram=max(self.peak_vram,torch.cuda.max_memory_allocated(self.device))
        elapsed=time.perf_counter()-started
        self.train_seconds += elapsed
        self.last_aux={
            "base_loss":float(base.detach()),
            "structure_loss_128":float(l128.detach()),
            "structure_loss_64":float(l64.detach()),
            "weighted_structure_128":float((self.lambda128*l128).detach()),
            "weighted_structure_64":float((self.lambda64*l64).detach()),
            "image_aux_loss":float(image_aux.detach()),
            "weighted_image_aux":float((self.lambda_img*image_aux).detach()),
            "total_loss":float(total.detach()),
            "pred128_mean":float(preds["128"].detach().mean()),
            "pred128_spatial_std":float(preds["128"].detach().flatten(2).std(dim=-1,unbiased=False).mean()),
            "pred64_mean":float(preds["64"].detach().mean()),
            "pred64_spatial_std":float(preds["64"].detach().flatten(2).std(dim=-1,unbiased=False).mean()),
            "target_stats":self.target_maker.last_stats,
            "lambda128":self.lambda128,"lambda64":self.lambda64,"lambda_img":self.lambda_img,
        }
        scalar={k:self.last_aux[k] for k in ("base_loss","structure_loss_128",
                "structure_loss_64","weighted_structure_128","weighted_structure_64","image_aux_loss","weighted_image_aux","total_loss")}
        if self.loss_running is None:
            self.loss_running=scalar.copy()
        else:
            self.loss_running={k:.99*self.loss_running[k]+.01*v for k,v in scalar.items()}
        self.last={
            "step":self.step,"train_loss":float(total.detach()),
            "base_loss":float(base.detach()),
            "structure_loss_128":float(l128.detach()),
            "structure_loss_64":float(l64.detach()),
            "weighted_structure_128":self.last_aux["weighted_structure_128"],
            "weighted_structure_64":self.last_aux["weighted_structure_64"],
            "image_aux_loss":float(image_aux.detach()),
            "weighted_image_aux":float((self.lambda_img*image_aux).detach()),
            "gradient_norm_pre_clip":norm,
            "learning_rate":self.optimizer.param_groups[0]["lr"],
            "seconds_per_step":elapsed,"peak_vram_bytes":self.peak_vram,
            "train_stream_sha256":self.stream_digest.hex(),
        }
        return self.last

    def checkpoint(self,path):
        atomic_save(path,{
            "arm":ARM,"optimizer_step":self.step,
            "model_state_dict":self.model.state_dict(),
            "ema_state_dict":self.ema.model.state_dict(),
            "optimizer_state_dict":self.optimizer.state_dict(),
            "ema_updates":self.ema.updates,
            "rng":rng_state(),"stream":self.stream.state_dict(),
            "stream_sha256":self.stream_digest.hex(),
            "config":self.cfg,"main_hash":self.main_hash,
            "initialization":self.init_audit,
            "train_seconds":self.train_seconds,"peak_vram":self.peak_vram,
            "nonfinite_count":self.nonfinite_count,
            "grad_usage":self.grad_usage,"last":self.last,
            "last_aux":self.last_aux,"loss_running":self.loss_running,
        })

    def resume(self,path):
        ck=torch.load(path,map_location="cpu",weights_only=False)
        assert ck["arm"]==ARM and ck["config"]==self.cfg
        self.model.load_state_dict(ck["model_state_dict"],strict=True)
        self.ema.model.load_state_dict(ck["ema_state_dict"],strict=True)
        self.ema.updates=ck["ema_updates"]
        self.optimizer.load_state_dict(ck["optimizer_state_dict"])
        self.stream.load_state_dict(ck["stream"])
        restore_rng(ck["rng"])
        self.step=ck["optimizer_step"]
        self.stream_digest=bytes.fromhex(ck["stream_sha256"])
        self.train_seconds=ck["train_seconds"]
        self.peak_vram=ck["peak_vram"]
        self.nonfinite_count=ck["nonfinite_count"]
        self.grad_usage=ck["grad_usage"]
        self.last=ck["last"]
        self.last_aux=ck["last_aux"]
        self.loss_running=ck["loss_running"]
        return self
