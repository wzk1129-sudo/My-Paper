"""Patient-local three-slice LDCT reads with center-only NDCT supervision."""
import hashlib
import json
import os
import re
import sys
from pathlib import Path
import numpy as np
import torch
from fast_dev.data import FastPairs
from fast_dev.protocol import FAST_TRAIN, MEMORY, validation_indices
from mayo2016_smallset_v1.data import COUNTS
from mayo2016_smallset_v1.frozen_protocol import normalize_stored_ct


class ContextPairs(FastPairs):
    def __init__(self,cfg,split,mode="FAST"):
        super().__init__(cfg,split,mode)
        assert split in ("train","validation")
        self.center_keys=tuple((r[0],r[1]) for r in self.records)
        self.guard=None
        for patient,index in self.center_keys:
            for neighbor in (max(0,index-1),min(COUNTS[patient]-1,index+1)):
                path=self.root/f"{patient}_25_{neighbor:03d}_img.npy"
                if path.parent!=self.root or path.is_symlink() or not path.is_file():
                    raise RuntimeError("Missing/nonlocal neighbor LDCT: "+str(path))

    def bind_guard(self,guard):
        self.guard=guard

    def _neighbor(self,patient,index):
        path=self.root/f"{patient}_25_{index:03d}_img.npy"
        if path.parent!=self.root or path.is_symlink():
            raise PermissionError("nonlocal neighbor")
        raw=np.load(path,allow_pickle=False)
        if raw.shape!=(512,512) or not np.isfinite(raw).all():
            raise ValueError("invalid neighbor LDCT")
        value=normalize_stored_ct(raw,self.config["data"])[None]
        assert value.dtype==np.float32 and np.isfinite(value).all()
        assert 0<=float(value.min())<=float(value.max())<=1
        return value

    def load(self,position):
        patient,index,_,_=self.records[int(position)]
        prev_index=max(0,index-1)
        next_index=min(COUNTS[patient]-1,index+1)
        if self.guard is None:raise RuntimeError("ContextPairs requires an active view-aware guard")
        self.guard.begin_sample(patient,index)
        try:
            center,target,meta=super().load(position)
            prev=self._neighbor(patient,prev_index)
            next_=self._neighbor(patient,next_index)
            return prev,center,next_,target,{**meta,"prev_index":prev_index,"next_index":next_index}
        finally:self.guard.end_sample()


class ContextBatchStream:
    """Same NumPy RandomState calls and augmentation order as frozen BatchStream."""
    def __init__(self,dataset,seed):
        self.dataset=dataset
        self.rng=np.random.RandomState(seed)
        self.permutation=self.rng.permutation(len(dataset))
        self.cursor=0;self.epoch=0

    def next(self,batch_size=2):
        prevs,centers,nexts,targets,metadata=[],[],[],[],[]
        for _ in range(batch_size):
            if self.cursor==len(self.permutation):
                self.permutation=self.rng.permutation(len(self.dataset))
                self.cursor=0;self.epoch+=1
            position=int(self.permutation[self.cursor]);self.cursor+=1
            prev,center,next_,target,meta=self.dataset.load(position)
            flip=bool(self.rng.rand()<.5)
            rotation=int(self.rng.randint(4))
            def transform(x):
                if flip:x=np.flip(x,axis=-1)
                return np.ascontiguousarray(np.rot90(x,rotation,axes=(-2,-1)))
            prevs.append(transform(prev));centers.append(transform(center))
            nexts.append(transform(next_));targets.append(transform(target))
            metadata.append({**meta,"horizontal_flip":flip,"rotation_degrees":rotation*90})
        return (torch.from_numpy(np.stack(prevs)),torch.from_numpy(np.stack(centers)),
                torch.from_numpy(np.stack(nexts)),torch.from_numpy(np.stack(targets)),metadata)

    def state_dict(self):
        return {"rng":self.rng.get_state(),"permutation":self.permutation.copy(),
                "cursor":self.cursor,"epoch":self.epoch,"manifest_sha256":self.dataset.manifest_hash}

    def load_state_dict(self,state):
        assert state["manifest_sha256"]==self.dataset.manifest_hash
        assert sorted(state["permutation"].tolist())==list(range(len(self.dataset)))
        assert 0<=state["cursor"]<=len(self.dataset)
        self.rng.set_state(state["rng"]);self.permutation=state["permutation"].copy()
        self.cursor=state["cursor"];self.epoch=state["epoch"]


class ContextDataGuard:
    """Audit actual opens using a patient/index/view allowlist; no directory enumeration."""
    def __init__(self,cfg,mode="FAST",allow_training=True):
        self.root=os.path.abspath(cfg["data"]["root"])
        self.mode=mode;self.active=False;self.blocked=0;self.reads=[];self.roles=[];self.active_sample=None
        self.validation_centers=set(validation_indices(mode))
        self.train_centers={(p,i) for p in FAST_TRAIN for i in range(COUNTS[p])} if allow_training else set()
        self.validation_ldct=set(self.validation_centers)
        for index in self.validation_centers:
            self.validation_ldct.add(max(0,index-1))
            self.validation_ldct.add(min(COUNTS["L333"]-1,index+1))
        self.allow_training=allow_training

    def begin_sample(self,patient,index):
        assert self.active and self.active_sample is None
        self.active_sample=(patient,index)
    def end_sample(self):
        self.active_sample=None

    def _hook(self,event,args):
        if not self.active or event not in ("open","os.listdir","os.scandir") or not args:
            return
        value=args[0]
        if not isinstance(value,(str,bytes,os.PathLike)):return
        path=os.path.abspath(os.fsdecode(value))
        if path!=self.root and not path.startswith(self.root+os.sep):return
        if event!="open":
            self.blocked+=1;raise PermissionError("medical data directory enumeration forbidden")
        match=re.fullmatch(r"(L\d{3})_(25|target)_(\d{3})_img\.npy",os.path.basename(path))
        flags=args[2] if len(args)>2 and isinstance(args[2],int) else 0
        writing=bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
        valid=False
        if match and os.path.dirname(path)==self.root and not writing:
            p,v,i=match[1],match[2],int(match[3])
            sample=self.active_sample
            if sample is not None and p==sample[0]:
                center=sample[1]
                permitted=(max(0,center-1),center,min(COUNTS[p]-1,center+1))
                role_allowed=((p,i) in self.train_centers if p in FAST_TRAIN else
                              p=="L333" and center in self.validation_centers)
                valid=role_allowed and ((v=="target" and i==center) or
                                        (v=="25" and i in permitted))
        if not valid:
            self.blocked+=1;raise PermissionError("unapproved patient/index/view or data write")
        self.reads.append((p,i,v))
        self.roles.append("center_target" if v=="target" and i==center else
                          "neighbor_target" if v=="target" else
                          "center_ldct" if i==center else "neighbor_ldct")

    def __enter__(self):
        self.active=True;sys.addaudithook(self._hook);return self
    def __exit__(self,*args):self.active=False

    def report(self):
        rows=self.reads;centers=self.validation_centers
        return {
          "mode":self.mode,"allowed_train_patients":list(FAST_TRAIN) if self.allow_training else [],
          "training_patients_opened":sorted({p for p,i,v in rows if p in FAST_TRAIN}),
          "validation_center_indices_opened":sorted({i for p,i,v in rows if p=="L333" and v=="target"}),
          "validation_ldct_indices_opened":sorted({i for p,i,v in rows if p=="L333" and v=="25"}),
          "center_target_opens":sum(p=="L333" and role=="center_target" for (p,i,v),role in zip(rows,self.roles)),
          "center_ldct_opens":sum(p=="L333" and role=="center_ldct" for (p,i,v),role in zip(rows,self.roles)),
          "neighbor_ldct_opens":sum(p=="L333" and role=="neighbor_ldct" for (p,i,v),role in zip(rows,self.roles)),
          "neighbor_target_opens":sum(role=="neighbor_target" for role in self.roles),
          "training_target_opens":sum(p in FAST_TRAIN and v=="target" for p,i,v in rows),
          "total_opens":len(rows),"L506_opens":sum(p=="L506" for p,i,v in rows),
          "external_memory_opens":sum(p in MEMORY for p,i,v in rows),
          "blocked_attempts":self.blocked,"data_directory_enumeration":False,
        }
