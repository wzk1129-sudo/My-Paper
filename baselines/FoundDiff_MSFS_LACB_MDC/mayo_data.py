"""Patient-local 2.5D Mayo loader; fixed historical Fast64 and explicit eight-patient train."""
import json, hashlib
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from frozen_metrics import normalize_stored_ct
ROOT=Path(__file__).resolve().parent
OUT=ROOT
CFG=json.loads((OUT/"config.json").read_text())
TRAIN=tuple(CFG["data"]["train_patients"])
COUNTS=CFG["allowed_slice_counts"]
VALID=tuple(map(int,(OUT/"L333_fast64.txt").read_text().splitlines()))
assert len(VALID)==len(set(VALID))==64
assert TRAIN==("L067","L096","L109","L143","L192","L286","L291","L310")
READS=Counter()

class Pairs:
    def __init__(self,split):
        assert split in ("train","validation")
        self.records=([(p,i) for p in TRAIN for i in range(COUNTS[p])] if split=="train"
                      else [("L333",i) for i in VALID])
        self.root=Path(CFG["data"]["root"])
        self.manifest_hash=hashlib.sha256(json.dumps(self.records).encode()).hexdigest()
    def __len__(self):return len(self.records)
    def read(self,p,i,view):
        assert p in (*TRAIN,"L333") and 0<=i<COUNTS[p] and view in ("25","target")
        path=self.root/f"{p}_{view}_{i:03d}_img.npy"
        assert not path.is_symlink() and path.parent==self.root
        a=np.load(path,allow_pickle=False)
        assert a.shape==(512,512) and np.isfinite(a).all()
        READS[(p,view)]+=1
        return normalize_stored_ct(a,CFG["data"])[None]
    def load(self,pos):
        p,i=self.records[pos]
        center=self.read(p,i,"25")
        prev=center if i==0 else self.read(p,i-1,"25")
        next_=center if i==COUNTS[p]-1 else self.read(p,i+1,"25")
        target=self.read(p,i,"target")
        return np.concatenate((prev,center,next_),axis=0),target,{"patient_id":p,"slice_index":i}
class Stream:
    def __init__(self,data):
        self.data=data;self.rng=np.random.RandomState(10)
        self.order=self.rng.permutation(len(data));self.cursor=0;self.epoch=0
    def next(self):
        contexts=[];targets=[];metas=[]
        for _ in range(2):
            if self.cursor==len(self.order):
                self.order=self.rng.permutation(len(self.data));self.cursor=0;self.epoch+=1
            x,y,m=self.data.load(int(self.order[self.cursor]));self.cursor+=1
            contexts.append(x);targets.append(y);metas.append(m)
        return torch.from_numpy(np.stack(contexts)),torch.from_numpy(np.stack(targets)),metas
    def state_dict(self):
        return {"rng":self.rng.get_state(),"order":self.order,"cursor":self.cursor,"epoch":self.epoch,"manifest":self.data.manifest_hash}
    def load_state_dict(self,s):
        assert s["manifest"]==self.data.manifest_hash
        self.rng.set_state(s["rng"]);self.order=s["order"];self.cursor=s["cursor"];self.epoch=s["epoch"]
def access_report():
    return {"reads":{p+"_"+v:n for (p,v),n in READS.items()},"L506_opens":sum(n for (p,v),n in READS.items() if p=="L506"),"augmentation":False}
