from pathlib import Path
import re
import numpy as np
import torch

TRAIN_PATIENTS=("L067","L096","L109","L143","L192","L286","L291","L310")
VAL_PATIENTS=("L333",)
FAST64=[0,10,19,29,39,48,58,68,77,87,97,106,116,126,135,145,155,164,174,184,193,203,213,222,232,242,251,261,271,280,290,300,309,319,329,338,348,358,367,377,387,396,406,416,425,435,445,454,464,474,483,493,503,512,522,532,541,551,561,570,580,590,599,609]

class Mayo:
    def __init__(self,root,patients):
        self.root=Path(root); self.records=[]; self.counts={}; self.accesses=0
        assert self.root.is_dir()
        for patient in patients:
            assert patient in TRAIN_PATIENTS+VAL_PATIENTS
            low={int(re.fullmatch(patient+r"_25_([0-9]+)_img.npy",p.name).group(1)):p
                 for p in self.root.glob(patient+"_25_*_img.npy")}
            high={int(re.fullmatch(patient+r"_target_([0-9]+)_img.npy",p.name).group(1)):p
                  for p in self.root.glob(patient+"_target_*_img.npy")}
            assert low and low.keys()==high.keys(),(patient,"pair mismatch")
            indices=sorted(low); self.counts[patient]=len(indices)
            for j,i in enumerate(indices):
                self.records.append((patient,i,low[indices[max(0,j-1)]],low[i],
                                     low[indices[min(len(indices)-1,j+1)]],high[i]))
    def read(self,path):
        assert path.name.split("_")[0] in TRAIN_PATIENTS+VAL_PATIENTS
        a=np.load(path,allow_pickle=False).astype(np.float32)
        assert a.shape==(512,512) and np.isfinite(a).all(),str(path)
        self.accesses+=1
        hu=a-1024.0
        return (np.clip(hu,-1024,3072)+1024)/4096
    def get(self,k,augmentation=None):
        rec=self.records[k]
        arrays=np.stack([self.read(p) for p in rec[2:]])
        if augmentation is not None:
            if augmentation.random()<0.5: arrays=arrays[:,:,::-1]
            arrays=np.rot90(arrays,int(augmentation.integers(4)),axes=(1,2))
        arrays=np.ascontiguousarray(arrays,dtype=np.float32)
        return torch.from_numpy(arrays[:3]),torch.from_numpy(arrays[3:])

class Stream:
    def __init__(self,data,seed):
        self.data=data
        self.shuffle=np.random.default_rng(seed)
        self.augmentation=np.random.default_rng(10)
        self.order=self.shuffle.permutation(len(data.records)); self.cursor=0
    def batch(self):
        ids=[]
        for _ in range(2):
            if self.cursor==len(self.order):
                self.order=self.shuffle.permutation(len(self.data.records)); self.cursor=0
            ids.append(int(self.order[self.cursor])); self.cursor+=1
        pairs=[self.data.get(i,self.augmentation) for i in ids]
        return tuple(torch.stack([pair[j] for pair in pairs]).cuda() for j in range(2))
    def state(self):
        return {"data_shuffle_rng_state":self.shuffle.bit_generator.state,
                "augmentation_rng_state":self.augmentation.bit_generator.state,
                "data_order":self.order,"data_cursor":self.cursor}
    def restore(self,s):
        self.shuffle.bit_generator.state=s["data_shuffle_rng_state"]
        self.augmentation.bit_generator.state=s["augmentation_rng_state"]
        self.order=s["data_order"]; self.cursor=s["data_cursor"]
