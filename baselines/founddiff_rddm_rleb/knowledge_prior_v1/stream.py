"""Fixed metadata/noise manifest shared by all 6000-step RDDM arms."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from mayo2016_smallset_v1.data import COUNTS

def make_manifest(dataset,steps=6000,seed=20260829):
    rng=np.random.RandomState(seed)
    permutation=rng.permutation(len(dataset));cursor=0
    diffusion=np.random.RandomState(seed+771)
    rows=[]
    for step in range(1,steps+1):
        batch=[]
        for _ in range(2):
            if cursor==len(permutation):permutation=rng.permutation(len(dataset));cursor=0
            position=int(permutation[cursor]);cursor+=1
            patient,index=dataset.records[position][:2]
            flip=bool(rng.rand()<.5);rotation=int(rng.randint(4))
            batch.append({"patient_id":patient,"slice_index":index,"prev_index":max(0,index-1),
                "next_index":min(COUNTS[patient]-1,index+1),"flip":flip,"rotation_degrees":rotation*90,
                "t":int(diffusion.randint(0,1000)),"noise_seed":int(diffusion.randint(0,2**31-1))})
        rows.append(batch)
    return rows

def digest(manifest):
    payload=json.dumps(manifest,sort_keys=True,separators=(",",":")).encode()
    return hashlib.sha256(payload).hexdigest()

def assert_batch(metadata,expected):
    assert len(metadata)==len(expected)==2
    for found,want in zip(metadata,expected):
        for key in ("patient_id","slice_index","prev_index","next_index"):
            assert found[key]==want[key],(key,found[key],want[key])
        assert found["horizontal_flip"]==want["flip"]
        assert found["rotation_degrees"]==want["rotation_degrees"]

def timestep_noise(batch,shape,device):
    assert shape==(2,1,512,512)
    t=torch.tensor([r["t"] for r in batch],device=device,dtype=torch.long)
    parts=[]
    for r in batch:
        gen=torch.Generator(device=device).manual_seed(r["noise_seed"])
        parts.append(torch.randn((1,1,512,512),generator=gen,device=device))
    return t,torch.cat(parts,dim=0)
