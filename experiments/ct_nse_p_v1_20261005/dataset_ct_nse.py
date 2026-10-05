"""Strict NDCT-only Mayo Fast-Dev reader and access audit for CT-NSE-P."""
from __future__ import annotations
import hashlib,json,os,re,sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
from ct_nse import sample_anchors

COUNTS={'L067':560,'L143':585,'L333':610}
TRAIN=('L067','L143')
VAL=('L333',)
NAME=re.compile(r'^(L\d{3})_target_(\d{3})_img\.npy$')

class NDCTAccessAudit:
    def __init__(self,root:Path):
        self.root=str(root.resolve())
        self.active=False;self.split=None;self.blocked=0
        self.opens=Counter();self.unique={p:set() for p in COUNTS};self.ldct_opens=0;self.l506_opens=0
        self.external_memory_opens=0
        sys.addaudithook(self._hook)
    def _hook(self,event,args):
        if not self.active or event not in ('open','os.listdir','os.scandir') or not args:return
        name=args[0]
        if not isinstance(name,(str,bytes,os.PathLike)):return
        path=os.path.abspath(os.fsdecode(name))
        if path!=self.root and not path.startswith(self.root+os.sep):return
        if event!='open':
            self.blocked+=1
            raise PermissionError('data directory enumeration forbidden')
        match=NAME.fullmatch(os.path.basename(path))
        flags=args[2] if len(args)>2 and isinstance(args[2],int) else 0
        writing=bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
        patient=match.group(1) if match else None
        index=int(match.group(2)) if match else None
        valid=(match is not None and os.path.dirname(path)==self.root and
            not writing and self.split in ('train','val') and
            patient in (TRAIN if self.split=='train' else VAL) and
            0<=index<COUNTS[patient])
        if not valid:
            self.blocked+=1
            if '_25_' in path:self.ldct_opens+=1
            if 'L506' in path:self.l506_opens+=1
            raise PermissionError('unapproved CT data access: '+path)
        self.opens[(self.split,patient)]+=1
        self.unique[patient].add(index)
    @contextmanager
    def role(self,split):
        assert split in ('train','val') and self.split is None
        self.split=split
        try:yield
        finally:self.split=None
    def __enter__(self):
        self.active=True;return self
    def __exit__(self,*_):
        self.active=False
    def report(self):
        return {'train_ndct_opens':{p:self.opens[('train',p)] for p in TRAIN},
            'val_ndct_opens':{p:self.opens[('val',p)] for p in VAL},
            'train_ndct_unique_opened':{p:len(self.unique[p]) for p in TRAIN},
            'val_ndct_unique_opened':{p:len(self.unique[p]) for p in VAL},
            'train_ndct_available':{p:COUNTS[p] for p in TRAIN},
            'val_ndct_available':{p:COUNTS[p] for p in VAL},
            'ldct_opens':self.ldct_opens,'L506_opens':self.l506_opens,
            'external_memory_opens':self.external_memory_opens,
            'blocked_attempts':self.blocked,'only_target_npy_allowed':True}

class NDCTOnlyDataset:
    def __init__(self,root:Path,audit:NDCTAccessAudit):
        self.root=Path(root).resolve();self.audit=audit
        assert str(self.root)==audit.root and self.root.is_dir()
        self.train_keys=tuple((p,i) for p in TRAIN for i in range(COUNTS[p]))
        self.val_keys=tuple(('L333',i) for i in range(COUNTS['L333']))
        for patient,index in (*self.train_keys,*self.val_keys):
            path=self.path(patient,index)
            assert path.parent==self.root and not path.is_symlink() and path.is_file(),str(path)
        serial=[f'{p}_target_{i:03d}_img.npy' for p,i in (*self.train_keys,*self.val_keys)]
        self.manifest_sha256=hashlib.sha256(json.dumps(serial,separators=(',',':')).encode()).hexdigest()
    def path(self,patient,index):
        assert patient in COUNTS and 0<=index<COUNTS[patient]
        return self.root/f'{patient}_target_{index:03d}_img.npy'
    def load(self,patient,index,split):
        assert (patient,index) in (self.train_keys if split=='train' else self.val_keys)
        with self.audit.role(split):
            raw=np.load(self.path(patient,index),allow_pickle=False)
        assert raw.shape==(512,512) and np.isfinite(raw).all()
        hu=raw.astype(np.float32)-1024.0
        result=(np.clip(hu,-1024.0,3072.0)+1024.0)/4096.0
        assert result.dtype==np.float32 and np.isfinite(result).all()
        assert 0.0<=float(result.min())<=float(result.max())<=1.0
        return result
    def train_batch(self,indices,rng,crop=420):
        arrays=[]
        for pos in indices:
            patient,index=self.train_keys[int(pos)]
            image=self.load(patient,index,'train')
            top=int(rng.integers(0,512-crop+1))
            left=int(rng.integers(0,512-crop+1))
            patch=image[top:top+crop,left:left+crop]
            if bool(rng.integers(0,2)):patch=patch[:,::-1]
            arrays.append(np.ascontiguousarray(patch[None]))
        return torch.from_numpy(np.stack(arrays))
    def val_manifest(self,seed,crop=420,k=16):
        rng=np.random.default_rng(int(seed)+500000)
        out=[]
        for patient,index in self.val_keys:
            top=int(rng.integers(0,512-crop+1))
            left=int(rng.integers(0,512-crop+1))
            anchors=sample_anchors(rng,1,crop//4,crop//4,k)[0].tolist()
            out.append({'patient':patient,'index':index,'crop_top':top,
                        'crop_left':left,'anchors_yx':anchors})
        return out
    def val_batch(self,records,crop=420):
        arrays=[];anchors=[]
        for item in records:
            image=self.load(item['patient'],item['index'],'val')
            top=item['crop_top'];left=item['crop_left']
            arrays.append(np.ascontiguousarray(image[top:top+crop,left:left+crop][None]))
            anchors.append(np.asarray(item['anchors_yx'],dtype=np.int64))
        return torch.from_numpy(np.stack(arrays)),torch.from_numpy(np.stack(anchors))

