"""Center-only datasets and a separate, opt-in external-memory interface."""
import hashlib
import json
import os
import re
import sys
from pathlib import Path
import numpy as np
from mayo2016_smallset_v1.data import CenterPairs, BatchStream, COUNTS
from mayo2016_smallset_v1.frozen_protocol import normalize_stored_ct
from .protocol import FAST_TRAIN, MEMORY, validation_indices, check_config

class FastPairs(CenterPairs):
    # Reuse the exact formal load/normalization and paired augmentation implementation.
    def __init__(self, cfg, split, mode='FAST'):
        if split not in ('train', 'validation'): raise ValueError('Only fast train or L333 validation')
        check_config(cfg)
        self.config = cfg
        self.root = Path(cfg['data']['root']).resolve()
        self.role = split
        self.mode = mode
        self.records = []
        if split == 'train': keys = [(p,i) for p in FAST_TRAIN for i in range(COUNTS[p])]
        else: keys = [('L333',i) for i in validation_indices(mode)]
        for patient,index in keys:
            paths = tuple(self.root / '{}_{}_{:03d}_img.npy'.format(patient,v,index) for v in ('25','target'))
            for path in paths:
                if path.parent != self.root or path.is_symlink() or not path.is_file():
                    raise RuntimeError('Missing/nonlocal approved pair: '+str(path))
            self.records.append((patient,index,*paths))
        self.manifest_hash = hashlib.sha256(json.dumps([(p,i,str(a),str(b)) for p,i,a,b in self.records],separators=(',',':')).encode()).hexdigest()

class ExternalMemory:
    """Future LDCT structure-key / NDCT detail-value pairs; never an optimizer dataset.

    No model, embeddings or retrieval is implemented here. Constructing this object
    does not read/stat images. Iteration produces allowlisted pair paths only.
    """
    def __init__(self,cfg):
        check_config(cfg); self.cfg=cfg; self.root=Path(cfg['data']['root']).resolve()
    def pair_paths(self,patient,index):
        if patient not in MEMORY: raise ValueError('Patient is not an external-memory donor')
        if not isinstance(index,int) or not 0 <= index < COUNTS[patient]: raise ValueError('Invalid memory slice')
        return {'patient':patient,'patient_id':patient,'slice_index':index,'role':'external_memory_only','eligible_as_query_supervision':False,
            'ldct_structure_key_source':self.root/'{}_25_{:03d}_img.npy'.format(patient,index),
            'ndct_detail_value_source':self.root/'{}_target_{:03d}_img.npy'.format(patient,index)}
    def iter_pairs(self):
        for p in MEMORY:
            for i in range(COUNTS[p]): yield self.pair_paths(p,i)
    def load_pair(self,patient,index):
        record=self.pair_paths(patient,index)  # reject invalid roles before any filesystem access
        result={}
        for key in ('ldct_structure_key_source','ndct_detail_value_source'):
            path=record[key]
            if path.is_symlink() or path.resolve().parent!=self.root: raise ValueError('Nonlocal memory file')
            raw=np.load(path,allow_pickle=False)
            if raw.shape!=(512,512) or not np.isfinite(raw).all(): raise ValueError('Invalid memory pair')
            result[key]=normalize_stored_ct(raw,self.cfg['data'])[None]
        return result

class DataAccessGuard:
    """Audit-hook allowlist for actual NPY opens; forbid data directory enumeration.

    Training enables just two optimization patients plus the selected validation
    indices. External memory is opt-in for future retrieval callers and is off here.
    """
    def __init__(self,cfg,mode='FAST',allow_training=True,allow_memory=False):
        self.root=os.path.abspath(cfg['data']['root']);self.mode=mode
        self.allowed={('L333',i) for i in validation_indices(mode)}
        if allow_training:self.allowed.update((p,i) for p in FAST_TRAIN for i in range(COUNTS[p]))
        if allow_memory:self.allowed.update((p,i) for p in MEMORY for i in range(COUNTS[p]))
        self.reads=[];self.blocked=0;self.active=False;self.allow_memory=allow_memory
    def _hook(self,event,args):
        if not self.active or event not in ('open','os.listdir','os.scandir') or not args:return
        value=args[0]
        if not isinstance(value,(str,bytes,os.PathLike)):return
        path=os.path.abspath(os.fsdecode(value))
        if path!=self.root and not path.startswith(self.root+os.sep):return
        if event!='open':
            self.blocked+=1;raise PermissionError('Data directory enumeration prohibited')
        match=re.fullmatch(r'(L\d{3})_(25|target)_(\d{3})_img\.npy',os.path.basename(path))
        flags=args[2] if len(args)>2 and isinstance(args[2],int) else 0
        writing=flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)
        key=(match[1],int(match[3])) if match else None
        if os.path.dirname(path)!=self.root or key not in self.allowed or writing:
            self.blocked+=1;raise PermissionError('Read outside approved fast-dev pairs or attempted data write')
        self.reads.append((key[0],key[1],match[2]))
    def __enter__(self):self.active=True;sys.addaudithook(self._hook);return self
    def __exit__(self,*args):self.active=False
    def report(self):
        pats=sorted(set(p for p,i,v in self.reads))
        return {'patients_opened':pats,'opens':len(self.reads),'blocked_attempts':self.blocked,
            'validation_indices_opened':sorted(set(i for p,i,v in self.reads if p=='L333')),
            'training_patients_opened':sorted(set(p for p,i,v in self.reads if p in FAST_TRAIN)),
            'external_memory_opens':sum(p in MEMORY for p,i,v in self.reads),
            'test_opens':sum(p not in set(FAST_TRAIN)|set(MEMORY)|{'L333'} for p,i,v in self.reads),
            'external_memory_enabled':self.allow_memory,'mode':self.mode}
