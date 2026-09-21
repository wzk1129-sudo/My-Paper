"""Center-only paired CT access with a fixed patient allowlist and no glob/listdir."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .frozen_protocol import normalize_stored_ct

COUNTS = {'L067':560,'L096':823,'L109':318,'L143':585,'L192':600,'L286':525,'L291':856,'L310':533,'L333':610}
TRAIN = tuple(p for p in COUNTS if p != 'L333')
VALIDATION = ('L333',)

def validate_protocol(config):
    data = config['data']
    assert tuple(data['train_patients']) == TRAIN
    assert tuple(data['validation_patients']) == VALIDATION
    assert not set(TRAIN).intersection(VALIDATION)
    assert config['allowed_slice_counts'] == COUNTS
    assert (data['dose_label'], data['image_size'], data['stored_value_offset_hu'], data['hu_min'], data['hu_max']) == ('25',512,-1024.,-1024.,3072.)
    assert config['evaluation']['hu_window'] == [-1000.,1000.]

class CenterPairs:
    def __init__(self, config, split):
        validate_protocol(config)
        if split not in ('train','validation'):
            raise ValueError('Only train and validation are accessible')
        self.config = config
        self.root = Path(config['data']['root']).resolve()
        self.records = []
        for patient in TRAIN if split == 'train' else VALIDATION:
            for index in range(COUNTS[patient]):
                paths = tuple(self.root / ('{}_{}_{:03d}_img.npy'.format(patient,view,index)) for view in ('25','target'))
                for path in paths:
                    if path.parent != self.root or not path.is_file():
                        raise RuntimeError('Missing allowed pair: ' + str(path))
                self.records.append((patient,index,*paths))
        self.manifest_hash = hashlib.sha256(json.dumps([(p,i,str(a),str(b)) for p,i,a,b in self.records],separators=(',',':')).encode()).hexdigest()

    def __len__(self):
        return len(self.records)

    def load(self,index):
        patient,slice_index,ldct_path,ndct_path = self.records[int(index)]
        arrays = []
        for path in (ldct_path,ndct_path):
            raw = np.load(path,allow_pickle=False)
            if raw.shape != (512,512) or not np.isfinite(raw).all():
                raise ValueError('Invalid pair image: ' + str(path))
            normalized = normalize_stored_ct(raw,self.config['data'])
            if normalized.dtype != np.float32 or not np.isfinite(normalized).all() or normalized.min()<0 or normalized.max()>1:
                raise ValueError('Invalid normalized image: ' + str(path))
            arrays.append(normalized[None])
        return arrays[0],arrays[1],{'patient_id':patient,'slice_index':slice_index}

    def audit(self):
        rng = np.random.RandomState(self.config['seed'])
        samples = []
        for patient in dict.fromkeys(r[0] for r in self.records):
            positions = [i for i,r in enumerate(self.records) if r[0]==patient]
            for index in rng.choice(positions,2,replace=False):
                ldct,ndct,meta = self.load(index)
                samples.append({**meta,'shape':list(ldct.shape),'dtype':str(ldct.dtype),'ldct_min':float(ldct.min()),'ldct_max':float(ldct.max()),'ndct_min':float(ndct.min()),'ndct_max':float(ndct.max()),'finite':True,'pair_alignment':'identical patient and slice keys; center only'})
        return {'total':len(self),'counts':{p:sum(r[0]==p for r in self.records) for p in dict.fromkeys(r[0] for r in self.records)},'manifest_sha256':self.manifest_hash,'samples':samples,'directory_enumeration':False}

class BatchStream:
    """Explicit serial stream: permutation, cursor and augmentation RNG are resumable."""
    def __init__(self,dataset,seed):
        self.dataset = dataset
        self.rng = np.random.RandomState(seed)
        self.permutation = self.rng.permutation(len(dataset))
        self.cursor = 0
        self.epoch = 0

    def next(self,batch_size=2):
        ldcts,ndcts,metadata = [],[],[]
        for _ in range(batch_size):
            if self.cursor == len(self.permutation):
                self.permutation = self.rng.permutation(len(self.dataset))
                self.cursor = 0
                self.epoch += 1
            position = int(self.permutation[self.cursor]); self.cursor += 1
            ldct,ndct,meta = self.dataset.load(position)
            flip = bool(self.rng.rand()<.5)
            rotation = int(self.rng.randint(4))
            def transform(x):
                if flip: x = np.flip(x,axis=-1)
                return np.ascontiguousarray(np.rot90(x,rotation,axes=(-2,-1)))
            ldcts.append(transform(ldct)); ndcts.append(transform(ndct))
            metadata.append({**meta,'horizontal_flip':flip,'rotation_degrees':rotation*90})
        return torch.from_numpy(np.stack(ldcts)),torch.from_numpy(np.stack(ndcts)),metadata

    def state_dict(self):
        return {'rng':self.rng.get_state(),'permutation':self.permutation.copy(),'cursor':self.cursor,'epoch':self.epoch,'manifest_sha256':self.dataset.manifest_hash}

    def load_state_dict(self,state):
        assert state['manifest_sha256']==self.dataset.manifest_hash
        assert sorted(state['permutation'].tolist())==list(range(len(self.dataset)))
        assert 0<=state['cursor']<=len(self.dataset)
        self.rng.set_state(state['rng']); self.permutation=state['permutation'].copy()
        self.cursor=state['cursor']; self.epoch=state['epoch']
