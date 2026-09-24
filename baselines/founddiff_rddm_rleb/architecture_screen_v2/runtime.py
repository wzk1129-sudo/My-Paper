"""Independent paired 12k training state for each V2 architecture arm."""
import hashlib
import json
import math
import time
from pathlib import Path
import torch
from fast_dev.data import FastPairs
from mayo2016_smallset_v1.data import BatchStream
from mayo2016_smallset_v1.runtime import EMA, rng_state, restore_rng
from .protocol import config, construct, source_manifest

def rng_hash(state):
    h=hashlib.sha256()
    h.update(repr(state['python']).encode())
    np_state=state['numpy']
    h.update(np_state[0].encode())
    h.update(np_state[1].tobytes())
    h.update(repr(np_state[2:]).encode())
    h.update(state['torch'].cpu().numpy().tobytes())
    for tensor in state['cuda']:
        h.update(tensor.cpu().numpy().tobytes())
    return h.hexdigest()

class ScreenState:
    def __init__(self,arm,device):
        self.cfg,self.base=config();self.arm=arm;self.device=torch.device(device)
        torch.set_num_threads(self.cfg['training']['cpu_threads'])
        if self.device.type=='cuda':torch.cuda.set_device(self.device)
        self.train_data=FastPairs(self.base,'train')
        self.validation_data=FastPairs(self.base,'validation','FAST')
        assert len(self.train_data)==1145 and len(self.validation_data)==64
        self.stream=BatchStream(self.train_data,self.cfg['seed'])
        self.model,self.init_audit=construct(arm,self.device)
        self._shape_hook=self.model.model.unet0.register_forward_hook(self._check_output)
        self.training_rng=rng_state()
        self.initial_rng_sha256=rng_hash(self.training_rng)
        tr=self.cfg['training']
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=tr['learning_rate'],
          betas=tuple(tr['betas']),weight_decay=tr['weight_decay'])
        self.ema=EMA(self.model,self.cfg['ema']['decay'])
        self.step=0;self.last=None;self.train_seconds=0.;self.peak_vram=0
        self.gradient_min=None;self.gradient_max=None;self.nonfinite_count=0
        self.stream_digest=hashlib.sha256(b'').digest()
        self.new_grad_abs={n:0. for n,p in self.model.named_parameters() if '.variant_modules.' in n}
        self.manifest=source_manifest()
        if self.device.type=='cuda':torch.cuda.reset_peak_memory_stats(self.device)

    def _check_output(self,module,inputs,output):
        assert output.shape==(inputs[0].shape[0],1,*inputs[0].shape[-2:])

    def update(self):
        if self.device.type=='cuda':torch.cuda.synchronize(self.device)
        started=time.perf_counter()
        ldct,target,metadata=self.stream.next(2)
        assert ldct.shape==target.shape==(2,1,512,512)
        assert ldct.dtype==target.dtype==torch.float32
        assert bool(torch.isfinite(ldct).all()) and bool(torch.isfinite(target).all())
        assert float(ldct.min())>=0 and float(ldct.max())<=1
        assert float(target.min())>=0 and float(target.max())<=1
        assert all(m['patient_id'] in ('L067','L143') for m in metadata)
        for meta in metadata:
            record={key:meta[key] for key in ('patient_id','slice_index','horizontal_flip','rotation_degrees')}
            self.stream_digest=hashlib.sha256(self.stream_digest+json.dumps(record,sort_keys=True,separators=(',',':')).encode()).digest()
        ldct=ldct.to(self.device);target=target.to(self.device)
        self.model.train();self.optimizer.zero_grad(set_to_none=True)
        loss=self.model([target,ldct])[0]
        if not bool(torch.isfinite(loss)):
            self.nonfinite_count+=1;raise FloatingPointError('nonfinite loss')
        loss.backward()
        missing=[n for n,p in self.model.named_parameters() if p.grad is None]
        if missing:raise RuntimeError('missing gradients: '+repr(missing))
        if self.step < 2:
            for name,p in self.model.named_parameters():
                if name in self.new_grad_abs:
                    self.new_grad_abs[name]+=float(p.grad.detach().abs().sum())
        norm=float(torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.))
        if not math.isfinite(norm):
            self.nonfinite_count+=1;raise FloatingPointError('nonfinite gradient')
        self.gradient_min=norm if self.gradient_min is None else min(self.gradient_min,norm)
        self.gradient_max=norm if self.gradient_max is None else max(self.gradient_max,norm)
        self.optimizer.step();self.ema.update(self.model);self.step+=1
        if any(not bool(torch.isfinite(p).all()) for p in self.model.parameters()):
            self.nonfinite_count+=1;raise FloatingPointError('nonfinite parameter')
        if self.device.type=='cuda':
            torch.cuda.synchronize(self.device)
            self.peak_vram=max(self.peak_vram,torch.cuda.max_memory_allocated(self.device))
        elapsed=time.perf_counter()-started;self.train_seconds+=elapsed
        self.last={'step':self.step,'train_loss':float(loss.detach()),
            'gradient_norm_pre_clip':norm,'learning_rate':self.optimizer.param_groups[0]['lr'],
            'seconds_per_step':elapsed,'peak_vram_bytes':self.peak_vram,
            'ema_updates':self.ema.updates,'samples':metadata}
        return self.last

    def payload(self):
        return {'arm':self.arm,'optimizer_step':self.step,'model_state_dict':self.model.state_dict(),
            'ema_state_dict':self.ema.model.state_dict(),'optimizer_state_dict':self.optimizer.state_dict(),
            'ema_updates':self.ema.updates,'rng':rng_state(),'data_stream':self.stream.state_dict(),
            'stream_digest_hex':self.stream_digest.hex(),'config':self.cfg,'source_manifest':self.manifest,
            'initialization_audit':self.init_audit,'initial_rng_sha256':self.initial_rng_sha256,
            'train_seconds':self.train_seconds,'peak_vram_bytes':self.peak_vram,
            'gradient_min':self.gradient_min,'gradient_max':self.gradient_max,
            'nonfinite_count':self.nonfinite_count,'last':self.last}

    def resume(self,path):
        ck=torch.load(str(path),map_location='cpu')
        assert ck['arm']==self.arm and ck['config']==self.cfg and ck['source_manifest']==self.manifest
        assert ck['optimizer_step']<12000 and ck['initialization_audit']==self.init_audit
        self.model.load_state_dict(ck['model_state_dict'],strict=True)
        self.ema.model.load_state_dict(ck['ema_state_dict'],strict=True)
        self.optimizer.load_state_dict(ck['optimizer_state_dict'])
        self.ema.updates=ck['ema_updates'];self.step=ck['optimizer_step']
        self.stream.load_state_dict(ck['data_stream'])
        self.stream_digest=bytes.fromhex(ck['stream_digest_hex'])
        self.train_seconds=ck['train_seconds'];self.peak_vram=ck['peak_vram_bytes']
        self.gradient_min=ck['gradient_min'];self.gradient_max=ck['gradient_max']
        self.nonfinite_count=ck['nonfinite_count'];self.last=ck['last']
        self.initial_rng_sha256=ck['initial_rng_sha256']
        assert self.ema.updates==self.step
        restore_rng(ck['rng'])
