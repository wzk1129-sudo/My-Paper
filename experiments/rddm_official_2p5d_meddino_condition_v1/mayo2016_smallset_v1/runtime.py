"""Training state, EMA, strict provenance and isolated deterministic validation."""
import contextlib
import copy
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from .data import CenterPairs, BatchStream, validate_protocol
from .frozen_protocol import slice_metrics
from src.rddm_rleb import build_baseline, ResnetBlock

ROOT=Path(__file__).resolve().parents[1]
METRICS=('psnr','ssim','rmse_hu','mae_hu')
CSV_FIELDS=('step','weights','patient','slices',*METRICS,'validation_loss','train_loss','gradient_norm','learning_rate','peak_vram_bytes','seconds_per_step','validation_seconds','checkpoint')

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def write_json(path,value):
    path=Path(path); temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w') as f:
        json.dump(value,f,indent=2,allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temporary,path)

def atomic_save(path,value):
    path=Path(path); temporary=path.with_name(path.name+'.tmp')
    with temporary.open('wb') as f:
        torch.save(value,f); f.flush(); os.fsync(f.fileno())
    os.replace(temporary,path)

def source_hashes():
    names=['src/rddm_rleb.py']+[str(p.relative_to(ROOT)) for p in sorted((ROOT/'mayo2016_smallset_v1').glob('*.py'))]
    return {name:sha256(ROOT/name) for name in names}

def provenance():
    def git(*args):return subprocess.check_output(['git',*args],cwd=str(ROOT),text=True).strip()
    return {'git_commit':git('rev-parse','HEAD'),'git_branch':git('branch','--show-current'),'git_status':git('status','--short'),'source_hashes':source_hashes(),'python':os.sys.version,'torch':torch.__version__,'cuda':torch.version.cuda}

def check_config(cfg):
    validate_protocol(cfg)
    assert sha256(ROOT/'src/rddm_rleb.py')==cfg['model']['source_sha256']=='e8ed93223fbdf4a0e2521f73c7a4f213561f6a191a1a7b2a5f95934b85ebf574'
    assert cfg['diffusion']=={'objective':'pred_res','loss_type':'l2','timesteps':1000,'sampling_timesteps':2,'condition':True,'sum_scale':.01}
    tr=cfg['training']
    assert (tr['optimizer'],tr['learning_rate'],tr['betas'],tr['weight_decay'])==('Adam',.0002,[.9,.99],1e-9)
    assert (tr['batch_size'],tr['gradient_accumulation'],tr['effective_batch'],tr['amp'],tr['gradient_clip'],tr['scheduler'])==(2,1,2,False,1.,None)
    assert cfg['seed']==20260829 and 0<tr['max_optimizer_steps']<=60000
    if cfg['mode']=='formal':
        assert tr['max_optimizer_steps']==60000 and tr['validation_every']==10000
        assert tr['matched_checkpoint_steps']==[30000,40000,50000,60000]
    assert cfg['ema']['decay']==.995 and cfg['ema']['update_every']==1
    assert cfg['augmentation']=={'horizontal_flip_probability':.5,'rotation_degrees':[0,90,180,270],'paired':True,'crop':False}

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)

class EMA:
    def __init__(self,model,decay):
        self.model=copy.deepcopy(model).eval().requires_grad_(False)
        self.decay=decay; self.updates=0
    @torch.no_grad()
    def update(self,model):
        for ep,p in zip(self.model.parameters(),model.parameters()):ep.mul_(self.decay).add_(p,alpha=1-self.decay)
        for eb,b in zip(self.model.buffers(),model.buffers()):eb.copy_(b)
        self.updates+=1

def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda']])

@torch.no_grad()
def validate(model,dataset,cfg,device,limit=None,output=None):
    assert all(r[0]=='L333' for r in dataset.records) and len(dataset)==610
    model.eval(); rows=[]; started=time.perf_counter()
    devices=[device.index or 0] if device.type=='cuda' else []
    positions=range(len(dataset) if limit is None else min(limit,len(dataset)))
    # Preserve training CPU/CUDA RNG. Each slice's random realization is checkpoint-independent.
    with torch.random.fork_rng(devices=devices):
        with open(os.devnull,'w') as quiet,contextlib.redirect_stderr(quiet):
            for pos in positions:
                ldct,target,meta=dataset.load(pos)
                ldct=torch.from_numpy(ldct).unsqueeze(0).to(device)
                target=torch.from_numpy(target).unsqueeze(0).to(device)
                sampling_seed=cfg['evaluation']['validation_seed']+meta['slice_index']
                torch.manual_seed(sampling_seed)
                if device.type=='cuda':torch.cuda.manual_seed(sampling_seed)
                prediction=model.sample([ldct],last=True)[-1]
                assert prediction.shape==(1,1,512,512) and torch.isfinite(prediction).all()
                metrics=slice_metrics(prediction,target,cfg)
                if not all(math.isfinite(metrics[k]) for k in METRICS):raise FloatingPointError('Nonfinite validation metric')
                torch.manual_seed(sampling_seed+cfg['evaluation']['loss_seed_offset'])
                if device.type=='cuda':torch.cuda.manual_seed(sampling_seed+cfg['evaluation']['loss_seed_offset'])
                loss=float(model([target,ldct])[0])
                if not math.isfinite(loss):raise FloatingPointError('Nonfinite validation loss')
                rows.append({**meta,**{k:metrics[k] for k in METRICS},'validation_loss':loss,'sampling_seed':sampling_seed})
                if limit is None and (pos+1)%100==0:print('validation_progress={}/610'.format(pos+1),flush=True)
    if device.type=='cuda':torch.cuda.synchronize(device)
    if output is not None:write_json(output,rows)
    result={k:float(np.mean([r[k] for r in rows])) for k in (*METRICS,'validation_loss')}
    return {**result,'slices':len(rows),'patient':'L333','weights':'ema','validation_seconds':time.perf_counter()-started}

class TrainingState:
    def __init__(self,cfg,device):
        check_config(cfg);self.cfg=cfg;self.device=torch.device(device)
        torch.set_num_threads(cfg['training']['cpu_threads'])
        if self.device.type=='cuda':torch.cuda.set_device(self.device)
        seed_everything(cfg['seed'])
        self.data=CenterPairs(cfg,'train'); self.validation_data=CenterPairs(cfg,'validation')
        self.stream=BatchStream(self.data,cfg['seed'])
        self.model=build_baseline().to(self.device)
        assert sum(p.numel() for p in self.model.parameters())==13130945
        assert sum(isinstance(m,ResnetBlock) for m in self.model.modules())==10
        forbidden=('mamba','ss2d','cssm','dacb','clip','dose','anatom','adaln','attention','prompt','text_mlp')
        assert not any(any(x in (n+type(m).__name__).lower() for x in forbidden) for n,m in self.model.named_modules())
        tr=cfg['training'];self.optimizer=torch.optim.Adam(self.model.parameters(),lr=tr['learning_rate'],betas=tuple(tr['betas']),weight_decay=tr['weight_decay'])
        self.ema=EMA(self.model,cfg['ema']['decay'])
        self.step=0;self.history=[];self.best_psnr=None;self.best_step=None;self.best_checkpoint=None
        self.cumulative_train_seconds=0.;self.previous_wall_seconds=0.;self.start_wall=time.perf_counter()
        self.last_record={};self.peak_vram=0;self.gradient_stats={"min":None,"max":None,"gt_100_count":0,"nonfinite_count":0}
        self.source_hashes=source_hashes()
        self._shape_hook=self.model.model.unet0.register_forward_hook(self._check_output)
        self.model.model.unet0.init_conv.register_forward_pre_hook(self._check_input)
        if self.device.type=='cuda':torch.cuda.reset_peak_memory_stats(self.device)

    def _check_output(self,module,inputs,out):
        assert out.shape==(inputs[0].shape[0],1,512,512)
    def _check_input(self,module,inputs):
        assert inputs[0].shape==(2,2,512,512)

    def update(self):
        if self.device.type=='cuda':torch.cuda.synchronize(self.device)
        started=time.perf_counter()
        ldct,target,metadata=self.stream.next(2)
        ldct=ldct.to(self.device);target=target.to(self.device)
        self.model.train();self.optimizer.zero_grad(set_to_none=True)
        loss=self.model([target,ldct])[0]
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss at '+str(self.step+1))
        loss.backward()
        if any(p.grad is None for p in self.model.parameters()):raise RuntimeError('Missing parameter gradient')
        norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.)
        grad=float(norm)
        if not math.isfinite(grad):raise FloatingPointError('Nonfinite gradient at '+str(self.step+1))
        self.gradient_stats['min']=grad if self.gradient_stats['min'] is None else min(grad,self.gradient_stats['min'])
        self.gradient_stats['max']=grad if self.gradient_stats['max'] is None else max(grad,self.gradient_stats['max'])
        self.gradient_stats['gt_100_count']+=int(grad>100)
        self.optimizer.step();self.ema.update(self.model);self.step+=1
        if self.device.type=='cuda':torch.cuda.synchronize(self.device)
        elapsed=time.perf_counter()-started;self.cumulative_train_seconds+=elapsed
        if self.device.type=='cuda':self.peak_vram=max(self.peak_vram,torch.cuda.max_memory_allocated(self.device))
        self.last_record={'optimizer_step':self.step,'train_loss':float(loss.detach()),'gradient_norm_pre_clip':grad,'learning_rate':self.optimizer.param_groups[0]['lr'],'seconds_per_step':elapsed,'peak_vram_bytes':self.peak_vram,'vram_allocated_bytes':torch.cuda.memory_allocated(self.device) if self.device.type=='cuda' else 0,'vram_reserved_bytes':torch.cuda.memory_reserved(self.device) if self.device.type=='cuda' else 0,'ema_updates':self.ema.updates,'amp':False,'nonfinite':False,'epoch':self.stream.epoch,'samples':metadata}
        return self.last_record

    def payload(self):
        return {'format_version':1,'gradient_stats':self.gradient_stats,'model_state_dict':self.model.state_dict(),'ema_state_dict':self.ema.model.state_dict(),'ema_updates':self.ema.updates,'optimizer_state_dict':self.optimizer.state_dict(),'optimizer_step':self.step,'rng':rng_state(),'data_stream':self.stream.state_dict(),'config':self.cfg,'source_hashes':self.source_hashes,'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(ROOT),text=True).strip(),'scheduler_state_dict':None,'validation_history':self.history,'best_psnr':self.best_psnr,'best_step':self.best_step,'best_checkpoint':self.best_checkpoint,'last_train_record':self.last_record,'cumulative_train_seconds':self.cumulative_train_seconds,'elapsed_wall_seconds':self.previous_wall_seconds+time.perf_counter()-self.start_wall,'peak_vram_bytes':self.peak_vram}

    def resume(self,path):
        ck=torch.load(str(path),map_location='cpu')
        assert ck['config']==self.cfg,'Config changed on resume'
        assert ck['source_hashes']==self.source_hashes,'Source changed on resume'
        assert ck['optimizer_step']<self.cfg['training']['max_optimizer_steps']
        self.model.load_state_dict(ck['model_state_dict'],strict=True)
        self.ema.model.load_state_dict(ck['ema_state_dict'],strict=True);self.ema.updates=ck['ema_updates']
        self.optimizer.load_state_dict(ck['optimizer_state_dict'])
        self.step=ck['optimizer_step'];self.stream.load_state_dict(ck['data_stream'])
        self.history=ck['validation_history'];self.best_psnr=ck['best_psnr'];self.best_step=ck['best_step'];self.best_checkpoint=ck['best_checkpoint']
        self.last_record=ck['last_train_record'];self.cumulative_train_seconds=ck['cumulative_train_seconds'];self.previous_wall_seconds=ck['elapsed_wall_seconds'];self.peak_vram=ck['peak_vram_bytes'];self.gradient_stats=ck['gradient_stats']
        assert self.ema.updates==self.step
        restore_rng(ck['rng'])
        return self.step
