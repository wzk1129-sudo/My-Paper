"""Baseline state transitions and optimizer, with a variant-only model factory."""
import time,torch
from mayo2016_smallset_v1.runtime import TrainingState,EMA,seed_everything
from fast_dev.data import FastPairs,BatchStream
from .protocol import BASELINE,check_variant,variant_hashes
from .model import build_variant

class VariantState(TrainingState):
    def __init__(self,cfg,device):
        check_variant(cfg);self.cfg=cfg;self.device=torch.device(device)
        torch.set_num_threads(cfg['training']['cpu_threads'])
        if self.device.type=='cuda':torch.cuda.set_device(self.device)
        seed_everything(cfg['seed'])
        self.data=FastPairs(BASELINE,'train');self.validation_data=FastPairs(BASELINE,'validation','FAST')
        self.stream=BatchStream(self.data,cfg['seed'])
        self.model=build_variant(cfg['variant']).to(self.device)
        self.parameters_count=sum(p.numel() for p in self.model.parameters())
        tr=cfg['training']
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=tr['learning_rate'],betas=tuple(tr['betas']),weight_decay=tr['weight_decay'])
        self.ema=EMA(self.model,cfg['ema']['decay'])
        self.step=0;self.history=[];self.best_psnr=None;self.best_step=None;self.best_checkpoint=None
        self.cumulative_train_seconds=0.;self.previous_wall_seconds=0.;self.start_wall=time.perf_counter()
        self.last_record={};self.peak_vram=0
        self.gradient_stats={'min':None,'max':None,'gt_100_count':0,'nonfinite_count':0}
        self.source_hashes=variant_hashes(cfg['variant']['id'])
        if self.device.type=='cuda':torch.cuda.reset_peak_memory_stats(self.device)
    def update(self):
        record=super().update()
        if self.cfg['variant']['kind']=='F2' and self.step>=2:
            for name,p in self.model.named_parameters():
                if 'cond_' in name and (p.grad is None or not torch.isfinite(p.grad).all()):
                    raise FloatingPointError('Missing/nonfinite F2 condition gradient: '+name)
        if self.cfg['variant']['kind']=='F3':
            for name,p in self.model.named_parameters():
                if ('conv2.' in name or 'norm2.' in name) and (p.grad is None or not torch.isfinite(p.grad).all()):
                    raise FloatingPointError('Missing/nonfinite F3 second-conv gradient: '+name)
        return record
