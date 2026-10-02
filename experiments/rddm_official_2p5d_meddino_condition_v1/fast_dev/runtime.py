"""Same baseline optimization/EMA/RNG math, independently initialized data contract."""
import time
import torch
from mayo2016_smallset_v1.runtime import TrainingState, EMA, seed_everything
from src.rddm_rleb import build_baseline, ResnetBlock
from .protocol import check_config, source_hashes
from .data import FastPairs, BatchStream

class FastState(TrainingState):
    def __init__(self,cfg,device):
        check_config(cfg);self.cfg=cfg;self.device=torch.device(device)
        torch.set_num_threads(cfg['training']['cpu_threads'])
        if self.device.type=='cuda':torch.cuda.set_device(self.device)
        seed_everything(cfg['seed'])
        self.data=FastPairs(cfg,'train');self.validation_data=FastPairs(cfg,'validation','FAST')
        self.stream=BatchStream(self.data,cfg['seed'])
        self.model=build_baseline().to(self.device)
        assert sum(p.numel() for p in self.model.parameters())==13130945
        assert sum(isinstance(m,ResnetBlock) for m in self.model.modules())==10
        forbidden=('mamba','ss2d','cssm','dacb','clip','dose','anatom','adaln','attention','prompt','text_mlp')
        assert not any(any(x in (n+type(m).__name__).lower() for x in forbidden) for n,m in self.model.named_modules())
        tr=cfg['training']
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=tr['learning_rate'],betas=tuple(tr['betas']),weight_decay=tr['weight_decay'])
        self.ema=EMA(self.model,cfg['ema']['decay'])
        self.step=0;self.history=[];self.best_psnr=None;self.best_step=None;self.best_checkpoint=None
        self.cumulative_train_seconds=0.;self.previous_wall_seconds=0.;self.start_wall=time.perf_counter()
        self.last_record={};self.peak_vram=0
        self.gradient_stats={'min':None,'max':None,'gt_100_count':0,'nonfinite_count':0}
        self.source_hashes=source_hashes()
        self._shape_hook=self.model.model.unet0.register_forward_hook(self._check_output)
        self.model.model.unet0.init_conv.register_forward_pre_hook(self._check_input)
        if self.device.type=='cuda':torch.cuda.reset_peak_memory_stats(self.device)
    # update, payload, resume and shape hooks are inherited without alterations.
