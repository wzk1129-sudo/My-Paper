"""Read-only paired diagnosis of joint RDDM EMA sampling paths."""
import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
import numpy as np
import torch
from fast_dev.data import FastPairs, DataAccessGuard
from fast_dev.protocol import CONFIG, FAST_INDICES, check_config
from mayo2016_smallset_v1.frozen_protocol import slice_metrics
from mayo2016_smallset_v1.runtime import seed_everything, sha256, write_json
from res_noise_beta_time_v1.model import build_joint

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'experiments/res_noise_beta_time_v1/runs/beta_time_fast8k_20260929'
METRICS=('psnr','ssim','rmse_hu','mae_hu')

class Moments:
    def __init__(self):
        self.n=0;self.sx=0.;self.sy=0.;self.sxx=0.;self.syy=0.;self.sxy=0.;self.sdiff=0.
        self.xmin=float('inf');self.xmax=float('-inf');self.ymin=float('inf');self.ymax=float('-inf')
    def add(self,x,y):
        x=x.detach().double();y=y.detach().double()
        assert x.shape==y.shape and torch.isfinite(x).all() and torch.isfinite(y).all()
        self.n+=x.numel()
        self.sx+=float(x.sum());self.sy+=float(y.sum())
        self.sxx+=float((x*x).sum());self.syy+=float((y*y).sum())
        self.sxy+=float((x*y).sum());self.sdiff+=float(((x-y)**2).sum())
        self.xmin=min(self.xmin,float(x.min()));self.xmax=max(self.xmax,float(x.max()))
        self.ymin=min(self.ymin,float(y.min()));self.ymax=max(self.ymax,float(y.max()))
    def result(self):
        mx=self.sx/self.n;my=self.sy/self.n
        vx=max(0.,self.sxx/self.n-mx*mx);vy=max(0.,self.syy/self.n-my*my)
        cov=self.sxy/self.n-mx*my
        return {'pixels':self.n,'mse':self.sdiff/self.n,
                'pearson_correlation':cov/math.sqrt(vx*vy) if vx>0 and vy>0 else None,
                'learned_mean':mx,'learned_std':math.sqrt(vx),
                'learned_min':self.xmin,'learned_max':self.xmax,
                'analytic_mean':my,'analytic_std':math.sqrt(vy),
                'analytic_min':self.ymin,'analytic_max':self.ymax}

def evaluate_step(step,cfg,data,device,output):
    checkpoint=RUN/'checkpoints'/('step_{}.pt'.format(step))
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert ck['optimizer_step']==step and ck['ema_updates']==step
    assert ck['config']['training']['batch_size']==2
    assert ck['config']['training']['gradient_accumulation']==1
    assert ck['source_hashes']['src/rddm_rleb.py']==cfg['model']['source_sha256']
    model=build_joint().to(device)
    model.load_state_dict(ck['ema_state_dict'],strict=True)
    model.eval().requires_grad_(False)
    del ck
    assert model.is_ddim_sampling and model.sampling_timesteps==2
    assert model.model.num_unet==1 and model.model.objective=='pred_res_noise'
    assert model.model.unet0.noise_head.out_channels==1
    assert sum(p.numel() for p in model.parameters())==13131010
    raw=[]
    def capture(_module,_inputs,result):
        assert len(result)==2
        raw.append(result)
    handle=model.model.register_forward_hook(capture)
    original=model.model_predictions
    active='learned'
    states=[]
    moments={}
    def traced(x_input,x,t,x_input_condition=0,x_self_cond=None,clip_denoised=True):
        nonlocal states
        result=original(x_input,x,t,x_input_condition,x_self_cond,clip_denoised)
        assert len(raw)==1
        pred_res,pred_noise=raw.pop()
        timestep=int(t.item())
        analytic=model.predict_noise_from_res(x,t,x_input,pred_res.clamp(-1,1))
        if active=='learned':
            moments.setdefault(timestep,Moments()).add(pred_noise,analytic)
            states.append(x.detach().clone())
        else:
            assert states and torch.equal(x,states.pop(0)),'Sampler states differ between paths'
        return result
    model.model_predictions=traced
    rows=[]
    start=time.perf_counter()
    with torch.inference_mode():
        for position in range(len(data)):
            ldct,target,meta=data.load(position)
            assert meta['patient_id']=='L333' and meta['slice_index']==FAST_INDICES[position]
            ldct=torch.from_numpy(ldct).unsqueeze(0).to(device)
            target=torch.from_numpy(target).unsqueeze(0).to(device)
            assert ldct.shape==target.shape==(1,1,512,512)
            assert torch.isfinite(ldct).all() and torch.isfinite(target).all()
            seed=cfg['evaluation']['validation_seed']+meta['slice_index']
            active='learned';model.test_res_or_noise='res_noise';states=[]
            torch.manual_seed(seed)
            if device.type=='cuda':torch.cuda.manual_seed(seed)
            learned=model.sample([ldct],last=True)[-1]
            assert len(states)==2
            active='residual';model.test_res_or_noise='res'
            torch.manual_seed(seed)
            if device.type=='cuda':torch.cuda.manual_seed(seed)
            residual=model.sample([ldct],last=True)[-1]
            assert not states and not raw
            assert learned.shape==residual.shape==(1,1,512,512)
            assert torch.isfinite(learned).all() and torch.isfinite(residual).all()
            a=slice_metrics(learned,target,cfg)
            b=slice_metrics(residual,target,cfg)
            assert all(math.isfinite(a[k]) and math.isfinite(b[k]) for k in METRICS)
            rows.append({'slice_index':meta['slice_index'],'sampling_seed':seed,
                        **{'learned_'+k:a[k] for k in METRICS},
                        **{'residual_'+k:b[k] for k in METRICS},
                        'mean_abs_prediction_difference':float((learned-residual).abs().mean()),
                        'learned_clip_fraction':float(((learned<=0)|(learned>=1)).float().mean()),
                        'residual_clip_fraction':float(((residual<=0)|(residual>=1)).float().mean())})
    handle.remove()
    del model.model_predictions
    elapsed=time.perf_counter()-start
    assert len(rows)==64 and set(moments)=={999,499}
    metric={path:{k:float(np.mean([r[path+'_'+k] for r in rows])) for k in METRICS}
            for path in ('learned','residual')}
    summary={'step':step,'checkpoint':str(checkpoint),'checkpoint_sha256':sha256(checkpoint),
             'weights':'ema','patient':'L333','slices':64,'indices':list(FAST_INDICES),
             'initial_noise':'fixed validation_seed+slice_index, identical for both paths',
             'sampling_timesteps':2,'sum_scale':model.sum_scale,
             'sampler_state_identity':True,
             'metrics':metric,
             'delta_residual_minus_learned':{k:metric['residual'][k]-metric['learned'][k] for k in METRICS},
             'noise_comparison_by_timestep':{str(t):moments[t].result() for t in sorted(moments,reverse=True)},
             'mean_abs_prediction_difference':float(np.mean([r['mean_abs_prediction_difference'] for r in rows])),
             'learned_clip_fraction':float(np.mean([r['learned_clip_fraction'] for r in rows])),
             'residual_clip_fraction':float(np.mean([r['residual_clip_fraction'] for r in rows])),
             'evaluation_seconds':elapsed,'no_optimizer_step':True,
             'time_encoding':'same shared U-Net receives alpha-time for residual and beta-time for noise in separate forwards'}
    with (output/('step_{}_per_slice.csv'.format(step))).open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    write_json(output/('step_{}_summary.json'.format(step)),summary)
    print('STEP {} learned={:.6f} residual={:.6f}'.format(
        step,metric['learned']['psnr'],metric['residual']['psnr']),flush=True)
    return summary

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    cfg=json.loads(CONFIG.read_text())
    check_config(cfg)
    output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(cfg['training']['cpu_threads'])
    seed_everything(cfg['seed'])
    device=torch.device(args.device)
    guard=DataAccessGuard(cfg,'FAST',allow_training=False)
    with guard:
        data=FastPairs(cfg,'validation','FAST')
        results=[evaluate_step(step,cfg,data,device,output) for step in (4000,8000)]
    access=guard.report()
    assert access['validation_indices_opened']==list(FAST_INDICES)
    assert access['test_opens']==access['external_memory_opens']==access['blocked_attempts']==0
    write_json(output/'DATA_ACCESS.json',access)
    write_json(output/'SUMMARY.json',{'steps':results,'data_access':access,
               'formal_source_sha256':sha256(ROOT/'src/rddm_rleb.py'),
               'fast64_split_sha256':sha256(ROOT/'splits/L333_fast64.txt'),
               'evaluator_sha256':sha256(ROOT/'mayo2016_smallset_v1/frozen_protocol.py'),
               'no_training':True,'no_checkpoint_modification':True})
    print('COMPLETE',flush=True)
if __name__=='__main__':main()
