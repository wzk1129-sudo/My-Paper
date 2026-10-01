"""Matched 2-arm fast-dev run. Frozen data, diffusion, loss, optimizer and evaluator."""
import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path
import torch
from mayo2016_smallset_v1.runtime import (TrainingState, EMA, seed_everything,
    rng_state, restore_rng, atomic_save, write_json, sha256)
from mayo2016_smallset_v1.acceptance import nested_equal
from fast_dev.protocol import CONFIG, FAST_TRAIN, FAST_INDICES, check_config, source_hashes
from fast_dev.data import FastPairs, BatchStream, DataAccessGuard
from fast_dev.evaluate import validate
from src.rddm_rleb import build_baseline as build_control
from src.official_rddm_ct import build_baseline as build_candidate, LinearAttention, Attention

ROOT=Path(__file__).resolve().parent
ARMS=('CONTROL_STRIPPED','CANDIDATE_OFFICIAL_RDDM')

def params(model):
    return sum(p.numel() for p in model.parameters())

def finite_params(model):
    return all(bool(torch.isfinite(p).all()) for p in model.parameters())

def attention_grad(model):
    parts=[]
    for m in model.model.unet0.modules():
        if isinstance(m,(LinearAttention,Attention)):
            parts += [p.grad.norm() for p in m.parameters() if p.grad is not None]
    if not parts:return 0.0
    result=float(torch.stack(parts).norm())
    if not math.isfinite(result) or result<=0:raise FloatingPointError('Attention gradient missing/nonfinite')
    return result

def rng_digest():
    h=hashlib.sha256()
    h.update(torch.get_rng_state().numpy().tobytes())
    for state in torch.cuda.get_rng_state_all():h.update(state.cpu().numpy().tobytes())
    return h.hexdigest()

class MatchedState(TrainingState):
    def __init__(self,cfg,device,arm):
        check_config(cfg)
        assert arm in ARMS
        self.cfg=cfg;self.device=torch.device(device);self.arm=arm
        torch.set_num_threads(cfg['training']['cpu_threads'])
        if self.device.type=='cuda':torch.cuda.set_device(self.device)
        seed_everything(cfg['seed'])
        self.data=FastPairs(cfg,'train')
        self.validation_data=FastPairs(cfg,'validation','FAST')
        self.stream=BatchStream(self.data,cfg['seed'])
        preinit=rng_state()
        self.model=(build_control() if arm==ARMS[0] else build_candidate()).to(self.device)
        expected=13130945 if arm==ARMS[0] else 36248513
        assert params(self.model)==expected
        tr=cfg['training']
        assert tr['learning_rate']==2e-4 and tuple(tr['betas'])==(0.9,0.99)
        assert tr['weight_decay']==1e-9 and tr['batch_size']==2
        assert tr['amp'] is False and tr['gradient_clip']==1.
        assert cfg['ema']['decay']==.995 and cfg['seed']==20260829
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=tr['learning_rate'],
            betas=tuple(tr['betas']),weight_decay=tr['weight_decay'])
        self.ema=EMA(self.model,cfg['ema']['decay'])
        restore_rng(preinit)
        self.initial_rng_digest=rng_digest()
        self.step=0;self.history=[];self.best_psnr=None;self.best_step=None;self.best_checkpoint=None
        self.cumulative_train_seconds=0.;self.previous_wall_seconds=0.;self.start_wall=time.perf_counter()
        self.last_record={};self.peak_vram=0
        self.gradient_stats={'min':None,'max':None,'gt_100_count':0,'nonfinite_count':0}
        self.source_hashes=source_hashes()
        self.source_hashes.update({name:sha256(ROOT/name) for name in
            ('matched_train.py','src/official_rddm_ct.py','SOURCE_PROVENANCE.json')})
        self._shape_hook=self.model.model.unet0.register_forward_hook(self._check_output)
        self.model.model.unet0.init_conv.register_forward_pre_hook(self._check_input)
        if self.device.type=='cuda':torch.cuda.reset_peak_memory_stats(self.device)

    def update(self):
        pre=rng_digest()
        original_q_sample=self.model.q_sample
        random_audit={}
        if self.step==0:
            def observed_q_sample(x_start,x_res,t,noise=None):
                random_audit['timesteps']=t.detach().cpu().tolist()
                random_audit['noise_sha256']=hashlib.sha256(
                    noise.detach().cpu().numpy().tobytes()).hexdigest()
                return original_q_sample(x_start,x_res,t,noise=noise)
            self.model.q_sample=observed_q_sample
        try:
            record=super().update()
        finally:
            self.model.q_sample=original_q_sample
        record['attention_gradient_norm']=attention_grad(self.model)
        record['pre_step_rng_sha256']=pre if self.step==1 else None
        record['post_step_rng_sha256']=rng_digest() if self.step==1 else None
        record.update(random_audit)
        if self.step==1 or self.step%100==0:
            record['parameters_finite']=finite_params(self.model)
            if not record['parameters_finite']:raise FloatingPointError('Nonfinite parameter')
        return record

def metric_row(state,output):
    before=rng_state()
    result=validate(state.ema.model,state.validation_data,state.cfg,state.device,
        output/('validation_step_%d.json'%state.step))
    assert nested_equal(before,rng_state()),'Evaluator changed training RNG'
    row={'arm':state.arm,'step':state.step,**result,'train_loss':state.last_record.get('train_loss'),
        'attention_gradient_norm':state.last_record.get('attention_gradient_norm'),
        'parameters_finite':finite_params(state.model),
        'step_time_seconds':state.last_record.get('seconds_per_step'),
        'peak_vram_bytes':state.peak_vram}
    assert row['parameters_finite']
    return row

def save_metrics(path,history):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(history[0]))
        writer.writeheader();writer.writerows(history)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--arm',choices=ARMS,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--steps',type=int,default=8000)
    ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--resume',type=Path)
    args=ap.parse_args()
    if not 1<=args.steps<=8000:raise ValueError('1..8000 steps only')
    if args.smoke and args.steps>2:raise ValueError('Smoke capped at 2 steps')
    cfg=json.loads(CONFIG.read_text());check_config(cfg)
    assert cfg['data']['train_patients']==['L067','L143']
    assert cfg['data']['validation_patients']==['L333']
    assert cfg['evaluation']['indices_file']=='splits/L333_fast64.txt'
    output=args.output.resolve()
    if args.resume:
        assert output.is_dir() and args.resume.is_file()
    else:
        output.mkdir(parents=True,exist_ok=False)
        (output/'checkpoints').mkdir()
    write_json(output/'invocation.json',{'arm':args.arm,'steps':args.steps,
        'config_sha256':sha256(CONFIG),'official_commit':'524caea34cedb3733aa5bf6c27dc1cd6e0469216',
        'device':args.device,'smoke':args.smoke,'resume':str(args.resume) if args.resume else None})
    with DataAccessGuard(cfg,'FAST',allow_memory=False) as guard:
        state=MatchedState(cfg,args.device,args.arm)
        write_json(output/'initial_state.json',{'parameters':params(state.model),
            'initial_rng_sha256':state.initial_rng_digest,'manifest_sha256':state.data.manifest_hash,
            'validation_manifest_sha256':state.validation_data.manifest_hash})
        if args.resume:state.resume(args.resume)
        if not args.smoke and not args.resume:
            row=metric_row(state,output)
            state.history.append(row)
            save_metrics(output/'validation_metrics.csv',state.history)
            print('VALIDATION '+json.dumps(row),flush=True)
        logmode='a' if args.resume else 'x'
        with (output/'train.jsonl').open(logmode,buffering=1) as log:
            while state.step<args.steps:
                record=state.update()
                assert {m['patient_id'] for m in record['samples']}<=set(FAST_TRAIN)
                log.write(json.dumps(record,allow_nan=False)+'\n')
                if state.step==1 or state.step%100==0:
                    print('TRAIN '+json.dumps({k:record[k] for k in
                        ('optimizer_step','train_loss','attention_gradient_norm',
                         'seconds_per_step','peak_vram_bytes')}),flush=True)
                if not args.smoke and state.step%2000==0:
                    row=metric_row(state,output)
                    state.history.append(row)
                    save_metrics(output/'validation_metrics.csv',state.history)
                    cp=output/'checkpoints'/('step_%d.pt'%state.step)
                    if cp.exists():raise FileExistsError(cp)
                    atomic_save(cp,state.payload())
                    print('VALIDATION '+json.dumps(row),flush=True)
        report=guard.report()
        assert report['test_opens']==0 and report['external_memory_opens']==0
        report['L506_access_count']=sum(p=='L506' for p,i,v in guard.reads)
        assert report['L506_access_count']==0
        write_json(output/'data_access_audit.json',report)
        write_json(output/'final_state.json',{'arm':args.arm,'step':state.step,
            'parameters_finite':finite_params(state.model),'peak_vram_bytes':state.peak_vram,
            'initial_rng_sha256':state.initial_rng_digest,'data_stream':{
                'cursor':state.stream.cursor,'epoch':state.stream.epoch},
            'source_hashes':state.source_hashes})
        print('COMPLETE '+json.dumps({'arm':args.arm,'step':state.step,'access':report}),flush=True)
if __name__=='__main__':main()
