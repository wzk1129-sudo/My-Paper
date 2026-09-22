"""Isolated fast-dev trainer: 8k maximum; optional 50-100 step acceptance only."""
import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
import torch
from mayo2016_smallset_v1.runtime import atomic_save, write_json, sha256, rng_state, provenance
from mayo2016_smallset_v1.acceptance import nested_equal
from .protocol import CONFIG, FAST_TRAIN, FAST_INDICES, check_config, output_directory
from .data import DataAccessGuard
from .runtime import FastState
from .evaluate import validate
from .reporting import plateau

def cpu_tree(value):
    """Compare saved CPU tensors with restored device tensors without changing state."""
    if torch.is_tensor(value):return value.detach().cpu()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return value

def write_metrics(output,history):
    if not history:return
    path=output/'validation_metrics.csv';tmp=path.with_suffix('.csv.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(history[0]));w.writeheader();w.writerows(history)
        f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def link_best(source,destination):
    tmp=destination.with_name(destination.name+'.tmp')
    os.link(source,tmp);os.replace(tmp,destination)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default=str(CONFIG));p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--resume')
    p.add_argument('--smoke-steps',type=int,choices=range(50,101))
    a=p.parse_args();cfg=json.loads(Path(a.config).read_text());check_config(cfg)
    if a.resume and a.smoke_steps:raise ValueError('Use regular resume for incomplete fast runs; smoke is fresh acceptance')
    output=output_directory(a.output);(output/'checkpoints').mkdir()
    write_json(output/'CONFIG.json',cfg)
    write_json(output/'invocation.json',{'smoke_steps':a.smoke_steps,'resume':a.resume,'effective_stop':a.smoke_steps or 8000,'validation_mode':'FAST'})
    stop=a.smoke_steps or cfg['training']['max_optimizer_steps']
    state=None;manifest={'checkpoints':[],'best':None,'latest':None};durations=[];losses=[]
    guard=DataAccessGuard(cfg,'FAST')
    try:
        with guard:
            state=FastState(cfg,a.device)
            prov=provenance();prov['source_hashes']=state.source_hashes
            write_json(output/'source_manifest.json',prov)
            if a.resume:
                state.resume(a.resume)
                write_json(output/'resume.json',{'checkpoint':str(Path(a.resume).resolve()),'sha256':sha256(a.resume),'step':state.step})
                if state.best_checkpoint:
                    link_best(state.best_checkpoint,output/'best_psnr.pt')
                    manifest['best']={'step':state.best_step,'psnr':state.best_psnr,'immutable_source':state.best_checkpoint,'sha256':sha256(state.best_checkpoint),'weights':'ema'}
            if state.step>=stop:raise ValueError('Resume checkpoint already reached this run budget')
            with (output/'train.jsonl').open('x',buffering=1) as log:
                while state.step<stop:
                    record=state.update();durations.append(record['seconds_per_step']);losses.append(record['train_loss'])
                    assert {m['patient_id'] for m in record['samples']} <= set(FAST_TRAIN)
                    log.write(json.dumps(record,allow_nan=False)+'\n')
                    if state.step==1 or state.step%10==0 and a.smoke_steps or state.step%100==0:
                        print('TRAIN '+json.dumps({k:record[k] for k in ('optimizer_step','train_loss','gradient_norm_pre_clip','ema_updates')}),flush=True)
                    due=state.step%cfg['training']['validation_every']==0 or bool(a.smoke_steps and state.step==stop)
                    if due:
                        before=rng_state()
                        metric=validate(state.ema.model,state.validation_data,cfg,state.device,output/('validation_step_{}.json'.format(state.step)))
                        assert nested_equal(before,rng_state()),'Validation changed training RNG'
                        previous=state.history[-1]['psnr'] if state.history else None
                        cp=output/'checkpoints'/('step_{}.pt'.format(state.step))
                        improved=state.best_psnr is None or metric['psnr']>state.best_psnr
                        if improved:state.best_psnr=metric['psnr'];state.best_step=state.step;state.best_checkpoint=str(cp)
                        row={'step':state.step,**metric,'best_step':state.best_step,'delta_psnr':None if previous is None else metric['psnr']-previous,
                            'train_loss':record['train_loss'],'gradient_norm_pre_clip':record['gradient_norm_pre_clip'],
                            'learning_rate':record['learning_rate'],'peak_vram_bytes':state.peak_vram,
                            'seconds_per_step':sum(durations[-100:])/len(durations[-100:]),'checkpoint':str(cp)}
                        state.history.append(row)
                        if cp.exists():raise FileExistsError(cp)
                        atomic_save(cp,state.payload())
                        entry={'step':state.step,'path':str(cp),'sha256':sha256(cp),'validation':row}
                        manifest['checkpoints'].append(entry)
                        if improved:
                            link_best(cp,output/'best_psnr.pt');manifest['best']={'step':state.step,'psnr':state.best_psnr,'immutable_source':str(cp),'sha256':entry['sha256'],'weights':'ema'}
                        write_metrics(output,state.history)
                        write_json(output/'plateau_report.json',plateau(state.history))
                        print('VALIDATION '+json.dumps(row),flush=True)
                    if state.step%cfg['training']['recovery_checkpoint_every']==0 or state.step==stop:
                        latest=output/'checkpoints/latest.pt';atomic_save(latest,state.payload())
                        manifest['latest']={'step':state.step,'path':str(latest),'sha256':sha256(latest)}
                        write_json(output/'checkpoint_manifest.json',manifest)
                    if due or state.step%100==0:
                        write_json(output/'progress.json',{'status':'RUNNING','step':state.step,'stop':stop,'best_step':state.best_step})
            smoke_checks=None
            if a.smoke_steps:
                # Same fixed 64 slices and seeded sampling, then exact checkpoint state round trip.
                before=rng_state()
                repeat=validate(state.ema.model,state.validation_data,cfg,state.device,output/'validation_repeat.json')
                assert nested_equal(before,rng_state())
                assert all(repeat[k]==state.history[-1][k] for k in ('psnr','ssim','rmse_hu','mae_hu','validation_loss'))
                assert state.ema.updates==stop and all(torch.isfinite(v).all() for v in state.ema.model.state_dict().values())
                assert any(not torch.equal(a,b) for a,b in zip(state.model.parameters(),state.ema.model.parameters()))
                payload=torch.load(output/'checkpoints/latest.pt',map_location='cpu')
                restored=FastState(cfg,a.device);restored.resume(output/'checkpoints/latest.pt')
                again=restored.payload()
                keys=('model_state_dict','ema_state_dict','optimizer_state_dict','optimizer_step','rng','data_stream','ema_updates')
                equality={k:nested_equal(payload[k],cpu_tree(again[k])) for k in keys}
                assert all(equality.values()),equality
                smoke_checks={'checkpoint_round_trip':equality,'deterministic_fast64':True,'training_rng_preserved':True,'ema_updates':state.ema.updates,'ema_finite_and_distinct_from_raw':True}
                del restored,payload,again
        access=guard.report()
        assert access['test_opens']==access['external_memory_opens']==access['blocked_attempts']==0
        assert access['validation_indices_opened']==list(FAST_INDICES)
        assert set(access['training_patients_opened'])==set(FAST_TRAIN)
        status='SMOKE_PASS' if a.smoke_steps else 'COMPLETE'
        result={'status':status,'optimizer_step':state.step,'training_patients':list(FAST_TRAIN),'training_pairs':len(state.data),
            'validation_mode':'FAST','validation_slices':64,'history':state.history,'best_step':state.best_step,'best_psnr':state.best_psnr,
            'plateau':plateau(state.history),'automatic_early_stop':False,'automatic_extension':False,'smoke_only':bool(a.smoke_steps),
            'smoke_checks':smoke_checks,'data_access':access,'source_hashes':state.source_hashes,
            'first_train_loss':losses[0],'last_train_loss':losses[-1],'loss_finite':all(math.isfinite(x) for x in losses),
            'gradient_statistics':state.gradient_stats,'training_peak_vram_bytes':state.peak_vram,
            'process_peak_vram_bytes_including_resume_check':max(state.peak_vram,torch.cuda.max_memory_allocated(state.device)) if state.device.type=='cuda' else 0,
            'seconds_per_step':sum(durations)/len(durations),'elapsed_seconds':time.perf_counter()-state.start_wall}
        write_json(output/'data_access.json',access);write_json(output/'final_summary.json',result)
        write_json(output/'progress.json',{'status':status,'step':state.step,'stop':stop,'best_step':state.best_step})
        print('RUN_COMPLETE '+json.dumps(result),flush=True)
    except BaseException as exc:
        write_json(output/'data_access.json',guard.report())
        write_json(output/'progress.json',{'status':'FAILED','step':state.step if state else 0,'error':repr(exc)})
        raise

if __name__=='__main__':main()
