"""Formal 60k trainer. Resume into a NEW directory; no implicit extension."""
import argparse
import csv
import json
import math
import os
import shutil
import time
from pathlib import Path
from .runtime import (TrainingState,ROOT,CSV_FIELDS,atomic_save,write_json,sha256,provenance,source_hashes,validate)

def rewrite_metrics(output,history):
    path=output/'validation_metrics.csv'
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=CSV_FIELDS);writer.writeheader();writer.writerows(history)
        f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)

def checkpoint_entry(path,step,kind,metrics=None):
    return {'path':str(path.resolve()),'optimizer_step':step,'kind':kind,'sha256':sha256(path),'bytes':path.stat().st_size,'validation':metrics}

def summary(state,status,output,reason=None):
    matched=[r for r in state.history if r['step'] in (30000,40000,50000,60000)]
    by_step={r['step']:r for r in matched}
    trend={}
    for a,b in [(30000,40000),(40000,50000),(50000,60000)]:
        if a in by_step and b in by_step:trend['{}-{}'.format(a,b)]={m:by_step[b][m]-by_step[a][m] for m in ('psnr','ssim','rmse_hu','mae_hu')}
    plateau=None
    if '40000-50000' in trend and '50000-60000' in trend:plateau=all(abs(trend[k]['psnr'])<.05 for k in ('40000-50000','50000-60000'))
    extend=None
    if '50000-60000' in trend:extend=trend['50000-60000']['psnr']>=.1 and trend['50000-60000']['ssim']>=0
    result={'status':status,'experiment':state.cfg['description'],'optimizer_step':state.step,'max_optimizer_steps':state.cfg['training']['max_optimizer_steps'],'matched_L333_metrics':matched,'best_psnr':state.best_psnr,'best_step':state.best_step,'best_checkpoint':state.best_checkpoint,'weights':'EMA','trend':trend,'plateau_heuristic_abs_delta_under_0_05db_for_two_intervals':plateau,'extension_worth_discussing_delta_at_least_0_1db_and_ssim_non_decreasing':extend,'automatic_extension':False,'total_training_compute_seconds':state.cumulative_train_seconds,'total_wall_seconds':state.previous_wall_seconds+time.perf_counter()-state.start_wall,'peak_vram_bytes':state.peak_vram,'gradient_statistics_pre_clip':state.gradient_stats,'nonfinite_or_abnormal_failure':reason,'amp_overflow':'not applicable; AMP disabled','test_access':False,'checkpoint_manifest':str(output/'checkpoint_manifest.json')}
    write_json(output/'final_training_summary.json',result)
    text=['# Training summary', '', 'Status: '+status, 'Step: '+str(state.step), state.cfg['description'], '', '| Step | PSNR | SSIM | RMSE HU | MAE HU |','|---|---:|---:|---:|---:|']
    text += ['| {step} | {psnr:.6f} | {ssim:.6f} | {rmse_hu:.6f} | {mae_hu:.6f} |'.format(**r) for r in matched]
    if not matched:text+=['','Matched validation checkpoints are not available yet; no final metric is claimed.']
    text+=['','Best step: '+str(state.best_step),'Best checkpoint: '+str(state.best_checkpoint),'','This run stops at its configured optimizer step. No automatic extension.']
    (output/'final_training_summary.md').write_text('\n'.join(text)+'\n')
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--resume');p.add_argument('--gate')
    args=p.parse_args();cfg=json.loads(Path(args.config).read_text())
    if cfg['mode']=='formal':
        if not args.gate:raise RuntimeError('Formal run requires completed preflight gate')
        gate=json.loads(Path(args.gate).read_text())
        assert gate['status']=='PASS' and gate['smoke_steps']>=20 and gate['benchmark_steps']==200
        assert gate['source_hashes']==source_hashes() and gate['model_hash']==cfg['model']['source_sha256']
        assert gate['formal_config_sha256']==sha256(args.config)
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    (output/'checkpoints').mkdir()
    write_json(output/'CONFIG.json',cfg);write_json(output/'source_manifest.json',provenance())
    for name in source_hashes():
        target=output/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    state=TrainingState(cfg,args.device)
    resumed_from=None
    if args.resume:
        resumed_from={'path':str(Path(args.resume).resolve()),'sha256':sha256(args.resume),'step':state.resume(args.resume)}
        write_json(output/'resume_provenance.json',resumed_from)
        if state.best_checkpoint:
            shutil.copyfile(state.best_checkpoint,output/'best_psnr.pt')
    write_json(output/'data_audit.json',{'train':state.data.audit(),'validation':state.validation_data.audit(),'no_test_access':True})
    manifest={'experiment':cfg['experiment_name'],'source_hashes':state.source_hashes,'resumed_from':resumed_from,'checkpoints':[],'best':None,'latest':None}
    rewrite_metrics(output,state.history);summary(state,'RUNNING',output)
    write_json(output/'checkpoint_manifest.json',manifest)
    print('RUN_STARTED '+json.dumps({'step':state.step,'mode':cfg['mode'],'max_steps':cfg['training']['max_optimizer_steps'],'output':str(output),'parameters':13130945}),flush=True)
    training_started=time.perf_counter();durations=[];gradients=[];losses=[]
    try:
        with (output/'train.jsonl').open('x',buffering=1) as log:
            while state.step<cfg['training']['max_optimizer_steps']:
                record=state.update();durations.append(record['seconds_per_step']);gradients.append(record['gradient_norm_pre_clip']);losses.append(record['train_loss'])
                log.write(json.dumps(record,allow_nan=False)+'\n')
                if state.step%cfg['training']['console_every']==0 or state.step==1:print('TRAIN '+json.dumps({k:record[k] for k in ('optimizer_step','train_loss','gradient_norm_pre_clip','seconds_per_step','peak_vram_bytes')}),flush=True)
                val_every=cfg['training']['validation_every']
                due_validation=bool(val_every and state.step%val_every==0)
                is_last=state.step==cfg['training']['max_optimizer_steps']
                if due_validation:
                    cp=output/'checkpoints'/('step_{}.pt'.format(state.step))
                    metrics=validate(state.ema.model,state.validation_data,cfg,state.device,output=output/('validation_step_{}.json'.format(state.step)))
                    assert metrics['slices']==610
                    if state.device.type=='cuda':
                        import torch
                        state.peak_vram=max(state.peak_vram,torch.cuda.max_memory_allocated(state.device))
                    row={'step':state.step,**metrics,'train_loss':record['train_loss'],'gradient_norm':record['gradient_norm_pre_clip'],'learning_rate':record['learning_rate'],'peak_vram_bytes':state.peak_vram,'seconds_per_step':float(sum(durations[-100:])/len(durations[-100:])),'checkpoint':str(cp)}
                    state.history.append(row)
                    improved=state.best_psnr is None or row['psnr']>state.best_psnr
                    if improved:state.best_psnr=row['psnr'];state.best_step=state.step;state.best_checkpoint=str(cp)
                    if cp.exists():raise FileExistsError(cp)
                    payload=state.payload();atomic_save(cp,payload)
                    manifest['checkpoints'].append(checkpoint_entry(cp,state.step,'matched' if state.step in cfg['training']['matched_checkpoint_steps'] else 'validation',row))
                    if improved:
                        temp=output/'best_psnr.pt.tmp';os.link(cp,temp);os.replace(temp,output/'best_psnr.pt')
                        manifest['best']={'step':state.step,'psnr':state.best_psnr,'path':str(output/'best_psnr.pt'),'immutable_source':str(cp),'sha256':sha256(cp),'weights':'ema'}
                    rewrite_metrics(output,state.history)
                    print('VALIDATION '+json.dumps(row,allow_nan=False),flush=True)
                    summary(state,'RUNNING',output)
                if state.step%cfg['training']['recovery_checkpoint_every']==0 or is_last:
                    latest=output/'checkpoints'/'latest.pt';atomic_save(latest,state.payload())
                    manifest['latest']=checkpoint_entry(latest,state.step,'recovery')
                    # Smoke / timing retain the exact milestones used for resume comparison.
                    if cfg['mode']!='formal':
                        cp=output/'checkpoints'/('step_{}.pt'.format(state.step))
                        if cp.exists():raise FileExistsError(cp)
                        os.link(latest,cp);manifest['checkpoints'].append(checkpoint_entry(cp,state.step,'acceptance'))
                    write_json(output/'checkpoint_manifest.json',manifest)
                    write_json(output/'progress.json',{'status':'RUNNING','step':state.step,'max_steps':cfg['training']['max_optimizer_steps'],'last_train':record,'best_step':state.best_step})
        warm=durations[20:] if len(durations)>20 else durations
        timing={'optimizer_steps_executed':len(durations),'all_steps_mean_seconds':sum(durations)/len(durations),'post_20_warmup_mean_seconds':sum(warm)/len(warm),'peak_vram_bytes':state.peak_vram,'training_wall_seconds':time.perf_counter()-training_started,'estimated_60000_compute_hours':sum(warm)/len(warm)*60000/3600,'loss_first':losses[0],'loss_last':losses[-1],'gradient_norm_min':min(gradients),'gradient_norm_max':max(gradients),'all_finite':all(math.isfinite(v) for v in losses+gradients+durations),'gradient_norm_gt_100_count':sum(v>100 for v in gradients),'note':'Estimate excludes validation, checkpoint I/O, interruptions; raw pre-clip gradient statistics retained.'}
        write_json(output/'timing.json',timing)
        summary(state,'COMPLETE',output);write_json(output/'progress.json',{'status':'COMPLETE','step':state.step,'best_step':state.best_step})
        print('RUN_COMPLETE '+json.dumps(timing),flush=True)
    except BaseException as exc:
        summary(state,'FAILED',output,repr(exc));write_json(output/'progress.json',{'status':'FAILED','step':state.step,'error':repr(exc)})
        raise

if __name__=='__main__':main()
