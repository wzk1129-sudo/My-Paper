"""Two-step smoke, 12k matched training, gated Full-L333 evaluation."""
import argparse
import contextlib
import csv
import json
import math
import os
import subprocess
import time
from pathlib import Path
import torch
from fast_dev.data import DataAccessGuard, FastPairs
from fast_dev.evaluate import validate
from fast_dev.protocol import FAST_INDICES
from mayo2016_smallset_v1.runtime import ROOT, atomic_save, rng_state, sha256, write_json, seed_everything
from src.rddm_rleb import build_baseline
from .variants import ARMS, SPECS
from .protocol import HERE, STEPS, BASE_FULL, config, references, source_manifest
from .runtime import ScreenState, rng_hash

def csv_write(path,rows):
    if not rows:return
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def audit_access(access,full=False,smoke=False):
    assert access['test_opens']==access['external_memory_opens']==access['blocked_attempts']==0
    assert access['external_memory_enabled'] is False
    if smoke:
        assert access['validation_indices_opened']==[]
        assert set(access['training_patients_opened']) <= {'L067','L143'}
    else:
        assert access['validation_indices_opened']==(list(range(610)) if full else list(FAST_INDICES))
        if not full: assert access['training_patients_opened']==['L067','L143']

def smoke(arm,device):
    cfg,base=config();HERE.mkdir(parents=True,exist_ok=True)
    out=HERE/'smoke_pass_v4'/arm;out.mkdir(parents=True,exist_ok=False)
    guard=DataAccessGuard(base,'FAST',allow_training=True,allow_memory=False)
    try:
        with guard:
            state=ScreenState(arm,device)
            seed_everything(cfg['seed']);canonical=build_baseline().to(state.device)
            # Step-0 equality on a small synthetic field isolates the zero-residual branches.
            with torch.random.fork_rng(devices=[0] if state.device.type=='cuda' else []):
                torch.manual_seed(41)
                x=torch.randn(1,2,64,64,device=state.device);t=torch.tensor([10],device=state.device)
                with torch.no_grad():
                    control=canonical.model.unet0(x,t)
                    candidate=state.model.model.unet0(x,t)
                assert torch.equal(control,candidate)
            del canonical,x,t,control,candidate
            # Restore the canonical post-construction RNG before the first training update.
            from mayo2016_smallset_v1.runtime import restore_rng
            # ScreenState was created before the temporary canonical model.
            restore_rng(state.training_rng)
            start_hash=rng_hash(rng_state())
            assert start_hash==state.initial_rng_sha256
            rows=[state.update() for _ in range(2)]
            missing=[k for k,v in state.new_grad_abs.items() if not math.isfinite(v) or v<=0]
            assert not missing,'new parameters without nonzero gradient across two steps: '+repr(missing)
            assert all(math.isfinite(r['train_loss']) and math.isfinite(r['gradient_norm_pre_clip']) for r in rows)
            result={'arm':arm,'status':'SMOKE_PASS','optimizer_steps':2,'parameter_count':state.init_audit['total_parameter_count'],
              'init_audit':state.init_audit,'step0_output_bitwise_equal':True,'losses':[r['train_loss'] for r in rows],
              'gradient_norms':[r['gradient_norm_pre_clip'] for r in rows],
              'new_parameter_nonzero_gradients':state.new_grad_abs,
              'output_shape':[2,1,512,512],'initial_training_rng_sha256':start_hash,
              'post_smoke_rng_sha256':rng_hash(rng_state()),'train_stream_sha256':state.stream_digest.hex(),
              'training_state_discarded':True,'data_access':guard.report(),'source_manifest':state.manifest}
        audit_access(guard.report(),smoke=True)
        write_json(out/'SMOKE_RESULT.json',result)
        print('SMOKE_PASS',arm,json.dumps({'losses':result['losses'],'gradient_norms':result['gradient_norms']}),flush=True)
    except BaseException as exc:
        write_json(out/'SMOKE_RESULT.json',{'arm':arm,'status':'FAILED','error':repr(exc),'data_access':guard.report()})
        raise

def full_validate(arm,checkpoint,base,device,out):
    guard=DataAccessGuard(base,'FULL',allow_training=False,allow_memory=False)
    with guard:
        dataset=FastPairs(base,'validation','FULL')
        state=ScreenState(arm,device)
        ck=torch.load(checkpoint,map_location='cpu')
        assert ck['arm']==arm and ck['optimizer_step']==12000 and ck['source_manifest']==state.manifest
        state.ema.model.load_state_dict(ck['ema_state_dict'],strict=True)
        summary=validate(state.ema.model,dataset,base,state.device,out/'full_l333_rows.json')
    access=guard.report();audit_access(access,full=True)
    observed=json.loads((out/'full_l333_rows.json').read_text())
    with BASE_FULL.open(newline='') as f: control=list(csv.DictReader(f))
    assert len(control)==len(observed)==610
    paired=[]
    for a,b in zip(observed,control):
        assert a['patient_id']==b['patient_id']=='L333' and a['slice_index']==int(b['slice_index'])
        row={'patient_id':'L333','slice_index':a['slice_index']}
        for metric in ('psnr','ssim','rmse_hu','mae_hu'):
            row['candidate_'+metric]=a[metric]
            row['control_'+metric]=float(b['control_'+metric])
            row['delta_'+metric]=a[metric]-float(b['control_'+metric])
        paired.append(row)
    csv_write(out/'full_l333_paired.csv',paired)
    report={'candidate':summary,'control_mean':{m:sum(float(r['control_'+m]) for r in control)/610 for m in ('psnr','ssim','rmse_hu','mae_hu')},
      'delta_mean':{m:sum(r['delta_'+m] for r in paired)/610 for m in ('psnr','ssim','rmse_hu','mae_hu')},
      'comparison':'same L333 slices and archived CONTROL_RDDM 12k per-slice metrics; within-patient stack only',
      'control_source_sha256':sha256(BASE_FULL),'data_access':access}
    write_json(out/'full_l333_summary.json',report)
    return report

def train(arm,device,resume=None):
    cfg,base=config();ref=references()
    smoke_path=HERE/'smoke_pass_v4'/arm/'SMOKE_RESULT.json'
    smoke_result=json.loads(smoke_path.read_text())
    assert smoke_result['status']=='SMOKE_PASS' and smoke_result['source_manifest']==source_manifest()
    out=HERE/'runs'/arm
    if resume:
        assert out.is_dir()
    else:
        out.mkdir(parents=True,exist_ok=False);(out/'checkpoints').mkdir()
    state=None;history=[];guard=DataAccessGuard(base,'FAST',allow_training=True,allow_memory=False)
    try:
        with guard:
            state=ScreenState(arm,device)
            assert state.init_audit==smoke_result['init_audit']
            if resume:
                state.resume(resume)
                if (out/'validation_history.csv').is_file():
                    with (out/'validation_history.csv').open(newline='') as f:history=list(csv.DictReader(f))
            else:
                write_json(out/'CONFIG.json',{**cfg,'arm':arm,'only_architecture_change':SPECS[arm]})
                write_json(out/'provenance.json',{'git_parent':subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(ROOT),text=True).strip(),
                  'source_manifest':state.manifest,'python':os.sys.version,'torch':torch.__version__,
                  'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(state.device) if state.device.type=='cuda' else None,
                  'seed':cfg['seed'],'fast_indices':list(FAST_INDICES)})
                write_json(out/'INITIALIZATION_AUDIT.json',state.init_audit)
                write_json(out/'TRAIN_RNG_AUDIT.json',{'initial_rng_sha256':state.initial_rng_sha256,
                    'canonical_restored':True,'smoke_initial_rng_sha256':smoke_result['initial_training_rng_sha256']})
                assert state.initial_rng_sha256==smoke_result['initial_training_rng_sha256']
            assert state.step<12000
            manifest={'arm':arm,'matched':[],'latest':None}
            mode='a' if resume else 'x'
            with (out/'train.jsonl').open(mode,buffering=1) as log:
                while state.step<12000:
                    row=state.update();log.write(json.dumps(row,allow_nan=False)+'\n')
                    if state.step%100==0:
                        write_json(out/'progress.json',{'status':'RUNNING','step':state.step,'stop':12000})
                        print('TRAIN',arm,state.step,row['train_loss'],flush=True)
                    if state.step in STEPS:
                        before=rng_hash(rng_state())
                        metrics=validate(state.ema.model,state.validation_data,base,state.device,out/('validation_step_%d.json'%state.step))
                        assert rng_hash(rng_state())==before
                        assert metrics['slices']==64 and metrics['weights']=='ema'
                        assert state.stream_digest.hex()==ref[state.step]['train_stream_sha256'], 'Train stream differs from archived control'
                        delta=metrics['psnr']-ref[state.step]['psnr']
                        record={'step':state.step,**metrics,'baseline_psnr':ref[state.step]['psnr'],'delta_psnr':delta,
                          'train_loss':row['train_loss'],'gradient_norm_pre_clip':row['gradient_norm_pre_clip'],
                          'learning_rate':row['learning_rate'],'parameter_count':state.init_audit['total_parameter_count'],
                          'peak_vram_bytes':state.peak_vram,'seconds_per_step':state.train_seconds/state.step,
                          'train_stream_sha256':state.stream_digest.hex()}
                        assert all(math.isfinite(record[k]) for k in ('psnr','ssim','rmse_hu','mae_hu','validation_loss'))
                        history.append(record);csv_write(out/'validation_history.csv',history)
                        print('VALIDATION',arm,json.dumps(record),flush=True)
                    if state.step%1000==0:
                        latest=out/'checkpoints/latest.pt';atomic_save(latest,state.payload())
                        manifest['latest']={'step':state.step,'sha256':sha256(latest)}
                        if state.step in STEPS:
                            matched=out/'checkpoints'/('step_%d.pt'%state.step)
                            assert not matched.exists();os.link(latest,matched)
                            manifest['matched'].append({'step':state.step,'sha256':sha256(matched)})
                        write_json(out/'checkpoint_manifest.json',manifest)
                    if state.step==4000 and history[-1]['delta_psnr']<=-0.5:
                        status='EARLY_STOP_PERFORMANCE';break
                else:status='COMPLETE'
        access=guard.report();audit_access(access)
        assert [int(r['step']) for r in history]==[s for s in STEPS if s<=state.step]
        flags={'promotion':'NO_PROMOTION','strong_fast_signal':False}
        if status=='COMPLETE':
            deltas={int(r['step']):float(r['delta_psnr']) for r in history}
            if all(deltas[s]>0 for s in (8000,10000,12000)) and deltas[12000]>=0.05:
                flags['promotion']='PROMOTE_FULL_L333'
                flags['strong_fast_signal']=deltas[12000]>=0.10
        full=None
        if flags['promotion']=='PROMOTE_FULL_L333':
            full_dir=HERE/arm;full_dir.mkdir(exist_ok=True)
            full=full_validate(arm,out/'checkpoints/step_12000.pt',base,device,full_dir)
        summary={'arm':arm,'status':status,'optimizer_steps':state.step,'history':history,'flags':flags,
          'full_l333':full,'parameter_count':state.init_audit['total_parameter_count'],
          'peak_vram_bytes':state.peak_vram,'training_seconds':state.train_seconds,
          'seconds_per_step':state.train_seconds/state.step,'gradient_min':state.gradient_min,
          'gradient_max':state.gradient_max,'nonfinite_count':state.nonfinite_count,
          'train_stream_sha256':state.stream_digest.hex(),'initial_rng_sha256':state.initial_rng_sha256,
          'data_access':access,'source_manifest':state.manifest}
        write_json(out/'final_summary.json',summary)
        write_json(out/'DATA_ACCESS_AUDIT.json',access)
        write_json(out/'TRAIN_STREAM_AUDIT.json',{'stream_sha256':state.stream_digest.hex(),'samples':state.step*2})
        write_json(out/'TRAIN_RNG_AUDIT.json',{'initial_rng_sha256':state.initial_rng_sha256,
          'final_rng_sha256':rng_hash(rng_state()),'validation_rng_preserved':True})
        write_json(out/'progress.json',{'status':status,'step':state.step,'stop':12000})
        light=HERE/arm;light.mkdir(exist_ok=True)
        for name in ('CONFIG.json','INITIALIZATION_AUDIT.json','TRAIN_RNG_AUDIT.json','DATA_ACCESS_AUDIT.json',
                     'TRAIN_STREAM_AUDIT.json','validation_history.csv','final_summary.json','provenance.json'):
            source=out/name
            if source.is_file(): (light/name).write_bytes(source.read_bytes())
        print('ARM_END',arm,status,flags,flush=True)
    except BaseException as exc:
        if out.is_dir():
            write_json(out/'DATA_ACCESS_AUDIT.json',guard.report())
            write_json(out/'progress.json',{'status':'FAILED','step':state.step if state else 0,'error':repr(exc)})
        raise

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('smoke','train'))
    p.add_argument('--arm',choices=ARMS,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--resume')
    a=p.parse_args()
    if a.mode=='smoke':smoke(a.arm,a.device)
    else:train(a.arm,a.device,a.resume)
if __name__=='__main__':main()
