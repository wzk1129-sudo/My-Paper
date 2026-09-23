"""One fresh smoke or 8k run; called by the serial queue."""
import argparse,csv,json,math,os,time,traceback
from pathlib import Path
import torch
from mayo2016_smallset_v1.runtime import atomic_save,write_json,sha256,rng_state
from mayo2016_smallset_v1.acceptance import nested_equal
from fast_dev.data import DataAccessGuard
from fast_dev.evaluate import validate
from fast_dev.reporting import plateau
from .protocol import BASELINE,CANDIDATES,SPECS,FAST_TRAIN,FAST_INDICES,canonical_checks,artifact_pins,variant_config,variant_hashes
from .runtime import VariantState

def write_metrics(path,history):
    if not history:return
    with path.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)

def _cpu(value):
    if torch.is_tensor(value):return value.detach().cpu()
    if isinstance(value,dict):return {k:_cpu(v) for k,v in value.items()}
    if isinstance(value,list):return [_cpu(v) for v in value]
    if isinstance(value,tuple):return tuple(_cpu(v) for v in value)
    return value

def run(job,smoke):
    assert job in SPECS
    canonical_checks();artifact_pins()
    cfg=variant_config(job)
    assert cfg==json.loads((CANDIDATES/job/'CONFIG.json').read_text())
    source=variant_hashes(job)
    path=CANDIDATES/job/('smoke50' if smoke else 'train8k')
    path.mkdir(exist_ok=False)
    (path/'checkpoints').mkdir()
    write_json(path/'CONFIG.json',cfg)
    write_json(path/'source_hashes.json',source)
    guard=DataAccessGuard(BASELINE,'FAST',allow_memory=False)
    state=None;durations=[];losses=[];manifest={'checkpoints':[],'latest':None,'best':None}
    stop=50 if smoke else 8000
    try:
        with guard:
            state=VariantState(cfg,'cuda:0')
            assert state.source_hashes==source
            with (path/'train.jsonl').open('x',buffering=1) as log:
                while state.step<stop:
                    rec=state.update();durations.append(rec['seconds_per_step']);losses.append(rec['train_loss'])
                    assert {s['patient_id'] for s in rec['samples']}==set(FAST_TRAIN) or {s['patient_id'] for s in rec['samples']}<=set(FAST_TRAIN)
                    log.write(json.dumps(rec,allow_nan=False)+'\n')
                    if state.step%100==0 or smoke and state.step%10==0:
                        canonical_checks()
                        print('TRAIN '+json.dumps({'job':job,'step':state.step,'loss':rec['train_loss'],'gradient_norm':rec['gradient_norm_pre_clip']}),flush=True)
                    due=state.step%2000==0 or smoke and state.step==stop
                    if due:
                        before=rng_state()
                        metric=validate(state.ema.model,state.validation_data,BASELINE,state.device,
                            path/('validation_step_{}.json'.format(state.step)))
                        assert nested_equal(before,rng_state()),'validation changed training RNG'
                        previous=state.history[-1]['psnr'] if state.history else None
                        cp=path/'checkpoints'/('step_{}.pt'.format(state.step))
                        improved=state.best_psnr is None or metric['psnr']>state.best_psnr
                        if improved:state.best_psnr=metric['psnr'];state.best_step=state.step;state.best_checkpoint=str(cp)
                        row={'step':state.step,**metric,'best_step':state.best_step,
                            'delta_psnr':None if previous is None else metric['psnr']-previous,
                            'train_loss':rec['train_loss'],'gradient_norm_pre_clip':rec['gradient_norm_pre_clip'],
                            'learning_rate':rec['learning_rate'],'peak_vram_bytes':state.peak_vram,
                            'seconds_per_step':sum(durations[-100:])/len(durations[-100:]),
                            'parameters':state.parameters_count,'checkpoint':str(cp)}
                        state.history.append(row)
                        atomic_save(cp,state.payload())
                        manifest['checkpoints'].append({'step':state.step,'path':str(cp),'sha256':sha256(cp),'validation':row})
                        if improved:
                            best=path/'best_psnr.pt';os.link(cp,best.with_suffix('.tmp'))
                            os.replace(best.with_suffix('.tmp'),best)
                            manifest['best']={'step':state.step,'path':str(best),'source':str(cp),'sha256':sha256(cp),'weights':'ema'}
                        print('VALIDATION '+json.dumps({'job':job,**row}),flush=True)
                    if state.step%1000==0 or state.step==stop:
                        latest=path/'checkpoints/latest.pt';atomic_save(latest,state.payload())
                        manifest['latest']={'step':state.step,'path':str(latest),'sha256':sha256(latest)}
                        write_json(path/'checkpoint_manifest.json',manifest)
                    if state.step%100==0 or due:
                        write_json(path/'progress.json',{'status':'RUNNING','step':state.step,'target':stop})
            smoke_checks=None
            if smoke:
                saved=torch.load(path/'checkpoints/latest.pt',map_location='cpu')
                restored=VariantState(cfg,'cuda:0');restored.resume(path/'checkpoints/latest.pt')
                again=restored.payload()
                keys=('model_state_dict','ema_state_dict','optimizer_state_dict','optimizer_step','rng','data_stream','ema_updates')
                equality={k:nested_equal(saved[k],_cpu(again[k])) for k in keys}
                assert all(equality.values()),equality
                assert state.ema.updates==50
                if cfg['variant']['kind']=='F2':
                    assert any(abs(float(p))>0 for p in state.model.model.unet0.cond_gammas)
                    assert all(p.grad is not None and torch.isfinite(p.grad).all() for n,p in state.model.named_parameters() if 'cond_' in n)
                if cfg['variant']['kind']=='F3':
                    assert all(p.grad is not None and torch.isfinite(p.grad).all() for n,p in state.model.named_parameters() if 'conv2.' in n or 'norm2.' in n)
                smoke_checks={'checkpoint_round_trip':equality,'ema_updates':50,'new_parameter_gradients_finite':True}
                del restored,saved,again
        access=guard.report()
        assert access['test_opens']==access['external_memory_opens']==access['blocked_attempts']==0
        assert access['validation_indices_opened']==list(FAST_INDICES)
        assert set(access['training_patients_opened'])==set(FAST_TRAIN)
        canonical_checks();artifact_pins()
        assert variant_hashes(job)==source
        assert len(state.history)==(1 if smoke else 4)
        assert all(math.isfinite(v) for v in losses)
        summary={'status':'SMOKE_PASS' if smoke else 'COMPLETE','job':job,'optimizer_step':state.step,
            'parameters':state.parameters_count,'training_patients':list(FAST_TRAIN),'validation_mode':'FAST',
            'validation_slices':64,'history':state.history,'best_step':state.best_step,'best_psnr':state.best_psnr,
            'plateau':plateau(state.history),'smoke_checks':smoke_checks,'data_access':access,
            'source_hashes':source,'first_train_loss':losses[0],'last_train_loss':losses[-1],
            'loss_finite':True,'gradient_statistics':state.gradient_stats,'peak_vram_bytes':state.peak_vram,
            'seconds_per_step':sum(durations)/len(durations),'elapsed_seconds':time.perf_counter()-state.start_wall}
        write_metrics(path/'validation_metrics.csv',state.history)
        write_json(path/'data_access.json',access)
        write_json(path/'final_summary.json',summary)
        write_json(path/'progress.json',{'status':summary['status'],'step':stop,'target':stop})
        print('RUN_COMPLETE '+json.dumps({'job':job,'status':summary['status'],'best_psnr':state.best_psnr}),flush=True)
    except BaseException as e:
        write_json(path/'data_access.json',guard.report())
        write_json(path/'progress.json',{'status':'FAILED','step':state.step if state else 0,'error':repr(e)})
        traceback.print_exc()
        raise

def main():
    p=argparse.ArgumentParser();p.add_argument('--job',choices=tuple(SPECS),required=True)
    p.add_argument('--smoke',action='store_true');a=p.parse_args()
    run(a.job,a.smoke)
if __name__=='__main__':main()
