"""Single-GPU, serial F1/F2/F3/F4 screening queue; stops on the first failed smoke/run."""
import csv,datetime,json,os,subprocess,sys,time,traceback
from pathlib import Path
from mayo2016_smallset_v1.runtime import write_json,sha256
from .protocol import ROOT,BASELINE,SPECS,CANDIDATES,canonical_checks,artifact_pins,variant_hashes,FAST_INDICES

ORDER=tuple(SPECS)
BASE_RUN=ROOT/'experiments/fast_dev/runs/rddm_fast_baseline_8k_20260922'
BASE_SUMMARY=json.loads((BASE_RUN/'final_summary.json').read_text())
BASE_METRICS={r['step']:r for r in BASE_SUMMARY['history']}
STATUS_PATH=CANDIDATES/'queue_status.json'
QUEUE_LOG=CANDIDATES/'queue.log'
FIELDS=('experiment','main_change','parameters','step_2k_psnr','delta_2k','step_4k_psnr','delta_4k','step_6k_psnr','delta_6k','step_8k_psnr','delta_8k','step_8k_ssim','step_8k_rmse','step_8k_mae','peak_vram','seconds_per_step','status')

def log(message):
    with QUEUE_LOG.open('a',buffering=1) as f:f.write(datetime.datetime.now().astimezone().isoformat()+' '+message+'\n')
    print(message,flush=True)

def verify_summary(job,mode):
    p=CANDIDATES/job/('smoke50' if mode=='SMOKE' else 'train8k')
    d=json.loads((p/'final_summary.json').read_text())
    expected='SMOKE_PASS' if mode=='SMOKE' else 'COMPLETE'
    assert d['status']==expected and d['optimizer_step']==(50 if mode=='SMOKE' else 8000)
    assert d['data_access']['test_opens']==d['data_access']['external_memory_opens']==d['data_access']['blocked_attempts']==0
    assert d['data_access']['validation_indices_opened']==list(FAST_INDICES)
    assert d['source_hashes']==variant_hashes(job)
    steps=[50] if mode=='SMOKE' else [2000,4000,6000,8000]
    assert [r['step'] for r in d['history']]==steps
    for row in d['history']:
        assert row['slices']==64 and row['weights']=='ema' and row['validation_mode']=='FAST'
        cp=Path(row['checkpoint']);assert cp.is_file()
        entry=next(e for e in json.loads((p/'checkpoint_manifest.json').read_text())['checkpoints'] if e['step']==row['step'])
        assert sha256(cp)==entry['sha256']
    assert d['loss_finite'] and d['gradient_statistics']['nonfinite_count']==0
    if mode=='SMOKE':assert all(d['smoke_checks']['checkpoint_round_trip'].values())
    return d

def main():
    assert BASE_SUMMARY['status']=='COMPLETE' and BASE_SUMMARY['optimizer_step']==8000
    assert [r['step'] for r in BASE_SUMMARY['history']]==[2000,4000,6000,8000]
    canonical_checks();artifact_pins()
    if STATUS_PATH.exists() or QUEUE_LOG.exists():raise FileExistsError('Queue outputs exist: refuse duplicate launch')
    state={'status':'RUNNING','started':datetime.datetime.now().astimezone().isoformat(),
        'order':list(ORDER),'jobs':{job:{'status':'PENDING'} for job in ORDER},
        'baseline_reference':str(BASE_RUN),'baseline_step_psnr':{str(k):v['psnr'] for k,v in BASE_METRICS.items()},
        'stopped_on_failure':None,'gpu_concurrency':1,'retry_count':0}
    write_json(STATUS_PATH,state)
    log('QUEUE_START '+','.join(ORDER))
    try:
        for job in ORDER:
            canonical_checks();artifact_pins()
            assert variant_hashes(job)
            for mode in ('SMOKE','RUNNING'):
                state['jobs'][job]['status']=mode
                state['jobs'][job]['started_'+mode.lower()]=datetime.datetime.now().astimezone().isoformat()
                write_json(STATUS_PATH,state)
                cmd=[sys.executable,'-u','-m','fast_dev_variants.candidate','--job',job]
                if mode=='SMOKE':cmd.append('--smoke')
                log(job+' '+mode+' START')
                path=CANDIDATES/job/('smoke50.log' if mode=='SMOKE' else 'train8k.log')
                with path.open('x') as f:
                    result=subprocess.run(cmd,cwd=str(ROOT),stdout=f,stderr=subprocess.STDOUT,check=False)
                if result.returncode:
                    raise RuntimeError('{} {} exit {}'.format(job,mode,result.returncode))
                artifact_pins()
                data=verify_summary(job,mode)
                state['jobs'][job][mode.lower()+'_summary']=str(CANDIDATES/job/('smoke50' if mode=='SMOKE' else 'train8k')/'final_summary.json')
                if mode=='RUNNING':
                    state['jobs'][job]['status']='COMPLETE'
                    state['jobs'][job]['psnr_8k']=data['history'][-1]['psnr']
                    state['jobs'][job]['delta_8k']=data['history'][-1]['psnr']-BASE_METRICS[8000]['psnr']
                    state['jobs'][job]['completed']=datetime.datetime.now().astimezone().isoformat()
                write_json(STATUS_PATH,state)
                log(job+' '+mode+' PASS')
        rows=[]
        for job in ORDER:
            d=verify_summary(job,'RUNNING');hist={r['step']:r for r in d['history']}
            r={'experiment':job,'main_change':SPECS[job]['main_change'],'parameters':d['parameters'],
                'step_2k_psnr':hist[2000]['psnr'],'delta_2k':hist[2000]['psnr']-BASE_METRICS[2000]['psnr'],
                'step_4k_psnr':hist[4000]['psnr'],'delta_4k':hist[4000]['psnr']-BASE_METRICS[4000]['psnr'],
                'step_6k_psnr':hist[6000]['psnr'],'delta_6k':hist[6000]['psnr']-BASE_METRICS[6000]['psnr'],
                'step_8k_psnr':hist[8000]['psnr'],'delta_8k':hist[8000]['psnr']-BASE_METRICS[8000]['psnr'],
                'step_8k_ssim':hist[8000]['ssim'],'step_8k_rmse':hist[8000]['rmse_hu'],
                'step_8k_mae':hist[8000]['mae_hu'],'peak_vram':d['peak_vram_bytes'],
                'seconds_per_step':d['seconds_per_step'],'status':d['status']}
            rows.append(r)
        with (CANDIDATES/'FAST_SCREEN_V1.csv').open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
        report={'status':'COMPLETE','baseline_step_psnr':state['baseline_step_psnr'],'rows':rows,
            'positive_both_6k_8k':[r['experiment'] for r in rows if r['delta_6k']>0 and r['delta_8k']>0],
            'threshold_8k':[r['experiment'] for r in rows if r['delta_6k']>0 and r['delta_8k']>=.05],
            'test_access_count':0,'external_memory_access_count':0,'model_selection':'none',
            'formal_source_sha256':sha256(ROOT/'src/rddm_rleb.py')}
        write_json(CANDIDATES/'FAST_SCREEN_V1.json',report)
        lines=['# Fast backbone screen v1','','All candidates: independent random initialization, L067/L143, 8k maximum, frozen L333-fast64, EMA, 2-step sampler and sum_scale=0.01.','',
            'Fast baseline PSNR: '+', '.join('{}: {:.6f}'.format(k,v['psnr']) for k,v in BASE_METRICS.items()),'',
            '| Candidate | Parameters | PSNR 2k / Δ | PSNR 4k / Δ | PSNR 6k / Δ | PSNR 8k / Δ | SSIM 8k | RMSE HU 8k | MAE HU 8k | VRAM GiB | sec/step |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for r in rows:
            lines.append('| {experiment} | {parameters} | {step_2k_psnr:.4f} / {delta_2k:+.4f} | {step_4k_psnr:.4f} / {delta_4k:+.4f} | {step_6k_psnr:.4f} / {delta_6k:+.4f} | {step_8k_psnr:.4f} / {delta_8k:+.4f} | {step_8k_ssim:.6f} | {step_8k_rmse:.4f} | {step_8k_mae:.4f} | {vram:.3f} | {seconds_per_step:.3f} |'.format(**dict(r,vram=r['peak_vram']/1073741824)))
        lines+=['','Positive at both 6k and 8k: '+', '.join(report['positive_both_6k_8k']),
            'At least +0.05 dB at 8k with positive 6k: '+', '.join(report['threshold_8k']),
            'SSIM/RMSE/MAE must also be checked for consistent improvement before promotion.','',
            'L506 image accesses: 0. External-memory image accesses: 0. Formal baseline source SHA256: '+report['formal_source_sha256'],
            'No final paper backbone selected. No Git push.']
        (CANDIDATES/'FAST_SCREEN_V1_REPORT.md').write_text('\n'.join(lines)+'\n')
        state['status']='COMPLETE';state['completed']=datetime.datetime.now().astimezone().isoformat();write_json(STATUS_PATH,state)
        log('QUEUE_COMPLETE')
    except BaseException as e:
        state['status']='FAILED';state['stopped_on_failure']=repr(e)
        current=next((j for j in ORDER if state['jobs'][j]['status'] in ('SMOKE','RUNNING')),None)
        if current:state['jobs'][current]['status']='FAILED'
        write_json(STATUS_PATH,state);log('QUEUE_FAILED '+repr(e))
        traceback.print_exc()
        raise
if __name__=='__main__':main()
