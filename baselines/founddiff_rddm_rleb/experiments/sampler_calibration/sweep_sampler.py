"""Inference-only runtime sampler calibration; frozen source and EMA buffers unchanged."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys, json, csv, time, hashlib, subprocess, traceback
from pathlib import Path
ROOT=Path('/home/xujian/projects/CoreDiff')
BASE=ROOT/'experiments/founddiff_table_v_20260921/rddm_rleb'
RUN=BASE/'mayo2016_smallset_v1/runs/FoundDiff_RDDM_RLEB_Mayo2016_smallset_v1_60k_20260921'
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
import torch, numpy as np, skimage
from mayo2016_smallset_v1.runtime import validate, seed_everything, check_config, sha256, source_hashes, write_json, METRICS
from mayo2016_smallset_v1.data import CenterPairs
from src.rddm_rleb import build_baseline

def state_hash(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():
        h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def main():
    assert not (OUT/'sampler_sweep.json').exists(), 'Refuse to overwrite an existing experiment'
    cfg=json.loads((RUN/'CONFIG.json').read_text());check_config(cfg)
    ckpath=RUN/'checkpoints/step_60000.pt'
    protected={str(p):sha256(p) for p in [ckpath,RUN/'CONFIG.json',RUN/'validation_metrics.csv']}
    before_sources=source_hashes()
    ck=torch.load(str(ckpath),map_location='cpu')
    assert ck['optimizer_step']==60000 and ck['ema_updates']==60000
    assert ck['config']==cfg and ck['source_hashes']==before_sources
    torch.set_num_threads(cfg['training']['cpu_threads']);seed_everything(cfg['seed'])
    model=build_baseline().eval().requires_grad_(False)
    model.load_state_dict(ck['ema_state_dict'],strict=True)
    del ck
    model=model.to('cuda:0');original_state=state_hash(model)
    data=CenterPairs(cfg,'validation');assert len(data)==610
    refs=list(csv.DictReader((RUN/'validation_metrics.csv').open()))
    ref=next(r for r in refs if int(r['step'])==60000)
    assert ref['weights']=='ema' and int(ref['slices'])==610
    tolerances={'psnr':1e-5,'ssim':1e-7,'rmse_hu':1e-4,'mae_hu':1e-4}
    result={'status':'RUNNING','checkpoint':str(ckpath),'checkpoint_sha256':protected[str(ckpath)],'weights':'ema_state_dict','optimizer_steps_performed':0,'source_hashes':before_sources,'config':cfg,'environment':{'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'numpy':np.__version__,'skimage':skimage.__version__,'gpu':torch.cuda.get_device_name(0)},'protocol':{'patient':'L333','slices':610,'stride':1,'seed':'20260829 + original slice_index','original_validation_function':True,'timing_includes_validation_loss':True,'sum_scale_semantics':'Runtime initial-noise scale only; loaded diffusion coefficient buffers remain unchanged'},'control_tolerances':tolerances,'rows':[]}
    write_json(OUT/'sampler_sweep.json',result)
    def evaluate(stage,n,s):
        model.sampling_timesteps=int(n);model.is_ddim_sampling=n<model.num_timesteps;model.sum_scale=float(s)
        print('START stage={} sampling_timesteps={} sum_scale={}'.format(stage,n,s),flush=True)
        metrics=validate(model,data,cfg,torch.device('cuda:0'),output=OUT/('slices_{}_{}_{}.json'.format(stage,n,s)))
        assert state_hash(model)==original_state
        row={'stage':stage,'sampling_timesteps':n,'sum_scale':s,**metrics}
        result['rows'].append(row);write_json(OUT/'sampler_sweep.json',result)
        print('RESULT '+json.dumps(row),flush=True)
        return row
    control=evaluate('control',2,.01)
    result['control_deltas']={k:control[k]-float(ref[k]) for k in METRICS}
    result['control_pass']=all(abs(result['control_deltas'][k])<=tolerances[k] for k in METRICS)
    write_json(OUT/'sampler_sweep.json',result)
    assert result['control_pass'], 'CONTROL_FAILED: stop sweep'
    # Reuse the identical full-volume control measurement for the shared configuration.
    step_rows=[]
    for n in [1,2,4,8]:
        r=dict(control,stage='timesteps',reused_from='control') if n==2 else evaluate('timesteps',n,.01)
        if n==2:result['rows'].append(r)
        step_rows.append(r)
    best_n=max(step_rows,key=lambda r:r['psnr'])['sampling_timesteps']
    scale_rows=[]
    for s in [0.,.0025,.005,.01,.02,.04]:
        if s==.01:
            r=dict(next(r for r in step_rows if r['sampling_timesteps']==best_n),stage='sum_scale',reused_from='timesteps');result['rows'].append(r)
        else:r=evaluate('sum_scale',best_n,s)
        scale_rows.append(r)
    best=max(result['rows'],key=lambda r:r['psnr'])
    assert source_hashes()==before_sources
    assert all(sha256(p)==v for p,v in protected.items())
    assert state_hash(model)==original_state
    result.update(status='COMPLETE',best=best,deltas_vs_control={k:best[k]-control[k] for k in METRICS},model_weights_unchanged=True,source_config_checkpoint_unchanged=True,L506_untouched=True,no_training_performed=True,evaluation_protocol_unchanged=True)
    write_json(OUT/'sampler_sweep.json',result)
    fields=['stage','sampling_timesteps','sum_scale',*METRICS,'validation_seconds','slices','weights']
    with (OUT/'sampler_sweep.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(result['rows'])
    lines=['# Sampler calibration','', 'FoundDiff RDDM+RLEB backbone adapted to our fixed Mayo-2016 protocol.','', 'Control reproduced: PASS. Full L333, 610 slices, stride=1, original seed and HU metrics.','', '| Stage | Steps | sum_scale | PSNR | SSIM | RMSE HU | MAE HU | Validation seconds |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in result['rows']:lines.append('| {stage} | {sampling_timesteps} | {sum_scale} | {psnr:.8f} | {ssim:.8f} | {rmse_hu:.8f} | {mae_hu:.8f} | {validation_seconds:.3f} |'.format(**r))
    lines+=['','Best: steps={}, sum_scale={}.'.format(best['sampling_timesteps'],best['sum_scale']),'','Deltas vs control: '+json.dumps(result['deltas_vs_control']),'','Model weights unchanged; no training performed; L506 untouched; evaluation protocol unchanged.','Frozen source, CONFIG, checkpoint and original metrics SHA256 unchanged.','Runtime sum_scale affects initial noise only; diffusion coefficient buffers remain unchanged.','Timing uses the original validate function and includes validation loss, data loading and metrics. Shared control / scale=0.01 measurements are reused, not rerun.','Selection is on L333; this is sampler calibration, not an independent generalization result. No baseline artifacts were overwritten.']
    (OUT/'SAMPLER_CALIBRATION_REPORT.md').write_text('\n'.join(lines)+'\n')
    print('COMPLETE '+json.dumps(result['deltas_vs_control']),flush=True)

if __name__=='__main__':
    try:
        with torch.no_grad():main()
    except Exception:
        (OUT/'FAILED.txt').write_text(traceback.format_exc());raise
