"""Summarize completed 8k runs without copying CT images or checkpoints."""
import argparse
import csv
import json
import math
from pathlib import Path
from fast_dev.protocol import FAST_INDICES

ROOT=Path(__file__).resolve().parent
RUNS={'CONTROL_STRIPPED':ROOT/'runs/control_8k_20261001',
      'CANDIDATE_OFFICIAL_RDDM':ROOT/'runs/candidate_8k_20261001'}
METRICS=('psnr','ssim','rmse_hu','mae_hu')
STEPS=(0,2000,4000,6000,8000)

def rows(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))

def main():
    metrics={};first={};summary={}
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='results',help='new output directory; existing results are never overwritten')
    args=parser.parse_args()
    output=Path(args.output)
    if not output.is_absolute():output=ROOT/output
    output.mkdir(exist_ok=False)
    with (output/'training_diagnostics.csv').open('w',newline='') as f:
        fields=('arm','step','train_loss','attention_gradient_norm',
                'parameters_finite','seconds_per_step','peak_vram_bytes')
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");writer.writeheader()
        for arm,run in RUNS.items():
            assert (run/'final_state.json').is_file()
            access=json.loads((run/'data_access_audit.json').read_text())
            assert access['L506_access_count']==0 and access['test_opens']==0
            assert access['external_memory_opens']==0 and not access['external_memory_enabled']
            assert access['patients_opened']==['L067','L143','L333']
            assert access['validation_indices_opened']==list(FAST_INDICES)
            state=json.loads((run/'final_state.json').read_text())
            assert state['step']==8000 and state['parameters_finite']
            traincount=0;loss_min=math.inf;loss_max=-math.inf;attn_min=math.inf;attn_max=0.
            step_time_sum=0.;finite_audits=0
            with (run/'train.jsonl').open() as log:
                for line in log:
                    rec=json.loads(line);traincount+=1
                    assert rec['optimizer_step']==traincount
                    assert not rec['nonfinite'] and math.isfinite(rec['train_loss'])
                    assert math.isfinite(rec['gradient_norm_pre_clip'])
                    if traincount==1:first[arm]=rec
                    loss_min=min(loss_min,rec['train_loss']);loss_max=max(loss_max,rec['train_loss'])
                    attn=float(rec['attention_gradient_norm'])
                    if arm=='CANDIDATE_OFFICIAL_RDDM':assert math.isfinite(attn) and attn>0
                    else:assert attn==0
                    attn_min=min(attn_min,attn);attn_max=max(attn_max,attn)
                    if rec.get('parameters_finite') is not None:
                        assert rec['parameters_finite'];finite_audits+=1
                    step_time_sum+=rec['seconds_per_step']
                    writer.writerow({'arm':arm,'step':traincount,'train_loss':rec['train_loss'],
                        'attention_gradient_norm':attn,
                        'parameters_finite':rec.get('parameters_finite'),
                        'seconds_per_step':rec['seconds_per_step'],
                        'peak_vram_bytes':rec['peak_vram_bytes']})
            assert traincount==8000 and finite_audits>=81
            val=rows(run/'validation_metrics.csv')
            assert tuple(int(r['step']) for r in val)==STEPS
            for r in val:
                assert r['arm']==arm and r['patient']=='L333'
                assert r['weights']=='ema' and r['validation_mode']=='FAST'
                assert int(r['slices'])==64 and r['parameters_finite']=='True'
                assert all(math.isfinite(float(r[k])) for k in METRICS)
                indices=[v['slice_index'] for v in json.loads(
                    (run/('validation_step_%s.json'%r['step'])).read_text())]
                assert indices==list(FAST_INDICES)
            metrics[arm]=val
            summary[arm]={'steps':traincount,'loss_min':loss_min,'loss_max':loss_max,
                'attention_grad_min':attn_min,'attention_grad_max':attn_max,
                'parameter_finite_audits':finite_audits,'mean_step_seconds':step_time_sum/traincount,
                'peak_vram_bytes':state['peak_vram_bytes'],'data_access':access}
    checks={k:first['CONTROL_STRIPPED'].get(k)==first['CANDIDATE_OFFICIAL_RDDM'].get(k)
        for k in ('samples','timesteps','noise_sha256',
                  'pre_step_rng_sha256','post_step_rng_sha256')}
    assert all(checks.values())
    with (output/'fast_dev_metrics.csv').open('w',newline='') as f:
        fields=('step',)+tuple('control_'+m for m in METRICS)+tuple('candidate_'+m for m in METRICS)+tuple('delta_'+m for m in METRICS)
        w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");w.writeheader()
        for c,a in zip(metrics['CONTROL_STRIPPED'],metrics['CANDIDATE_OFFICIAL_RDDM']):
            assert c['step']==a['step']
            row={'step':c['step']}
            for m in METRICS:
                row['control_'+m]=c[m];row['candidate_'+m]=a[m]
                row['delta_'+m]=float(a[m])-float(c[m])
            w.writerow(row)
    summary['first_batch_matches']=checks
    summary['official_commit']=json.loads((ROOT/'SOURCE_PROVENANCE.json').read_text())['official_commit']
    (output/'audit_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('Complete, 8000 steps per arm; all first-batch and access checks passed.')
if __name__=='__main__':main()
