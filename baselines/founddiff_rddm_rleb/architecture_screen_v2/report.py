"""Lightweight, observation-only V2 aggregation after the serial queue."""
import csv
import json
from mayo2016_smallset_v1.runtime import write_json, sha256, ROOT
from .protocol import HERE, STEPS, references, source_manifest
from .variants import ARMS

def build_report():
    queue=json.loads((HERE/'queue_status.json').read_text())
    ref=references();rows=[];arms={}
    for arm in ARMS:
        final_path=HERE/'runs'/arm/'final_summary.json'
        if final_path.is_file():
            result=json.loads(final_path.read_text())
            arms[arm]=result
            for rec in result['history']:
                step=int(rec['step'])
                rows.append({'arm':arm,'step':step,'status':result['status'],
                  'psnr':rec['psnr'],'baseline_psnr':ref[step]['psnr'],'delta_psnr':rec['delta_psnr'],
                  'ssim':rec['ssim'],'rmse_hu':rec['rmse_hu'],'mae_hu':rec['mae_hu'],
                  'validation_loss':rec['validation_loss'],'train_loss':rec['train_loss'],
                  'gradient_norm_pre_clip':rec['gradient_norm_pre_clip'],
                  'learning_rate':rec['learning_rate'],'parameter_count':rec['parameter_count'],
                  'peak_vram_bytes':rec['peak_vram_bytes'],'seconds_per_step':rec['seconds_per_step'],
                  'promotion':result['flags']['promotion'],'strong_fast_signal':result['flags']['strong_fast_signal'],
                  'test_opens':result['data_access']['test_opens'],
                  'external_memory_opens':result['data_access']['external_memory_opens'],
                  'blocked_attempts':result['data_access']['blocked_attempts']})
        else:
            arms[arm]={'status':queue['arms'][arm]['status'],'error':'No final summary; inspect progress and queue_process.log'}
    csv_path=HERE/'ARCHITECTURE_SCREEN_V2.csv'
    if rows:
        with csv_path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={'queue_status':queue['status'],'arms':arms,
      'matched_baseline':ref,'source_manifest':source_manifest(),
      'queue_source_sha256':sha256(ROOT/'architecture_screen_v2/queue.py'),
      'report_source_sha256':sha256(ROOT/'architecture_screen_v2/report.py'),
      'rules':{'promotion':'delta at 8k/10k/12k > 0 and 12k >= +0.05 dB',
       'strong_fast_signal':'delta at 12k >= +0.10 dB, only after promotion',
       'early_stop':'4k delta <= -0.5 dB',
       'full_validation':'only promoted arms; full L333 610 EMA; no test/external memory'},
      'no_combination':True,'no_formal_training':True}
    write_json(HERE/'ARCHITECTURE_SCREEN_V2_SUMMARY.json',summary)
    lines=['# RDDM Architecture Fast Screen V2','',
      'Four independent additions to the frozen FoundDiff RDDM+RLEB backbone. '
      'All Fast results use L067/L143 optimization and the same fixed L333 Fast64 EMA protocol. '
      'No candidate combinations were tested.','',
      '| Arm | 2k PSNR | 4k | 6k | 8k | 10k | 12k | Delta 12k | Promotion | Params | Peak VRAM GiB | s/step |',
      '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |']
    for arm in ARMS:
        r=arms[arm];by={int(x['step']):x for x in r.get('history',[])}
        values=[('{:.5f}'.format(by[s]['psnr']) if s in by else '—') for s in STEPS]
        last=by.get(12000)
        lines.append('| '+arm+' | '+' | '.join(values)+' | '+
          ('{:+.5f}'.format(last['delta_psnr']) if last else '—')+' | '+
          r.get('flags',{}).get('promotion',r['status'])+' | '+str(r.get('parameter_count','—'))+' | '+
          ('{:.2f}'.format(r['peak_vram_bytes']/2**30) if 'peak_vram_bytes' in r else '—')+' | '+
          ('{:.3f}'.format(r['seconds_per_step']) if 'seconds_per_step' in r else '—')+' |')
    lines += ['', 'Full metric curves, losses, gradient norms, and matched baseline deltas are in ARCHITECTURE_SCREEN_V2.csv.',
      'Promotion thresholds are engineering screening rules, not statistical significance.',
      'Full-L333 paired comparisons refer only to one patient volume.','']
    for arm in ARMS:
        r=arms[arm]
        if r.get('full_l333'):
            full=r['full_l333'];c=full['candidate'];d=full['delta_mean']
            lines.append('## '+arm+' Full-L333')
            lines.append('PSNR {:.5f} dB, SSIM {:.6f}, RMSE {:.4f} HU, MAE {:.4f} HU; paired delta PSNR {:+.5f} dB.'.format(c['psnr'],c['ssim'],c['rmse_hu'],c['mae_hu'],d['psnr']))
            lines.append('')
    lines += ['## Access and integrity','']
    for arm in ARMS:
        r=arms[arm];a=r.get('data_access',{})
        lines.append('- {}: status {}; test opens {}; external-memory opens {}; blocked attempts {}; nonfinite count {}.'.format(
            arm,r['status'],a.get('test_opens','unknown'),a.get('external_memory_opens','unknown'),
            a.get('blocked_attempts','unknown'),r.get('nonfinite_count','unknown')))
    lines += ['', 'Formal baseline source, checkpoint, frozen protocol, and split remain unchanged. '
      'No 60k formal training or candidate combination is part of this screen.','']
    (HERE/'ARCHITECTURE_SCREEN_V2_REPORT.md').write_text('\n'.join(lines))
    return summary

if __name__=='__main__':build_report()
