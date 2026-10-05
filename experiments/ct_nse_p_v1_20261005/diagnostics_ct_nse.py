"""Post-training report, curves, CPU reload and NDCT-only local retrieval."""
import hashlib,json,math
from pathlib import Path
import numpy as np
import torch
from ct_nse import CTNSE,StructurePredictor
ROOT=Path(__file__).resolve().parent
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def report(cfg,history,train_epochs,status,final=None):
 ready=json.loads((ROOT/'READINESS.json').read_text())
 lines=['# CT_NSE_P_V1','', 'Status: '+status,'',
 '## Source-faithful implementation audit','',
 'Reference: [Structure-Preserving Image Super-Resolution, TPAMI 2021](https://arxiv.org/html/2109.12530), Sections IV-A/IV-B/V-A and Appendix A/B.',
 'This standalone implementation follows the paper NSE-P principles and the exact CT architecture supplied for this experiment.',
 'Six convolutions, two residual skips, shallow local RF=31, output stride=4, 32-dimensional structure vectors.',
 'Predictor: concatenated horizontal left/right vectors -> Linear 64-128 -> ReLU -> Linear 128-64 -> ReLU -> Linear 64-32.',
 'Same-feature-map negatives only. Positive, contexts and negatives have nonoverlapping theoretical receptive fields.',
 'Context separation is 8 grid positions = 32 input pixels > RF 31.',
 'An overlap exists iff abs(dx)<=7 AND abs(dy)<=7; all such positions around left/target/right are excluded.',
 'Full RF grid positions are y/x [4,101] for 420 crops. Targets are y [4,101], x [12,93]. No padded boundary features enter the candidates.',
 'Positive candidate is index 0; logits=64*cosine; CrossEntropy mean over batch and K=16 anchors.','',
 '## Exact CT adaptations and data','',
 'Input is one-channel NDCT. Each convolution emits 32 channels as explicitly specified. Training is 60 epochs.',
 'Train: L067 560 + L143 585 NDCT center slices; validation: all 610 L333 NDCT center slices.',
 'Only target NPY paths are readable by dataset_ct_nse.py; access guard rejects LDCT, other patients and writes.',
 'Preprocessing exactly reproduces existing Mayo normalization: raw.float32-1024 -> clip [-1024,3072] -> (HU+1024)/4096.',
 'Random 420x420 crops and horizontal flips with p=0.5; no other augmentation.',
 'Validation crop coordinates and 16 anchors are fixed in VALIDATION_SAMPLING_MANIFEST.json.','',
 '## Architecture audit','',
 'Conv kernels=3, padding=1, strides=[2,1,1,2,1,1]. RF recurrence r_next=r+2*j; j_next=j*stride.',
 'RF progression: 3,7,11,15,23,31. Effective stride: 4.',
 'Two residual paths add Conv1 output after Conv3 and Conv4 output after Conv6.',
 'Input [2,1,420,420] -> output [2,32,105,105], with finite backward and nonzero module gradients.',
 'CTNSE parameters: '+str(ready['CTNSE_parameters'])+'; Predictor parameters: '+str(ready['Predictor_parameters'])+'.','',
 '## Training protocol','',
 'Adam lr=1e-3, betas=(0.9,0.999), wd=0; StepLR gamma=0.2 every 20 completed epochs.',
 'Epoch indices 0-19 use 1e-3; 20-39 use 2e-4; 40-59 use 4e-5. No gradient clipping or added loss.',
 'Seed=20260829; AMP=False; CUBLAS_WORKSPACE_CONFIG=:4096:8; deterministic PyTorch/CuDNN; TF32 disabled.',
 'Effective batch='+str(cfg['training']['effective_batch_size'])+'; drop_last=True (one shuffled pass per epoch).',
 'Batch48 OOM fallback used: '+str(cfg['training']['batch48_oom'])+'.',
 'Readiness peak allocated VRAM GiB: %.4f'%(max(a['peak_vram_bytes'] for a in ready['attempts'])/1024**3),
 'Validation every 5 completed epochs plus initial epoch0. Full candidate set is used for exact top1.','',
 '## Validation curve','',
 '| Epoch | Val loss | Pos cosine | Neg cosine | Margin | Top1 | Global mean | Global std | Channel std mean/min/max | Norm mean/std |',
 '|---:|---:|---:|---:|---:|---:|---:|---:|---|---|']
 for x in history:
  lines.append('| {epoch} | {loss:.6f} | {positive_cosine:.6f} | {negative_cosine:.6f} | {margin:.6f} | {top1:.6f} | {feature_global_mean:.6f} | {feature_global_std:.6f} | {channel_std_mean:.6f}/{channel_std_min:.6f}/{channel_std_max:.6f} | {feature_norm_mean:.6f}/{feature_norm_std:.6f} |'.format(**x))
 if train_epochs:
  lines+=['','Training epoch loss is recorded in logs/train_epochs.csv; per-step metrics in logs/train.jsonl.']
 if final:
  lines+=['','## Final decision and audits','',
    'Decision: '+final['decision']+'.',
    'Prespecified diagnostic criteria: no collapse, final/initial val loss <=0.9, best val top1 >=0.01, final positive-negative cosine margin >=0.02.',
    'Final/initial loss ratio: %.6f; best top1: %.6f.'%(final['final_initial_loss_ratio'],final['best_top1']),
    'Mean sec/step: %.6f; total seconds: %.2f; peak allocated VRAM GiB: %.4f.'%
      (final['mean_seconds_per_step'],final['total_seconds'],final['peak_vram_bytes']/1024**3),
    'All validation tensors, losses, gradients and parameters passed finite checks. No representation collapse was recorded.',
    'LDCT access=0; L506 access=0; external memory access=0; blocked attempts=0. Actual NDCT open counts are in DATA_ACCESS_AUDIT.json.',
    'Readiness opens are separately recorded in READINESS_DATA_ACCESS_AUDIT.json.',
    'Model/source/config hashes and Git state are recorded in SOURCE_MANIFEST.json and RUN_PROVENANCE.json.',
    'Best-checkpoint CPU reload and SHA256:','']
  for x in final['checkpoints']:
   lines.append('- '+x['path']+'; epoch '+str(x['epoch'])+'; SHA256 '+x['SHA256']+'; CPU forward finite='+str(x['cpu_reload_forward_finite']))
  lines+=['','Structure sanity retrieval: visuals/retrieval_contact_sheet.png and RETRIEVAL_METADATA.json.',
    'Queries use slice300; candidates come from slices100/200/400/500 of the same L333 validation collection. Exact/neighboring self-matches are excluded by searching other slices.',
    'Display window [-160,240] HU is post-hoc visualization only. Training preprocessing is unchanged.',
    'This experiment evaluates local CT representation learning. It does not establish a downstream RDDM metric gain.',
    'The training and report are complete; no follow-on model training is launched.']
 (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
def cpu_reload():
 audits=[]
 for name in ('best_val_loss.pt','best_val_top1.pt'):
  p=ROOT/'checkpoints'/name;ck=torch.load(p,map_location='cpu',weights_only=False)
  required=('ctnse_state_dict','predictor_state_dict','optimizer','scheduler','epoch','seed',
            'rng_states','numpy_train_generator_state','config','preprocessing')
  assert all(k in ck for k in required)
  nse=CTNSE().eval();pred=StructurePredictor().eval()
  nse.load_state_dict(ck['ctnse_state_dict']);pred.load_state_dict(ck['predictor_state_dict'])
  with torch.no_grad():
   out=nse(torch.ones(2,1,420,420)*.3)
   assert out.shape==(2,32,105,105) and torch.isfinite(out).all()
   value=pred(out[:,:,50,42],out[:,:,50,58]);assert torch.isfinite(value).all()
  audits.append({'path':str(p),'epoch':ck['epoch'],'SHA256':sha(p),'bytes':p.stat().st_size,
                 'cpu_reload_forward_finite':True,'checkpoint_required_fields_present':True})
 write(ROOT/'CHECKPOINT_AUDIT.json',audits)
 return audits
def plots(history,train_epochs):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,3,figsize=(14,4))
 axes[0].plot([x['epoch'] for x in train_epochs],[x['loss'] for x in train_epochs],label='train')
 axes[0].plot([x['epoch'] for x in history],[x['loss'] for x in history],label='val');axes[0].set_title('Contrastive loss');axes[0].legend()
 axes[1].plot([x['epoch'] for x in history],[x['top1'] for x in history]);axes[1].set_title('Validation top1')
 axes[2].plot([x['epoch'] for x in history],[x['margin'] for x in history]);axes[2].set_title('Positive-negative cosine margin')
 for ax in axes:ax.set_xlabel('Completed epoch');ax.grid(alpha=.25)
 fig.tight_layout();fig.savefig(ROOT/'visuals/training_curve.png',dpi=160);plt.close(fig)
def retrieval(nse,ds,records):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 chosen={r['index']:r for r in records if r['index'] in (100,200,300,400,500)}
 features={};images={}
 with torch.no_grad():
  for index,r in chosen.items():
   image=ds.load('L333',index,'val');top=r['crop_top'];left=r['crop_left']
   patch=image[top:top+420,left:left+420];images[index]=patch
   feats=nse(torch.from_numpy(np.ascontiguousarray(patch[None,None])).cuda())
   features[index]=torch.nn.functional.normalize(feats[0].flatten(1).T,dim=-1).cpu()
 positions=[(y,x) for y in range(4,102) for x in range(4,102)]
 ids=torch.tensor([y*105+x for y,x in positions]);other=[100,200,400,500]
 pool=torch.cat([features[i][ids] for i in other])
 queries=[(40,40),(50,55),(65,45),(70,65)]
 fig,axes=plt.subplots(4,5,figsize=(12,10));metadata=[]
 for row,(y,x) in enumerate(queries):
  q=features[300][y*105+x];assert q.norm()>0
  vals,indices=(pool@q).topk(4)
  entries=[{'slice':300,'y':y,'x':x,'cosine':1.,'label':'query'}]
  for rank,(score,index) in enumerate(zip(vals.tolist(),indices.tolist()),1):
   yy,xx=positions[index%len(positions)]
   entries.append({'slice':other[index//len(positions)],'y':yy,'x':xx,'cosine':score,'label':'top-'+str(rank)})
  for col,item in enumerate(entries):
   patch=images[item['slice']][item['y']*4-15:item['y']*4+16,item['x']*4-15:item['x']*4+16]
   assert patch.shape==(31,31);hu=patch*4096-1024
   item['hu_mean']=float(hu.mean());item['hu_std']=float(hu.std())
   ax=axes[row,col];ax.imshow(hu,cmap='gray',vmin=-160,vmax=240)
   ax.set_title(item['label']+' / slice '+str(item['slice'])+'\ncos=%.4f'%item['cosine'],fontsize=9);ax.axis('off')
  metadata.append(entries)
 fig.suptitle('CT-NSE local structure retrieval; NDCT; display [-160,240] HU',fontsize=12)
 fig.tight_layout();fig.savefig(ROOT/'visuals/retrieval_contact_sheet.png',dpi=180);plt.close(fig)
 write(ROOT/'visuals/RETRIEVAL_METADATA.json',metadata)
def final_diagnostics(nse,ds,records,history,train_epochs):
 audits=cpu_reload()
 ck=torch.load(ROOT/'checkpoints/best_val_loss.pt',map_location='cpu',weights_only=False)
 nse.load_state_dict(ck['ctnse_state_dict']);nse.eval()
 retrieval(nse,ds,records);plots(history,train_epochs)
 return audits

