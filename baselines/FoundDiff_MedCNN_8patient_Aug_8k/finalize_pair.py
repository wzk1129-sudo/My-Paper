"""Read-only completed-run verification and final report generation."""
from pathlib import Path
import sys,csv,json,hashlib,math
import numpy as np,torch
ROOT=Path(__file__).resolve().parent;ARM=ROOT/'experiments/MedCNN_8patient_Aug_8k'
sys.path.insert(0,str(ARM))
from train import sources
from mayo_data import CFG,Pairs,Stream,VALID
from reporting import write_report,KEYS
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 done=json.loads((ARM/'COMPLETE.json').read_text())
 assert done['step']==8000 and [x['step'] for x in done['history']]==[2000,4000,6000,8000]
 assert done['source_hashes']==sources()
 for p,h in json.loads((ROOT/'ORIGINAL_SOURCE_HASHES.json').read_text()).items():assert sha(Path(p))==h
 for row in done['history']:
  st=row['step']
  with (ARM/f'validation_step_{st}.csv').open() as f:rr=list(csv.DictReader(f))
  assert len(rr)==64 and [int(x['slice_index']) for x in rr]==list(VALID) and all(x['patient_id']=='L333' for x in rr)
  for k in KEYS:assert abs(float(np.mean([float(x[k]) for x in rr]))-row[k])<1e-10
  ck=torch.load(ARM/f'checkpoints/step_{st}.pt',map_location='cpu',weights_only=False)
  assert ck['step']==ck['ema_updates']==st and ck['config']==CFG and ck['source_hashes']==done['source_hashes']
  assert all(torch.isfinite(v).all() for v in ck['ema_state_dict'].values());del ck
 logs=[json.loads(x) for x in (ARM/'train.jsonl').read_text().splitlines()]
 assert [x['step'] for x in logs]==list(range(1,8001))
 records=Pairs('train').records;rng=np.random.RandomState(10);arng=np.random.RandomState(10)
 order=rng.permutation(len(records));cursor=0;counts={p:0 for p in CFG['data']['train_patients']};transforms={}
 for row in logs:
  assert math.isfinite(row['loss']) and math.isfinite(row['gradient_norm']) and len(row['samples'])==4
  for meta in row['samples']:
   if cursor==len(order):order=rng.permutation(len(records));cursor=0
   p,i=records[int(order[cursor])];cursor+=1;flip=bool(arng.rand()<.5);rot=int(arng.randint(4))*90
   assert meta=={'patient_id':p,'slice_index':i,'horizontal_flip':flip,'rotation_degrees':rot}
   counts[p]+=1;key=str((flip,rot));transforms[key]=transforms.get(key,0)+1
 access=json.loads((ARM/'DATA_ACCESS.json').read_text())
 assert access['L506_opens']==0 and access['augmentation_counts']==transforms
 # Frozen MedDINO states from saved step0 and8k must match exactly.
 s0=torch.load(ARM/'checkpoints/step_0.pt',map_location='cpu',weights_only=False)
 s8=torch.load(ARM/'checkpoints/step_8000.pt',map_location='cpu',weights_only=False)
 medkeys=[k for k in s0['model_state_dict'] if '.structure.med.' in k];assert medkeys
 assert all(torch.equal(s0['model_state_dict'][k],s8['model_state_dict'][k]) and torch.equal(s8['model_state_dict'][k],s8['ema_state_dict'][k]) for k in medkeys)
 del s0,s8
 pre=json.loads((ROOT/'PREFLIGHT_DETAILS.json').read_text())
 write_report(done['history'],pre['parameters']['B']['total'],done['peak_allocated'],done['peak_reserved'])
 result={'status':'PASS','steps':8000,'validations':4,'validation_slices':64,'L506_opens':0,'source_hashes_unchanged':True,'meddino_frozen_unchanged':True,'all_training_samples_and_augmentations_replayed':True,'patient_center_counts':counts,'augmentation_counts':transforms,'best':json.loads((ARM/'BEST_CHECKPOINT.json').read_text())}
 (ARM/'FINAL_AUDIT.json').write_text(json.dumps(result,indent=2));print('FINAL_AUDIT_PASS',flush=True)
if __name__=='__main__':main()
