"""Single MedCNN eight-patient preflight, with synchronized augmentation checks."""
from pathlib import Path
import sys,json,hashlib,copy,ast,time
import numpy as np,torch
ROOT=Path(__file__).resolve().parent
ARM=ROOT/'experiments/MedCNN_8patient_Aug_8k'
sys.path.insert(0,str(ARM))
from train import build,setup,sources
from mayo_data import CFG,Pairs,Stream,VALID,access_report,apply_augmentation
from src.DADiff import TimeOnlyDACB,ResnetBlock
OLD=Path('/home/xujian/research/FoundDiff_TimeOnly_PairedFastDev')
def sh(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def tensor_hash(state):
 h=hashlib.sha256()
 for n,v in sorted(state.items()):
  h.update(n.encode());h.update(v.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
 return h.hexdigest()
def grad(mod):
 gs=[p.grad for p in mod.parameters() if p.requires_grad and p.grad is not None]
 assert gs and all(torch.isfinite(g).all() for g in gs)
 return sum(float(g.abs().sum()) for g in gs)
def main():
 started=time.perf_counter();setup()
 for n in ['train.py','src/DADiff.py','src/medcnn.py','frozen_metrics.py']:
  assert (ARM/n).read_bytes()==(OLD/'experiments/Fast2P_TimeOnly_MedCNN_8k'/n).read_bytes(),n
 assert len(Pairs('train'))==4800 and len(Pairs('validation'))==64
 assert 'L506' not in CFG['data']['train_patients']
 # Exhaust all 8 exact index permutations; synthetic marker values reveal channel/label misalignment.
 a=np.arange(3*13*13,dtype=np.float32).reshape(3,13,13);b=np.arange(13*13,dtype=np.float32).reshape(1,13,13)+9999
 for flip in [False,True]:
  for rot in range(4):
   x,y=apply_augmentation(a,b,flip,rot)
   expected=np.rot90(np.flip(a,axis=-1) if flip else a,rot,axes=(-2,-1))
   ey=np.rot90(np.flip(b,axis=-1) if flip else b,rot,axes=(-2,-1))
   assert np.array_equal(x,expected) and np.array_equal(y,ey)
   assert x.flags.c_contiguous and y.flags.c_contiguous
   assert np.array_equal(np.sort(x.ravel()),np.sort(a.ravel()))
   assert np.array_equal(np.sort(y.ravel()),np.sort(b.ravel()))
 # Test actual augmented batch equality against unaugmented dataset using logged transform metadata.
 data=Pairs('train');stream=Stream(data)
 context,target,metas=stream.next()
 for j,meta in enumerate(metas):
  pos=data.records.index((meta['patient_id'],meta['slice_index']))
  x,y,_=data.load(pos);ex,ey=apply_augmentation(x,y,meta['horizontal_flip'],meta['rotation_degrees']//90)
  assert np.array_equal(ex,context[j].numpy()) and np.array_equal(ey,target[j].numpy())
 state=copy.deepcopy(stream.state_dict());batch_a=stream.next();stream.load_state_dict(state);batch_b=stream.next()
 assert torch.equal(batch_a[0],batch_b[0]) and torch.equal(batch_a[1],batch_b[1]) and batch_a[2]==batch_b[2]
 # Validation stays unaugmented and deterministic.
 val=Pairs('validation');v1=val.load(5);v2=val.load(5)
 assert np.array_equal(v1[0],v2[0]) and v1[2]==v2[2] and 'horizontal_flip' not in v1[2]
 setup();m=build()
 ck=torch.load(Path(CFG['paired_control_dir'])/'checkpoints/step_0.pt',map_location='cpu',weights_only=False)
 assert ck['step']==0 and ck['ema_updates']==0
 rr=m.load_state_dict(ck['model_state_dict'],strict=False)
 assert not rr.unexpected_keys and all('.structure.' in k or '.adapters.' in k for k in rr.missing_keys)
 del ck
 # Compare with original MedCNN untrained step0, never use any trained checkpoint.
 old=torch.load(OLD/'experiments/Fast2P_TimeOnly_MedCNN_8k/checkpoints/step_0.pt',map_location='cpu',weights_only=False)
 assert old['step']==0 and old['ema_updates']==0
 assert all(torch.equal(m.state_dict()[k],v) for k,v in old['model_state_dict'].items())
 initial_hash=tensor_hash(m.state_dict());del old
 assert sum(isinstance(x,TimeOnlyDACB) for x in m.modules())==9
 assert sum(isinstance(x,ResnetBlock) for x in m.modules())==10
 assert len(m.model.unet0.adapters)==9
 u=m.model.unet0;med=u.structure.med;med_before=tensor_hash(med.state_dict())
 assert all(not p.requires_grad for p in med.parameters())
 m=m.cuda().train();assert not med.training and not med.net.training
 context=context.cuda();target=target.cuda()
 opt=torch.optim.Adam((p for p in m.parameters() if p.requires_grad),lr=2e-4,betas=(.9,.99))
 calls={};hooks=[]
 for name,mod in u.adapters.items():
  def hook(mod,args,out,_name=name):
   assert out.shape==args[0].shape and torch.isfinite(out).all();calls[_name]=calls.get(_name,0)+1
  hooks.append(mod.register_forward_hook(hook))
 torch.cuda.reset_peak_memory_stats();rows=[]
 for update in range(2):
  opt.zero_grad(set_to_none=True)
  # Match actual gradient accumulation=2 while keeping batch2.
  for micro in range(2):
   loss=m([target,context[:,1:2],context])[0];assert torch.isfinite(loss);(loss/2).backward()
  st=u.structure
  norms={'cnn':grad(st.cnn),'projections':grad(st.projections),'med_fuse':grad(st.med_fuse),'fuse64':grad(st.fuse64),'fuse128':grad(st.fuse128),'fuse256':grad(st.fuse256),'fuse512':grad(st.fuse512),**{'adapter_'+n:grad(x) for n,x in u.adapters.items()}}
  if update:assert all(x>0 for x in norms.values())
  assert all(p.grad is None for p in med.parameters())
  gn=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True));opt.step()
  rows.append({'update':update+1,'loss':float(loss.detach()),'gradient_norm':gn,'groups':norms})
 for h in hooks:h.remove()
 assert set(calls)==set(u.adapters)
 m.eval();seen=[]
 def stem(mod,args):
  assert torch.equal(args[0][:,1:],context[:1]*2-1);seen.append(1)
 h=u.init_conv.register_forward_pre_hook(stem)
 with torch.no_grad():pred=m.sample([context[:1,1:2],context[:1]],last=True)[-1]
 h.remove();assert len(seen)==2 and pred.shape==(1,1,512,512) and torch.isfinite(pred).all()
 assert tensor_hash(med.state_dict())==med_before
 peak=torch.cuda.max_memory_allocated();assert peak<24*2**30
 assert access_report()['L506_opens']==0
 result={'status':'PASS','augmentation_all_8_combinations_exact':True,'augmentation_actual_batch_synchronized':True,'augmentation_resume_rng_exact':True,'validation_unaugmented':True,'original_medcnn_step0_exact':True,'initial_state_sha256':initial_hash,'time_dacb_count':9,'rleb_count':10,'adapter_count':9,'adapter_calls':calls,'checks':rows,'meddino':med.load_confirmation,'meddino_frozen_unchanged':True,'peak_allocated':peak,'seconds':time.perf_counter()-started,'parameters':{'B':{'total':sum(p.numel() for p in m.parameters()),'trainable':sum(p.numel() for p in m.parameters() if p.requires_grad)}},'data_access':access_report(),'environment':{'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'numpy':np.__version__}}
 (ROOT/'PREFLIGHT_DETAILS.json').write_text(json.dumps(result,indent=2))
 (ARM/'PREFLIGHT.json').write_text(json.dumps({'status':'PASS','source_hashes':sources(),'report':str(ROOT/'PREFLIGHT_DETAILS.json')},indent=2))
 print('PREFLIGHT_PASS',result['parameters'],peak,'seconds',result['seconds'],flush=True)
if __name__=='__main__':main()
