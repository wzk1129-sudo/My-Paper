"""Independent NDCT-only CT-NSE-P trainer."""
import argparse,csv,gc,hashlib,json,math,os,random,statistics,subprocess,sys,time,traceback
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from ct_nse import CTNSE,StructurePredictor,receptive_field_audit,sample_anchors,contrastive_prediction
from dataset_ct_nse import NDCTOnlyDataset,NDCTAccessAudit
ROOT=Path(__file__).resolve().parent
def now():return datetime.now().astimezone().isoformat()
def write(p,x):
 p=Path(p);tmp=p.with_name(p.name+'.tmp')
 tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def sources():
 names=('ct_nse.py','dataset_ct_nse.py','train_ct_nse.py','diagnostics_ct_nse.py','test_ct_nse.py','config.json')
 return {str(ROOT/n):sha(ROOT/n) for n in names}
def setup(seed):
 assert os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
def rng_state():
 return {'python':random.getstate(),'numpy':np.random.get_state(),
         'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}
def finite(models):
 return all(torch.isfinite(p).all().item() for m in models for p in m.parameters())
def grads(models,required=True):
 out={}
 for label,m in zip(('CTNSE','Predictor'),models):
  local=[]
  for n,p in m.named_parameters():
   if p.grad is None:assert not required,(label,n,'missing gradient')
   else:
    assert torch.isfinite(p.grad).all(),(label,n,'nonfinite gradient')
    value=float(p.grad.norm());out[label+'.'+n]=value;local.append(value)
  if required:assert local and sum(local)>0,(label,'all gradients zero')
 return out
def savecsv(p,rows):
 with Path(p).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def probe_batch(ds,cfg,batch):
 rng=np.random.default_rng(cfg['training']['seed'])
 torch.manual_seed(cfg['training']['seed']);torch.cuda.manual_seed_all(cfg['training']['seed'])
 nse=CTNSE().cuda();pred=StructurePredictor().cuda()
 opt=torch.optim.Adam(list(nse.parameters())+list(pred.parameters()),lr=.001,betas=(.9,.999),weight_decay=0)
 torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
 images=ds.train_batch(np.arange(batch),rng).cuda()
 anchors=torch.from_numpy(sample_anchors(rng,batch,105,105,16)).cuda()
 feats=nse(images);assert feats.shape==(batch,32,105,105)
 loss,metrics=contrastive_prediction(feats,pred,anchors,64.)
 loss.backward();norms=grads((nse,pred));opt.step();assert finite((nse,pred))
 torch.cuda.synchronize()
 return {'batch':batch,'status':'PASS','loss':float(loss.detach()),
         'peak_vram_bytes':torch.cuda.max_memory_allocated(),'seconds':time.perf_counter()-tick,
         'gradient_norms':norms,'metrics':metrics,
         'CTNSE_parameters':sum(p.numel() for p in nse.parameters()),
         'Predictor_parameters':sum(p.numel() for p in pred.parameters())}
def readiness():
 cfg=json.loads((ROOT/'config.json').read_text());setup(cfg['training']['seed'])
 with NDCTAccessAudit(Path(cfg['data_root'])) as audit:
  ds=NDCTOnlyDataset(Path(cfg['data_root']),audit)
  result={'status':'IN_PROGRESS','geometry':receptive_field_audit(),
          'train_ndct_count':len(ds.train_keys),'val_ndct_count':len(ds.val_keys),
          'manifest_sha256':ds.manifest_sha256,'attempts':[]}
  chosen=None
  for batch in (48,32):
   success=False
   try:
    item=probe_batch(ds,cfg,batch);result['attempts'].append(item);success=True
   except torch.OutOfMemoryError as e:
    result['attempts'].append({'batch':batch,'status':'OOM','error':str(e),
                              'peak_vram_bytes':torch.cuda.max_memory_allocated()})
   gc.collect();torch.cuda.empty_cache()
   if success:
    chosen=batch;result['CTNSE_parameters']=item['CTNSE_parameters']
    result['Predictor_parameters']=item['Predictor_parameters'];break
   if batch==32:raise RuntimeError('Both allowed batch sizes OOM')
  assert chosen in (48,32)
  cfg['training'].update({'effective_batch_size':chosen,'batch48_oom':chosen==32,
                         'readiness_updates_discarded':True})
  cfg['decision']={'val_loss_ratio_max':.9,'best_top1_min':.01,'final_similarity_margin_min':.02}
  result.update({'status':'PASS','effective_batch':chosen,'data_access':audit.report(),
                 'source_hashes':sources(),'end_time':now()})
  assert all(result['data_access'][k]==0 for k in ('ldct_opens','L506_opens','external_memory_opens','blocked_attempts'))
  write(ROOT/'EFFECTIVE_CONFIG.json',cfg)
  result['effective_config_sha256']=sha(ROOT/'EFFECTIVE_CONFIG.json')
  write(ROOT/'READINESS.json',result)
  write(ROOT/'READINESS_DATA_ACCESS_AUDIT.json',audit.report())
  print('READINESS '+json.dumps(result),flush=True)
def validate(nse,pred,ds,records,cfg,epoch,step):
 nse.eval();pred.eval()
 totals={k:0. for k in ('loss','positive_cosine','negative_cosine','margin','top1','negative_count_mean')}
 count=0;s=torch.zeros(32,dtype=torch.float64,device='cuda');sq=torch.zeros_like(s);n=0
 ns=0.;nsq=0.;nn=0;tick=time.perf_counter()
 with torch.no_grad():
  bs=cfg['validation']['batch_size']
  for start in range(0,len(records),bs):
   chosen=records[start:start+bs];images,anchors=ds.val_batch(chosen)
   feats=nse(images.cuda());assert torch.isfinite(feats).all()
   loss,metrics=contrastive_prediction(feats,pred,anchors.cuda(),64.);assert torch.isfinite(loss)
   for k,v in metrics.items():totals[k]+=v*len(chosen)
   count+=len(chosen);s+=feats.sum((0,2,3),dtype=torch.float64)
   sq+=feats.square().sum((0,2,3),dtype=torch.float64);n+=len(chosen)*105*105
   norm=feats.norm(dim=1);ns+=float(norm.sum(dtype=torch.float64))
   nsq+=float(norm.square().sum(dtype=torch.float64));nn+=norm.numel()
 assert count==610 and finite((nse,pred));grads((nse,pred),False)
 channel_mean=s/n;std=(sq/n-channel_mean.square()).clamp_min(0).sqrt()
 mean=float(s.sum()/(32*n));global_std=math.sqrt(max(0,float(sq.sum()/(32*n))-mean**2))
 norm_mean=ns/nn;norm_std=math.sqrt(max(0,nsq/nn-norm_mean**2))
 collapse=global_std<1e-4 or float(std.max())<1e-4
 row={'epoch':epoch,'optimizer_step':step,**{k:v/count for k,v in totals.items()},
      'feature_global_mean':mean,'feature_global_std':global_std,
      'channel_std_mean':float(std.mean()),'channel_std_min':float(std.min()),'channel_std_max':float(std.max()),
      'feature_norm_mean':norm_mean,'feature_norm_std':norm_std,'representation_collapse':collapse,
      'all_tensors_parameters_gradients_finite':True,'slices':count,'validation_seconds':time.perf_counter()-tick}
 write(ROOT/f'logs/validation_epoch_{epoch}.json',row)
 print('VALIDATION '+json.dumps(row),flush=True)
 if collapse:raise RuntimeError('REPRESENTATION_COLLAPSE')
 return row
def checkpoint(path,nse,pred,opt,scheduler,epoch,step,rng,cfg,history,bests,audit,hashes):
 assert sources()==hashes
 payload={'ctnse_state_dict':nse.state_dict(),'predictor_state_dict':pred.state_dict(),
          'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch,
          'optimizer_step':step,'seed':cfg['training']['seed'],'rng_states':rng_state(),
          'numpy_train_generator_state':rng.bit_generator.state,'config':cfg,
          'preprocessing':{k:cfg[k] for k in ('input_view','stored_value_offset_hu','hu_clip','normalization','input_size','crop_size')},
          'validation_history':history,'best_metrics':bests,'data_access_audit':audit,'source_hashes':hashes}
 tmp=path.with_name(path.name+'.tmp');torch.save(payload,tmp);os.replace(tmp,path)
def train():
 from diagnostics_ct_nse import report,final_diagnostics
 cfg=json.loads((ROOT/'EFFECTIVE_CONFIG.json').read_text())
 ready=json.loads((ROOT/'READINESS.json').read_text())
 assert ready['status']=='PASS' and sources()==ready['source_hashes']
 assert sha(ROOT/'EFFECTIVE_CONFIG.json')==ready['effective_config_sha256']
 assert not (ROOT/'logs/train.jsonl').exists(),'Existing run: do not restart from zero'
 setup(cfg['training']['seed']);batch=cfg['training']['effective_batch_size']
 started=time.perf_counter();hashes=sources()
 write(ROOT/'SOURCE_MANIFEST.json',hashes)
 write(ROOT/'RUN_PROVENANCE.json',{'command':sys.argv,'python':sys.executable,'pytorch':torch.__version__,
   'CUDA':torch.version.cuda,'GPU':torch.cuda.get_device_name(0),'cublas_workspace_config':os.environ['CUBLAS_WORKSPACE_CONFIG'],
   'git_branch':subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
   'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
   'git_status_short':subprocess.check_output(['git','status','--short'],text=True),
   'start_time':now(),'source_hashes':hashes,'effective_config':cfg})
 nse=CTNSE().cuda();pred=StructurePredictor().cuda();params=list(nse.parameters())+list(pred.parameters())
 opt=torch.optim.Adam(params,lr=.001,betas=(.9,.999),weight_decay=0)
 scheduler=torch.optim.lr_scheduler.StepLR(opt,step_size=20,gamma=.2)
 rng=np.random.default_rng(cfg['training']['seed'])
 history=[];train_epochs=[];times=[];step=0
 torch.cuda.reset_peak_memory_stats()
 with NDCTAccessAudit(Path(cfg['data_root'])) as audit:
  ds=NDCTOnlyDataset(Path(cfg['data_root']),audit);records=ds.val_manifest(cfg['training']['seed'])
  write(ROOT/'VALIDATION_SAMPLING_MANIFEST.json',records)
  initial=validate(nse,pred,ds,records,cfg,0,0);history.append(initial)
  bests={'loss':initial['loss'],'top1':initial['top1']}
  for name in ('best_val_loss.pt','best_val_top1.pt'):
   checkpoint(ROOT/'checkpoints'/name,nse,pred,opt,scheduler,0,0,rng,cfg,history,bests,audit.report(),hashes)
  savecsv(ROOT/'logs/validation_history.csv',history);report(cfg,history,train_epochs,'RUNNING')
  write(ROOT/'STATUS.json',{'status':'RUNNING','epoch':0,'optimizer_step':0,'max_epochs':60})
  with (ROOT/'logs/train.jsonl').open('x',buffering=1) as log:
   for epoch in range(60):
    order=rng.permutation(len(ds.train_keys));losses=[];lr=opt.param_groups[0]['lr']
    assert math.isclose(lr,.001*(.2**(epoch//20)),rel_tol=1e-12)
    for start in range(0,len(order)-batch+1,batch):
     tick=time.perf_counter();nse.train();pred.train();opt.zero_grad(set_to_none=True)
     images=ds.train_batch(order[start:start+batch],rng).cuda()
     anchors=torch.from_numpy(sample_anchors(rng,batch,105,105,16)).cuda()
     feats=nse(images);assert feats.shape==(batch,32,105,105) and torch.isfinite(feats).all()
     loss,metrics=contrastive_prediction(feats,pred,anchors,64.)
     loss.backward();norms=grads((nse,pred));opt.step();assert finite((nse,pred))
     torch.cuda.synchronize();seconds=time.perf_counter()-tick
     times.append(seconds);losses.append(metrics['loss']);step+=1
     row={'epoch_index':epoch,'optimizer_step':step,'batch_size':batch,'lr':lr,**metrics,
       'seconds_per_step':seconds,'peak_vram_bytes':torch.cuda.max_memory_allocated(),
       'loss_finite':True,'gradients_finite_nonzero':True,'parameters_finite':True,
       'AMP':False,'NaN_Inf':False,'gradient_overflow':False}
     log.write(json.dumps(row,allow_nan=False)+'\n')
     if step==1:write(ROOT/'HEALTH_CHECK.json',{'status':'PASS','step':step,'row':row,'gradient_norms':norms})
     if step==1 or step%10==0:
      write(ROOT/'STATUS.json',{'status':'RUNNING','epoch_index':epoch,'optimizer_step':step,'max_epochs':60,'last_train':row})
      write(ROOT/'DATA_ACCESS_AUDIT.json',audit.report());print('TRAIN '+json.dumps(row),flush=True)
    scheduler.step();train_epochs.append({'epoch':epoch+1,'loss':statistics.mean(losses),'lr_used':lr})
    savecsv(ROOT/'logs/train_epochs.csv',train_epochs)
    if (epoch+1)%5==0:
     val=validate(nse,pred,ds,records,cfg,epoch+1,step);history.append(val)
     savecsv(ROOT/'logs/validation_history.csv',history)
     if val['loss']<bests['loss']:
      bests['loss']=val['loss']
      checkpoint(ROOT/'checkpoints/best_val_loss.pt',nse,pred,opt,scheduler,epoch+1,step,rng,cfg,history,bests,audit.report(),hashes)
     if val['top1']>bests['top1']:
      bests['top1']=val['top1']
      checkpoint(ROOT/'checkpoints/best_val_top1.pt',nse,pred,opt,scheduler,epoch+1,step,rng,cfg,history,bests,audit.report(),hashes)
     checkpoint(ROOT/f'checkpoints/epoch_{epoch+1}.pt',nse,pred,opt,scheduler,epoch+1,step,rng,cfg,history,bests,audit.report(),hashes)
     write(ROOT/'DATA_ACCESS_AUDIT.json',audit.report());report(cfg,history,train_epochs,'RUNNING')
  assert step==60*(len(ds.train_keys)//batch) and sources()==hashes
  ck_audits=final_diagnostics(nse,ds,records,history,train_epochs)
  ratio=history[-1]['loss']/history[0]['loss']
  passed=(ratio<=cfg['decision']['val_loss_ratio_max'] and bests['top1']>=cfg['decision']['best_top1_min']
          and history[-1]['margin']>=cfg['decision']['final_similarity_margin_min']
          and not any(x['representation_collapse'] for x in history))
  access=audit.report()
  assert all(access[k]==0 for k in ('ldct_opens','L506_opens','external_memory_opens','blocked_attempts'))
  final={'status':'COMPLETE','epoch':60,'optimizer_step':step,'decision':'PASS' if passed else 'FAIL',
         'final_initial_loss_ratio':ratio,'best_top1':bests['top1'],'final_validation':history[-1],
         'mean_seconds_per_step':statistics.mean(times),'total_seconds':time.perf_counter()-started,
         'peak_vram_bytes':torch.cuda.max_memory_allocated(),'checkpoints':ck_audits,'data_access':access,
         'source_hashes_unchanged':True,'end_time':now()}
  write(ROOT/'FINAL_STATE.json',final);write(ROOT/'STATUS.json',final);write(ROOT/'DATA_ACCESS_AUDIT.json',access)
  report(cfg,history,train_epochs,'COMPLETE',final);print('COMPLETE '+json.dumps(final),flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--readiness',action='store_true');parser.add_argument('--train',action='store_true')
 args=parser.parse_args()
 try:
  if args.readiness:readiness()
  elif args.train:train()
  else:parser.error('choose --readiness or --train')
 except Exception as e:
  failure={'status':'FAILED','error':repr(e),'traceback':traceback.format_exc(),'time':now()}
  write(ROOT/'FAILURE.json',failure);write(ROOT/'STATUS.json',failure);print('FAILED '+json.dumps(failure),flush=True)
  raise

