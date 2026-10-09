"""Read-only MedDINO feature and V1 condition audit; no parameter updates."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,sys,time,traceback
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch.nn import functional as F
from scipy.ndimage import gaussian_filter,convolve
OUT=Path(__file__).resolve().parent
SRC=OUT.parent/"rddm_official_2p5d_meddino_condition_v1"
sys.path.insert(0,str(SRC))
from model import build_model,load_state_without_med,STAGES,GLOBAL_STAGES
from med_loader import MED_CK,MED_SHA,sha as med_sha
from provenance import ROOT
from fast_dev.protocol import CONFIG,check_config,validation_indices
from context_data import ContextPairs,ContextDataGuard
from mayo2016_smallset_v1.frozen_protocol import slice_metrics,normalized_to_hu,normalize_stored_ct
from mayo2016_smallset_v1.runtime import seed_everything,METRICS
from evaluate import validate
NAMES=("C0_NORMAL","C1_OFF","C2_NEAR_SWAP","C3_FAR_SWAP","C4_SHIFT2_ALL","C5_MEAN_ONLY","C6_E2_SHIFT2")
LEVELS=("L2","L5","L8","L11","FUSED")
REGIONS=("all","foreground","soft")
EXPECTED={"psnr":40.908237245382026,"ssim":0.9488474637550007,"rmse_hu":18.11923421427757,"mae_hu":11.986736714839935}
def digest(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for chunk in iter(lambda:f.read(1048576),b""):h.update(chunk)
 return h.hexdigest()
def jwrite(name,obj):
 p=OUT/name;tmp=p.with_name(p.name+".tmp")
 tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
 os.replace(tmp,p)
def csvwrite(name,rows):
 if not rows:raise RuntimeError("Empty CSV "+name)
 p=OUT/name;tmp=p.with_name(p.name+".tmp")
 with tmp.open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n")
  w.writeheader();w.writerows(rows)
 os.replace(tmp,p)
def statehash(model,category):
 h=hashlib.sha256()
 for key,t in sorted(model.state_dict().items()):
  is_med=".med." in key
  is_adapter=".layer_adapters." in key or ".fusion." in key
  is_film=".conditions." in key
  selected={"all":True,"MedDINO":is_med,"AdapterFusion":is_adapter,"FiLM":is_film,"RDDM":not(is_med or is_adapter or is_film)}[category]
  if selected:
   x=t.detach().cpu().contiguous()
   h.update(key.encode());h.update(str(x.dtype).encode());h.update(str(tuple(x.shape)).encode())
   h.update(x.view(torch.uint8).numpy().tobytes())
 return h.hexdigest()
def preflight():
 cfg=json.loads(CONFIG.read_text());check_config(cfg)
 assert ROOT==SRC and cfg["evaluation"]["validation_mode"]=="FAST"
 assert cfg["evaluation"]["hu_window"]==[-1000.0,1000.0]
 assert cfg["diffusion"]=={"objective":"pred_res","loss_type":"l2","timesteps":1000,"sampling_timesteps":2,"condition":True,"sum_scale":0.01}
 indices=tuple(validation_indices("FAST"))
 assert len(indices)==64 and indices==tuple(map(int,(SRC/"splits/L333_fast64.txt").read_text().splitlines()))
 runpaths=json.loads((SRC/"RUN_PATHS.json").read_text())
 run=Path(runpaths["OFFICIAL_RDDM_2P5D_MEDDINO_CONDITION"]).resolve()
 assert run==SRC/"runs/OFFICIAL_RDDM_2P5D_MEDDINO_CONDITION_retry1"
 checkpoint=run/"checkpoints/step_8000.pt"
 assert checkpoint.is_file() and digest(checkpoint)=="1ea09e9d0a4f2d56a636a62dc4a02afe80ff6342f506628077dd0003af748ce1"
 assert med_sha(MED_CK)==MED_SHA
 assert json.loads((SRC/"READINESS.json").read_text())["status"]=="PASS"
 original_access=json.loads((SRC/"DATA_ACCESS_AUDIT.json").read_text())
 assert original_access["arms"]["OFFICIAL_RDDM_2P5D_MEDDINO_CONDITION"]["L506_opens"]==0
 old=json.loads((SRC/"DIAGNOSTIC_NORMAL.json").read_text())["metrics"]
 for k,v in EXPECTED.items():assert abs(old[k]-v)<1e-9
 pins=json.loads((SRC/"SOURCE_PROVENANCE.json").read_text())["new_source_hashes"]
 for name in ("model.py","med_loader.py","evaluate.py","context_data.py","src/official_rddm_ct.py"):
  assert name in pins and digest(SRC/name)==pins[name],name+" source changed"
 seed_everything(20260829);torch.set_num_threads(2)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 assert torch.cuda.is_available() and torch.cuda.get_device_name(0)=="NVIDIA GeForce RTX 3090"
 model=build_model(anatomy=True)
 ck=torch.load(checkpoint,map_location="cpu",weights_only=False)
 assert ck["optimizer_step"]==ck["ema_updates"]==8000
 load_state_without_med(model,ck["ema_state_dict"])
 del ck
 model.cuda().eval().requires_grad_(False)
 u=model.model.unet0
 assert u.anatomy and len(u.conditions)==9 and set(u.conditions)==set(STAGES)
 assert set(s for s,b in u.conditions.items() if b.global_mode)==GLOBAL_STAGES
 assert all(not p.requires_grad and p.grad is None for p in model.parameters())
 assert model.objective=="pred_res" and model.sampling_timesteps==2 and model.sum_scale==0.01
 hashes={c:statehash(model,c) for c in ("all","MedDINO","AdapterFusion","FiLM","RDDM")}
 info={"status":"PASS","checkpoint":str(checkpoint),"checkpoint_sha256":digest(checkpoint),
       "med_checkpoint_sha256":MED_SHA,"source_hashes":{n:digest(SRC/n) for n in ("model.py","med_loader.py","evaluate.py","context_data.py","src/official_rddm_ct.py")},
       "fast64_manifest_sha256":digest(SRC/"splits/L333_fast64.txt"),"fast64_indices":list(indices),
       "before_parameter_hashes":hashes,"python":sys.version,"torch":torch.__version__,"cuda":torch.version.cuda,
       "environment":sys.executable,"git_commit":"1c258e6706a1d0fe23c8319a3ffad18738594278",
       "GPU":torch.cuda.get_device_name(0),"AMP":False,"TF32":False,"seed":20260829,
       "command":"CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 python -u run_audit.py --mode all",
       "read_only":True,"no_optimizer_backward_or_ema":True}
 return cfg,indices,model,info
def masks(center):
 hu=(center.detach().float()*4096-1024).clamp(-1000,1000)
 fg=hu>-500
 soft=fg & (hu>=-160) & (hu<=240)
 fg32=F.avg_pool2d(fg.float(),16,16).flatten()>=0.5
 soft32=F.avg_pool2d(soft.float(),16,16).flatten()>=0.5
 return {"all":torch.ones_like(fg32,dtype=torch.bool),"foreground":fg32,"soft":soft32},hu.squeeze().cpu().numpy(),fg.squeeze().cpu().numpy(),soft.squeeze().cpu().numpy()
def features_for(center,target,u):
 ld=u.extract(center);nd=u.extract(target)
 assert len(ld)==len(nd)==4
 assert all(t.shape==(1,768,32,32) and torch.isfinite(t).all() for t in (*ld,*nd))
 al=u.fusion(torch.cat([m(p) for m,p in zip(u.layer_adapters,ld)],1))
 an=u.fusion(torch.cat([m(p) for m,p in zip(u.layer_adapters,nd)],1))
 assert al.shape==an.shape==(1,128,32,32) and torch.isfinite(al).all() and torch.isfinite(an).all()
 return list(ld)+[al],list(nd)+[an]
def region_stats(fl,fn,mask):
 if not bool(mask.any()):return (None,None,0)
 x=fl.float().reshape(fl.shape[1],-1)[:,mask];y=fn.float().reshape(fn.shape[1],-1)[:,mask]
 cos=F.cosine_similarity(x,y,dim=0,eps=1e-8)
 relative=float((x-y).abs().mean()/y.abs().mean().clamp_min(1e-8))
 return float(cos.mean()),relative,int(mask.sum())
COORD=torch.arange(1024,device="cuda")
DY=(COORD[:,None]//32-COORD[None,:]//32).abs()
DX=(COORD[:,None]%32-COORD[None,:]%32).abs()
CHEB=torch.maximum(DY,DX)
FAR=(DY.square()+DX.square())>=64
def matching(fl,fn,valid,query):
 x=F.normalize(fl.float().flatten(2)[0].T,dim=1,eps=1e-8)
 y=F.normalize(fn.float().flatten(2)[0].T,dim=1,eps=1e-8)
 sim=x@y.T
 qi=query.nonzero().flatten()
 if not qi.numel():return {"queries":0,"hard_valid":0,"far_valid":0,"positive_mean":None,"hardest_negative_mean":None,"far_negative_mean":None,"margin_hard_mean":None,"margin_far_mean":None,"margin_hard_median":None,"margin_far_median":None,"top1":None,"top5":None}
 s=sim[qi]
 pos=s.gather(1,qi[:,None]).flatten()
 hardmask=valid[None,:] & (CHEB[qi]>=1)&(CHEB[qi]<=2)
 farmask=valid[None,:] & FAR[qi]
 hard=s.masked_fill(~hardmask,-torch.inf).max(1).values
 far=s.masked_fill(~farmask,-torch.inf).max(1).values
 hv=torch.isfinite(hard);fv=torch.isfinite(far)
 top=s.masked_fill(~valid[None,:],-torch.inf).topk(min(5,int(valid.sum())),dim=1).indices
 def mean(t):return float(t.mean()) if t.numel() else None
 return {"queries":int(qi.numel()),"hard_valid":int(hv.sum()),"far_valid":int(fv.sum()),
         "positive_mean":mean(pos),"hardest_negative_mean":mean(hard[hv]),"far_negative_mean":mean(far[fv]),
         "margin_hard_mean":mean((pos-hard)[hv]),"margin_far_mean":mean((pos-far)[fv]),
         "margin_hard_median":float((pos-hard)[hv].median()) if hv.any() else None,
         "margin_far_median":float((pos-far)[fv].median()) if fv.any() else None,
         "top1":float((top[:,0]==qi).float().mean()),"top5":float((top==qi[:,None]).any(1).float().mean())}
def feature_item(item,u):
 prev,center,nxt,target,meta=item
 c=torch.from_numpy(center)[None].cuda();n=torch.from_numpy(target)[None].cuda()
 r,hu,fg,soft=masks(c)
 fl,fn=features_for(c,n,u)
 cross=[];match=[]
 for level,x,y in zip(LEVELS,fl,fn):
  for region in REGIONS:
   cos,relative,tokens=region_stats(x,y,r[region])
   cross.append({"slice_index":meta["slice_index"],"level":level,"region":region,"tokens":tokens,"cosine":cos,"relative_l1":relative})
  for region in ("foreground","soft"):
   m=matching(x,y,r["foreground"],r[region])
   match.append({"slice_index":meta["slice_index"],"level":level,"query_region":region,**m})
 return cross,match
def shift2(a):
 assert a.shape[-2:]==(32,32)
 return F.pad(a,(2,0,2,0),mode="replicate")[...,:32,:32]
def donor_for(mode,pos,indices):
 if mode=="C2_NEAR_SWAP":return indices[pos^1]
 if mode=="C3_FAR_SWAP":
  here=indices[pos]
  return min(indices,key=lambda k:(-abs(k-here),k))
 return None
def center_only(cfg,guard,index):
 root=Path(cfg["data"]["root"])
 guard.begin_sample("L333",index)
 try:
  raw=np.load(root/("L333_25_%03d_img.npy"%index),allow_pickle=False)
 finally:guard.end_sample()
 assert raw.shape==(512,512) and np.isfinite(raw).all()
 norm=normalize_stored_ct(raw,cfg["data"])
 return torch.from_numpy(norm[None,None]).cuda()
SCHARR_X=np.array([[-3,0,3],[-10,0,10],[-3,0,3]],dtype=np.float32)
SCHARR_Y=SCHARR_X.T
def pixel_regions(center_hu,fg,soft):
 sm=gaussian_filter(center_hu,sigma=.8,mode="reflect")
 g=np.hypot(convolve(sm,SCHARR_X,mode="reflect"),convolve(sm,SCHARR_Y,mode="reflect"))
 edge=fg & (g>=np.quantile(g[fg],.9))
 assert all(m.any() for m in (fg,soft,edge))
 return {"foreground":fg,"soft":soft,"high_gradient":edge}
def region_errors(pred,target,cfg,region):
 lo,hi=cfg["evaluation"]["hu_window"]
 p=normalized_to_hu(pred.detach().float().cpu(),cfg["data"]).clamp(lo,hi).squeeze().numpy()
 y=normalized_to_hu(target.detach().float().cpu(),cfg["data"]).clamp(lo,hi).squeeze().numpy()
 e=p-y
 out={}
 for k,m in region.items():
  out[k+"_rmse_hu"]=float(np.sqrt(np.mean(e[m]**2)))
  out[k+"_mae_hu"]=float(np.mean(np.abs(e[m])))
 return out
@torch.no_grad()
def infer_one(mode,pos,item,indices,cfg,model,guard):
 prev,center,nxt,target,meta=item
 prev=torch.from_numpy(prev)[None].cuda();center=torch.from_numpy(center)[None].cuda()
 nxt=torch.from_numpy(nxt)[None].cuda();target=torch.from_numpy(target)[None].cuda()
 u=model.model.unet0
 donor=donor_for(mode,pos,indices)
 supplied=None
 if donor is not None:supplied=u.extract(center_only(cfg,guard,donor))
 original_mod=u.modulate
 changed=defaultdict(int);calls=defaultdict(int)
 def hook(stage,x,a):
  chosen=a
  if mode=="C4_SHIFT2_ALL" or (mode=="C6_E2_SHIFT2" and stage=="E2"):chosen=shift2(a)
  elif mode=="C5_MEAN_ONLY":chosen=a.mean((-2,-1),keepdim=True).expand_as(a)
  calls[stage]+=1;changed[stage]+=int(not torch.equal(chosen,a))
  return original_mod(stage,x,chosen)
 try:
  u.modulate=hook
  for b in u.conditions.values():b.record=True;b.rows=[];b.force_off=mode=="C1_OFF"
  u.set_context(prev,center,nxt,supplied)
  s=20260829+meta["slice_index"]
  torch.manual_seed(s);torch.cuda.manual_seed(s)
  pred=model.sample([center],last=True)[-1]
  assert pred.shape==target.shape==(1,1,512,512) and torch.isfinite(pred).all()
  metrics=slice_metrics(pred,target,cfg)
  hu,fg,soft=None,None,None
  _,hu,fg,soft=masks(center)
  reg=pixel_regions(hu,fg,soft)
  local=region_errors(pred,target,cfg,reg)
  stage={}
  for name,b in u.conditions.items():
   assert len(b.rows)==2,(mode,name,len(b.rows))
   stage[name]=float(np.mean([r["delta_base_ratio"] for r in b.rows]))
  assert set(calls)==set(STAGES) and all(v==2 for v in calls.values())
  if mode=="C6_E2_SHIFT2":assert changed["E2"]==2 and all(changed[s]==0 for s in STAGES if s!="E2")
  if mode=="C4_SHIFT2_ALL":assert all(changed[s]==2 for s in STAGES)
  if mode in ("C0_NORMAL","C1_OFF","C2_NEAR_SWAP","C3_FAR_SWAP"):assert all(changed[s]==0 for s in STAGES)
  return {**meta,"mode":mode,"donor_index":donor,"sampling_seed":s,**{k:metrics[k] for k in METRICS},**local,
          **{"stage_"+k+"_delta_base_ratio":v for k,v in stage.items()},"condition_changed_stages":",".join(s for s in STAGES if changed[s])},hashlib.sha256(pred.detach().cpu().numpy().tobytes()).hexdigest()
 finally:
  u.clear_context()
  for b in u.conditions.values():b.record=False;b.force_off=False;b.rows=[]
  del u.modulate
def smoke(cfg,indices,model,data,guard):
 u=model.model.unet0;cross=[];match=[];inter=[]
 for pos in range(2):
  item=data.load(pos);a,b=feature_item(item,u);cross+=a;match+=b
  for mode in NAMES:
   row,ph=infer_one(mode,pos,item,indices,cfg,model,guard)
   if pos==0:
    repeated,rh=infer_one(mode,pos,item,indices,cfg,model,guard)
    assert ph==rh and all(row[k]==repeated[k] for k in METRICS)
   inter.append({"slice_index":row["slice_index"],"mode":mode,"psnr":row["psnr"],"prediction_sha256":ph})
 item=data.load(0)
 one,h1=infer_one("C0_NORMAL",0,item,indices,cfg,model,guard)
 two,h2=infer_one("C0_NORMAL",0,item,indices,cfg,model,guard)
 assert h1==h2 and all(one[k]==two[k] for k in METRICS)
 assert all(math.isfinite(r["psnr"]) for r in inter)
 result={"status":"PASS","slice_indices":list(indices[:2]),"feature_rows":len(cross),"matching_rows":len(match),
         "interventions":[r for r in inter],"same_mode_repeat_identical":True,"C6_E2_only_assertion":True}
 jwrite("SMOKE_RESULT.json",result)
 return result
def baseline(cfg,indices,model,data):
 summary,activity=validate(model,data,cfg,torch.device("cuda:0"),OUT/"BASELINE_PER_SLICE.json","NORMAL")
 assert summary["slices"]==64
 assert abs(summary["psnr"]-EXPECTED["psnr"])<=1e-4 and abs(summary["ssim"]-EXPECTED["ssim"])<=1e-5
 old=json.loads((SRC/"DIAGNOSTIC_NORMAL_PER_SLICE.json").read_text())
 now=json.loads((OUT/"BASELINE_PER_SLICE.json").read_text())
 assert [r["slice_index"] for r in now]==list(indices)==[r["slice_index"] for r in old]
 assert all(r["sampling_seed"]==20260829+r["slice_index"] for r in now)
 jwrite("BASELINE_RESULT.json",{"status":"PASS","metrics":summary,"delta_from_historical":{k:summary[k]-EXPECTED[k] for k in EXPECTED},"activity":activity})
 return now
def features(cfg,indices,model,data):
 rows=[];matches=[]
 for pos in range(64):
  item=data.load(pos);a,b=feature_item(item,model.model.unet0);rows+=a;matches+=b
  if pos%8==7:print("FEATURES",pos+1,flush=True)
 assert len(rows)==64*5*3 and len(matches)==64*5*2
 csvwrite("FEATURE_CROSSDOSE.csv",rows);csvwrite("SPATIAL_MATCHING.csv",matches)
 return rows,matches
def bootstrap(deltas,idx):
 samples=deltas[idx].mean(axis=1)
 return [float(np.quantile(samples,.025)),float(np.quantile(samples,.975))]
def interventions(cfg,indices,model,data,guard,baseline_rows):
 records=[]
 for pos in range(64):
  item=data.load(pos)
  for mode in NAMES:
   r,_=infer_one(mode,pos,item,indices,cfg,model,guard);records.append(r)
  if pos%8==7:print("INTERVENTIONS",pos+1,flush=True)
 bymode={m:[r for r in records if r["mode"]==m] for m in NAMES}
 assert all(len(v)==64 for v in bymode.values())
 assert [r["slice_index"] for r in bymode["C0_NORMAL"]]==list(indices)
 for a,b in zip(bymode["C0_NORMAL"],baseline_rows):
  assert a["slice_index"]==b["slice_index"]
  assert abs(a["psnr"]-b["psnr"])<1e-8 and abs(a["ssim"]-b["ssim"])<1e-8
 base=bymode["C0_NORMAL"]
 random=np.random.default_rng(20260829)
 idx=random.integers(0,64,size=(10000,64))
 per=[];aggregate=[]
 metric_names=list(METRICS)+["foreground_rmse_hu","foreground_mae_hu","soft_rmse_hu","soft_mae_hu","high_gradient_rmse_hu","high_gradient_mae_hu"]
 for mode in NAMES:
  group=bymode[mode]
  for r,b in zip(group,base):
   row={"mode":mode,"slice_index":r["slice_index"],"donor_index":r["donor_index"],"sampling_seed":r["sampling_seed"],"condition_changed_stages":r["condition_changed_stages"]}
   for k in metric_names:row[k]=r[k];row["delta_"+k]=r[k]-b[k]
   for k in STAGES:row["stage_"+k+"_delta_base_ratio"]=r["stage_"+k+"_delta_base_ratio"]
   per.append(row)
  summary={"mode":mode,"slices":64,"donor_rule":"near fixed pair" if mode=="C2_NEAR_SWAP" else "farthest fixed Fast64" if mode=="C3_FAR_SWAP" else "none"}
  for k in metric_names:
   values=np.array([r[k] for r in group],dtype=float)
   delta=np.array([r[k]-b[k] for r,b in zip(group,base)],dtype=float)
   ci=bootstrap(delta,idx)
   summary[k]=float(values.mean())
   summary["delta_"+k]=float(delta.mean())
   summary["delta_"+k+"_ci_low"]=ci[0];summary["delta_"+k+"_ci_high"]=ci[1]
   summary[k+"_improved_fraction"]=float(np.mean(delta>0 if k in ("psnr","ssim") else delta<0))
  for k in STAGES:summary["stage_"+k+"_delta_base_ratio"]=float(np.mean([r["stage_"+k+"_delta_base_ratio"] for r in group]))
  aggregate.append(summary)
 csvwrite("PER_SLICE_INTERVENTIONS.csv",per);csvwrite("CONDITION_INTERVENTIONS.csv",aggregate)
 return per,aggregate
def groupstats(rows,keys,metrics):
 groups=defaultdict(list)
 for r in rows:groups[tuple(r[k] for k in keys)].append(r)
 result=[]
 for key,items in sorted(groups.items()):
  v={k:x for k,x in zip(keys,key)};v["slices"]=len(items)
  for metric in metrics:
   vals=np.array([float(i[metric]) for i in items if i[metric] is not None])
   v[metric+"_mean"]=float(np.mean(vals)) if vals.size else None
   v[metric+"_median"]=float(np.median(vals)) if vals.size else None
  result.append(v)
 return result
def finish(model,info,guard,feature_rows,matching_rows,intervention_rows,mode):
 assert all(p.grad is None and not p.requires_grad for p in model.parameters())
 after={c:statehash(model,c) for c in info["before_parameter_hashes"]}
 assert after==info["before_parameter_hashes"]
 access=guard.report()
 assert access["L506_opens"]==access["external_memory_opens"]==access["neighbor_target_opens"]==access["training_target_opens"]==access["blocked_attempts"]==0
 assert access["validation_center_indices_opened"]==(info["fast64_indices"][:2] if mode=="smoke" else info["fast64_indices"])
 jwrite("DATA_ACCESS_AUDIT.json",access)
 info["after_parameter_hashes"]=after;info["all_parameter_hashes_equal"]=True
 info["actual_access"]=access;info["status"]="PASS"
 jwrite("AUDIT_MANIFEST.json",info)
 if feature_rows is not None:
  jwrite("FEATURE_SUMMARY.json",groupstats(feature_rows,("level","region"),("cosine","relative_l1")))
 if matching_rows is not None:
  jwrite("MATCHING_SUMMARY.json",groupstats(matching_rows,("level","query_region"),("positive_mean","hardest_negative_mean","far_negative_mean","margin_hard_mean","margin_far_mean","top1","top5")))
 if intervention_rows is not None:
  jwrite("INTERVENTION_SUMMARY.json",intervention_rows)
def main():
 parser=argparse.ArgumentParser();parser.add_argument("--mode",choices=("smoke","features","interventions","all"),required=True)
 args=parser.parse_args();cfg,indices,model,info=preflight()
 print("PREFLIGHT_PASS",flush=True)
 feature_rows=matching_rows=intervention_rows=None
 with ContextDataGuard(cfg,"FAST",allow_training=False) as guard:
  data=ContextPairs(cfg,"validation","FAST");data.bind_guard(guard)
  assert tuple((r[0],r[1]) for r in data.records)==tuple(("L333",i) for i in indices)
  if args.mode in ("smoke","all"):
   smoke(cfg,indices,model,data,guard);print("SMOKE_PASS",flush=True)
  if args.mode=="all":
   base=baseline(cfg,indices,model,data);print("BASELINE_PASS",flush=True)
   feature_rows,matching_rows=features(cfg,indices,model,data);print("FEATURES_PASS",flush=True)
   _,intervention_rows=interventions(cfg,indices,model,data,guard,base);print("INTERVENTIONS_PASS",flush=True)
  elif args.mode=="features":
   assert (OUT/"BASELINE_RESULT.json").exists()
   feature_rows,matching_rows=features(cfg,indices,model,data)
  elif args.mode=="interventions":
   assert (OUT/"BASELINE_RESULT.json").exists()
   base=json.loads((OUT/"BASELINE_PER_SLICE.json").read_text())
   _,intervention_rows=interventions(cfg,indices,model,data,guard,base)
  finish(model,info,guard,feature_rows,matching_rows,intervention_rows,args.mode)
 print("AUDIT_COMPLETE",args.mode,flush=True)
if __name__=="__main__":
 try:main()
 except Exception:
  jwrite("AUDIT_FAILURE.json",{"status":"FAIL","traceback":traceback.format_exc()})
  raise
