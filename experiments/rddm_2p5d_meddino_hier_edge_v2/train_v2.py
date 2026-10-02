"""Fresh matched V2 training; every stochastic training step checked against verified baseline."""
import bootstrap
import argparse,csv,gc,hashlib,inspect,json,math,os,sys,time,traceback
from pathlib import Path
import torch,numpy as np
from bootstrap import ROOT,BASE
from model_v2 import build_v2,MAP
from model import native_attention_grad,state_without_med,load_state_without_med
from context_data import ContextPairs,ContextBatchStream,ContextDataGuard
from fast_dev.protocol import CONFIG,check_config
from mayo2016_smallset_v1.runtime import seed_everything,rng_state,restore_rng,atomic_save
from evaluate import validate
from train import MatchedEMA,tensor_state_hash,rng_digest,finite_model
from med_loader import sha
METRICS=('psnr','ssim','rmse_hu','mae_hu')
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def save_csv(p,rows):
    if not rows:return
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def sources():
    paths=[ROOT/n for n in ('model_v2.py','train_v2.py','bootstrap.py','readiness_v2.py','CONFIG.json','BASELINE_PROVENANCE.json')]
    paths += [BASE/n for n in json.loads((ROOT/'BASELINE_PROVENANCE.json').read_text())['baseline_source_sha256']]
    import fast_dev.data,fast_dev.protocol,mayo2016_smallset_v1.runtime,mayo2016_smallset_v1.frozen_protocol
    paths += [Path(inspect.getfile(m)) for m in (fast_dev.data,fast_dev.protocol,mayo2016_smallset_v1.runtime,mayo2016_smallset_v1.frozen_protocol)]
    return {str(p):sha(p) for p in sorted(set(paths))}
def fixed_hash(u):return tensor_state_hash(u.fusion.edge.fixed.state_dict())
def audit_access(g):
    a=g.report();assert all(a[k]==0 for k in ('L506_opens','external_memory_opens','neighbor_target_opens','blocked_attempts'));return a
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=('sanity','formal'),required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    ready=json.loads((ROOT/'READINESS.json').read_text());visual=json.loads((ROOT/'VISUAL_REVIEW.json').read_text())
    assert ready['status']==visual['status']=='PASS'
    manifest=json.loads((ROOT/'READINESS_SOURCE_MANIFEST.json').read_text())
    for n,h in manifest.items():assert sha(n)==h,('readiness source changed',n)
    if args.phase=='formal':assert json.loads((ROOT/'SANITY_SUMMARY.json').read_text())['status']=='PASS'
    cfg=json.loads(CONFIG.read_text());check_config(cfg);v2=json.loads((ROOT/'CONFIG.json').read_text());bp=json.loads((ROOT/'BASELINE_PROVENANCE.json').read_text())
    assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8' and os.environ['CUDA_VISIBLE_DEVICES']=='0'
    for n,h in bp['baseline_source_sha256'].items():assert sha(BASE/n)==h
    assert sha(bp['parent_checkpoint'])==bp['parent_checkpoint_sha256']
    torch.set_num_threads(2);seed_everything(20260829);device=torch.device('cuda:0')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False);(output/'checkpoints').mkdir()
    hashes=sources();write(output/'SOURCE_MANIFEST.json',hashes);write(output/'EFFECTIVE_CONFIG.json',{'v2':v2,'matched_protocol':cfg,'phase':args.phase})
    limit=200 if args.phase=='sanity' else 8000;nodes=(200,) if args.phase=='sanity' else (0,2000,4000,6000,8000)
    expected=[json.loads(l) for l in (Path(bp['baseline_run'])/'train.jsonl').read_text().splitlines()]
    assert len(expected)==8000
    with ContextDataGuard(cfg) as guard:
        train=ContextPairs(cfg,'train');val=ContextPairs(cfg,'validation');train.bind_guard(guard);val.bind_guard(guard)
        stream=ContextBatchStream(train,20260829);initial_rng=rng_state()
        model=build_v2().to(device).train();u=model.model.unet0;params=[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.Adam(params,lr=2e-4,betas=(.9,.99),weight_decay=1e-9);ema=MatchedEMA(model);restore_rng(initial_rng)
        shared={n:t for n,t in state_without_med(model).items() if not any(x in n for x in ('.layer_adapters.','.fusion.','.conditions.'))}
        initial={'shared_initial_parameters_sha256':tensor_state_hash(shared),'initial_rng_sha256':rng_digest(),'train_manifest_sha256':train.manifest_hash,'validation_manifest_sha256':val.manifest_hash,'meddino_initial_weights_sha256':tensor_state_hash(u.med.state_dict()),'fresh_optimizer':True,'ema_updates':0,'batch':2,'accum':1,'trainable_parameters':sum(p.numel() for p in params),'total_parameters':sum(p.numel() for p in model.parameters())};del shared
        for k in ('shared_initial_parameters_sha256','initial_rng_sha256','train_manifest_sha256','validation_manifest_sha256','meddino_initial_weights_sha256'):assert initial[k]==bp['baseline_initial_state'][k],(k,initial[k])
        kernels_before=fixed_hash(u);write(output/'INITIAL_STATE.json',initial)
        med_before=initial['meddino_initial_weights_sha256'];history=[];step=0;stream_digest=hashlib.sha256().hexdigest();last={};strength=[];feature_stats=[];times=[]
        torch.cuda.reset_peak_memory_stats()
        def evaluate_step():
            before=rng_digest();row,activity=validate(ema.model,val,cfg,device,output/f'validation_step_{step}.json');assert before==rng_digest()
            assert row['slices']==64
            row={'step':step,**row,**{'delta_'+k+'_vs_baseline_8k':row[k]-bp['baseline_candidate_metrics'][k] for k in METRICS},'train_stream_sha256':stream_digest,'peak_vram_bytes':torch.cuda.max_memory_allocated()}
            history.append(row);save_csv(output/'validation_metrics.csv',history)
            write(output/f'condition_activity_step_{step}.json',activity)
            per_slice=json.loads((output/f'validation_step_{step}.json').read_text());save_csv(output/f'L333_fast64_step_{step}_metrics.csv',per_slice)
            write(output/'DATA_ACCESS_AUDIT.json',audit_access(guard));write(output/'MEDDINO_FROZEN_AUDIT.json',{'before':med_before,'after':tensor_state_hash(u.med.state_dict()),'unchanged':tensor_state_hash(u.med.state_dict())==med_before,'all_grad_none':all(p.grad is None and not p.requires_grad for p in u.med.parameters()),'eval_mode':not u.med.training})
            assert tensor_state_hash(u.med.state_dict())==med_before
            print('VALIDATION '+json.dumps(row),flush=True)
        if 0 in nodes:evaluate_step()
        with (output/'train.jsonl').open('x',buffering=1) as log:
            while step<limit:
                started=time.perf_counter();optimizer.zero_grad(set_to_none=True)
                stream_hash=hashlib.sha256(bytes.fromhex(stream_digest))
                prev,center,next_,target,meta=stream.next(2)
                for x in (prev,center,next_,target):stream_hash.update(x.numpy().tobytes())
                prev,center,next_,target=[x.to(device) for x in (prev,center,next_,target)]
                t=torch.randint(0,model.num_timesteps,(2,),device=device).long();noise=torch.randn_like(target)
                stream_hash.update(json.dumps(meta,sort_keys=True,separators=(',',':')).encode());stream_hash.update(t.detach().cpu().numpy().tobytes());stream_hash.update(noise.detach().cpu().numpy().tobytes());stream_digest=stream_hash.hexdigest()
                assert stream_digest==expected[step]['train_stream_sha256'],('matched stochastic stream mismatch',step+1)
                diagnostic=(step==0 or (step+1)%100==0);u.fusion.record=diagnostic
                for b in u.conditions.values():b.record=diagnostic;b.rows=[]
                u.set_context(prev,center,next_);loss=model.p_losses([2*target-1,2*center-1],t,noise=noise)[0];assert bool(torch.isfinite(loss))
                if diagnostic:
                    stats=u.fusion.latest_stats.copy();assert all(r['finite'] and r['max_abs']<1000 for r in stats.values())
                    feature_stats.append({'step':step+1,'features':stats});write(output/'FEATURE_STATISTICS.json',feature_stats)
                    for s,b in u.conditions.items():
                        assert len(b.rows)==1
                        strength.append({'step':step+1,'stage':s,**b.rows[0]})
                # Recomputed checkpoint functions must not duplicate diagnostic rows during backward.
                u.fusion.record=False
                for b in u.conditions.values():b.record=False;b.rows=[]
                loss.backward();u.clear_context();attn=native_attention_grad(model)
                assert all(p.grad is None and not p.requires_grad for p in u.med.parameters()) and not u.med.training
                assert all(b.grad is None and not b.requires_grad for b in u.fusion.edge.fixed.buffers())
                grad=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True));optimizer.step();ema.update(model);step+=1
                finite=None
                if diagnostic:
                    finite=finite_model(model);assert finite
                    save_csv(output/'train_condition_strength.csv',strength)
                torch.cuda.synchronize();sec=time.perf_counter()-started;times.append(sec)
                last={'optimizer_step':step,'train_loss':float(loss.detach()),'gradient_norm_before_clip':grad,'attention_gradient_norm':math.sqrt(sum(x*x for x in attn.values())),'parameters_finite':finite,'seconds_per_step':sec,'peak_vram_bytes':torch.cuda.max_memory_allocated(),'train_stream_sha256':stream_digest,'baseline_stream_match':True,'nonfinite_count':0,'samples':meta}
                log.write(json.dumps(last,allow_nan=False)+'\n')
                if diagnostic:print('TRAIN '+json.dumps({k:last[k] for k in ('optimizer_step','train_loss','seconds_per_step','peak_vram_bytes','gradient_norm_before_clip')}),flush=True)
                if step in nodes:
                    evaluate_step()
                    if args.phase=='formal':
                        atomic_save(output/'checkpoints'/f'step_{step}.pt',{'arm':'2P5D_MEDDINO_HIER_EDGE_V2_8K','optimizer_step':step,'ema_updates':ema.updates,'model_state_dict':state_without_med(model),'ema_state_dict':state_without_med(ema.model),'optimizer_state_dict':optimizer.state_dict(),'rng_state':rng_state(),'stream_state':stream.state_dict(),'history':history,'train_stream_sha256':stream_digest,'source_hashes':hashes,'diffusion':cfg['diffusion'],'parent_checkpoint_sha256':bp['parent_checkpoint_sha256'],'batch_config':{'batch':2,'accum':1,'effective_batch':2}})
        med_after=tensor_state_hash(u.med.state_dict());assert med_after==med_before
        assert fixed_hash(u)==kernels_before and finite_model(model)
        if args.phase=='formal':assert stream_digest==bp['baseline_final_stream_sha256']
        summary={'status':'PASS','phase':args.phase,'optimizer_step':step,'ema_updates':ema.updates,'all_steps_stochastic_stream_match':True,'train_stream_sha256':stream_digest,'initial':initial,'meddino_hash_before':med_before,'meddino_hash_after':med_after,'meddino_grad_none':all(p.grad is None for p in u.med.parameters()),'fixed_kernel_hash_before':kernels_before,'fixed_kernel_hash_after':fixed_hash(u),'nonfinite_count':0,'parameters_finite':True,'peak_vram_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'seconds_per_step_mean':float(np.mean(times)),'seconds_per_step_median':float(np.median(times)),'validation':history,'data_access':audit_access(guard),'source_hashes':hashes}
        write(output/'FINAL_STATE.json',summary);write(output/'DATA_ACCESS_AUDIT.json',summary['data_access']);write(output/'RANDOM_STREAM_AUDIT.json',{'status':'PASS','matched_steps':step,'all_steps_match':True,'baseline_log':str(Path(bp['baseline_run'])/'train.jsonl'),'final_digest':stream_digest})
        if args.phase=='sanity':write(ROOT/'SANITY_SUMMARY.json',summary)
        else:write(ROOT/'TRAINING_SUMMARY.json',summary)
        print('COMPLETE '+args.phase,flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        write(ROOT/('TRAIN_FAILURE_'+time.strftime('%Y%m%d_%H%M%S')+'.json'),{'status':'FAIL','traceback':traceback.format_exc()});raise
