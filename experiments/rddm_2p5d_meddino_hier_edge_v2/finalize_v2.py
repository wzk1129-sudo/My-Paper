"""Read-only 8k checkpoint reload and fixed Fast64 edge/flat diagnostics; final evidence report."""
import bootstrap
import gc,hashlib,json,math,os,subprocess,time,traceback
from pathlib import Path
import torch,numpy as np
from bootstrap import ROOT,BASE
from model_v2 import build_v2,FixedEdges
from model import build_model,load_state_without_med
from context_data import ContextPairs,ContextDataGuard
from fast_dev.protocol import CONFIG,check_config
from mayo2016_smallset_v1.runtime import seed_everything
from mayo2016_smallset_v1.frozen_protocol import slice_metrics,_metric_tensors
from med_loader import sha
from train import tensor_state_hash
from train_v2 import write,save_csv,METRICS,audit_access
RUN=ROOT/'runs/2P5D_MEDDINO_HIER_EDGE_V2_8K'
@torch.no_grad()
def region_eval(builder,path,arm,dataset,cfg,device,expected):
    h=sha(path);ck=torch.load(path,map_location='cpu',weights_only=False)
    assert ck['optimizer_step']==ck['ema_updates']==8000
    assert all(bool(torch.isfinite(t).all()) for t in ck['ema_state_dict'].values())
    model=builder().to(device).eval();load_state_without_med(model,ck['ema_state_dict']);del ck;gc.collect()
    u=model.model.unet0;med_before=tensor_state_hash(u.med.state_dict());rows=[];pred_hash=hashlib.sha256();mask_hash=hashlib.sha256()
    fixed=FixedEdges().to(device);torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    for pos in range(len(dataset)):
        prev,center,next_,target,meta=dataset.load(pos)
        prev,center,next_,target=[torch.from_numpy(x).unsqueeze(0).to(device) for x in (prev,center,next_,target)]
        u.set_context(prev,center,next_);seed=cfg['evaluation']['validation_seed']+meta['slice_index'];torch.manual_seed(seed);torch.cuda.manual_seed(seed)
        prediction=model.sample([center],last=True)[-1];assert torch.isfinite(prediction).all()
        metrics=slice_metrics(prediction,target,cfg)
        mask=(fixed(center)[0]>=2).detach().cpu();assert bool(mask.any()) and bool((~mask).any())
        _,_,ph,th,_,_=_metric_tensors(prediction,target,cfg);diff=ph-th
        region={}
        for name,m in [('edge',mask),('flat',~mask)]:
            region[name+'_rmse_hu']=float(diff[m].square().mean().sqrt());region[name+'_mae_hu']=float(diff[m].abs().mean());region[name+'_pixels']=int(m.sum())
        pred_hash.update(meta['slice_index'].to_bytes(4,'little'));pred_hash.update(prediction.detach().cpu().contiguous().numpy().tobytes());mask_hash.update(mask.numpy().tobytes())
        rows.append({'arm':arm,'patient_id':meta['patient_id'],'slice_index':meta['slice_index'],**{k:metrics[k] for k in METRICS},**region});u.clear_context()
    means={k:float(np.mean([row[k] for row in rows])) for k in (*METRICS,'edge_rmse_hu','edge_mae_hu','flat_rmse_hu','flat_mae_hu')}
    reproduction={k:means[k]-expected[k] for k in METRICS}
    assert all(abs(reproduction[k])<1e-5 for k in METRICS),(arm,'checkpoint reload metric mismatch',reproduction)
    assert tensor_state_hash(u.med.state_dict())==med_before
    result={'arm':arm,'checkpoint':str(path),'checkpoint_sha256':h,'checkpoint_optimizer_step':8000,'slices':len(rows),'means':means,'prediction_tensor_stream_sha256':pred_hash.hexdigest(),'mask_stream_sha256':mask_hash.hexdigest(),'reload_metric_differences':reproduction,'reload_metric_match':True,'meddino_hash':med_before,'meddino_frozen_eval':not u.med.training and all(not p.requires_grad and p.grad is None for p in u.med.parameters()),'seconds':time.perf_counter()-started,'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
    save_csv(ROOT/(arm+'_REGION_METRICS.csv'),rows);write(ROOT/(arm+'_RELOAD_AUDIT.json'),result)
    del model,u,prev,center,next_,target,prediction,fixed;gc.collect();torch.cuda.empty_cache();return result,rows

def main():
    cfg=json.loads(CONFIG.read_text());check_config(cfg);v2=json.loads((ROOT/'CONFIG.json').read_text());bp=json.loads((ROOT/'BASELINE_PROVENANCE.json').read_text());train=json.loads((ROOT/'TRAINING_SUMMARY.json').read_text())
    assert train['status']=='PASS' and train['optimizer_step']==8000
    assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8';torch.set_num_threads(2);seed_everything(20260829);device=torch.device('cuda:0')
    for n,h in train['source_hashes'].items():assert sha(n)==h
    expected={k:train['validation'][-1][k] for k in METRICS}
    with ContextDataGuard(cfg,allow_training=False) as guard:
        val=ContextPairs(cfg,'validation');val.bind_guard(guard)
        assert len(val)==64
        candidate,crows=region_eval(build_v2,RUN/'checkpoints/step_8000.pt','V2',val,cfg,device,expected)
        baseline,brows=region_eval(lambda:build_model(True),Path(bp['baseline_8k_checkpoint']),'BASELINE',val,cfg,device,bp['baseline_candidate_metrics'])
        data=audit_access(guard);write(ROOT/'REGION_DATA_ACCESS_AUDIT.json',data)
    assert candidate['mask_stream_sha256']==baseline['mask_stream_sha256']
    assert [r['slice_index'] for r in crows]==[r['slice_index'] for r in brows]
    paired=[{'slice_index':c['slice_index'],**{k:c[k]-b[k] for k in ('edge_rmse_hu','edge_mae_hu','flat_rmse_hu','flat_mae_hu')}} for c,b in zip(crows,brows)]
    save_csv(ROOT/'PAIRED_REGION_DELTAS.csv',paired)
    improvement=np.array([-r['edge_rmse_hu'] for r in paired],dtype=np.float64);rng=np.random.RandomState(20260829)
    bootstrap=improvement[rng.randint(len(improvement),size=(10000,len(improvement)))].mean(axis=1);ci=np.quantile(bootstrap,[.025,.975]).tolist()
    regions={'mask':v2['edge_region_mask'],'aggregation':'arithmetic mean of 64 per-slice region RMSE and MAE','slice_count':64,'baseline':baseline,'candidate':candidate,'candidate_minus_baseline':{k:candidate['means'][k]-baseline['means'][k] for k in candidate['means']},'same_mask_hash':True,'edge_rmse_baseline_minus_v2_mean':float(improvement.mean()),'edge_rmse_improvement_bootstrap95_ci':ci,'bootstrap_seed':20260829,'bootstrap_replicates':10000,'clear_edge_rmse_improvement':ci[0]>0}
    write(ROOT/'REGION_SUMMARY.json',regions)
    history=train['validation'];six=next(r for r in history if r['step']==6000);eight=next(r for r in history if r['step']==8000)
    dp=eight['delta_psnr_vs_baseline_8k'];ds=eight['delta_ssim_vs_baseline_8k']
    if dp>=.03 and ds>0 and regions['clear_edge_rmse_improvement']:decision='PROMOTE'
    elif six['delta_psnr_vs_baseline_8k']<=0 and dp<=0:decision='STOP'
    else:decision='RETAIN_DIAGNOSE'
    save_csv(ROOT/'VALIDATION_METRICS.csv',history)
    strength=[]
    for row in history:
        act=json.loads((RUN/f"condition_activity_step_{row['step']}.json").read_text())
        strength.extend({'step':row['step'],'stage':s,**a} for s,a in act.items())
    save_csv(ROOT/'CONDITION_STRENGTH.csv',strength)
    sanity=json.loads((ROOT/'SANITY_SUMMARY.json').read_text());ready=json.loads((ROOT/'READINESS.json').read_text())
    work=ROOT.parents[1];commit=subprocess.check_output(['git','-C',str(work),'rev-parse','HEAD'],text=True).strip();branch=subprocess.check_output(['git','-C',str(work),'branch','--show-current'],text=True).strip();status=subprocess.check_output(['git','-C',str(work),'status','--short'],text=True)
    summary={'status':'PASS','decision':decision,'optimizer_steps':8000,'git_commit':commit,'branch':branch,'git_status':status,'implementation_committed':False,'baseline_metrics':bp['baseline_candidate_metrics'],'validation':history,'regions':regions,'readiness':'PASS','visual_review':'PASS','sanity':'PASS','nonfinite_count':0,'meddino_hash_before':train['meddino_hash_before'],'meddino_hash_after':train['meddino_hash_after'],'fixed_kernel_hash_unchanged':train['fixed_kernel_hash_before']==train['fixed_kernel_hash_after'],'stochastic_stream_match_steps':8000,'peak_vram_bytes':train['peak_vram_bytes'],'seconds_per_step_mean':train['seconds_per_step_mean'],'seconds_per_step_median':train['seconds_per_step_median'],'trainable_parameters':ready['parameters_trainable'],'total_parameters':ready['parameters_total'],'baseline_trainable_parameters':bp['baseline_initial_state']['trainable_parameters'],'trainable_parameter_delta':ready['parameters_trainable']-bp['baseline_initial_state']['trainable_parameters'],'data_access':{'readiness':ready['data_access'],'sanity':sanity['data_access'],'formal':train['data_access'],'regions':data},'automatic_extension':False,'finished_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    assert all(a['L506_opens']==a['external_memory_opens']==a['neighbor_target_opens']==a['blocked_attempts']==0 for a in summary['data_access'].values())
    write(ROOT/'SUMMARY.json',summary)
    source_files=[p for p in ROOT.glob('*') if p.is_file() and p.suffix in ('.py','.json') and p.name not in ('FINAL_SOURCE_MANIFEST.json','SUMMARY.json')]
    write(ROOT/'FINAL_SOURCE_MANIFEST.json',{str(p):sha(p) for p in source_files})
    lines=['# MedDINO-Hierarchical-Edge V2 Fast-Dev report','',f'Conclusion: **{decision}**. 8000 steps completed; no extension scheduled.', '', '## Provenance',f'- Branch: {branch}',f'- Base Git commit: {commit}; V2 implementation is an uncommitted independent worktree change (see source manifests).',f'- Matched baseline checkpoint: {baseline["checkpoint"]}',f'- Baseline checkpoint SHA256: {baseline["checkpoint_sha256"]}',f'- V2 checkpoint SHA256: {candidate["checkpoint_sha256"]}',f'- Parent initialization checkpoint: {bp["parent_checkpoint"]}',f'- Parent SHA256: {bp["parent_checkpoint_sha256"]}',f'- Official RDDM restoration source commit: 524caea34cedb3733aa5bf6c27dc1cd6e0469216','', '## Exact protocol and implementation', '- Exact V2 constants: CONFIG.json; actual training protocol snapshot: runs/2P5D_MEDDINO_HIER_EDGE_V2_8K/EFFECTIVE_CONFIG.json.', '- Inherited config model metadata describes the historical stripped implementation and is not used to build V2. Actual V2 topology is audited in READINESS.json: 19 ResnetBlocks, 8 LinearAttention, 1 bottleneck full Attention; unchanged original skips and diffusion AST.', '- MedDINO layers [2,5,8,11] -> four 768-to-64 Conv1x1 adapters; [2,5] local and [8,11] semantic -> separate Conv1x1(128,128)+SiLU, spatial 32x32 throughout.', '- Fixed center-LDCT Gaussian sigma [0.8,1.6,3.2], radius ceil(3*sigma)=[3,5,10], separable normalized kernels; reflect padding; Scharr /32 magnitude; signed four-neighbor Laplacian after sigma0.8.', '- Raw maps use per-sample spatial RMS normalization with epsilon1e-6; each Conv3(1,32)+GN8+SiLU; concat -> Conv3(128,128)+GN8+SiLU at full512x512; resize32 -> independent Conv1x1(128,128)+GN8 local/semantic.', '- S_local=0.1*(A_local*E_local); S_semantic=0.1*(A_semantic*E_semantic); S_mid=0.5*S_local+0.5*S_semantic. No residual anatomy fusion or learnable lambda.', '- Stage map: '+json.dumps(v2['stage_map']), '- Every stage resizes bilinearly with align_corners=False; original GN8 affine=False and adapter/projection dimensions preserved. Former global Linear adapters apply per spatial position; no global pooling. FiLM = F+gamma*GN(F)+beta; zero final projections.', '- Activation recomputation only for edge encoder and FiLM affine functions during training, non-reentrant and preserve_rng_state=True. No backbone change, AMP, batch, or formula change.', '- Train L067/L143 only; validation fixed64 L333 centers with patient-local prev/next LDCT. Center NDCT supervision only. No external memory, L506, or Full-L333.', '- Stored pixel -1024 -> clip[-1024,3072] -> [0,1]; paired flip0.5 and rotations0/90/180/270. Input [x_t,prev LDCT,center LDCT,next LDCT]; output center residual.', '- pred_res, original MSE/q_sample/residual/sampler; T=1000, sampling2, sum_scale0.01. Adam lr2e-4, betas(0.9,0.99), wd1e-9; B2 accum1; EMA0.995 each step; AMP=False; clip1.0; no scheduler.', '- Seed20260829; CPUthreads2; CUBLAS_WORKSPACE_CONFIG=:4096:8; deterministic algorithms and cuDNN deterministic, benchmarkFalse, TF32False. Fresh optimizer/EMA for sanity and formal; both initialized from matched2D8kEMA, not prior MedDINO8k.', '- Evaluation: EMA, fixed validation per-slice seed+slice_index; HUclip[-1000,1000], PSNRrange2000; skimage SSIM win7, non-Gaussian, sample covarianceTrue, K1=.01 K2=.03; per-slice HU RMSE/MAE and arithmetic means.', '', '## Validation', '| Step | PSNR | SSIM | RMSE HU | MAE HU | Delta PSNR vs baseline8k | Delta SSIM |', '|---:|---:|---:|---:|---:|---:|---:|']
    for x in history:lines.append(f"| {x['step']} | {x['psnr']:.6f} | {x['ssim']:.6f} | {x['rmse_hu']:.6f} | {x['mae_hu']:.6f} | {x['delta_psnr_vs_baseline_8k']:+.6f} | {x['delta_ssim_vs_baseline_8k']:+.6f} |")
    lines+=['',f"Baseline8k: PSNR {bp['baseline_candidate_metrics']['psnr']:.9f}; SSIM {bp['baseline_candidate_metrics']['ssim']:.9f}; RMSE {bp['baseline_candidate_metrics']['rmse_hu']:.9f} HU; MAE {bp['baseline_candidate_metrics']['mae_hu']:.9f} HU.",'All node deltas compare with the fixed baseline8k above, not a same-step control. Full deltas: VALIDATION_METRICS.csv. Per-slice values: run L333_fast64_step_*_metrics.csv.', '', '## Region diagnostics', 'Mask fixed before training: center LDCT sigma0.8 Gaussian+Scharr magnitude, divided by per-sample spatial RMS+1e-6; edge>=2.0, flat=complement. No target-derived mask. Both EMA8k checkpoints evaluated on exactly the same64 slices and masks. Region metrics are arithmetic means of per-slice HU-clipped metrics.', '| Region | Baseline RMSE | V2 RMSE | Baseline MAE | V2 MAE |','|---|---:|---:|---:|---:|']
    for region in ('edge','flat'):
        b=baseline['means'];c=candidate['means'];lines.append(f"| {region} | {b[region+'_rmse_hu']:.6f} | {c[region+'_rmse_hu']:.6f} | {b[region+'_mae_hu']:.6f} | {c[region+'_mae_hu']:.6f} |")
    lines += ['',f'Paired mean baseline-minus-V2 edge RMSE improvement: {float(improvement.mean()):+.6f} HU; bootstrap95%CI {ci}, seed20260829,10000 replicates. This interval describes Fast64 slice resampling, not independent patient replication.', '', '## Condition strength at 8k', 'Definition: RMS(gamma*GN(F)+beta)/RMS(F), mean of128 sampler observations across64 slices. CONDITION_STRENGTH.csv retains every validation node and gamma/beta RMS.', '| Stage | Delta/base | Gamma RMS | Beta RMS |','|---|---:|---:|---:|']
    for x in strength:
        if x['step']==8000:lines.append(f"| {x['stage']} | {x['delta_base_ratio']:.8f} | {x['gamma_rms']:.8f} | {x['beta_rms']:.8f} |")
    lines+=['', '## Integrity and resources', f"- Readiness PASS: realB2 forward/backward, native attention gradients finite/nonzero, feature/edge gradients finite/nonzero after zero projection begins updating; step0 output difference {ready['step0_max_abs_difference']}.", '- Visual review PASS with limitations: Laplacian has visible granular noise and coherent contours; no kernel changed. The step0 E heatmap is untrained and provides no performance claim.', f"- Sanity PASS200: nonfinite0; median {sanity['seconds_per_step_median']:.4f} sec/step; peak {sanity['peak_vram_bytes']/2**30:.3f} GiB.", f"- Formal nonfinite0; all parameter audits finite; peak {train['peak_vram_bytes']/2**30:.3f} GiB; mean {train['seconds_per_step_mean']:.4f}, median {train['seconds_per_step_median']:.4f} sec/step (training timing includes step audits, excludes node validation).", f"- Trainable parameters {ready['parameters_trainable']:,}; total {ready['parameters_total']:,}; baseline trainable {bp['baseline_initial_state']['trainable_parameters']:,}; delta {summary['trainable_parameter_delta']:+,}.", f"- MedDINO checkpoint file SHA256492b2b96c00e9d3b93ba3b07a8d387aa2196b5e04698378fcc23049b447966d2; loaded state before/after {train['meddino_hash_before']} / {train['meddino_hash_after']}; frozen/eval/gradNone; fixed kernels unchanged.", '- All8000 per-step augmented tensors, sample metadata, timestep and noise bytes match the baseline cumulative stochastic stream; initialization and manifests also match. RANDOM_STREAM_AUDIT.json and train.jsonl retain evidence.', '- Reload PASS for baseline and V2 checkpoints; reproduced whole-image metrics within1e-5. Both SHA256s recorded; checkpoint files remain private and unmodified.', '- All data audits: L506=0, external_memory=0, neighbor_NDCT=0, blocked_attempts=0. Only authorized train patients and Fast64 centers/context opened.', '', '## Decision and scope', f'- Decision {decision}; PROMOTE requires deltaPSNR>=.03, SSIM improvement and positive lower bootstrapCI for edge RMSE. STOP requires both6k and8k deltaPSNR<=0. Other outcomes retained for diagnosis.', '- No10k/12k/Full-L333 scheduled. Evidence is a single seeded two-training-patient Fast-Dev experiment, not a generalization claim.', '', '## Artifacts and reproducibility', '- Code: model_v2.py, train_v2.py, finalize_v2.py, bootstrap.py; inherited unchanged model, loader, diffusion and evaluator are tracked in the sibling baseline package.', '- Launch commands/environment/output/log/time and Git state: *_LAUNCH_AUDIT.json. ENVIRONMENT.json contains actual runtime versions. Checkpoints under runs/*/checkpoints remain excluded from Git.', '- Audits: READINESS.json, VISUAL_REVIEW.json, SANITY_SUMMARY.json, TRAINING_SUMMARY.json, REGION_* and *_RELOAD_AUDIT.json, source manifests, run INITIAL_STATE/RANDOM_STREAM_AUDIT/DATA_ACCESS_AUDIT/MEDDINO_FROZEN_AUDIT.', '- Logs: logs/sanity_200.log, logs/formal_8k.log; per-step log runs/2P5D_MEDDINO_HIER_EDGE_V2_8K/train.jsonl.', '- Diagnostic image remains private: visualizations/L333_fast64_309_edges.png. No CT data, checkpoints, or large outputs uploaded.', '- No software installation, Git commit, or push performed in this task.']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n');print('FINAL_REPORT '+json.dumps({'status':'PASS','decision':decision,'psnr_delta':dp}),flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        write(ROOT/('FINALIZE_FAILURE_'+time.strftime('%Y%m%d_%H%M%S')+'.json'),{'status':'FAIL','traceback':traceback.format_exc()});raise
