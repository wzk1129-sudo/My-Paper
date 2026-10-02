"""Final-only L506 inference for frozen V2@8k EMA; one prediction, two post-hoc HU windows."""
import bootstrap
import copy,csv,hashlib,json,math,os,re,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np,torch,skimage
from bootstrap import ROOT,BASE
from model_v2 import build_v2
from model import load_state_without_med,state_without_med
from train import tensor_state_hash
from med_loader import sha
from fast_dev.protocol import CONFIG,check_config
from mayo2016_smallset_v1.frozen_protocol import normalize_stored_ct,slice_metrics
from mayo2016_smallset_v1.runtime import seed_everything
PATIENT='L506';TOTAL=526;INDICES=tuple(range(1,525));METRICS=('psnr','ssim','rmse_hu','mae_hu')
OUTPUT=ROOT/'L506_FINAL_EVAL_8K'
CHECKPOINT=ROOT/'runs/2P5D_MEDDINO_HIER_EDGE_V2_8K/checkpoints/step_8000.pt'
OFFICIAL=Path('/home/xujian/projects/CoreDiff/experiments/corediff_official_checkpoint_eval')

def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def file_sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save_rows(path,rows):
    with Path(path).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)

class L506OnlyGuard:
    """Audit actual medical-data opens: only authorized L506 center NDCT and 3 LDCT views."""
    def __init__(self,data_root):
        self.root=os.path.abspath(data_root);self.active=False;self.current=None;self.reads=[];self.blocked=0
    def begin(self,index):
        assert self.active and self.current is None and index in INDICES;self.current=index
    def end(self):self.current=None
    def _audit(self,event,args):
        if not self.active or event not in ('open','os.listdir','os.scandir') or not args:return
        raw=args[0]
        if not isinstance(raw,(str,bytes,os.PathLike)):return
        path=os.path.abspath(os.fsdecode(raw))
        if path!=self.root and not path.startswith(self.root+os.sep):return
        if event!='open':
            self.blocked+=1;raise PermissionError('Mayo image directory enumeration prohibited')
        m=re.fullmatch(r'(L\d{3})_(25|target)_(\d{3})_img\.npy',os.path.basename(path))
        flags=args[2] if len(args)>2 and isinstance(args[2],int) else 0
        writing=bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
        valid=bool(m) and os.path.dirname(path)==self.root and not writing and self.current is not None and m[1]==PATIENT
        if valid:
            i=int(m[3]);c=self.current
            valid=(m[2]=='25' and i in (c-1,c,c+1)) or (m[2]=='target' and i==c)
        if not valid:
            self.blocked+=1;raise PermissionError('Out-of-scope image read or data write')
        self.reads.append((m[1],m[2],int(m[3]),self.current))
    def __enter__(self):self.active=True;sys.addaudithook(self._audit);return self
    def __exit__(self,*args):self.active=False
    def report(self):
        a=self.reads
        return {'patient':PATIENT,'authorized_center_count':len(INDICES),'actual_opens':len(a),
                'center_target_opens':sum(v=='target' and i==c for _,v,i,c in a),
                'center_ldct_opens':sum(v=='25' and i==c for _,v,i,c in a),
                'neighbor_ldct_opens':sum(v=='25' and i!=c for _,v,i,c in a),
                'neighbor_target_opens':sum(v=='target' and i!=c for _,v,i,c in a),
                'patients_opened':sorted({p for p,v,i,c in a}),
                'center_indices_opened':sorted({i for p,v,i,c in a if v=='target'}),
                'other_patient_opens':sum(p!=PATIENT for p,v,i,c in a),
                'blocked_attempts':self.blocked,'directory_enumeration':False,
                'training':False,'external_memory':False}

class L506Centers:
    def __init__(self,cfg):
        self.root=Path(cfg['data']['root']).resolve();self.data_cfg=cfg['data'];self.guard=None
        assert cfg['data']['dose_label']=='25' and cfg['data']['image_size']==512
        assert (cfg['data']['stored_value_offset_hu'],cfg['data']['hu_min'],cfg['data']['hu_max'])==(-1024.,-1024.,3072.)
        # The earlier official L506 evaluation verified 526 originals and 524 valid context centers.
        for i in range(TOTAL):
            for view in ('25','target'):
                p=self.path(i,view)
                assert p.parent==self.root and not p.is_symlink() and p.is_file(),('missing L506 view',str(p))
        for view in ('25','target'):assert not self.path(TOTAL,view).is_file(),('unexpected extra L506 slice',view)
    def path(self,index,view):return self.root/f'{PATIENT}_{view}_{index:03d}_img.npy'
    def _load(self,index,view):
        p=self.path(index,view);raw=np.load(p,allow_pickle=False)
        assert raw.shape==(512,512) and bool(np.isfinite(raw).all())
        n=normalize_stored_ct(raw,self.data_cfg)
        assert n.dtype==np.float32 and bool(np.isfinite(n).all()) and float(n.min())>=0 and float(n.max())<=1
        return n[None]
    def load(self,index):
        assert self.guard is not None and index in INDICES
        self.guard.begin(index)
        try:
            prev=self._load(index-1,'25');center=self._load(index,'25');next_=self._load(index+1,'25');target=self._load(index,'target')
            return prev,center,next_,target
        finally:self.guard.end()

@torch.no_grad()
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0' and os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
    assert CHECKPOINT.is_file() and not OUTPUT.exists()
    pipeline=json.loads((ROOT/'PIPELINE_STATUS.json').read_text());assert pipeline['status']=='COMPLETE'
    train=json.loads((ROOT/'TRAINING_SUMMARY.json').read_text());assert train['status']=='PASS' and train['optimizer_step']==train['ema_updates']==8000
    final=json.loads((ROOT/'SUMMARY.json').read_text());assert final['status']=='PASS' and final['optimizer_steps']==8000
    cfg=json.loads(CONFIG.read_text());check_config(cfg);soft_cfg=copy.deepcopy(cfg);soft_cfg['evaluation']['hu_window']=[-160.,240.]
    assert cfg['evaluation']['hu_window']==[-1000.,1000.]
    reference=json.loads((OFFICIAL/'SUMMARY.json').read_text());soft_reference=json.loads((OFFICIAL/'SOFT_TISSUE_160_240_SUMMARY.json').read_text())
    assert reference['patient']==soft_reference['patient']==PATIENT and reference['slice_count']==soft_reference['slice_count']==len(INDICES)==524
    ck_sha_before=file_sha(CHECKPOINT);ck=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    assert ck['arm']=='2P5D_MEDDINO_HIER_EDGE_V2_8K' and ck['optimizer_step']==ck['ema_updates']==8000
    assert ck['train_stream_sha256']==train['train_stream_sha256']
    assert ck['source_hashes']==train['source_hashes']
    for source,h in ck['source_hashes'].items():assert file_sha(source)==h,('training source changed',source)
    assert all(bool(torch.isfinite(t).all()) for t in ck['ema_state_dict'].values())
    expected_l333=next(x for x in train['validation'] if x['step']==8000)
    assert all(abs(expected_l333[k]-final['validation'][-1][k])<1e-9 for k in METRICS)
    OUTPUT.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(cfg['training']['cpu_threads']);seed_everything(20260829);device=torch.device('cuda:0')
    model=build_v2().to(device).eval();load_state_without_med(model,ck['ema_state_dict']);model.requires_grad_(False)
    del ck
    u=model.model.unet0
    med_hash_before=tensor_state_hash(u.med.state_dict());state_hash_before=tensor_state_hash(state_without_med(model))
    assert med_hash_before==train['meddino_hash_after'] and not u.med.training
    dataset=L506Centers(cfg);guard=L506OnlyGuard(dataset.root);dataset.guard=guard
    pred_path=OUTPUT/'predictions_l506_524.npy'
    prediction_map=np.lib.format.open_memmap(pred_path,mode='w+',dtype=np.float32,shape=(len(INDICES),1,512,512))
    cfg_out={'patient':PATIENT,'dose':25,'checkpoint':str(CHECKPOINT),'checkpoint_sha256':ck_sha_before,'checkpoint_step':8000,'weights':'EMA','context':['prev LDCT','center LDCT','next LDCT'],'valid_center_indices':[1,524],'valid_center_count':len(INDICES),'excluded_center_indices':[0,525],
             'input_preprocessing':'stored pixel -> subtract 1024 -> clip [-1024,3072] HU -> (HU+1024)/4096; no [-160,240] pre-model clipping',
             'condition_preprocessing':'unchanged MedDINO center-LDCT preprocess in model.py','model_objective':cfg['diffusion']['objective'],'diffusion_timesteps':cfg['diffusion']['timesteps'],'sampling_timesteps':cfg['diffusion']['sampling_timesteps'],'sum_scale':cfg['diffusion']['sum_scale'],
             'sampler':'unmodified model.sample([center],last=True)[-1]','sampler_calls_expected':len(INDICES),'batch_size':1,'seed_per_slice':'20260829 + center slice index','augmentation':False,'training':False,'checkpoint_selection':False,
             'evaluation':{'main':{'window_hu':[-1000,1000],'data_range_hu':2000},'soft_tissue':{'window_hu':[-160,240],'data_range_hu':400},'ssim':{'implementation':'skimage.metrics.structural_similarity via frozen slice_metrics','win_size':7,'gaussian_weights':False,'use_sample_covariance':True,'K1':0.01,'K2':0.03},'hu_dtype':'float32 before float64 SSIM','mean':'arithmetic mean of per-slice metrics'},
             'prediction_file':str(pred_path),'prediction_dtype':'float32','prediction_shape':[len(INDICES),1,512,512],
             'source_hashes':ck_sha_before,'official_corediff_reference':{'unified_summary':str(OFFICIAL/'SUMMARY.json'),'soft_tissue_summary':str(OFFICIAL/'SOFT_TISSUE_160_240_SUMMARY.json'),'official_checkpoint_sha256':reference['checkpoint_sha256']}}
    # The source-hash map is kept separate from the checkpoint hash; preserve exact effective config.
    cfg_out['source_hashes']=train['source_hashes'];cfg_out['evaluator_source_sha256']=file_sha(Path(__file__))
    write(OUTPUT/'EVALUATION_CONFIG.json',cfg_out)
    environment={'python':sys.version,'executable':sys.executable,'torch':torch.__version__,'cuda':torch.version.cuda,'numpy':np.__version__,'skimage':skimage.__version__,'gpu':torch.cuda.get_device_name(0),'cublas_workspace_config':os.environ['CUBLAS_WORKSPACE_CONFIG'],'seed':20260829,'cpu_threads':torch.get_num_threads()}
    write(OUTPUT/'ENVIRONMENT.json',environment)
    torch.cuda.reset_peak_memory_stats();started=time.perf_counter();main_rows=[];soft_rows=[];pred_stream=hashlib.sha256();sample_calls=0
    with guard:
        for pos,index in enumerate(INDICES):
            prev,center,next_,target=dataset.load(index)
            prev,center,next_,target=[torch.from_numpy(x).unsqueeze(0).to(device) for x in (prev,center,next_,target)]
            u.set_context(prev,center,next_)
            seed=cfg['evaluation']['validation_seed']+index;torch.manual_seed(seed);torch.cuda.manual_seed(seed)
            pred=model.sample([center],last=True)[-1];sample_calls+=1
            assert pred.shape==(1,1,512,512) and bool(torch.isfinite(pred).all())
            prediction_map[pos]=pred.detach().float().cpu().numpy()[0]
            stored=np.array(prediction_map[pos:pos+1],copy=True)
            bytes_=stored.tobytes();digest=hashlib.sha256(bytes_).hexdigest()
            pred_stream.update(index.to_bytes(4,'little'));pred_stream.update(bytes_)
            scored=torch.from_numpy(stored)
            main=slice_metrics(scored,target,cfg)
            soft=slice_metrics(scored,target,soft_cfg)
            assert all(math.isfinite(main[k]) and math.isfinite(soft[k]) for k in METRICS)
            common={'patient_id':PATIENT,'slice_index':index,'sampling_seed':seed,'prediction_sha256':digest}
            main_rows.append({**common,**{k:main[k] for k in METRICS}})
            soft_rows.append({**common,**{k:soft[k] for k in METRICS}})
            u.clear_context()
            if sample_calls==1 or sample_calls%50==0 or sample_calls==len(INDICES):
                print('PROGRESS '+json.dumps({'slices':sample_calls,'total':len(INDICES),'elapsed_seconds':time.perf_counter()-started}),flush=True)
    prediction_map.flush();del prediction_map
    access=guard.report();assert access['patients_opened']==[PATIENT]
    assert access['center_target_opens']==access['center_ldct_opens']==len(INDICES)
    assert access['neighbor_ldct_opens']==2*len(INDICES)
    assert access['neighbor_target_opens']==access['other_patient_opens']==access['blocked_attempts']==0
    assert access['center_indices_opened']==list(INDICES)
    assert sample_calls==len(INDICES)
    # Reopen the saved float32 array and verify every scored prediction byte sequence.
    reopened=np.load(pred_path,mmap_mode='r',allow_pickle=False);assert reopened.shape==(len(INDICES),1,512,512) and reopened.dtype==np.float32
    reread=hashlib.sha256()
    for pos,index in enumerate(INDICES):
        b=np.asarray(reopened[pos:pos+1]).tobytes()
        assert hashlib.sha256(b).hexdigest()==main_rows[pos]['prediction_sha256']==soft_rows[pos]['prediction_sha256']
        reread.update(index.to_bytes(4,'little'));reread.update(b)
    assert reread.hexdigest()==pred_stream.hexdigest()
    del reopened
    pred_file_sha=file_sha(pred_path);ck_sha_after=file_sha(CHECKPOINT)
    state_hash_after=tensor_state_hash(state_without_med(model));med_hash_after=tensor_state_hash(u.med.state_dict())
    assert ck_sha_after==ck_sha_before and state_hash_after==state_hash_before and med_hash_after==med_hash_before
    assert all(p.grad is None for p in model.parameters())
    save_rows(OUTPUT/'MAIN_1000_METRICS.csv',main_rows);save_rows(OUTPUT/'SOFT_TISSUE_160_240_METRICS.csv',soft_rows)
    write(OUTPUT/'DATA_ACCESS_AUDIT.json',access)
    main_mean={k:float(np.mean([x[k] for x in main_rows])) for k in METRICS};soft_mean={k:float(np.mean([x[k] for x in soft_rows])) for k in METRICS}
    official_main={'psnr':reference['unified_psnr'],'ssim':reference['unified_ssim'],'rmse_hu':reference['unified_rmse_hu'],'mae_hu':reference['unified_mae_hu']}
    official_soft={'psnr':soft_reference['PSNR'],'ssim':soft_reference['SSIM'],'rmse_hu':soft_reference['RMSE_HU'],'mae_hu':soft_reference['MAE_HU']}
    result={'status':'COMPLETE','patient':PATIENT,'slice_count':len(INDICES),'actual_center_indices':[1,524],'checkpoint':str(CHECKPOINT),'checkpoint_sha256':ck_sha_before,'checkpoint_sha256_after':ck_sha_after,'checkpoint_step':8000,'ema':True,
            'l333_fast64_8k':{k:expected_l333[k] for k in METRICS},'l506_main_1000':main_mean,'l506_soft_tissue_160_240':soft_mean,
            'official_corediff_l506_unified_1000':official_main,'official_corediff_l506_soft_tissue_160_240':official_soft,
            'v2_minus_official_main_1000':{k:main_mean[k]-official_main[k] for k in METRICS},'v2_minus_official_soft_tissue_160_240':{k:soft_mean[k]-official_soft[k] for k in METRICS},
            'prediction_file':str(pred_path),'prediction_file_sha256':pred_file_sha,'prediction_file_bytes':pred_path.stat().st_size,'prediction_stream_sha256':pred_stream.hexdigest(),'same_predictions_both_protocols':True,'reopened_prediction_hash_match_count':len(INDICES),
            'sample_calls':sample_calls,'meddino_hash_before':med_hash_before,'meddino_hash_after':med_hash_after,'model_state_hash_before':state_hash_before,'model_state_hash_after':state_hash_after,'no_gradients':True,'data_access':access,
            'elapsed_seconds':time.perf_counter()-started,'peak_vram_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_vram_reserved_bytes':torch.cuda.max_memory_reserved(),
            'git_commit':subprocess.check_output(['git','-C',str(ROOT.parents[1]),'rev-parse','HEAD'],text=True).strip(),'branch':subprocess.check_output(['git','-C',str(ROOT.parents[1]),'branch','--show-current'],text=True).strip(),
            'evaluation_config':str(OUTPUT/'EVALUATION_CONFIG.json'),'finished_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    write(OUTPUT/'SUMMARY.json',result)
    lines=['# V2@8k EMA final-only L506 evaluation','','Status: COMPLETE. One read-only inference pass over all524 valid center slices, with both HU windows applied to the same saved float32 prediction array.','',
           '## Exact input and model protocol',f'- Checkpoint: {CHECKPOINT}',f'- Checkpoint SHA256: {ck_sha_before}','- V2 official RDDM restoration backbone, frozen MedDINOv3 layers[2,5,8,11], hierarchical edge condition; step8000 EMA.','- Mayo 2016 dose25 NPY series, full valid centers1..524; first/last slices0/525 used only as LDCT context, not target.','- Per center: prev/center/next LDCT, center NDCT target; stored pixel -1024 -> clip[-1024,3072] HU -> [0,1]. No soft-tissue input clipping.','- Unmodified V2 sampler, 2 sampling steps, sum_scale0.01; seed20260829+slice_index; batch1; no augmentation.','',
           '## Numeric results','Arithmetic means of 524 per-slice scores. Main and soft-tissue metrics reuse the identical prediction bytes.','| Model / window | PSNR dB | SSIM | RMSE HU | MAE HU |','|---|---:|---:|---:|---:|']
    for label,m in [('V2@8k L333-fast64 [-1000,1000]',result['l333_fast64_8k']),('V2@8k L506 [-1000,1000]',main_mean),('Official CoreDiff L506 unified [-1000,1000]',official_main),('V2@8k L506 [-160,240]',soft_mean),('Official CoreDiff L506 [-160,240]',official_soft)]:
        lines.append('| '+label+' | '+' | '.join(f'{m[k]:.8f}' for k in METRICS)+' |')
    lines+=['','V2 minus official CoreDiff on L506, matching evaluation window:','| Window | Delta PSNR dB | Delta SSIM | Delta RMSE HU | Delta MAE HU |','|---|---:|---:|---:|---:|']
    for label,m in [('[-1000,1000]',result['v2_minus_official_main_1000']),('[-160,240]',result['v2_minus_official_soft_tissue_160_240'])]:
        lines.append('| '+label+' | '+' | '.join(f'{m[k]:+.8f}' for k in METRICS)+' |')
    lines+=['','CoreDiff reference is its official released150k EMA model, scored with the unified HU-window metrics for numeric comparison. The two models have different training and sampling protocols; these values alone do not isolate an architecture effect. No L506 result affected training, hyperparameters or checkpoint selection.','',
            '## Metric implementation','Both normalized prediction and target are converted to HU, then clipped to the stated post-hoc window. PSNR ranges2000 and400 HU. SSIM reuses the project frozen slice_metrics: skimage win7, gaussian_weights=False, use_sample_covariance=True, K1=.01, K2=.03. RMSE and MAE use clipped HU. Scores are per-slice then averaged.',
            '', '## Integrity and artifacts',f'- Prediction: {pred_path}; shape[524,1,512,512], float32 normalized [0,1] before HU windowing; SHA256 {pred_file_sha}.',f'- Prediction stream SHA256: {pred_stream.hexdigest()}; all524 reopened per-slice prediction hashes match both CSVs.',f'- Checkpoint before/after SHA256: {ck_sha_before} / {ck_sha_after}. Model and MedDINO parameter hashes unchanged; no gradients.',
            '- Data audit: only L506 views opened, center targets only, no other-patient reads or data writes. See DATA_ACCESS_AUDIT.json.',
            '- Exact runtime config EVALUATION_CONFIG.json; environment ENVIRONMENT.json; metrics MAIN_1000_METRICS.csv and SOFT_TISSUE_160_240_METRICS.csv; summary SUMMARY.json.','- Prediction remains private on the remote machine and is ignored by Git. No training, tuning, OSL, L333 re-evaluation or other checkpoint evaluation.']
    (OUTPUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    # Add a compact link to the new post-hoc evidence only after all L506 audits pass.
    original=(ROOT/'REPORT.md').read_text();heading='## Post-hoc L506 final-only evaluation'
    assert heading not in original
    with (ROOT/'REPORT.md').open('a') as f:
        f.write('\n'+heading+'\n\nV2 step8000 EMA, full valid L506 centers, one saved prediction array, two post-hoc HU windows. See L506_FINAL_EVAL_8K/REPORT.md, SUMMARY.json and per-slice CSVs. No L506 result was used for training or checkpoint selection.\n')
    print('L506_COMPLETE '+json.dumps({'count':len(INDICES),'main':main_mean,'soft':soft_mean,'prediction_sha256':pred_file_sha}),flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        write(ROOT/('L506_EVAL_FAILURE_'+time.strftime('%Y%m%d_%H%M%S')+'.json'),{'status':'FAIL','traceback':traceback.format_exc()})
        raise
