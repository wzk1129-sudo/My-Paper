#!/usr/bin/env python
"""Read-only official CoreDiff EMA evaluation; sample each L506 center exactly once."""
import argparse, csv, hashlib, importlib.metadata, json, os, platform, subprocess, sys, time
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from skimage.metrics import structural_similarity
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from models.corediff.corediff_wrapper import Network
from models.corediff.diffusion_modules import Diffusion
from models.basic_template import TrainTask
from utils.dataset import CTDataset
from utils.measure import compute_measure
from utils.ops import load_network

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8388608),b''): h.update(b)
    return h.hexdigest()
def dump(path,value):
    with open(path,'x') as f: json.dump(value,f,indent=2,allow_nan=False)
def tensor_sha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def state_sha(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():
        h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def write_csv(path,rows):
    with open(path,'x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def run():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--output',required=True)
    args=ap.parse_args()
    os.chdir(ROOT);out=Path(args.output).resolve();checkpoint=Path(args.checkpoint).resolve()
    assert out.is_dir() and not (out/'SUMMARY.json').exists()
    provenance=json.load(open(out/'PRECHECK_PROVENANCE.json'))
    git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    source_audit={}
    for rel,expected in provenance['official_source_sha256'].items():
        actual=sha(ROOT/rel);assert actual==expected,(rel,actual,expected)
        source_audit[rel]={'sha256':actual,'official_byte_match':True}
    dump(out/'SOURCE_AUDIT.json',{'official_commit':provenance['official_commit'],'project_commit':git_commit,'files':source_audit,'evaluator_sha256':sha(__file__)})
    config={
        'patient':'L506','dataset':'mayo_2016','dose':25,'test_id':9,
        'checkpoint_step':150000,'context':True,'context_order':['prev','center','next'],
        'T':10,'sampling_routine':'ddim','only_adjust_two_step':True,
        'only_adjust_two_step_note':'Official flag controls training forward only; official sample has no such argument. Inference uses unmodified official sample.',
        'start_adjust_iter':1,'n_iter':150000,'test_batch_size':1,
        'loader_shuffle':False,'loader_num_workers':0,
        'loader_note':'Direct official CTDataset iteration in official sorted order; worker count affects IO only.',
        'normalization':'stored pixel -> subtract 1024 -> clip [-1024,3072] HU -> (HU+1024)/4096',
        'official_metrics':{'window_hu':[-1000,1000],'map_to':[0,255],'data_range':'target.max()-target.min() per slice','implementation':'utils.measure.compute_measure','ssim_window':11,'ssim_gaussian_sigma':1.5,'ssim_padding':5,'rmse_units':'HU'},
        'unified_metrics':{'window_hu':[-1000,1000],'data_range':2000,'ssim':'skimage.metrics.structural_similarity','win_size':7,'gaussian_weights':False,'use_sample_covariance':True,'K1':0.01,'K2':0.03,'numeric_dtype':'float64 after same float32 HU conversion and clipping'},
        'checkpoint':str(checkpoint),'output':str(out),'git_commit':git_commit,
        'training':False,'OSL':False,'amp':False,'device':'cuda:0',
        'full_command':provenance['command'],'tmux_session':provenance['tmux_session'],
        'conda_environment':'corediff','seed':20261002,'prediction_persistence':'memory only; per-slice SHA256 saved, no image files',
    }
    dump(out/'RUN_CONFIG.json',config)
    torch.set_num_threads(4)
    torch.manual_seed(config['seed']);np.random.seed(config['seed']);torch.cuda.manual_seed_all(config['seed'])
    environment={'python':sys.version,'python_executable':sys.executable,'pytorch':torch.__version__,'cuda_runtime':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'gpu':torch.cuda.get_device_name(0),'hostname':platform.node(),'user':os.environ.get('USER'),'packages':{p:importlib.metadata.version(p) for p in ['numpy','torchvision','scikit-image','pydicom','natsort']},'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'nvidia_smi':subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi'],text=True),'started_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    dump(out/'ENVIRONMENT.json',environment)
    checksha=sha(checkpoint)
    assert checksha==provenance['official_checkpoint_sha256']
    model=Diffusion(Network(in_channels=3,context=True),image_size=512,timesteps=10,context=True)
    weights=load_network(str(checkpoint))
    incompatible=model.load_state_dict(weights,strict=True)
    assert not incompatible.missing_keys and not incompatible.unexpected_keys
    assert all(torch.isfinite(v).all().item() for v in weights.values())
    audit={'selected_path':str(checkpoint),'bytes':checkpoint.stat().st_size,'sha256_before':checksha,
           'official_public_sha256':provenance['official_checkpoint_sha256'],'official_public_byte_match':True,
           'source_url':provenance['official_checkpoint_url'],'source_folder':provenance['official_checkpoint_folder'],
           'verification':provenance['checkpoint_verification'],'candidates':provenance['candidates'],
           'loader':'utils.ops.load_network -> Diffusion.load_state_dict(strict=True)',
           'weight_format':'official standalone EMA state_dict','key_count':len(weights),'missing_keys':[], 'unexpected_keys':[],
           'finite':True,'parameter_count':sum(p.numel() for p in model.parameters()),
           'checkpoint_step':150000,'EMA':True,'checkpoint_modified':False}
    dump(out/'CHECKPOINT_AUDIT.json',audit)
    del weights
    model=model.cuda().eval()
    for p in model.parameters():p.requires_grad_(False)
    before_state=state_sha(model)
    dataset=CTDataset(dataset='mayo_2016',mode='test',test_id=9,dose=25,context=True)
    data_dir=ROOT/'data_preprocess/gen_data/mayo_2016_npy'
    lows=sorted(data_dir.glob('L506_25_*_img.npy'));targets=sorted(data_dir.glob('L506_target_*_img.npy'))
    assert len(lows)==len(targets)==len(dataset)+2
    assert dataset.target==['./'+str(p.relative_to(ROOT)) for p in targets[1:-1]]
    for i,entry in enumerate(dataset.input):
        assert entry.split('~')[1:]==['./'+str(p.relative_to(ROOT)) for p in lows[i:i+3]]
        assert lows[i+1].name.split('_')[2]==targets[i+1].name.split('_')[2]
    # Verify original DICOM preprocessing by exact stored-pixel equality at three fixed indices.
    import pydicom
    from natsort import natsorted
    raw_root=Path('/home/xujian/data/Mayo2016/L506')
    raw_audit=[]
    for kind,paths in [('quarter_1mm',lows),('full_1mm',targets)]:
        raw=natsorted((raw_root/kind).glob('*.IMA'),key=str)
        assert len(raw)==len(paths)
        for index in [0,len(paths)//2,len(paths)-1]:
            dicom=pydicom.dcmread(str(raw[index]))
            stored=np.load(paths[index])
            assert np.array_equal(np.array(dicom.pixel_array).astype(np.uint16),stored)
            raw_audit.append({'series':kind,'sorted_index':index,'npy':paths[index].name,'dicom_to_npy_exact':True,'npy_dtype':str(stored.dtype),'shape':list(stored.shape)})
    dump(out/'DATA_PROTOCOL_AUDIT.json',{'patient':'L506','raw_ldct_slices':len(lows),'raw_ndct_slices':len(targets),'valid_center_slices':len(dataset),'excluded_center_first':lows[0].name,'excluded_center_last':lows[-1].name,'first_evaluated_center':lows[1].name,'last_evaluated_center':lows[-2].name,'boundary_note':'First and last slices are excluded as centers, but used as neighbors for the first/last valid centers.','original_dicom_checks':raw_audit,'test_split_unmodified':True,'normalization_implementation':'utils.dataset.CTDataset.normalize_'})
    accesses=Counter();orig_load=np.load
    def guarded_load(file,*a,**kw):
        p=Path(file).resolve()
        assert p.parent==data_dir.resolve() and p.name.startswith(('L506_25_','L506_target_')),str(p)
        accesses[p.name]+=1
        return orig_load(file,*a,**kw)
    np.load=guarded_load
    forward_count=0
    def forward_hook(module,inputs,output):
        nonlocal forward_count
        forward_count+=1
    hook=model.denoise_fn.register_forward_hook(forward_hook)
    official_rows=[];unified_rows=[];identities=[];start=time.time();sample_calls=0
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for i in range(len(dataset)):
            low,target=dataset[i]
            low=torch.from_numpy(low).unsqueeze(0).cuda();target=torch.from_numpy(target).unsqueeze(0).cuda()
            assert low.shape==(1,3,512,512) and target.shape==(1,1,512,512)
            pred,direct,steps=model.sample(batch_size=low.shape[0],img=low,t=10,sampling_routine='ddim',n_iter=150000,start_adjust_iter=1)
            sample_calls+=1
            assert torch.isfinite(pred).all().item()
            identity=tensor_sha(pred)
            # Exact official test conversion and exact official measurement on CUDA.
            official_target=TrainTask.transfer_calculate_window(None,target)
            official_pred=TrainTask.transfer_calculate_window(None,pred)
            dynamic_range=official_target.max()-official_target.min()
            psnr,ssim,rmse=compute_measure(official_target,official_pred,dynamic_range)
            pred_hu=(pred*4096.-1024.).clamp(-1000.,1000.)[0,0].cpu().numpy().astype(np.float64)
            target_hu=(target*4096.-1024.).clamp(-1000.,1000.)[0,0].cpu().numpy().astype(np.float64)
            error=pred_hu-target_hu;mse=float(np.mean(error**2))
            upsnr=float(10*np.log10(2000.**2/mse))
            ussim=float(structural_similarity(target_hu,pred_hu,data_range=2000,win_size=7,gaussian_weights=False,use_sample_covariance=True,K1=.01,K2=.03))
            urmse=float(np.sqrt(mse));umae=float(np.mean(np.abs(error)))
            assert tensor_sha(pred)==identity
            assert all(np.isfinite(x) for x in [psnr,ssim,rmse,upsnr,ussim,urmse,umae])
            common={'slice_index':i,'center_slice':lows[i+1].name,'target_slice':targets[i+1].name,'prediction_sha256':identity}
            official_rows.append(dict(common,PSNR=psnr,SSIM=ssim,RMSE_HU=rmse,data_range_255=float(dynamic_range.item())))
            unified_rows.append(dict(common,PSNR=upsnr,SSIM=ussim,RMSE_HU=urmse,MAE_HU=umae))
            identities.append(identity)
            del pred,direct,steps,official_pred,official_target,low,target
            if (i+1)%25==0 or i==len(dataset)-1:
                print(json.dumps({'completed':i+1,'total':len(dataset),'elapsed_seconds':time.time()-start,'official_psnr_running':float(np.mean([x['PSNR'] for x in official_rows])),'unified_psnr_running':float(np.mean([x['PSNR'] for x in unified_rows]))}),flush=True)
    np.load=orig_load;hook.remove()
    assert sample_calls==len(dataset) and forward_count==10*len(dataset)
    assert all(p.grad is None for p in model.parameters())
    after_state=state_sha(model);after_sha=sha(checkpoint)
    assert after_state==before_state and after_sha==checksha
    for rel,expected in provenance['official_source_sha256'].items():assert sha(ROOT/rel)==expected
    write_csv(out/'OFFICIAL_METRICS.csv',official_rows);write_csv(out/'UNIFIED_METRICS.csv',unified_rows)
    summary={'patient':'L506','dose':25,'checkpoint_step':150000,'slice_count':len(dataset),
             'official_psnr':float(np.mean([x['PSNR'] for x in official_rows])),'official_ssim':float(np.mean([x['SSIM'] for x in official_rows])),'official_rmse_hu':float(np.mean([x['RMSE_HU'] for x in official_rows])),
             'unified_psnr':float(np.mean([x['PSNR'] for x in unified_rows])),'unified_ssim':float(np.mean([x['SSIM'] for x in unified_rows])),'unified_rmse_hu':float(np.mean([x['RMSE_HU'] for x in unified_rows])),'unified_mae_hu':float(np.mean([x['MAE_HU'] for x in unified_rows])),
             'checkpoint_sha256':checksha,'git_commit':git_commit,'official_source_commit':provenance['official_commit'],
             'sample_calls':sample_calls,'denoiser_forward_calls':forward_count,'same_prediction_all_slices':True,
             'prediction_stream_sha256':hashlib.sha256(''.join(identities).encode()).hexdigest(),
             'checkpoint_sha256_after':after_sha,'model_state_sha256_before':before_state,'model_state_sha256_after':after_state,
             'elapsed_seconds':time.time()-start,'peak_vram_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_vram_reserved_bytes':torch.cuda.max_memory_reserved(),'finished_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'status':'COMPLETE'}
    dump(out/'SUMMARY.json',summary)
    dump(out/'READ_ONLY_AUDIT.json',{'checkpoint_unchanged':True,'model_state_unchanged':True,'official_sources_unchanged':True,'training_steps':0,'optimizer_instances':0,'backward_calls':0,'OSL_calls':0,'L333_access_count':0,'other_patient_access_count':0,'l506_evaluation_npy_reads':sum(accesses.values()),'l506_preflight_npy_reads':6,'l506_preflight_dicom_reads':6,'l506_unique_evaluation_npy_files':len(accesses),'access_by_file':dict(accesses),'same_prediction_sha256_in_both_csv':True,'sample_calls':sample_calls,'denoiser_forward_calls':forward_count})
    report=f"""# Official CoreDiff Mayo 2016 25% dose / 150k EMA: L506 dual-protocol evaluation

Status: COMPLETE. Read-only inference; no training, OSL, parameter selection, or L333 evaluation.

## Results

Arithmetic means of per-slice scores over {len(dataset)} valid centers:

| Protocol | PSNR (dB) | SSIM | RMSE (HU) | MAE (HU) |
|---|---:|---:|---:|---:|
| OFFICIAL | {summary['official_psnr']:.8f} | {summary['official_ssim']:.8f} | {summary['official_rmse_hu']:.8f} | Not requested |
| UNIFIED | {summary['unified_psnr']:.8f} | {summary['unified_ssim']:.8f} | {summary['unified_rmse_hu']:.8f} | {summary['unified_mae_hu']:.8f} |

## Checkpoint and source verification

Existing checkpoint: {checkpoint}
Bytes: {checkpoint.stat().st_size}
SHA256: {checksha}
Public official release SHA256 matches exactly. The two local reproduce copies have a different SHA256 and were rejected.
Official repository: https://github.com/qgao21/CoreDiff
Official source commit: {provenance['official_commit']}
Actual project commit: {git_commit}
Eight relevant source files match the official commit byte for byte (SOURCE_AUDIT.json).
Official load_network and strict Diffusion.load_state_dict loaded 84 state entries; 4,825,953 trainable-architecture parameters, all finite.
File and loaded model state hashes are unchanged after evaluation.

## Data and inference

Original Mayo 2016 stored-pixel DICOM preprocessing; dataset=mayo_2016, dose=25, test_id=9, patient=L506.
Both raw series contain {len(lows)} slices. Official dataset excludes first/last as centers; first/last remain context neighbors.
The evaluated centers are {lows[1].name} through {lows[-2].name}.
Exact DICOM stored-pixel versus existing NPY equality was checked at first, middle, last for both series, without rewriting data.
Normalization uses the unmodified CTDataset.normalize_: subtract 1024, clip [-1024,3072], map to [0,1].
Batch=1, context=True (prev/center/next), T=10, sampling_routine=ddim (official default), n_iter=150000, start_adjust_iter=1.
only_adjust_two_step=True is the official training flag; it is not an argument to sample and does not change inference.
The existing official Network and Diffusion.sample are called directly. No optimizer is instantiated.
Each center is sampled once: {sample_calls} sample calls, {forward_count} denoiser calls.
The identical normalized prediction tensor is used by both protocols and hash-checked before and after scoring.
Predictions are retained in memory only; no CT image/prediction array export. Both CSVs carry matching per-slice prediction hashes.
Direct dataset iteration replaces worker-based loading only; slice ordering and returned arrays are unchanged.

## Evaluation protocols

OFFICIAL: use the exact TrainTask.transfer_calculate_window and utils.measure.compute_measure on CUDA.
Normalized images become HU, clipped [-1000,1000], then mapped [0,255].
Per slice data_range=target.max()-target.min() in this mapped space.
Official SSIM uses an 11x11 Gaussian (sigma 1.5), official convolution/padding and dynamic range.
Official RMSE converts mapped values back to HU inside compute_RMSE, and is reported in HU.

UNIFIED: reuse the identical prediction; convert to HU and clip [-1000,1000].
PSNR=10*log10(2000^2/MSE_HU); RMSE=sqrt(MSE_HU); MAE=mean(abs(error)).
SSIM is skimage.structural_similarity with data_range=2000, win_size=7,
gaussian_weights=False, use_sample_covariance=True, K1=0.01, K2=0.03.
HU conversion/clipping is float32 like the official model; unified accumulation is float64.
These are two scoring protocols applied to the same outputs; SSIM differences do not represent a model change.
No parameters, data_range, windows, or slice counts were tuned to match paper numbers.

## Execution and audit

Command:
~~~bash
{provenance['command']}
~~~
tmux: {provenance['tmux_session']}
Conda: corediff; GPU: {environment['gpu']}; PyTorch: {environment['pytorch']}; CUDA runtime: {environment['cuda_runtime']}.
Started: {environment['started_at']}; finished: {summary['finished_at']}.
Inference/scoring elapsed: {summary['elapsed_seconds']:.2f} seconds.
Peak allocated GPU memory: {summary['peak_vram_allocated_bytes']/1024**3:.3f} GiB.
Peak reserved GPU memory: {summary['peak_vram_reserved_bytes']/1024**3:.3f} GiB.
Detailed environment, command/options, source, checkpoint, data boundary and access/state audits are in the neighboring JSON files.
No original source, checkpoint, or medical data was modified. No other-patient file reads occurred in this evaluation.
No claim is made about independent performance on L333.
"""
    with open(out/'REPORT.md','x') as f:f.write(report)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':run()
