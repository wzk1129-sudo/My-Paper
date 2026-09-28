"""Independent 100-step calibration and 8k optimizer-step Fast-dev run."""
import argparse
import copy
import csv
import json
import math
import os
import statistics
import time
from pathlib import Path
import numpy as np
import torch
from fast_dev.data import FastPairs, DataAccessGuard, BatchStream
from fast_dev.evaluate import validate
from fast_dev.protocol import CONFIG as BASE_CONFIG, check_config, FAST_TRAIN, FAST_INDICES, source_hashes
from mayo2016_smallset_v1.runtime import EMA, seed_everything, rng_state, restore_rng, atomic_save, write_json, sha256
from .model import build_joint

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
BASE_PSNR = {2000:38.47870036920836,4000:40.14538588663511,6000:40.38720219986538,8000:40.484568141498166}
PARTS = ('loss_total','loss_res','loss_noise','loss_recon')

def configuration():
    base=json.loads(BASE_CONFIG.read_text())
    check_config(base)
    cfg=json.loads((HERE/'CONFIG.json').read_text())
    expected=copy.deepcopy(base)
    expected['experiment_name']=cfg['experiment_name']
    expected['description']=cfg['description']
    expected['model'].update(parameters=13131010,additional_noise_head_parameters=65,shared_backbone=True,time_routing=cfg['model']['time_routing'],shared_unet_calls_per_iteration=2)
    expected['diffusion'].update(objective='pred_res_noise',test_res_or_noise='res_noise')
    expected['training'].update(gradient_accumulation=1,effective_batch=2)
    expected['joint_loss']=cfg['joint_loss']
    expected['reference']=cfg['reference']
    assert cfg==expected, 'Unexpected change to frozen Fast-dev settings'
    assert cfg['training']['batch_size']==2 and cfg['training']['gradient_accumulation']==1
    return base,cfg

def hashes():
    result=source_hashes()
    for rel in ('res_noise_beta_time_v1/model.py','res_noise_beta_time_v1/run.py','res_noise_beta_time_v1/CONFIG.json'):
        result[rel]=sha256(ROOT/rel)
    return result

def read_calibration(path):
    cal=json.loads(Path(path).read_text())
    assert cal['status']=='PASS' and cal['optimizer_steps']==100
    assert cal['source_hashes']==hashes()
    assert cal['gradient_finite'] and cal['loss_finite']
    return cal

def initialize(base,cfg,device,lambda_noise,lambda_recon):
    torch.set_num_threads(cfg['training']['cpu_threads'])
    if device.type=='cuda':torch.cuda.set_device(device)
    seed_everything(cfg['seed'])
    data=FastPairs(base,'train')
    val=FastPairs(base,'validation','FAST')
    stream=BatchStream(data,cfg['seed'])
    model=build_joint(lambda_noise,lambda_recon).to(device)
    assert len(data)==1145 and len(val)==64
    assert sum(p.numel() for p in model.parameters())==13131010
    opt=torch.optim.Adam(model.parameters(),lr=2e-4,betas=(.9,.99),weight_decay=1e-9)
    ema=EMA(model,.995)
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    return data,val,stream,model,opt,ema

def update(stream,model,opt,ema,device,step,calibration=False):
    if device.type=='cuda':torch.cuda.synchronize(device)
    start=time.perf_counter()
    model.train()
    opt.zero_grad(set_to_none=True)
    components=np.zeros(4,dtype=np.float64)
    metadata=[]
    for micro in range(1):
        ldct,target,meta=stream.next(2)
        assert all(item['patient_id'] in FAST_TRAIN for item in meta)
        assert ldct.shape==target.shape==(2,1,512,512)
        assert ldct.dtype==target.dtype==torch.float32
        ldct=ldct.to(device);target=target.to(device)
        assert torch.isfinite(ldct).all() and torch.isfinite(target).all()
        assert ldct.min()>=0 and ldct.max()<=1 and target.min()>=0 and target.max()<=1
        total,res,noise,recon=model([target,ldct])
        if calibration and step==0 and micro==0:
            # Positive provisional scale for calibration; final lambdas use
            # only the median of optimizer steps 51..100, then are frozen.
            lnoise=float(.2*res.detach()/noise.detach().clamp_min(1e-12))
            lrecon=float(.1*res.detach()/recon.detach().clamp_min(1e-12))
            model.set_loss_weights(lnoise,lrecon)
        total=res+model.lambda_noise*noise+model.lambda_recon*recon
        values=(total,res,noise,recon)
        if not all(torch.isfinite(v).all() for v in values):
            raise FloatingPointError('Nonfinite loss at optimizer step {}'.format(step+1))
        total.backward()
        components += np.array([float(v.detach()) for v in values])
        metadata.extend(meta)
    if any(p.grad is None for p in model.parameters()):
        raise RuntimeError('Missing gradient at optimizer step {}'.format(step+1))
    grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.0))
    if not math.isfinite(grad) or grad>100:
        raise FloatingPointError('Abnormal gradient at optimizer step {}: {}'.format(step+1,grad))
    opt.step()
    ema.update(model)
    if device.type=='cuda':torch.cuda.synchronize(device)
    elapsed=time.perf_counter()-start
    record={'optimizer_step':step+1,**dict(zip(PARTS,components.tolist())),
            'gradient_norm_pre_clip':grad,'learning_rate':opt.param_groups[0]['lr'],
            'seconds_per_step':elapsed,
            'peak_vram_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0,
            'ema_updates':ema.updates,'physical_batch':2,'gradient_accumulation':1,
            'effective_batch':2,'weighted_noise_loss':model.lambda_noise*components[2],
            'weighted_recon_loss':model.lambda_recon*components[3],
            'samples':metadata}
    assert ema.updates==step+1
    return record

def payload(cfg,model,opt,ema,stream,step,history,calibration):
    return {'model_state_dict':model.state_dict(),'ema_state_dict':ema.model.state_dict(),
            'optimizer_state_dict':opt.state_dict(),'optimizer_step':step,
            'ema_updates':ema.updates,'rng':rng_state(),'data_stream':stream.state_dict(),
            'config':cfg,'source_hashes':hashes(),'validation_history':history,
            'calibration_sha256':sha256(calibration)}

def save_csv(path,rows):
    if not rows:return
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
        file.flush();os.fsync(file.fileno())
    os.replace(temp,path)

def run_calibration(args,base,cfg,device):
    output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'CONFIG.json',cfg)
    guard=DataAccessGuard(base,'FAST')
    with guard:
        data,val,stream,model,opt,ema=initialize(base,cfg,device,0.0,0.0)
        records=[]
        with (output/'calibration_train.jsonl').open('x',buffering=1) as log:
            for step in range(100):
                row=update(stream,model,opt,ema,device,step,calibration=True)
                records.append(row)
                log.write(json.dumps(row,allow_nan=False)+'\n')
                if (step+1)%10==0:
                    print('CALIBRATION {}'.format(step+1),flush=True)
        late=records[50:]
        med={key:statistics.median(r[key] for r in late) for key in PARTS[1:]}
        assert all(math.isfinite(x) and x>0 for x in med.values())
        ln=.2*med['loss_res']/med['loss_noise']
        lr=.1*med['loss_res']/med['loss_recon']
        assert all(math.isfinite(x) and x>0 for x in (ln,lr))
        weighted={'res':med['loss_res'],'noise':ln*med['loss_noise'],'recon':lr*med['loss_recon']}
        denominator=sum(weighted.values())
        access=guard.report()
        assert set(access['training_patients_opened'])==set(FAST_TRAIN)
        assert access['test_opens']==access['external_memory_opens']==access['blocked_attempts']==0
        result={'status':'PASS','optimizer_steps':100,'median_optimizer_steps':[51,100],
                'median_unweighted_losses':med,'lambda_noise':ln,'lambda_recon':lr,
                'weighted_contributions':weighted,
                'weighted_contribution_fractions':{k:v/denominator for k,v in weighted.items()},
                'gradient_norm_min':min(r['gradient_norm_pre_clip'] for r in records),
                'gradient_norm_max':max(r['gradient_norm_pre_clip'] for r in records),
                'gradient_finite':True,'loss_finite':True,
                'ema_updates':ema.updates,'parameter_count':sum(p.numel() for p in model.parameters()),
                'peak_vram_bytes':max(r['peak_vram_bytes'] for r in records),
                'seconds_per_optimizer_step':statistics.mean(r['seconds_per_step'] for r in records),
                'train_patients':list(FAST_TRAIN),'data_access':access,
                'source_hashes':hashes(),'base_config_sha256':sha256(BASE_CONFIG),
                'config_sha256':sha256(HERE/'CONFIG.json')}
    write_json(output/'CALIBRATION.json',result)
    write_json(output/'DATA_ACCESS.json',access)
    print('CALIBRATION_PASS '+json.dumps({k:result[k] for k in ('lambda_noise','lambda_recon','gradient_norm_max','peak_vram_bytes')}),flush=True)

def run_formal(args,base,cfg,device):
    calibration=Path(args.calibration).resolve()
    cal=read_calibration(calibration)
    output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=bool(args.resume))
    if not args.resume and any(output.iterdir()):
        raise FileExistsError('Fresh output required: '+str(output))
    write_json(output/'CONFIG.json',{**cfg,'joint_loss':{**cfg['joint_loss'],
               'lambda_noise':cal['lambda_noise'],'lambda_recon':cal['lambda_recon']}})
    guard=DataAccessGuard(base,'FAST')
    with guard:
        data,val,stream,model,opt,ema=initialize(base,cfg,device,cal['lambda_noise'],cal['lambda_recon'])
        history=[]
        start_step=0
        if args.resume:
            ck=torch.load(args.resume,map_location='cpu',weights_only=False)
            assert ck['config']==cfg and ck['source_hashes']==hashes()
            assert ck['calibration_sha256']==sha256(calibration)
            model.load_state_dict(ck['model_state_dict'],strict=True)
            ema.model.load_state_dict(ck['ema_state_dict'],strict=True)
            opt.load_state_dict(ck['optimizer_state_dict'])
            stream.load_state_dict(ck['data_stream'])
            start_step=ck['optimizer_step'];ema.updates=ck['ema_updates']
            history=ck['validation_history']
            assert start_step<8000 and ema.updates==start_step
            restore_rng(ck['rng'])
        cpdir=output/'checkpoints';cpdir.mkdir(exist_ok=bool(args.resume))
        mode='a' if args.resume else 'x'
        with (output/'train.jsonl').open(mode,buffering=1) as log:
            for step in range(start_step,8000):
                row=update(stream,model,opt,ema,device,step)
                log.write(json.dumps(row,allow_nan=False)+'\n')
                if (step+1)%100==0:
                    write_json(output/'progress.json',{'status':'RUNNING','optimizer_step':step+1,
                        'ema_updates':ema.updates,'peak_vram_bytes':row['peak_vram_bytes'],
                        'seconds_per_optimizer_step':row['seconds_per_step']})
                    print('TRAIN {} loss={:.8f} grad={:.5f}'.format(step+1,row['loss_total'],row['gradient_norm_pre_clip']),flush=True)
                if (step+1)%2000==0:
                    metric=validate(ema.model,val,base,device,output/('validation_step_{}_slices.json'.format(step+1)))
                    assert metric['slices']==64 and metric['weights']=='ema'
                    entry={'step':step+1,**metric,'baseline_psnr':BASE_PSNR[step+1],
                           'delta_psnr':metric['psnr']-BASE_PSNR[step+1],
                           'train_loss':row['loss_total'],'loss_res':row['loss_res'],
                           'loss_noise':row['loss_noise'],'loss_recon':row['loss_recon'],
                           'weighted_noise_loss':row['weighted_noise_loss'],
                           'weighted_recon_loss':row['weighted_recon_loss'],
                           'gradient_norm_pre_clip':row['gradient_norm_pre_clip'],
                           'learning_rate':row['learning_rate'],'peak_vram_bytes':row['peak_vram_bytes'],
                           'seconds_per_step':row['seconds_per_step']}
                    history.append(entry)
                    save_csv(output/'validation_metrics.csv',history)
                    path=cpdir/('step_{}.pt'.format(step+1))
                    if path.exists():raise FileExistsError(path)
                    atomic_save(path,payload(cfg,model,opt,ema,stream,step+1,history,calibration))
                    write_json(output/'checkpoint_manifest.json',
                               {'checkpoints':[{'step':r['step'],'path':str(cpdir/('step_{}.pt'.format(r['step']))),
                                  'sha256':sha256(cpdir/('step_{}.pt'.format(r['step'])))} for r in history]})
                    print('VALIDATION '+json.dumps(entry,allow_nan=False),flush=True)
                if (step+1)%1000==0:
                    atomic_save(cpdir/'latest.pt',payload(cfg,model,opt,ema,stream,step+1,history,calibration))
        access=guard.report()
        assert access['validation_indices_opened']==list(FAST_INDICES)
        assert set(access['training_patients_opened'])==set(FAST_TRAIN)
        assert access['test_opens']==access['external_memory_opens']==access['blocked_attempts']==0
    best=max(history,key=lambda r:r['psnr'])
    result={'status':'COMPLETE','optimizer_steps':8000,'ema_updates':ema.updates,
            'history':history,'best_step':best['step'],'best_psnr':best['psnr'],
            'best_delta_psnr':best['delta_psnr'],'eight_k_delta_psnr':history[-1]['delta_psnr'],'data_access':access,
            'parameter_count':sum(p.numel() for p in model.parameters()),
            'peak_vram_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0,
            'calibration':str(calibration),'calibration_sha256':sha256(calibration),
            'source_hashes':hashes()}
    write_json(output/'DATA_ACCESS.json',access)
    write_json(output/'FINAL_TRAIN_SUMMARY.json',result)
    write_json(output/'progress.json',{'status':'COMPLETE','optimizer_step':8000,'best_step':best['step']})
    print('FORMAL_COMPLETE '+json.dumps({'best_step':best['step'],'best_psnr':best['psnr'],'best_delta_psnr':best['delta_psnr']}),flush=True)
    if history[-1]['delta_psnr']>=.05:
        full=output/'full_l333'
        full.mkdir(exist_ok=False)
        checkpoint=cpdir/'step_8000.pt'
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
        full_model=build_joint(cal['lambda_noise'],cal['lambda_recon']).to(device)
        full_model.load_state_dict(ck['ema_state_dict'],strict=True)
        full_guard=DataAccessGuard(base,'FULL',allow_training=False)
        with full_guard:
            full_data=FastPairs(base,'validation','FULL')
            full_cfg=copy.deepcopy(base)
            full_cfg['evaluation']['expected_slices']=610
            full_cfg['evaluation']['validation_mode']='FULL'
            full_metric=validate(full_model,full_data,full_cfg,device,full/'validation_slices.json')
        full_access=full_guard.report()
        assert full_metric['slices']==610 and full_access['test_opens']==full_access['external_memory_opens']==0
        write_json(full/'metrics.json',{'step':8000,**full_metric})
        write_json(full/'DATA_ACCESS.json',full_access)
        print('FULL_L333 '+json.dumps({'step':8000,**full_metric}),flush=True)
    report=['# RDDM residual/noise joint Fast-dev V1','',
            'One canonical shared RLEB U-Net plus a 65-parameter noise head. Physical batch 2, accumulation 2, effective batch 4.',
            'The 100 optimizer-step calibration is separate from the fresh 8k run.','',
            '| Step | PSNR | Delta vs matched Fast baseline | SSIM | RMSE HU | MAE HU |',
            '|---:|---:|---:|---:|---:|---:|']
    for r in history:
        report.append('| {} | {:.6f} | {:+.6f} | {:.6f} | {:.4f} | {:.4f} |'.format(
            r['step'],r['psnr'],r['delta_psnr'],r['ssim'],r['rmse_hu'],r['mae_hu']))
    report += ['', 'Best step: {}. Best matched delta: {:+.6f} dB.'.format(best['step'],best['delta_psnr'])]
    if history[-1]['delta_psnr']<.02:report.append('Decision: stop this direction.')
    elif history[-1]['delta_psnr']<.05:report.append('Decision: record, no additional optimization.')
    else:report.append('Decision: Full-L333 triggered; no further sweep.')
    report.append('L506 untouched; baseline source, split and evaluator unchanged.')
    (output/'REPORT.md').write_text('\n'.join(report)+'\n')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=('calibrate','formal'),required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--calibration')
    parser.add_argument('--resume')
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    base,cfg=configuration()
    device=torch.device(args.device)
    if args.phase=='calibrate':
        assert not args.resume and not args.calibration
        run_calibration(args,base,cfg,device)
    else:
        assert args.calibration
        run_formal(args,base,cfg,device)

if __name__=='__main__':main()
