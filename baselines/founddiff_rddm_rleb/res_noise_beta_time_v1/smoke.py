"""Pre-training routing and numerical acceptance; no checkpoint is written."""
import argparse
import json
import math
import time
from pathlib import Path
import torch
from fast_dev.data import DataAccessGuard
from mayo2016_smallset_v1.runtime import seed_everything, write_json
from src.rddm_rleb import build_baseline
from .model import build_joint
from .run import configuration, initialize, update

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda:0')
    args=p.parse_args()
    output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    base,cfg=configuration()
    device=torch.device(args.device)
    torch.set_num_threads(2)
    seed_everything(cfg['seed'])
    canonical=build_baseline()
    seed_everything(cfg['seed'])
    candidate=build_joint()
    identical=all(torch.equal(v,candidate.state_dict()[k]) for k,v in canonical.state_dict().items())
    assert identical and not hasattr(candidate.model,'unet1')
    del canonical,candidate
    guard=DataAccessGuard(base,'FAST')
    with guard:
        data,val,stream,model,opt,ema=initialize(base,cfg,device,0.0,0.0)
        shared=model.model.unet0
        assert not hasattr(model.model,'unet1')
        assert sum(p.numel() for p in model.parameters())==13131010
        expected=[];seen=[];outputs=[]
        def outer_pre(_module,inputs):
            x,times=inputs[0],inputs[1]
            expected.append((x.data_ptr(),times[0].detach().clone(),times[1].detach().clone()))
        def shared_pre(module,inputs):
            seen.append((id(module),inputs[0].data_ptr(),inputs[1].detach().clone()))
        def outer_post(_module,inputs,result):
            outputs.append((inputs[0].detach(),result[0],result[1]))
        hooks=[model.model.register_forward_pre_hook(outer_pre),
               shared.register_forward_pre_hook(shared_pre),
               model.model.register_forward_hook(outer_post)]
        ldct,target,meta=stream.next(2)
        ldct=ldct.to(device);target=target.to(device)
        assert ldct.shape==target.shape==(2,1,512,512)
        assert all(x['patient_id'] in ('L067','L143') for x in meta)
        t=torch.tensor([0,999],device=device,dtype=torch.long)
        start=time.perf_counter()
        values=model.p_losses([target*2-1,ldct*2-1],t)
        assert len(values)==4 and all(torch.isfinite(v).all() for v in values)
        x_in,pred_res,pred_noise=outputs[-1]
        assert pred_res.shape==pred_noise.shape==(2,1,512,512)
        x0_hat=model.predict_start_from_res_noise(x_in[:,:1],t,pred_res,pred_noise)
        assert x0_hat.shape==(2,1,512,512) and torch.isfinite(x0_hat).all()
        assert len(expected)==1 and len(seen)==2
        ptr,alpha,beta=expected[0]
        assert torch.equal(alpha,model.alphas_cumsum[t]*model.num_timesteps)
        assert torch.equal(beta,model.betas_cumsum[t]*model.num_timesteps)
        assert seen[0][0]==seen[1][0]==id(shared)
        assert seen[0][1]==seen[1][1]==ptr
        assert torch.equal(seen[0][2],alpha) and torch.equal(seen[1][2],beta)
        assert not torch.equal(alpha,beta)
        (values[1]+.2*values[2]+.1*values[3]).backward()
        grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.0))
        assert math.isfinite(grad) and grad>0
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        opt.zero_grad(set_to_none=True)
        routing_seconds=time.perf_counter()-start
        expected.clear();seen.clear();outputs.clear()
        rows=[]
        for step in range(2):
            row=update(stream,model,opt,ema,device,step,calibration=True)
            rows.append(row)
            assert len(expected)==step+1 and len(seen)==2*(step+1)
            ptr,alpha,beta=expected[-1]
            a,b=seen[-2:]
            assert a[0]==b[0]==id(shared) and a[1]==b[1]==ptr
            assert torch.equal(a[2],alpha) and torch.equal(b[2],beta)
            assert all(torch.isfinite(z).all() for z in outputs[-1][1:])
        for h in hooks:h.remove()
        access=guard.report()
        assert access['test_opens']==access['external_memory_opens']==access['blocked_attempts']==0
        assert set(access['training_patients_opened'])=={'L067','L143'}
    result={'status':'PASS','formal_source_parameters_unchanged_at_initialization':identical,
            'shared_unet_id':id(shared),'unet_forward_calls_per_loss':2,
            'residual_receives_alpha_time':True,'noise_receives_beta_time':True,
            'same_input_and_parameters':True,'predictions_and_x0_finite':True,
            'loss_and_gradients_finite':True,'gradient_norm_pre_clip':grad,
            'parameter_count':13131010,'routing_forward_backward_seconds':routing_seconds,
            'optimizer_step_seconds':[r['seconds_per_step'] for r in rows],
            'peak_vram_bytes':max(r['peak_vram_bytes'] for r in rows),
            'ema_updates':ema.updates,'data_access':access,'no_checkpoint_written':True}
    write_json(output/'SMOKE.json',result)
    print('SMOKE_PASS '+json.dumps({k:result[k] for k in
        ('parameter_count','optimizer_step_seconds','peak_vram_bytes','ema_updates')}),flush=True)
if __name__=='__main__':main()
