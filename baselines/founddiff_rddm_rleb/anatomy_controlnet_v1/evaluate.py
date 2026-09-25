"""Frozen Fast64 deterministic sampling and two HU windows."""
import contextlib
import math
import os
import time
import numpy as np
import torch
from skimage.metrics import structural_similarity
from fast_dev.protocol import FAST_INDICES
from context_2p5d.model import restored_slice, training_loss
from mayo2016_smallset_v1.frozen_protocol import slice_metrics, normalized_to_hu

def soft_metrics(pred,target,data):
    p=normalized_to_hu(pred.detach().float().cpu(),data).clamp(-160.,240.)
    g=normalized_to_hu(target.detach().float().cpu(),data).clamp(-160.,240.)
    p01=(p+160.)/400.;g01=(g+160.)/400.
    mse=float((p01-g01).square().mean())
    psnr=10.*math.log10(1./mse) if mse else float("inf")
    ssim=float(structural_similarity(g01.squeeze().numpy().astype(np.float64),
        p01.squeeze().numpy().astype(np.float64),win_size=7,gradient=False,
        data_range=1.,channel_axis=None,gaussian_weights=False,full=False,
        use_sample_covariance=True,K1=.01,K2=.03))
    delta=p-g
    return {"soft_psnr":psnr,"soft_ssim":ssim,
            "soft_rmse_hu":math.sqrt(float(delta.square().mean())),
            "soft_mae_hu":float(delta.abs().mean())}

@torch.no_grad()
def validate(diffusion,dataset,cfg,device,activity=False):
    assert dataset.mode=="FAST" and len(dataset)==64
    assert tuple((r[0],r[1]) for r in dataset.records)==tuple(("L333",i) for i in FAST_INDICES)
    diffusion.eval()
    u=diffusion.model.unet0
    u.collect_activity=activity if hasattr(u,"collect_activity") else False
    rows=[];activities=[];started=time.perf_counter()
    devices=[device.index or 0] if device.type=="cuda" else []
    with torch.random.fork_rng(devices=devices):
        with open(os.devnull,"w") as quiet,contextlib.redirect_stderr(quiet):
            for pos in range(64):
                prev,center,next_,target,meta=dataset.load(pos)
                prev=torch.from_numpy(prev).unsqueeze(0).to(device)
                center=torch.from_numpy(center).unsqueeze(0).to(device)
                next_=torch.from_numpy(next_).unsqueeze(0).to(device)
                target=torch.from_numpy(target).unsqueeze(0).to(device)
                seed=cfg["evaluation"]["validation_seed"]+meta["slice_index"]
                torch.manual_seed(seed)
                if device.type=="cuda":torch.cuda.manual_seed(seed)
                pred=restored_slice(diffusion,center,prev,next_)
                assert pred.shape==(1,1,512,512) and bool(torch.isfinite(pred).all())
                original=slice_metrics(pred,target,cfg)
                soft=soft_metrics(pred,target,cfg["data"])
                if activity:activities.append(u.activity.copy())
                loss_seed=seed+cfg["evaluation"]["loss_seed_offset"]
                torch.manual_seed(loss_seed)
                if device.type=="cuda":torch.cuda.manual_seed(loss_seed)
                loss=float(training_loss(diffusion,target,center,prev,next_))
                row={"patient_id":"L333","slice_index":meta["slice_index"],
                     **{k:original[k] for k in ("psnr","ssim","rmse_hu","mae_hu")},
                     **soft,"validation_loss":loss}
                assert all(math.isfinite(float(v)) for k,v in row.items() if k not in ("patient_id","slice_index"))
                rows.append(row)
    if hasattr(u,"collect_activity"):u.collect_activity=False
    keys=("psnr","ssim","rmse_hu","mae_hu","soft_psnr","soft_ssim",
          "soft_rmse_hu","soft_mae_hu","validation_loss")
    summary={k:float(np.mean([r[k] for r in rows])) for k in keys}
    summary.update(patient="L333",slices=64,weights="ema",mode="FAST",
                   validation_seconds=time.perf_counter()-started)
    avg=None
    if activities:
        avg={}
        for key in activities[0]:
            if activities[0][key] is None:avg[key]=None
            else:avg[key]=np.mean([a[key] for a in activities],axis=0).tolist()
    return summary,rows,avg
