"""The frozen L333 metric and seed protocol for 2.5D context."""
import contextlib
import math
import os
import time
import numpy as np
import torch
from mayo2016_smallset_v1.runtime import write_json, METRICS
from mayo2016_smallset_v1.frozen_protocol import slice_metrics
from fast_dev.protocol import validation_indices
from .model import restored_slice, training_loss


@torch.no_grad()
def validate(model,dataset,cfg,device,output=None):
    expected=validation_indices(dataset.mode)
    assert tuple((r[0],r[1]) for r in dataset.records)==tuple(("L333",i) for i in expected)
    model.eval();rows=[];started=time.perf_counter()
    devices=[device.index or 0] if device.type=="cuda" else []
    with torch.random.fork_rng(devices=devices):
        with open(os.devnull,"w") as quiet,contextlib.redirect_stderr(quiet):
            for pos in range(len(dataset)):
                prev,center,next_,target,meta=dataset.load(pos)
                prev=torch.from_numpy(prev).unsqueeze(0).to(device)
                center=torch.from_numpy(center).unsqueeze(0).to(device)
                next_=torch.from_numpy(next_).unsqueeze(0).to(device)
                target=torch.from_numpy(target).unsqueeze(0).to(device)
                seed=cfg["evaluation"]["validation_seed"]+meta["slice_index"]
                torch.manual_seed(seed)
                if device.type=="cuda":torch.cuda.manual_seed(seed)
                prediction=restored_slice(model,center,prev,next_)
                assert prediction.shape==(1,1,512,512) and bool(torch.isfinite(prediction).all())
                metrics=slice_metrics(prediction,target,cfg)
                assert all(math.isfinite(metrics[k]) for k in METRICS)
                loss_seed=seed+cfg["evaluation"]["loss_seed_offset"]
                torch.manual_seed(loss_seed)
                if device.type=="cuda":torch.cuda.manual_seed(loss_seed)
                loss=float(training_loss(model,target,center,prev,next_))
                assert math.isfinite(loss)
                rows.append({**meta,**{k:metrics[k] for k in METRICS},
                             "validation_loss":loss,"sampling_seed":seed})
    if device.type=="cuda":torch.cuda.synchronize(device)
    if output is not None:write_json(output,rows)
    result={k:float(np.mean([r[k] for r in rows])) for k in (*METRICS,"validation_loss")}
    return {**result,"slices":len(rows),"patient":"L333","weights":"ema",
            "validation_mode":dataset.mode,"validation_seconds":time.perf_counter()-started}
