"""The matched EMA evaluator, with only LDCT neighbor/anatomy context adaptation."""
import contextlib,os,time,math
import numpy as np
import torch
from fast_dev.protocol import validation_indices
from mayo2016_smallset_v1.frozen_protocol import slice_metrics
from mayo2016_smallset_v1.runtime import METRICS
from provenance import write_json
@torch.no_grad()
def validate(model,dataset,cfg,device,output=None,condition_mode="NORMAL"):
    expected=validation_indices(dataset.mode)
    assert tuple((r[0],r[1]) for r in dataset.records)==tuple(("L333",i) for i in expected)
    model.eval();u=model.model.unet0
    assert condition_mode in ("NORMAL","CONDITION_OFF","SHUFFLED_MEDDINO")
    assert condition_mode=="NORMAL" or u.anatomy
    if u.anatomy:
        for b in u.conditions.values():
            b.rows=[];b.record=True;b.force_off=condition_mode=="CONDITION_OFF"
    rows=[];started=time.perf_counter()
    devices=[device.index or 0] if device.type=="cuda" else []
    with torch.random.fork_rng(devices=devices):
        with open(os.devnull,"w") as quiet,contextlib.redirect_stderr(quiet):
            for pair_start in range(0,len(dataset),2):
                items=[dataset.load(p) for p in range(pair_start,min(pair_start+2,len(dataset)))]
                prepared=[]
                for prev,center,next_,target,meta in items:
                    prepared.append(([torch.from_numpy(x).unsqueeze(0).to(device) for x in (prev,center,next_,target)],meta))
                patches=None
                if condition_mode=="SHUFFLED_MEDDINO":
                    assert len(prepared)==2
                    patches=u.extract(torch.cat([x[0][1] for x in prepared],dim=0))
                for j,(tensors,meta) in enumerate(prepared):
                    prev,center,next_,target=tensors
                    supplied=None if patches is None else tuple(x[1-j:2-j] for x in patches)
                    u.set_context(prev,center,next_,supplied)
                    sampling_seed=cfg["evaluation"]["validation_seed"]+meta["slice_index"]
                    torch.manual_seed(sampling_seed)
                    if device.type=="cuda":torch.cuda.manual_seed(sampling_seed)
                    prediction=model.sample([center],last=True)[-1]
                    assert prediction.shape==(1,1,512,512) and torch.isfinite(prediction).all()
                    metrics=slice_metrics(prediction,target,cfg)
                    assert all(math.isfinite(metrics[k]) for k in METRICS)
                    if u.anatomy:
                        for b in u.conditions.values():b.record=False
                    torch.manual_seed(sampling_seed+cfg["evaluation"]["loss_seed_offset"])
                    if device.type=="cuda":torch.cuda.manual_seed(sampling_seed+cfg["evaluation"]["loss_seed_offset"])
                    loss=float(model([target,center])[0])
                    assert math.isfinite(loss)
                    if u.anatomy:
                        for b in u.conditions.values():b.record=True
                    rows.append({**meta,**{k:metrics[k] for k in METRICS},"validation_loss":loss,"sampling_seed":sampling_seed})
                    u.clear_context()
    if device.type=="cuda":torch.cuda.synchronize()
    if output is not None:write_json(output,rows)
    activity={}
    if u.anatomy:
        for s,b in u.conditions.items():
            assert len(b.rows)==len(dataset)*2
            activity[s]={k:float(np.mean([r[k] for r in b.rows])) for k in b.rows[0]}
            activity[s]["observations"]=len(b.rows)
            b.record=False;b.force_off=False;b.rows=[]
    summary={k:float(np.mean([r[k] for r in rows])) for k in (*METRICS,"validation_loss")}
    return {**summary,"slices":len(rows),"patient":"L333","weights":"ema","validation_mode":dataset.mode,
        "condition_mode":condition_mode,"validation_seconds":time.perf_counter()-started},activity
