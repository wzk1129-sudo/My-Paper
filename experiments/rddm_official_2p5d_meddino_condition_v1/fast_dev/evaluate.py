"""FAST and FULL use this same evaluator and the exact formal metric function."""
import argparse
import contextlib
import json
import math
import os
import time
import numpy as np
import torch
from mayo2016_smallset_v1.runtime import write_json, seed_everything, sha256, METRICS
from mayo2016_smallset_v1.frozen_protocol import slice_metrics
from src.rddm_rleb import build_baseline
from .protocol import CONFIG, check_config, validation_indices, output_directory, source_hashes
from .data import FastPairs, DataAccessGuard

@torch.no_grad()
def validate(model,dataset,cfg,device,output=None):
    expected=validation_indices(dataset.mode)
    assert tuple((r[0],r[1]) for r in dataset.records)==tuple(('L333',i) for i in expected)
    model.eval();rows=[];started=time.perf_counter()
    devices=[device.index or 0] if device.type=='cuda' else []
    with torch.random.fork_rng(devices=devices):
        with open(os.devnull,'w') as quiet,contextlib.redirect_stderr(quiet):
            for pos in range(len(dataset)):
                ldct,target,meta=dataset.load(pos)
                ldct=torch.from_numpy(ldct).unsqueeze(0).to(device)
                target=torch.from_numpy(target).unsqueeze(0).to(device)
                sampling_seed=cfg['evaluation']['validation_seed']+meta['slice_index']
                torch.manual_seed(sampling_seed)
                if device.type=='cuda':torch.cuda.manual_seed(sampling_seed)
                prediction=model.sample([ldct],last=True)[-1]
                assert prediction.shape==(1,1,512,512) and torch.isfinite(prediction).all()
                metrics=slice_metrics(prediction,target,cfg)
                if not all(math.isfinite(metrics[k]) for k in METRICS):raise FloatingPointError('Nonfinite validation metric')
                torch.manual_seed(sampling_seed+cfg['evaluation']['loss_seed_offset'])
                if device.type=='cuda':torch.cuda.manual_seed(sampling_seed+cfg['evaluation']['loss_seed_offset'])
                loss=float(model([target,ldct])[0])
                if not math.isfinite(loss):raise FloatingPointError('Nonfinite validation loss')
                rows.append({**meta,**{k:metrics[k] for k in METRICS},'validation_loss':loss,'sampling_seed':sampling_seed})
    if device.type=='cuda':torch.cuda.synchronize(device)
    if output is not None:write_json(output,rows)
    result={k:float(np.mean([r[k] for r in rows])) for k in (*METRICS,'validation_loss')}
    return {**result,'slices':len(rows),'patient':'L333','weights':'ema','validation_mode':dataset.mode,
        'validation_seconds':time.perf_counter()-started}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default=str(CONFIG));p.add_argument('--checkpoint',required=True)
    p.add_argument('--mode',choices=('FAST','FULL'),required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda:0');a=p.parse_args()
    cfg=json.loads(open(a.config).read());check_config(cfg)
    # Only checkpoints produced by this implementation and unchanged protocol.
    ck=torch.load(a.checkpoint,map_location='cpu')
    assert ck['config']==cfg and ck['source_hashes']==source_hashes()
    output=output_directory(a.output);device=torch.device(a.device)
    torch.set_num_threads(cfg['training']['cpu_threads']);seed_everything(cfg['seed'])
    model=build_baseline().to(device);model.load_state_dict(ck['ema_state_dict'],strict=True)
    with DataAccessGuard(cfg,a.mode,allow_training=False) as guard:
        dataset=FastPairs(cfg,'validation',a.mode)
        result=validate(model,dataset,cfg,device,output/'validation_slices.json')
    result.update(optimizer_step=ck['optimizer_step'],checkpoint_sha256=sha256(a.checkpoint),selection_changed=False)
    write_json(output/'metrics.json',result);write_json(output/'data_access.json',guard.report())
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
