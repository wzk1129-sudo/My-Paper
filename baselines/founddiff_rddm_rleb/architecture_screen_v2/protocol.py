"""Frozen-protocol contract and matched reference for Architecture Fast Screen V2."""
import copy
import csv
import hashlib
import json
import subprocess
from pathlib import Path
import torch
from fast_dev.protocol import CONFIG, FAST_INDICES, FAST_TRAIN, protocol as frozen_protocol, check_config
from mayo2016_smallset_v1.runtime import ROOT, seed_everything, rng_state, restore_rng, sha256
from src.rddm_rleb import build_baseline, ResnetBlock
from .variants import ARMS, build_candidate

HERE = ROOT / 'experiments/architecture_screen_v2'
STEPS = (2000,4000,6000,8000,10000,12000)
SOURCE_PIN = 'e8ed93223fbdf4a0e2521f73c7a4f213561f6a191a1a7b2a5f95934b85ebf574'
BASE_HISTORY = ROOT / 'experiments/rddm_vs_ldct_regression_v1/CONTROL_HISTORY.csv'
BASE_FULL = ROOT / 'experiments/rddm_vs_ldct_regression_v1/FULL_L333_PER_SLICE.csv'

def config():
    frozen_protocol()
    base = json.loads(CONFIG.read_text())
    check_config(base)
    assert sha256(ROOT/'src/rddm_rleb.py') == SOURCE_PIN
    cfg = copy.deepcopy(base)
    cfg['experiment_name'] = 'RDDM_Architecture_Fast_Screen_V2'
    cfg['description'] = 'Four independent RDDM backbone additions on frozen Mayo Fast protocol'
    cfg['training']['max_optimizer_steps'] = 12000
    cfg['training']['matched_checkpoint_steps'] = list(STEPS)
    cfg['safety']['maximum_steps'] = 12000
    assert {k:v for k,v in cfg.items() if k not in ('experiment_name','description','training','safety')} == {k:v for k,v in base.items() if k not in ('experiment_name','description','training','safety')}
    for key in base['training']:
        if key not in ('max_optimizer_steps','matched_checkpoint_steps'):
            assert cfg['training'][key] == base['training'][key]
    for key in base['safety']:
        if key != 'maximum_steps': assert cfg['safety'][key] == base['safety'][key]
    assert tuple(base['data']['train_patients']) == FAST_TRAIN
    assert tuple(base['evaluation']['validation_seed']+i for i in FAST_INDICES)
    return cfg,base

def references():
    with BASE_HISTORY.open(newline='') as f: rows=list(csv.DictReader(f))
    assert [int(r['step']) for r in rows] == list(STEPS)
    assert all(r['weights']=='ema' and r['patient']=='L333' and r['slices']=='64' for r in rows)
    return {int(r['step']): dict({k:float(r[k]) for k in ('psnr','ssim','rmse_hu','mae_hu','validation_loss')},train_stream_sha256=r['train_stream_sha256']) for r in rows}

def source_manifest():
    names=['src/rddm_rleb.py','configs/rddm_fast_dev.json','configs/fast_dev_protocol.json',
      'splits/L333_fast64.txt','splits/fast_train_2patients.txt',
      'fast_dev/data.py','fast_dev/evaluate.py','fast_dev/protocol.py',
      'mayo2016_smallset_v1/runtime.py','mayo2016_smallset_v1/frozen_protocol.py',
      'experiments/rddm_vs_ldct_regression_v1/CONTROL_HISTORY.csv',
      'experiments/rddm_vs_ldct_regression_v1/FULL_L333_PER_SLICE.csv']
    names += ['architecture_screen_v2/variants.py','architecture_screen_v2/protocol.py',
              'architecture_screen_v2/runtime.py','architecture_screen_v2/train.py']
    return {name:sha256(ROOT/name) for name in names}

def state_hash(state):
    digest=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        digest.update(name.encode());digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

def construct(arm, device='cpu'):
    cfg,_=config(); seed=cfg['seed']
    seed_everything(seed)
    canonical=build_baseline()
    canonical_rng=rng_state()
    candidate=build_candidate(arm,seed)
    common=canonical.state_dict(); observed=candidate.state_dict()
    assert all(name in observed and torch.equal(tensor,observed[name]) for name,tensor in common.items())
    assert len(common)==106
    new=[name for name in observed if name not in common]
    assert new and all('.variant_modules.' in name for name in new)
    assert sum(isinstance(m,ResnetBlock) for m in candidate.modules())==10
    assert candidate.objective=='pred_res' and candidate.num_timesteps==1000
    # Reset after all candidate-only parameter initializers have run.
    restore_rng(canonical_rng)
    audit={'arm':arm,'seed':seed,'canonical_state_sha256':state_hash(common),
      'shared_tensor_count':len(common),'shared_bitwise_identical':True,
      'new_parameter_keys':new,'new_parameter_count':sum(p.numel() for n,p in candidate.named_parameters() if '.variant_modules.' in n),
      'total_parameter_count':sum(p.numel() for p in candidate.parameters()),
      'rleb_count':10,'canonical_rng_restored_before_training':True,
      'baseline_source_sha256':SOURCE_PIN}
    return candidate.to(device),audit
