"""Exact fast protocol plus a single allowlisted variant field."""
import copy,json,hashlib
from pathlib import Path
from fast_dev.protocol import CONFIG,ROOT,FAST_TRAIN,FAST_INDICES,protocol as frozen_protocol,source_hashes as baseline_hashes
from mayo2016_smallset_v1.runtime import sha256

SPECS={
 'F1_noise_050':{'kind':'F1','value':.5,'main_change':'q_sample noise amplitude 0.5'},
 'F1_noise_025':{'kind':'F1','value':.25,'main_change':'q_sample noise amplitude 0.25'},
 'F1_noise_000':{'kind':'F1','value':0.,'main_change':'q_sample noise amplitude 0'},
 'F2_multiscale_ldct':{'kind':'F2','value':None,'main_change':'zero-gated four-level LDCT conditioning'},
 'F3_rleb2conv':{'kind':'F3','value':None,'main_change':'second WSConv3x3+GroupNorm in each RLEB'},
 'F4_window_010':{'kind':'F4','value':.1,'main_change':'metric-window loss lambda 0.1'},
 'F4_window_025':{'kind':'F4','value':.25,'main_change':'metric-window loss lambda 0.25'},
 'F4_window_050':{'kind':'F4','value':.5,'main_change':'metric-window loss lambda 0.5'},
}
CANDIDATES=ROOT/'experiments/fast_dev/candidates'
BASELINE=json.loads(CONFIG.read_text())

def canonical_checks():
    frozen_protocol()
    pins=json.loads((ROOT/'experiments/fast_dev/formal_source_pins.json').read_text())
    for name,expected in pins.items():
        assert sha256(ROOT/name)==expected,'formal source modified: '+name
    baseline=json.loads(CONFIG.read_text())
    assert baseline==BASELINE,'fast config modified'
    for name,expected in baseline_hashes().items():
        assert sha256(ROOT/name)==expected,'fast dependency modified: '+name
    assert baseline['data']['train_patients']==['L067','L143']
    assert baseline['evaluation']['expected_slices']==64
    assert baseline['diffusion']['sampling_timesteps']==2 and baseline['diffusion']['sum_scale']==.01
    return True

def artifact_pins():
    pins=json.loads((CANDIDATES/"protected_artifacts.json").read_text())
    for name,expected in pins.items():
        assert sha256(Path(name))==expected, "protected baseline artifact changed: "+name
    return pins

def variant_config(job):
    assert job in SPECS
    cfg=copy.deepcopy(BASELINE)
    cfg['experiment_name']='FoundDiff_RDDM_RLEB_fast_screen_v1_'+job
    cfg['variant']={'id':job,**SPECS[job]}
    return cfg

def check_variant(cfg):
    job=cfg['variant']['id'];assert job in SPECS
    expected=variant_config(job)
    assert cfg==expected,'config differs from frozen protocol or specified variant'
    canonical_checks()

def variant_hashes(job):
    paths=[ROOT/'fast_dev_variants'/n for n in ('__init__.py','model.py','protocol.py','runtime.py','candidate.py','queue.py','check_variants.py')]
    paths +=[CANDIDATES/job/'CONFIG.json',CANDIDATES/'protected_artifacts.json']
    base=baseline_hashes()
    return {**base,**{str(p.relative_to(ROOT)):sha256(p) for p in paths}}
