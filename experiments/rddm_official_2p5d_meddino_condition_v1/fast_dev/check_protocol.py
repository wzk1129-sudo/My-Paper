"""Cheap contract tests: no image read, no training, no test filesystem probe."""
import copy
import json
import tempfile
from pathlib import Path
from .protocol import CONFIG, FAST_TRAIN, MEMORY, FAST_INDICES, check_config, validation_indices
from .data import FastPairs, ExternalMemory, DataAccessGuard
from .runtime import FastState
from .evaluate import slice_metrics
from .reporting import plateau,promotion
from mayo2016_smallset_v1.runtime import TrainingState
from mayo2016_smallset_v1.data import CenterPairs
from mayo2016_smallset_v1.frozen_protocol import slice_metrics as formal_metrics

def must_reject(fn):
    try:fn()
    except (ValueError,AssertionError):return
    raise AssertionError('Invalid input accepted')

def run():
    cfg=json.loads(CONFIG.read_text());check_config(cfg)
    assert FastState.update is TrainingState.update
    assert FastState.resume is TrainingState.resume and FastState.payload is TrainingState.payload
    assert FastPairs.load is CenterPairs.load and slice_metrics is formal_metrics
    assert len(FAST_INDICES)==len(set(FAST_INDICES))==64 and FAST_INDICES[0]==0 and FAST_INDICES[-1]==609
    assert set(FAST_INDICES)<set(validation_indices('FULL'))
    memory=ExternalMemory(cfg);pairs=list(memory.iter_pairs())
    assert len(pairs)==3655 and {x['patient'] for x in pairs}==set(MEMORY)
    for patient in list(FAST_TRAIN)+['L333']:
        must_reject(lambda:memory.pair_paths(patient,0))
    must_reject(lambda:memory.pair_paths('not-an-approved-patient',0))
    must_reject(lambda:FastPairs(cfg,'test'))
    must_reject(lambda:FastPairs(cfg,'external_memory'))
    changed=copy.deepcopy(cfg);changed['data']['train_patients']=list(MEMORY[:2])
    must_reject(lambda:check_config(changed))
    changed=copy.deepcopy(cfg);changed['data']['hu_max']=2000
    must_reject(lambda:check_config(changed))
    # Header/path checks only, all explicitly approved patient/index paths.
    with DataAccessGuard(cfg) as guard:
        train=FastPairs(cfg,'train');fast=FastPairs(cfg,'validation','FAST');full=FastPairs(cfg,'validation','FULL')
    assert len(train)==1145 and len(fast)==64 and len(full)==610 and guard.report()['opens']==0
    row=lambda step,psnr:{'step':step,'psnr':psnr,'validation_mode':'FAST','slices':64,'ssim':.95,'rmse_hu':19.,'mae_hu':12.}
    assert plateau([row(6000,40),row(8000,40.009)])['status']=='PLATEAU'
    assert plateau([row(6000,40),row(8000,40.02)])['status']=='NOT_PLATEAU'
    assert plateau([row(50,30)])['status']=='INSUFFICIENT_EVIDENCE'
    assert promotion([row(6000,40.06),row(8000,40.07)],[row(6000,40),row(8000,40)])['status']=='PROMOTE_TO_L333_FULL'
    assert promotion([row(6000,39.99),row(8000,40.07)],[row(6000,40),row(8000,40)])['status']=='KEEP_IN_DEVELOPMENT'
    assert promotion([row(6000,40.01),row(8000,40.05)],[row(6000,40),row(8000,40)])['status']=='PROMOTE_TO_L333_FULL'
    bad=[row(6000,40.01),row(8000,40.05)];bad[-1]['ssim']=.94
    assert promotion(bad,[row(6000,40),row(8000,40)])['status']=='REVIEW_OTHER_METRICS'
    assert all(x['patient_id']==x['patient'] and not x['eligible_as_query_supervision'] for x in pairs)
    return {'status':'PASS','formal_update_resume_payload_reused':True,'formal_load_metrics_reused':True,
            'train_pairs':len(train),'fast_slices':len(fast),'full_slices':len(full),'external_memory_pairs':len(pairs),
            'invalid_roles_rejected_before_io':True,'images_read':0,'plateau_and_promotion_cases':'PASS'}

if __name__=='__main__':print(json.dumps(run(),indent=2))
