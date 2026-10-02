"""Bounded post-smoke checks: exact resume, paired transforms, deterministic L333."""
import argparse
import ast
import json
from pathlib import Path
import numpy as np
import torch
from .data import CenterPairs,BatchStream
from .runtime import TrainingState,validate,source_hashes,sha256,write_json,rng_state
from . import frozen_protocol

def nested_equal(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(nested_equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(nested_equal(x,y) for x,y in zip(a,b))
    return a==b

def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',required=True);p.add_argument('--resumed',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0')
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    a=torch.load(str(Path(args.smoke)/'checkpoints/step_20.pt'),map_location='cpu');b=torch.load(str(Path(args.resumed)/'checkpoints/step_20.pt'),map_location='cpu')
    exact_keys=('model_state_dict','ema_state_dict','optimizer_state_dict','optimizer_step','rng','data_stream','ema_updates')
    equality={key:nested_equal(a[key],b[key]) for key in exact_keys};assert all(equality.values()),equality
    cfg=a['config'];state=TrainingState(cfg,args.device)
    state.model.load_state_dict(a['model_state_dict']);state.ema.model.load_state_dict(a['ema_state_dict'])
    before=rng_state()
    first=validate(state.ema.model,state.validation_data,cfg,state.device,limit=2,output=out/'validation_repeat1.json')
    after=rng_state();assert nested_equal(before,after),'Validation changed training RNG'
    second=validate(state.ema.model,state.validation_data,cfg,state.device,limit=2,output=out/'validation_repeat2.json')
    assert all(first[k]==second[k] for k in ('psnr','ssim','rmse_hu','mae_hu','validation_loss'))
    dataset=CenterPairs(cfg,'train');stream=BatchStream(dataset,cfg['seed']);ldct,ndct,metadata=stream.next()
    # Compare both tensors against the same explicit transformation from original center pairs.
    for j,m in enumerate(metadata):
        pos=next(i for i,r in enumerate(dataset.records) if r[:2]==(m['patient_id'],m['slice_index']))
        x,y,_=dataset.load(pos)
        for raw,actual in [(x,ldct[j].numpy()),(y,ndct[j].numpy())]:
            if m['horizontal_flip']:raw=np.flip(raw,axis=-1)
            expected=np.rot90(raw,m['rotation_degrees']//90,axes=(-2,-1))
            assert np.array_equal(expected,actual)
    try:CenterPairs(cfg,'test')
    except ValueError:pass
    else:raise AssertionError('Test split accepted')
    provenance=json.loads((Path(__file__).parent/'protocol_provenance.json').read_text())
    adapted=ast.parse(Path(frozen_protocol.__file__).read_text())
    for info in provenance.values():
        original_path=Path(info['source']);assert sha256(original_path)==info['sha256']
        original=ast.parse(original_path.read_text())
        for name in info['functions']:
            old=next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name==name)
            new=next(n for n in adapted.body if isinstance(n,ast.FunctionDef) and n.name==name)
            assert ast.dump(old)==ast.dump(new),name
    result={'status':'PASS','exact_resume_comparison':equality,'optimizer_steps':20,'full_resolution':[2,1,512,512],'deterministic_validation_repeated':True,'training_rng_preserved':True,'paired_augmentation_exact':True,'test_split_rejected':True,'frozen_metrics_normalization_ast_identical':True,'source_hashes':source_hashes(),'validation_smoke_metrics_not_formal':first}
    write_json(out/'acceptance.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
