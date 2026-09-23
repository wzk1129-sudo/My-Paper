"""Read-only static and numerical equivalence checks before any data-backed run."""
import gc,json,torch
from src.rddm_rleb import build_baseline
from .protocol import SPECS,CANDIDATES,canonical_checks,variant_config,variant_hashes
from .model import build_variant,NoiseDiffusion,WindowDiffusion,MultiScaleUnet,TwoConvRLEB

def main():
    torch.set_num_threads(2);canonical_checks()
    rows={}
    for job,spec in SPECS.items():
        assert variant_config(job)==json.loads((CANDIDATES/job/'CONFIG.json').read_text())
        assert variant_hashes(job)
        model=build_variant(spec)
        count=sum(p.numel() for p in model.parameters())
        assert model.sampling_timesteps==2 and model.sum_scale==.01 and model.objective=='pred_res'
        assert model.model.unet0.init_conv.in_channels==2
        if spec['kind']=='F1':assert isinstance(model,NoiseDiffusion) and model.noise_scale==spec['value']
        if spec['kind']=='F4':assert isinstance(model,WindowDiffusion) and model.lambda_window==spec['value']
        if spec['kind']=='F2':assert isinstance(model.model.unet0,MultiScaleUnet)
        if spec['kind']=='F3':assert sum(isinstance(m,TwoConvRLEB) for m in model.modules())==10
        rows[job]={'parameters':count,'change':spec['main_change'],'source_sha256':variant_hashes(job)}
        del model;gc.collect()
    torch.manual_seed(1729);base=build_baseline()
    torch.manual_seed(1234);f2=build_variant(SPECS['F2_multiscale_ldct'])
    missing,unexpected=f2.load_state_dict(base.state_dict(),strict=False)
    assert not unexpected and all('cond_' in k for k in missing)
    x=torch.randn(1,2,64,64);t=torch.tensor([524.],dtype=torch.float32)
    with torch.no_grad():
        a=base.model.unet0(x,t);b=f2.model.unet0(x,t)
    difference=float((a-b).abs().max())
    assert difference<1e-6,difference
    rows['F2_zero_gate_max_abs_difference']=difference
    del base,f2;gc.collect()
    print(json.dumps(rows,indent=2),flush=True)
if __name__=='__main__':main()
