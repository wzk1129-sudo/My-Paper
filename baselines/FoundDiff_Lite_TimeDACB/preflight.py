import ast,copy,hashlib,json,subprocess,sys,time
import torch,numpy,skimage,einops
from train import build,setup,OUT,write_json
from mayo_data import Pairs,Stream,CFG,VALID,access_report
from src.DADiff import TimeOnlyDACB
def main():
    setup()
    m=build().cuda();m.train()
    blocks=[(n,b) for n,b in m.named_modules() if isinstance(b,TimeOnlyDACB)]
    assert len(blocks)==9
    assert not any(any(t in type(b).__name__.lower() for t in ("attention","clip","mamba","ss2d")) for b in m.modules())
    assert m.channels==m.model.channels==m.model.out_dim==1 and m.condition
    assert m.model.unet0.init_conv.in_channels==4
    for n,b in blocks:
        x=torch.randn(2,b.norm.normalized_shape[0],9,11,device="cuda");t=torch.randn(2,256,device="cuda")
        assert torch.equal(b(x,t),x),n
    original=(OUT/"upstream_DADiff_c85a3a0.py").read_text()
    current=(OUT.parents[1]/"src/DADiff.py").read_text()
    def get(text,cls,name):
        tree=ast.parse(text)
        c=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
        return ast.dump(next(n for n in c.body if getattr(n,"name",None)==name),include_attributes=False)
    preserved=("q_sample","predict_noise_from_res","predict_start_from_xinput_noise","predict_start_from_res_noise",
               "q_posterior","q_posterior_from_res_noise","loss_fn")
    assert all(get(original,"ResidualDiffusion",n)==get(current,"ResidualDiffusion",n) for n in preserved)
    data=Pairs("train");stream=Stream(data);x,y,meta=stream.next();x=x.cuda();y=y.cuda()
    assert x.shape==(2,3,512,512) and y.shape==(2,1,512,512)
    shapes={};inputs=[]
    def hook(name):
        def f(module,args,result):
            assert args[0].shape==result.shape
            assert args[1].shape==(2,256)
            shapes[name]=list(result.shape)
        return f
    handles=[b.register_forward_hook(hook(n)) for n,b in blocks]
    def stem(module,args):
        assert args[0].shape==(2,4,512,512)
        assert torch.equal(args[0][:,1:],2*x-1)
        inputs.append(True)
    handles.append(m.model.unet0.init_conv.register_forward_pre_hook(stem))
    torch.cuda.reset_peak_memory_stats()
    opt=torch.optim.Adam(m.parameters(),lr=2e-4,betas=(.9,.99))
    gradients=[]
    for update in range(2):
        opt.zero_grad(set_to_none=True)
        loss=m([y,x[:,1:2],x])[0];assert torch.isfinite(loss);loss.backward()
        g={}
        for n,b in blocks:
            a=b.adaLN_modulation[-1].weight.grad
            assert a is not None and torch.isfinite(a).all() and a.abs().sum()>0,n
            c=b.norm.normalized_shape[0]
            chunks=[float(q.norm()) for q in a.chunk(3,dim=0)]
            assert chunks[2]>0
            if update==1:assert all(v>0 for v in chunks), (n,chunks)
            g[n]=chunks
        gradients.append(g)
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True)
        opt.step()
    for h in handles:h.remove()
    assert len(shapes)==9 and len(inputs)>=2
    assert all(torch.isfinite(q).all() for q in m.parameters())
    m.eval()
    with torch.no_grad():
        z=torch.cat((torch.zeros_like(y),2*x-1),dim=1)
        a=m.model.unet0(z,torch.full((2,),10.,device="cuda"))
        b=m.model.unet0(z,torch.full((2,),900.,device="cuda"))
        assert a.shape==b.shape==(2,1,512,512)
        difference=float((a-b).abs().max());assert difference>0
        calls=[]
        def sample_stem(module,args):
            assert args[0].shape==(1,4,512,512)
            assert torch.equal(args[0][:,1:],2*x[:1]-1)
            calls.append(True)
        handle=m.model.unet0.init_conv.register_forward_pre_hook(sample_stem)
        pred=m.sample([x[:1,1:2],x[:1]],last=True)[-1]
        handle.remove()
        assert pred.shape==(1,1,512,512) and torch.isfinite(pred).all() and len(calls)==2
    peak=torch.cuda.max_memory_allocated();assert peak<24*2**30
    result={"status":"PASS","count":len(blocks),"parameters":sum(q.numel() for q in m.parameters()),
       "shapes":shapes,"identity":"torch.equal all nine","gradients_shift_scale_gate":gradients,
       "time_prediction_max_difference":difference,"sampling_calls":len(calls),"peak_allocated":peak,
       "peak_reserved":torch.cuda.max_memory_reserved(),"train_samples":meta,
       "training_slices":len(data),"validation_indices":list(VALID),"preserved_diffusion_methods":preserved,
       "source_sha256":hashlib.sha256(current.encode()).hexdigest(),"data_access":access_report(),
       "python":sys.version,"torch":torch.__version__,"cuda":torch.version.cuda,"numpy":numpy.__version__,
       "skimage":skimage.__version__,"einops":einops.__version__,"gpu":torch.cuda.get_device_name(0)}
    write_json(OUT/"PREFLIGHT.json",result);print(json.dumps(result),flush=True)
if __name__=="__main__":main()
