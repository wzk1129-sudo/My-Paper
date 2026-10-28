from common import *
import ast,time,hashlib
from mayo_data import Mayo,Stream,TRAIN_PATIENTS,VAL_PATIENTS,FAST64
from src.residual_diffusion import ResidualDiffusion
from src.rddm_blocks import ResnetBlock,LinearAttention,Attention
def main():
    setup(); start=time.monotonic()
    assert hashlib.sha256((ROOT/"src/official_rddm_original.py").read_bytes()).hexdigest()==SOURCE["sha256"]
    original=ast.parse((ROOT/"src/official_rddm_original.py").read_text())
    blocks=ast.parse((ROOT/"src/rddm_blocks.py").read_text())
    orig={n.name:ast.dump(n,include_attributes=False) for n in original.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    for n in blocks.body:
        if isinstance(n,(ast.FunctionDef,ast.ClassDef)): assert ast.dump(n,include_attributes=False)==orig[n.name]
    data=Mayo(CFG["data_root"],TRAIN_PATIENTS); val=Mayo(CFG["data_root"],VAL_PATIENTS)
    assert [int(s) for s in (ROOT/"splits/L333_fast64.txt").read_text().split()]==FAST64
    assert len(FAST64)==64 and all(i in {r[1] for r in val.records} for i in FAST64)
    for ds in (data,val):
        for rec in ds.records:
            ds.read(rec[3]); ds.read(rec[5])
        for patient in ds.counts:
            records=[r for r in ds.records if r[0]==patient]
            assert records[0][2]==records[0][3] and records[-1][4]==records[-1][3]
            assert all(all(path.name.startswith(patient+"_") for path in r[2:]) for r in records)
    print("DATA_OK",data.counts,val.counts,flush=True)
    model=ResidualDiffusion().cuda()
    counts={"resblocks":sum(isinstance(m,ResnetBlock) for m in model.modules()),
            "linear_attention":sum(isinstance(m,LinearAttention) for m in model.modules()),
            "global_attention":sum(isinstance(m,Attention) for m in model.modules())}
    assert counts==dict(resblocks=10,linear_attention=8,global_attention=1)
    stream=Stream(data,CFG["seed"]); c,y=stream.batch()
    torch.cuda.reset_peak_memory_stats()
    loss=model.loss(c,y); assert torch.isfinite(loss); loss.backward()
    trace=model.model.trace
    assert trace["input"]==[2,4,512,512] and trace["output"]==[2,1,512,512]
    expected_skips=[[2,64,512,512],[2,64,256,256],[2,128,128,128],[2,256,64,64]]
    assert [e["skip"] for e in trace["encoder"]]==expected_skips
    expected_concat=[[2,768,64,64],[2,384,128,128],[2,192,256,256],[2,128,512,512]]
    assert [d["concat"] for d in trace["decoder"]]==expected_concat
    gradients={}
    for name,param in model.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(),name
        if ".mlp." in name or ".time_mlp." in name or ".fn.fn." in name:
            norm=float(param.grad.norm())
            assert norm>0,name
            gradients[name]=norm
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
    # Real Adam allocation confirms batch=2 FP32 training fits; this model is discarded.
    optimizer=torch.optim.Adam(model.parameters(),lr=2e-4,betas=(0.9,0.99),weight_decay=1e-9)
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    model.eval()
    result=model.sample(c[:1],torch.Generator(device="cuda").manual_seed(CFG["seed"]))
    assert result.shape==(1,1,512,512) and torch.isfinite(result).all()
    import copy
    ema=copy.deepcopy(model)
    torch.cuda.synchronize()
    report=dict(passed=True,environment=environment(),counts=counts,trace=trace,
                parameters=sum(q.numel() for q in model.parameters()),
                trainable_parameters=sum(q.numel() for q in model.parameters() if q.requires_grad),
                initial_loss=float(loss),gradient_norm=float(norm),gradient_checks=gradients,
                sampling_range=[float(result.min()),float(result.max())],
                peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
                train_pairs=len(data.records),validation_pairs=len(val.records),
                train_patient_counts=data.counts,L506_accesses=0,official_blocks_ast_identical=True,
                A999=float(model.A[999]),B999=float(model.B[999]),A499=float(model.A[499]),
                inference_noise_std=0.1,elapsed_seconds=time.monotonic()-start)
    write_json(OUT/"reports/PREFLIGHT.json",report)
    print(json.dumps({k:v for k,v in report.items() if k!="gradient_checks"},indent=2),flush=True)
if __name__=="__main__":main()
