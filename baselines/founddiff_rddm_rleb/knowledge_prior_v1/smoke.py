import json,torch
from context_2p5d.model import neighbor_context
from mayo2016_smallset_v1.runtime import ROOT,seed_everything,write_json
from knowledge_prior_v1.core import load_arm,matched_loss
from knowledge_prior_v1.teacher import Teacher,teacher_loss,corrupt
seed_everything(20260829)
torch.set_num_threads(2);device=torch.device("cuda:0")
control,h0=load_arm("CONTROL_CONTINUE",device)
a,h1=load_arm("A_MEDDINO_BCA",device)
assert h0==h1
u=a.model.unet0
assert not u.structure.med.training and all(not p.requires_grad for p in u.structure.med.parameters())
assert not control.model.unet0.self_condition
with torch.no_grad():
    torch.manual_seed(123)
    x=torch.randn(1,2,512,512,device=device)
    prev=torch.randn(1,1,512,512,device=device)
    next_=torch.randn(1,1,512,512,device=device)
    t=torch.tensor([10.],device=device)
    with neighbor_context(control,prev,next_): ref=control.model.unet0(x,t)
    with neighbor_context(a,prev,next_): out=a.model.unet0(x,t)
    delta=float((ref-out).abs().max())
assert delta<=1e-6,delta
del control
torch.cuda.empty_cache()
a.train()
torch.manual_seed(124)
prev=torch.rand(1,1,512,512,device=device)
center=torch.rand_like(prev);next_=torch.rand_like(prev);target=torch.rand_like(prev)
t=torch.tensor([500],device=device)
noise=torch.randn_like(prev)
loss,_,_,_=matched_loss(a,prev,center,next_,target,t,noise)
assert bool(torch.isfinite(loss));loss.backward()
grad=float(u.bca128.out.weight.grad.abs().sum()+u.bca64.out.weight.grad.abs().sum())
assert grad>0 and all(p.grad is None for p in u.structure.med.parameters())
del a
torch.cuda.empty_cache()
teacher=Teacher().to(device)
clean=torch.rand(2,1,256,256,device=device)*2-1
pred=teacher(corrupt(clean))
tl,li,bands=teacher_loss(pred,clean)
assert pred.shape==clean.shape and bool(torch.isfinite(tl))
tl.backward()
tg=float(teacher.s4[0].c1.weight.grad.abs().sum())
assert tg>0
write_json(ROOT/"experiments/knowledge_prior_serial_v1/SMOKE_AUDIT.json",{"status":"PASS","main_hash_equal":True,"step0_max_abs_diff":delta,"A_BCA_grad_sum":grad,"med_frozen":True,"teacher_grad_sum":tg,"teacher_loss":float(tl.detach()),"peak_vram_bytes":torch.cuda.max_memory_allocated(device)})
print(json.dumps({"status":"PASS","delta":delta,"bca_grad":grad,"teacher_grad":tg}),flush=True)
