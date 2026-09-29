import os,json,torch
from context_2p5d.protocol import config
from context_2p5d.data import ContextPairs,ContextDataGuard
from context_2p5d.evaluate import validate
from knowledge_prior_v1.core import load_direct
from mayo2016_smallset_v1.runtime import seed_everything,write_json,ROOT
cfg,base=config();seed_everything(cfg["seed"]);device=torch.device("cuda:0")
model=load_direct(device)
with ContextDataGuard(base,"FAST",allow_training=False) as guard:
    ds=ContextPairs(base,"validation","FAST");ds.bind_guard(guard)
    result=validate(model,ds,cfg,device)
result["access"]=guard.report();result["pass"]=abs(result["psnr"]-40.76762730415604)<=.005
path=ROOT/"experiments/knowledge_prior_serial_v1/STEP0_PARITY.json"
write_json(path,result)
print(json.dumps({"psnr":result["psnr"],"pass":result["pass"]}),flush=True)
if not result["pass"]:raise SystemExit(2)
