import json,torch
from pathlib import Path
from knowledge_prior_v1.hashing import exact_state_hash
from knowledge_prior_v1.engine import HERE,init_arm,read_json
from mayo2016_smallset_v1.runtime import write_json
a={"bf16":torch.tensor([1.,2.],dtype=torch.bfloat16),"f32":torch.tensor([3.],dtype=torch.float32)}
h1=exact_state_hash(a);h2=exact_state_hash(a)
a["bf16"][0]=4.
h3=exact_state_hash(a)
assert h1==h2 and h1!=h3 and len(h1)==64
control=read_json(HERE/"CONTROL_CONTINUE/final_summary.json")
assert control["status"]=="COMPLETE" and control["steps"]==6000 and control["nonfinite_count"]==0
assert control["access"]["L506_opens"]==control["access"]["external_memory_opens"]==control["access"]["neighbor_target_opens"]==control["access"]["blocked_attempts"]==0
assert (HERE/"CONTROL_CONTINUE/step_6000.pt").is_file()
model,ema,opt,audit=init_arm("A_MEDDINO_BCA",torch.device("cuda:0"))
assert audit["starting_ema_equal_model"] and audit["fresh_adam_state_count"]==0
assert audit["meddino_frozen"] and audit["bca128_out_zero"] and audit["bca64_out_zero"]
result={"status":"PASS","dtype_hash_reproducible":True,"dtype_hash_sensitivity":True,
 "control_6000_preserved":True,"control_psnr":control["history"][-1]["psnr"],
 "A_init_audit":audit}
write_json(HERE/"A_INIT_RETRY_AUDIT.json",result)
print(json.dumps({"status":"PASS","control_psnr":result["control_psnr"],"A_total_parameters":audit["total_parameters"]}),flush=True)
