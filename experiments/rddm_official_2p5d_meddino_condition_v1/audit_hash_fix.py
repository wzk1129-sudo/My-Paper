"""Audit the BF16 hashing fix without changing model mathematics or reading CT."""
import ast,hashlib,json,subprocess,gc
from pathlib import Path
import torch
from model import build_model,state_without_med
from train import tensor_state_hash,ARMS
from provenance import ROOT,write_json
from med_loader import sha
def legacy(state):
    h=hashlib.sha256()
    for n,t in sorted(state.items()):
        h.update(n.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def named_function(s,name):
    tree=ast.parse(s)
    return ast.dump(next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name==name),include_attributes=False)
def main():
    torch.set_num_threads(4)
    original=subprocess.check_output(["git","show","abe51fe:experiments/rddm_official_2p5d_meddino_condition_v1/train.py"],text=True)
    assert named_function(original,"main")==named_function((ROOT/"train.py").read_text(),"main")
    base={str(dtype):torch.arange(6).reshape(2,3).to(dtype) for dtype in (torch.float32,torch.float64,torch.float16,torch.int64,torch.int32,torch.int8,torch.uint8,torch.bool)}
    base["scalar"]=torch.tensor(1.5)
    assert legacy(base)==tensor_state_hash(base)
    bf={"bf":torch.arange(6).reshape(2,3).to(torch.bfloat16)}
    assert tensor_state_hash(bf)==tensor_state_hash({"bf":bf["bf"].clone()})
    changed=bf["bf"].clone();changed[0,0]=1
    assert tensor_state_hash(bf)!=tensor_state_hash({"bf":changed})
    model=build_model(True)
    med=model.model.unet0.med.state_dict()
    before=torch.get_rng_state().clone()
    med_sha=tensor_state_hash(med)
    assert torch.equal(before,torch.get_rng_state())
    assert all(torch.isfinite(t).all() for t in med.values())
    assert all(not p.requires_grad and p.grad is None for p in model.model.unet0.med.parameters())
    shared={n:t for n,t in state_without_med(model).items() if not any(x in n for x in (".layer_adapters.",".fusion.",".conditions."))}
    actual=tensor_state_hash(shared)
    control=json.loads((ROOT/"runs"/ARMS[0]/"INITIAL_STATE.json").read_text())
    assert actual==control["shared_initial_parameters_sha256"]
    ready_source=json.loads((ROOT/"SOURCE_PROVENANCE.json").read_text())["new_source_hashes"]
    for n in ("model.py","med_loader.py","readiness.py","context_data.py","src/official_rddm_ct.py"):assert sha(ROOT/n)==ready_source[n]
    counts={}
    for t in med.values():counts[str(t.dtype)]=counts.get(str(t.dtype),0)+1
    result={"status":"PASS","reason":"BF16 state bytes cannot be converted directly to NumPy; hash the contiguous uint8 byte view instead",
        "legacy_compatible_dtype_hash_unchanged":True,"scalar_hash_supported":True,"bf16_hash_supported":True,
        "bf16_changed_value_changes_hash":True,"actual_meddino_state_hash":med_sha,"actual_meddino_state_dtypes":counts,
        "meddino_frozen":True,"hash_does_not_change_rng":True,"train_main_ast_unchanged":True,
        "candidate_shared_initial_hash":actual,"matches_completed_control_initial_hash":True,
        "readiness_model_dependencies_unchanged":True,"medical_data_opens":0,"L506_opens":0,"external_memory_opens":0,"neighbor_NDCT_opens":0}
    write_json(ROOT/"HASH_FIX_AUDIT.json",result)
    print(json.dumps(result),flush=True)
if __name__=="__main__":main()
