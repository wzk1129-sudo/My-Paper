"""Source, environment and immutable input audits; no image reads."""
import ast,hashlib,json,subprocess,sys
from pathlib import Path
import torch,numpy,einops
from model import PARENT_CHECKPOINT,load_parent
from med_loader import MED_ROOT,MED_CK,MED_SHA,MED_COMMIT,sha
from fast_dev.protocol import CONFIG,check_config
ROOT=Path(__file__).resolve().parent
OFFICIAL_ROOT=Path("/home/xujian/projects/RDDM_CVPR2024/RDDM_Official")
OFFICIAL_COMMIT="524caea34cedb3733aa5bf6c27dc1cd6e0469216"
MED_CANONICAL=Path("/home/xujian/projects/CoreDiff/experiments/rddm_meddino_fulltoken_ca_v1_20261001/baselines/founddiff_rddm_rleb/knowledge_prior_v1/core.py")
CONTEXT_CANONICAL=Path("/home/xujian/projects/CoreDiff/experiments/rddm_2p5d_screen_v1_20260924/baselines/founddiff_rddm_rleb/context_2p5d/data.py")
PARENT=Path("/home/xujian/projects/CoreDiff_official_rddm_ct_adapt_v1/experiments/rddm_official_ct_adapt_v1")
def write_json(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n")
def digest(value):
    return hashlib.sha256(value).hexdigest()
def source_manifest():
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob("*")) if p.is_file()
        and p.suffix in (".py",".json",".txt",".sh") and not any(x in p.parts for x in ("runs","logs","__pycache__"))
        and p.name not in ("SOURCE_PROVENANCE.json","READINESS.json","ENVIRONMENT.json","RUN_MANIFEST.json","QUEUE_STATUS.json")}
def git(path,*args):
    return subprocess.check_output(["git","-C",str(path),*args],text=True).strip()
def function_ast(path,name):
    t=ast.parse(Path(path).read_text())
    return ast.dump(next(n for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name),include_attributes=False)
def audit_sources():
    cfg=json.loads(CONFIG.read_text());check_config(cfg)
    assert git(OFFICIAL_ROOT,"rev-parse","HEAD")==OFFICIAL_COMMIT
    assert not git(OFFICIAL_ROOT,"status","--short")
    assert git(MED_ROOT,"rev-parse","HEAD")==MED_COMMIT and not git(MED_ROOT,"status","--short")
    assert sha(MED_CK)==MED_SHA
    assert sha(ROOT/"src/official_rddm_ct.py")==sha(PARENT/"src/official_rddm_ct.py")
    assert sha(ROOT/"context_data.py")==sha(CONTEXT_CANONICAL)
    assert function_ast(ROOT/"med_loader.py","official_med")==function_ast(MED_CANONICAL,"official_med")
    for name in ("LinearAttention","Attention","ResidualDiffusion","UnetRes"):
        assert function_ast(ROOT/"src/official_rddm_ct.py",name)==function_ast(PARENT/"src/official_rddm_ct.py",name)
    inherited=json.loads((ROOT/"inherited_audits/DICOM_SPATIAL_AUDIT.json").read_text())
    assert inherited["result"]=="PASS" and inherited["scope"]["patients"]==["L067","L143","L333"]
    ck=torch.load(PARENT_CHECKPOINT,map_location="cpu",weights_only=False)
    assert ck["optimizer_step"]==ck["ema_updates"]==8000
    assert ck["source_hashes"]["src/official_rddm_ct.py"]==sha(PARENT/"src/official_rddm_ct.py")
    assert ck["config"]["diffusion"]==cfg["diffusion"]
    for p in ck["ema_state_dict"].values():assert torch.isfinite(p).all()
    info={"official_rddm_commit":OFFICIAL_COMMIT,"official_adapter_sha256":sha(ROOT/"src/official_rddm_ct.py"),
        "meddino_commit":MED_COMMIT,"meddino_checkpoint_sha256":MED_SHA,"official_med_loader_ast_identical":True,
        "context_data_sha256":sha(CONTEXT_CANONICAL),"context_data_byte_identical":True,
        "parent_worktree_commit":git(PARENT,"rev-parse","HEAD"),
        "parent_checkpoint":str(PARENT_CHECKPOINT),"parent_checkpoint_sha256":sha(PARENT_CHECKPOINT),
        "parent_optimizer_step":8000,"parent_ema_updates":8000,
        "parent_source_hashes":ck["source_hashes"],"parent_8k_metrics":json.loads((PARENT/"runs/candidate_8k_20261001/validation_step_8000.json").read_text()),
        "new_source_hashes":source_manifest()}
    del ck
    write_json(ROOT/"SOURCE_PROVENANCE.json",info)
    environment={"python":sys.version,"executable":sys.executable,"torch":torch.__version__,
        "cuda":torch.version.cuda,"numpy":numpy.__version__,"einops":einops.__version__,
        "gpu":torch.cuda.get_device_name(0),"environment":"/home/xujian/miniconda3/envs/flexict",
        "tf32":False,"amp":False,"cublas_workspace_config":":4096:8","cuda_visible_devices":"0"}
    write_json(ROOT/"ENVIRONMENT.json",environment)
    print("SOURCE_AND_PARENT_CHECKPOINT_AUDIT_PASS",flush=True)
    return cfg,info
if __name__=="__main__":audit_sources()
