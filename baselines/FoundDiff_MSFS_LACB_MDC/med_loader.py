import hashlib,sys
from pathlib import Path
import torch
MED_ROOT = Path("/home/xujian/projects/CoreDiff/experiments/rddm_prosac_add_fusion_v1_20260926/official_source/MedDINOv3")
MED_CK = Path("/home/xujian/models/ct_foundation/meddinov3/model.pth")
MED_SHA = "492b2b96c00e9d3b93ba3b07a8d387aa2196b5e04698378fcc23049b447966d2"
MED_COMMIT = "a2460c2547899283446e520b84523454f7e25608"
MEAN_HU = 65.1084213256836
STD_HU = 178.01663208007812

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1048576),b""): h.update(chunk)
    return h.hexdigest()

def official_med():
    assert sha(MED_CK)==MED_SHA
    parent=MED_ROOT/"nnUNet/nnunetv2/training/nnUNetTrainer/dinov3"
    if str(parent) not in sys.path: sys.path.insert(0,str(parent))
    from dinov3.models.vision_transformer import vit_base
    med=vit_base(drop_path_rate=.2,layerscale_init=1e-5,n_storage_tokens=4,qkv_bias=False,mask_k_bias=True)
    ck=torch.load(MED_CK,map_location="cpu",weights_only=False)
    state={k.replace("backbone.",""):v for k,v in ck["teacher"].items() if "ibot" not in k and "dino_head" not in k}
    med.load_state_dict(state,strict=True)
    del ck,state
    return med.eval().requires_grad_(False)
