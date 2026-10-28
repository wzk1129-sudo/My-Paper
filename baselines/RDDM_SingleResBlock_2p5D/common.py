import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
os.environ["OMP_NUM_THREADS"]="2"
os.environ["MKL_NUM_THREADS"]="2"
import random,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/"config.json").read_text())
OUT=ROOT/"experiments"/CFG["name"]
SOURCE=json.loads((ROOT/"SOURCE.json").read_text())
def setup():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    random.seed(20260829); np.random.seed(20260829)
    torch.manual_seed(20260829); torch.cuda.manual_seed_all(20260829)
def write_json(path,obj):
    with Path(path).open("x") as f: json.dump(obj,f,indent=2,allow_nan=False)
def environment():
    import einops,skimage
    return dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
                numpy=np.__version__,einops=einops.__version__,skimage=skimage.__version__,
                gpu=torch.cuda.get_device_name(0),tf32=False,amp=False,cpu_threads=2)
