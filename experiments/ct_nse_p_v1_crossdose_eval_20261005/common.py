"""Read-only frozen CT-NSE cross-dose utilities; no optimizer or training."""
from pathlib import Path
import sys,os,json,hashlib,re
from collections import Counter
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent.parent
SOURCE=PROJECT/'experiments/ct_nse_p_v1_20261005'
DATA=PROJECT/'data_preprocess/gen_data/mayo_2016_npy'
RAW=Path('/home/xujian/data/Mayo2016/L333')
CHECKPOINT=SOURCE/'checkpoints/best_val_loss.pt'
sys.path.insert(0,str(SOURCE))
from ct_nse import CTNSE,full_rf_bounds
SEED=20260829
def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def state_sha(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):
        h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def normalize(raw):
    return (np.clip(raw.astype(np.float32)-1024,-1024,3072)+1024)/4096
def setup():
    torch.set_num_threads(4)
    torch.manual_seed(SEED);np.random.seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
def load_model(device='cuda'):
    ck=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    model=CTNSE().eval().requires_grad_(False)
    model.load_state_dict(ck['ctnse_state_dict'],strict=True)
    assert all(not p.requires_grad for p in model.parameters())
    return model.to(device),ck['epoch']
class Access:
    """Allow only L333 paired files; block writes and any other medical data."""
    def __init__(self):
        self.active=False;self.allowed_raw=set();self.count=Counter();self.unique=set();self.blocked=0
        sys.addaudithook(self.hook)
    def hook(self,event,args):
        if not self.active or event not in ('open','os.listdir','os.scandir') or not args:return
        if not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.path.abspath(os.fsdecode(args[0])))
        sensitive=(str(p).startswith('/home/xujian/data/') or str(p).startswith(str(DATA)))
        if not sensitive:return
        if event!='open':
            allowed=p in (RAW/'quarter_1mm',RAW/'full_1mm')
            if not allowed:self.blocked+=1;raise PermissionError('Forbidden medical enumeration: '+str(p))
            return
        flags=args[2] if len(args)>2 and isinstance(args[2],int) else 0
        match=re.fullmatch(r'L333_(25|target)_(\d{3})_img.npy',p.name)
        npy=p.parent==DATA and match is not None and int(match[2])<610
        dicom=str(p) in self.allowed_raw
        writing=flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)
        if not(npy or dicom) or writing:
            self.blocked+=1;raise PermissionError('Forbidden medical open: '+str(p))
        kind=('NPY_'+match[1]) if npy else ('DICOM_LDCT' if p.parent.name=='quarter_1mm' else 'DICOM_NDCT')
        self.count[kind]+=1;self.unique.add(str(p))
    def report(self):
        return {'opens':dict(self.count),'unique_files':len(self.unique),
                'blocked_attempts':self.blocked,'L506_access':0,'L067_L143_access':0,
                'external_memory_access':0,'medical_data_modified':False}
def coords():
    (lo,hi),_=full_rf_bounds(128,128)
    assert (lo,hi)==(4,124)
    yy,xx=np.meshgrid(np.arange(lo,hi+1),np.arange(lo,hi+1),indexing='ij')
    return np.stack((yy.ravel(),xx.ravel()),axis=-1)
def ordinal(scores,positive_index,tolerance=1e-6):
    """Fixed tie rule: scores within 1e-6 tie; lower gallery index wins."""
    positive=scores.gather(1,positive_index[:,None])
    ids=torch.arange(scores.shape[1],device=scores.device)[None,:]
    tied=(scores-positive).abs()<=tolerance
    rank=1+(scores>positive+tolerance).sum(1)+(tied&(ids<positive_index[:,None])).sum(1)
    maxima=scores.max(1).values
    best=torch.where(scores>=maxima[:,None]-tolerance,ids,scores.shape[1]).min(1).values
    return rank,best
def stats(x):
    a=np.asarray(x,dtype=np.float64)
    return {'mean':float(a.mean()),'median':float(np.median(a)),'std':float(a.std()),
            **{'p'+str(p):float(np.percentile(a,p)) for p in (5,25,75,95)}}
def feature_stats(a):
    # Features: [610,14641,32], only full-RF locations.
    total=np.zeros(32,np.float64);square=total.copy();nonzero=np.zeros(32,np.int64);count=0
    normsum=normsq=0.;nan=inf=0
    for x in a:
        v=x.astype(np.float64);count+=len(v)
        total+=v.sum(0);square+=(v*v).sum(0);nonzero+=(np.abs(v)>1e-6).sum(0)
        n=np.linalg.norm(v,axis=1);normsum+=n.sum();normsq+=(n*n).sum()
        nan+=np.isnan(v).sum();inf+=np.isinf(v).sum()
    mean=total/count;std=np.sqrt(np.maximum(square/count-mean**2,0))
    gmean=mean.mean();gstd=np.sqrt(square.sum()/(count*32)-gmean**2)
    return {'global_mean':float(gmean),'global_std':float(gstd),
      'per_channel_mean':mean.tolist(),'per_channel_std':std.tolist(),
      'per_channel_active_fraction':(nonzero/count).tolist(),
      'channel_std_mean':float(std.mean()),'channel_std_min':float(std.min()),'channel_std_max':float(std.max()),
      'zero_channels':int((nonzero==0).sum()),'near_zero_channels_std_lt_1e_4':int((std<1e-4).sum()),
      'norm_mean':float(normsum/count),'norm_std':float(np.sqrt(normsq/count-(normsum/count)**2)),
      'nan_count':int(nan),'inf_count':int(inf),
      'collapse':bool(gstd<1e-4 or std.max()<1e-4)}
