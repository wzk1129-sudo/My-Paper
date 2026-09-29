"""Disjoint donor NDCT teacher and feature loss; no donor LDCT path is built."""
import os,re,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from mayo2016_smallset_v1.data import COUNTS
from mayo2016_smallset_v1.frozen_protocol import normalize_stored_ct
DONORS=("L096","L109","L192","L286","L291","L310")
class DonorGuard:
    def __init__(self,root):self.root=os.path.abspath(root);self.reads=[];self.blocked=0;self.active=False
    def hook(self,event,args):
        if not self.active or event not in ("open","os.listdir","os.scandir") or not args:return
        if not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=os.path.abspath(os.fsdecode(args[0]))
        if path!=self.root and not path.startswith(self.root+os.sep):return
        if event!="open":self.blocked+=1;raise PermissionError("data directory enumeration")
        m=re.fullmatch(r"(L\d{3})_target_(\d{3})_img\.npy",os.path.basename(path))
        flags=args[2] if len(args)>2 and isinstance(args[2],int) else 0
        if not m or m[1] not in DONORS or not 0<=int(m[2])<COUNTS[m[1]] or os.path.dirname(path)!=self.root or flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):
            self.blocked+=1;raise PermissionError("non-donor NDCT access")
        self.reads.append((m[1],int(m[2])))
    def __enter__(self):self.active=True;sys.addaudithook(self.hook);return self
    def __exit__(self,*args):self.active=False
    def report(self):return {"donor_patients_opened":sorted({p for p,i in self.reads}),"donor_ndct_opens":len(self.reads),"donor_ldct_opens":0,"L333_opens":0,"L506_opens":0,"blocked_attempts":self.blocked}
class DonorPatches:
    def __init__(self,cfg,seed=20260929):self.root=Path(cfg["data"]["root"]).resolve();self.norm=cfg["data"];self.rng=np.random.RandomState(seed)
    def batch(self,n=8):
        xs=[];meta=[]
        for _ in range(n):
            p=DONORS[int(self.rng.randint(len(DONORS)))];i=int(self.rng.randint(COUNTS[p]))
            path=self.root/f"{p}_target_{i:03d}_img.npy"
            assert path.parent==self.root and not path.is_symlink() and path.is_file()
            raw=np.load(path,allow_pickle=False)
            assert raw.shape==(512,512) and np.isfinite(raw).all()
            x=normalize_stored_ct(raw,self.norm)
            top=int(self.rng.randint(257));left=int(self.rng.randint(257))
            x=x[top:top+256,left:left+256]
            if self.rng.rand()<.5:x=np.flip(x,-1)
            x=np.ascontiguousarray(np.rot90(x,int(self.rng.randint(4))))
            assert x.shape==(256,256) and x.dtype==np.float32 and 0<=x.min()<=x.max()<=1
            xs.append(2*x-1);meta.append({"patient_id":p,"slice_index":i})
        return torch.from_numpy(np.stack(xs)[:,None].copy()),meta
class ResBlock(nn.Module):
    def __init__(self,c):
        super().__init__();self.c1=nn.Conv2d(c,c,3,padding=1);self.g1=nn.GroupNorm(min(8,c),c);self.c2=nn.Conv2d(c,c,3,padding=1);self.g2=nn.GroupNorm(min(8,c),c)
    def forward(self,x):return F.silu(self.g2(self.c2(F.silu(self.g1(self.c1(x)))))+x)
class Stage(nn.Sequential):
    def __init__(self,c):super().__init__(ResBlock(c),ResBlock(c))
class Up(nn.Module):
    def __init__(self,cin,skip,cout):
        super().__init__();self.conv=nn.Conv2d(cin,cout,3,padding=1);self.fuse=nn.Conv2d(cout+skip,cout,3,padding=1);self.stage=Stage(cout)
    def forward(self,x,skip):
        x=self.conv(F.interpolate(x,scale_factor=2,mode="bilinear",align_corners=False))
        return self.stage(self.fuse(torch.cat((x,skip),dim=1)))
class Teacher(nn.Module):
    def __init__(self):
        super().__init__();self.stem=nn.Conv2d(1,32,3,padding=1);self.s1=Stage(32);self.d1=nn.Conv2d(32,64,3,stride=2,padding=1);self.s2=Stage(64);self.d2=nn.Conv2d(64,128,3,stride=2,padding=1);self.s3=Stage(128);self.d3=nn.Conv2d(128,256,3,stride=2,padding=1);self.s4=Stage(256)
        self.u3=Up(256,128,128);self.u2=Up(128,64,64);self.u1=Up(64,32,32);self.out=nn.Conv2d(32,1,3,padding=1)
    def features(self,x):
        a=self.s1(self.stem(x));b=self.s2(self.d1(a));c=self.s3(self.d2(b));d=self.s4(self.d3(c));return a,b,c,d
    def forward(self,x):
        a,b,c,d=self.features(x);return self.out(self.u1(self.u2(self.u3(d,c),b),a))
    def prior(self,pred,gt):
        _,p2,p3,p4=self.features(pred)
        with torch.no_grad():_,g2,g3,g4=self.features(gt)
        terms=[(1-(F.normalize(p,p=2,dim=1)*F.normalize(g,p=2,dim=1)).sum(1)).mean() for p,g in zip((p2,p3,p4),(g2,g3,g4))]
        return .5*terms[0]+.3*terms[1]+.2*terms[2],terms
def gaussian(x,sigma,k):
    z=torch.arange(k,device=x.device,dtype=x.dtype)-k//2;w=torch.exp(-.5*(z/sigma).square());w=w/w.sum()
    x=F.pad(x,(k//2,k//2,k//2,k//2),mode="reflect")
    return F.conv2d(F.conv2d(x,w.view(1,1,k,1)),w.view(1,1,1,k))
def pyramid(x):
    a=gaussian(x,1,7);b=gaussian(x,2,13);c=gaussian(x,4,25);return x-a,a-b,b-c
def teacher_loss(pred,gt):
    img=F.l1_loss(pred,gt);bands=[F.l1_loss(a,b) for a,b in zip(pyramid(pred),pyramid(gt))]
    return img+.5*(.5*bands[0]+.3*bands[1]+.2*bands[2]),img,bands
def corrupt(x):
    y=x+torch.rand((x.shape[0],1,1,1),device=x.device)*.03*torch.randn_like(x)
    if bool(torch.rand((),device=x.device)<.5):
        mask=(torch.rand((x.shape[0],1,x.shape[-2]//16,x.shape[-1]//16),device=x.device)<.25).float()
        mask=F.interpolate(mask,scale_factor=16,mode="nearest").bool()
        y=torch.where(mask,gaussian(x,4,25),y)
    return y
