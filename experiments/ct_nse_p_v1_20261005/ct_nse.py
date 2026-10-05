"""CT adaptation of the shallow SPSR-P neural structure extractor."""
from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

RF = 31
STRIDES = (2, 1, 1, 2, 1, 1)
CONTEXT_DISTANCE = 8
EXCLUSION_RADIUS = 7

class CTNSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 2, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv4 = nn.Conv2d(32, 32, 3, 2, 1)
        self.conv5 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv6 = nn.Conv2d(32, 32, 3, 1, 1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        assert x.ndim == 4 and x.shape[1] == 1
        a = self.relu(self.conv1(x))
        b = self.relu(self.conv3(self.relu(self.conv2(a))) + a)
        c = self.relu(self.conv4(b))
        return self.relu(self.conv6(self.relu(self.conv5(c))) + c)

class StructurePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(64,128),nn.ReLU(),
                                 nn.Linear(128,64),nn.ReLU(),
                                 nn.Linear(64,32))
    def forward(self,left,right):
        return self.net(torch.cat((left,right),dim=-1))

def receptive_field_audit():
    rf=1; jump=1; progression=[]
    for stride in STRIDES:
        rf += 2*jump
        jump *= stride
        progression.append(rf)
    assert progression == [3,7,11,15,23,31] and jump == 4
    return {'strides':list(STRIDES),'rf_progression':progression,
            'receptive_field':rf,'effective_stride':jump,
            'residual_skips':2,'output_channels':32}

def full_rf_bounds(feature_h,feature_w):
    # With the specified odd kernels and symmetric padding, output (y,x)
    # is centered at input pixel (4*y,4*x), radius 15.
    def bounds(size):
        input_size=size*4
        low=math.ceil(15/4)
        high=math.floor((input_size-1-15)/4)
        assert 0<=low<=high<size
        return low,high
    return bounds(feature_h),bounds(feature_w)

def sample_anchors(rng:np.random.Generator,batch_size:int,feature_h:int,feature_w:int,k:int):
    (y0,y1),(x0,x1)=full_rf_bounds(feature_h,feature_w)
    assert x1-x0>=2*CONTEXT_DISTANCE and k>0
    y=rng.integers(y0,y1+1,size=(batch_size,k),dtype=np.int64)
    x=rng.integers(x0+CONTEXT_DISTANCE,x1-CONTEXT_DISTANCE+1,
                   size=(batch_size,k),dtype=np.int64)
    return np.stack((y,x),axis=-1)

def valid_negative_mask(anchors:torch.Tensor,feature_h:int,feature_w:int):
    assert anchors.ndim==3 and anchors.shape[-1]==2
    (y0,y1),(x0,x1)=full_rf_bounds(feature_h,feature_w)
    yy,xx=torch.meshgrid(torch.arange(feature_h,device=anchors.device),
                         torch.arange(feature_w,device=anchors.device),indexing='ij')
    yy=yy.flatten()[None,None,:]
    xx=xx.flatten()[None,None,:]
    ay=anchors[...,0,None];ax=anchors[...,1,None]
    valid=(yy>=y0)&(yy<=y1)&(xx>=x0)&(xx<=x1)
    for offset in (-CONTEXT_DISTANCE,0,CONTEXT_DISTANCE):
        overlap=((yy-ay).abs()<=EXCLUSION_RADIUS)&((xx-(ax+offset)).abs()<=EXCLUSION_RADIUS)
        valid=valid&(~overlap)
    assert bool(valid.any(dim=-1).all())
    return valid

def contrastive_prediction(features,predictor,anchors,scale=64.0):
    b,c,h,w=features.shape
    assert c==32 and anchors.shape[:2]==(b,anchors.shape[1])
    k=anchors.shape[1]
    assert anchors.shape==(b,k,2)
    batch=torch.arange(b,device=features.device)[:,None]
    y=anchors[...,0];x=anchors[...,1]
    left=features[batch,:,y,x-CONTEXT_DISTANCE]
    right=features[batch,:,y,x+CONTEXT_DISTANCE]
    predicted=F.normalize(predictor(left,right),dim=-1)
    flat=features.flatten(2).transpose(1,2)
    normalized=F.normalize(flat,dim=-1)
    cosine=torch.bmm(predicted,normalized.transpose(1,2))
    assert torch.isfinite(cosine).all()
    target_index=y*w+x
    positive=cosine.gather(2,target_index[...,None])
    mask=valid_negative_mask(anchors,h,w)
    assert not bool(mask.gather(2,target_index[...,None]).any())
    negatives=cosine.masked_fill(~mask,-1e9)
    logits=torch.cat((positive,negatives),dim=-1)*scale
    assert torch.isfinite(logits).all()
    labels=torch.zeros(b*k,dtype=torch.long,device=features.device)
    loss=F.cross_entropy(logits.reshape(b*k,-1),labels)
    assert torch.isfinite(loss)
    with torch.no_grad():
        count=mask.sum()
        neg_mean=cosine.masked_fill(~mask,0).sum()/count
        pos_mean=positive.mean()
        top1=(positive.squeeze(-1)>negatives.max(dim=-1).values).float().mean()
        metrics={'loss':float(loss.detach()),'positive_cosine':float(pos_mean),
                 'negative_cosine':float(neg_mean),'margin':float(pos_mean-neg_mean),
                 'top1':float(top1),'negative_count_mean':float(mask.sum(dim=-1).float().mean())}
    return loss,metrics

