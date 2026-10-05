"""Unit checks for CT-NSE-P architecture, geometry and negative exclusion."""
import numpy as np
import torch
from ct_nse import (CTNSE,StructurePredictor,receptive_field_audit,
    sample_anchors,valid_negative_mask,contrastive_prediction,full_rf_bounds)

def main():
    audit=receptive_field_audit()
    assert audit['rf_progression']==[3,7,11,15,23,31]
    assert audit['effective_stride']==4 and audit['receptive_field']==31
    assert len([m for m in CTNSE().modules() if isinstance(m,torch.nn.Conv2d)])==6
    torch.manual_seed(20260829)
    model=CTNSE();predictor=StructurePredictor()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device);predictor.to(device)
    x=torch.rand(2,1,420,420,device=device)
    f=model(x)
    assert f.shape==(2,32,105,105),f.shape
    rng=np.random.default_rng(20260829)
    a=torch.from_numpy(sample_anchors(rng,2,105,105,16)).to(device)
    assert a.shape==(2,16,2)
    (y0,y1),(x0,x1)=full_rf_bounds(105,105)
    assert (y0,y1,x0,x1)==(4,101,4,101)
    assert int(a[...,0].min())>=4 and int(a[...,0].max())<=101
    assert int(a[...,1].min())>=12 and int(a[...,1].max())<=93
    assert 8*4>31
    mask=valid_negative_mask(a,105,105)
    assert mask.shape==(2,16,105*105)
    for b in range(2):
        for k in range(16):
            y=int(a[b,k,0]);x=int(a[b,k,1])
            for center in (x-8,x,x+8):
                for dy in (-7,0,7):
                    for dx in (-7,0,7):
                        yy=y+dy;xx=center+dx
                        if 0<=yy<105 and 0<=xx<105:assert not bool(mask[b,k,yy*105+xx])
            far_y=y+8 if y+8<=101 else y-8
            assert bool(mask[b,k,far_y*105+x])
    loss,metrics=contrastive_prediction(f,predictor,a)
    assert torch.isfinite(loss) and metrics['negative_count_mean']>1000
    loss.backward()
    for m in (model,predictor):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    assert all(p.grad.norm().item()>0 for p in list(model.parameters())+list(predictor.parameters()))
    # Independent spatial enumeration verifies exact positive-at-index0 loss.
    fake=torch.rand(2,32,105,105,device=device)
    coords=torch.tensor([[[50,50]],[[30,40]]],device=device)
    actual,_=contrastive_prediction(fake[:1],predictor,coords[:1])
    legal=[(yy,xx) for yy in range(4,102) for xx in range(4,102)
           if not any(abs(yy-50)<=7 and abs(xx-cx)<=7 for cx in (42,50,58))]
    candidates=torch.stack([fake[0,:,50,50]]+[fake[0,:,yy,xx] for yy,xx in legal])
    prediction=predictor(fake[0,:,50,42][None],fake[0,:,50,58][None])[0]
    prediction=torch.nn.functional.normalize(prediction,dim=-1)
    logits=64.*(torch.nn.functional.normalize(candidates,dim=-1)@prediction)
    expected=torch.nn.functional.cross_entropy(logits[None],torch.zeros(1,dtype=torch.long,device=device))
    assert torch.allclose(actual,expected,atol=2e-5,rtol=1e-6),(actual,expected)
    both,_=contrastive_prediction(fake,predictor,coords)
    first,_=contrastive_prediction(fake[:1],predictor,coords[:1])
    second,_=contrastive_prediction(fake[1:],predictor,coords[1:])
    assert torch.allclose(both,(first+second)/2,atol=2e-5,rtol=1e-6)
    print('PASS',audit,'shape',tuple(f.shape),'loss',float(loss.detach()),'negative_count',metrics['negative_count_mean'])

if __name__=='__main__':main()

