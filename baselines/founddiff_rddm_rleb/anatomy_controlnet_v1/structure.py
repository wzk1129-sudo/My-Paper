"""Fixed smoothed CT edge hints and all-zero capacity control."""
import torch
from torch import nn
from torch.nn import functional as F

def gaussian(size, sigma):
    x = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    w = torch.exp(-0.5 * (x / sigma).square())
    w = w / w.sum()
    return torch.outer(w, w)[None, None]

class ZeroStructureProvider(nn.Module):
    def forward(self, center):
        b, c, h, w = center.shape
        assert (c, h, w) == (1, 512, 512)
        return {"s256": center.new_zeros((b,32,256,256)),
                "s128": center.new_zeros((b,64,128,128)),
                "s64": center.new_zeros((b,128,64,64))}

class EdgeStructureProvider(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("g5", gaussian(5, 1.))
        self.register_buffer("g9", gaussian(9, 2.))
        self.register_buffer("dx", torch.tensor([[[[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]]]])/8)
        self.register_buffer("dy", self.dx.transpose(-1,-2).contiguous())
        self.register_buffer("lap", torch.tensor([[[[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]]]]))
        self.p256 = nn.Sequential(nn.Conv2d(3,32,3,stride=2,padding=1),nn.SiLU(),
                                  nn.Conv2d(32,32,3,padding=1),nn.SiLU())
        self.p128 = nn.Sequential(nn.Conv2d(32,64,3,stride=2,padding=1),nn.SiLU())
        self.p64 = nn.Sequential(nn.Conv2d(64,128,3,stride=2,padding=1),nn.SiLU())
    def forward(self, center):
        assert center.shape[1:] == (1,512,512)
        with torch.no_grad():
            # Center is in model [-1,1]; frozen data normalization spans [-1024,3072] HU.
            hu = ((center+1.)*2048.-1024.).clamp(-1000.,1000.)
            v = (hu+1000.)/2000.
            small = F.conv2d(v,self.g5,padding=2)
            large = F.conv2d(v,self.g9,padding=4)
            gx1=F.conv2d(small,self.dx,padding=1);gy1=F.conv2d(small,self.dy,padding=1)
            gx2=F.conv2d(large,self.dx,padding=1);gy2=F.conv2d(large,self.dy,padding=1)
            edge=torch.cat(((gx1.square()+gy1.square()+1e-12).sqrt(),
                            (gx2.square()+gy2.square()+1e-12).sqrt(),
                            F.conv2d(large,self.lap,padding=1)),1)
        s256=self.p256(edge);s128=self.p128(s256);s64=self.p64(s128)
        assert s256.shape[1:]==(32,256,256) and s128.shape[1:]==(64,128,128)
        assert s64.shape[1:]==(128,64,64)
        return {"s256":s256,"s128":s128,"s64":s64}
