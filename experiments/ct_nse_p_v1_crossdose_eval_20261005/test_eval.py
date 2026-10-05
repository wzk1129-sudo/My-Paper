"""Independent low-cost tests for coordinates, masking, ranking, normalization."""
from common import *
setup()
xy=coords();assert len(xy)==14641 and xy[0].tolist()==[4,4] and xy[-1].tolist()==[124,124]
assert 4*4-15>=0 and 124*4+15<512
raw=np.array([0,1024,4096,5000],np.uint16)
assert np.array_equal(normalize(raw),np.array([0,.25,1,1],np.float32))
s=torch.tensor([[.2,.9,.9,.1],[.8,.3,.5,.1]])
rank,best=ordinal(s,torch.tensor([2,0]))
assert rank.tolist()==[2,1] and best.tolist()==[1,0]
for j in range(2):
    correct=[2,0][j];p=float(s[j,correct]);expected=1
    for k in range(4):
        value=float(s[j,k])
        expected+=int(value>p+1e-6 or (abs(value-p)<=1e-6 and k<correct))
    assert int(rank[j])==expected
rng=np.random.default_rng(SEED);ids=rng.integers(len(xy),size=len(xy))
mask=np.any(np.abs(xy[ids]-xy)>7,axis=1)
for a,b in zip(xy[:100],xy[ids[:100]]):
    nonoverlap=abs(4*int(a[0])-4*int(b[0]))>=31 or abs(4*int(a[1])-4*int(b[1]))>=31
    assert nonoverlap==bool(np.any(np.abs(a-b)>7))
m,ep=load_model()
before=state_sha(m)
with torch.inference_mode():
    y=m(torch.zeros(2,1,512,512,device='cuda'))
assert y.shape==(2,32,128,128) and torch.isfinite(y).all()
assert state_sha(m)==before and all(p.grad is None and not p.requires_grad for p in m.parameters())
assert not m.training and ep==60
dump(ROOT/'READINESS.json',{'PASS':True,'input':[2,1,512,512],'output':list(y.shape),
      'full_RF_bounds':[4,124],'valid_locations':14641,'scalar_rank_test':True,
      'nonoverlap_RF_test':True,'normalization_test':True,'model_frozen':True,
      'checkpoint_sha256':sha(CHECKPOINT),'model_state_sha256':before,
      'environment':{'python':sys.version,'executable':sys.executable,'torch':torch.__version__,
       'cuda':torch.version.cuda,'numpy':np.__version__,'GPU':torch.cuda.get_device_name(0)}})
print('READINESS_PASS',flush=True)
