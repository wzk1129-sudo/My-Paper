"""Synthetic tests only. No CT arrays, model checkpoint, or training."""
import unittest
import torch
from run_audit import shift2,donor_for,masks,matching
class AuditMathTests(unittest.TestCase):
 def test_shift_noncyclic_replicate(self):
  x=torch.arange(32*32,dtype=torch.float32).reshape(1,1,32,32)
  y=shift2(x)
  self.assertEqual(y.shape,x.shape)
  self.assertEqual(y[0,0,0,0].item(),x[0,0,0,0].item())
  self.assertEqual(y[0,0,2,2].item(),x[0,0,0,0].item())
  self.assertEqual(y[0,0,31,31].item(),x[0,0,29,29].item())
  self.assertNotEqual(y[0,0,0,0].item(),x[0,0,30,30].item())
 def test_donor_rules(self):
  indices=(0,10,20,30)
  self.assertEqual(donor_for("C2_NEAR_SWAP",0,indices),10)
  self.assertEqual(donor_for("C2_NEAR_SWAP",3,indices),20)
  self.assertEqual(donor_for("C3_FAR_SWAP",1,indices),30)
  self.assertEqual(donor_for("C3_FAR_SWAP",2,indices),0)
  self.assertIsNone(donor_for("C0_NORMAL",0,indices))
 def test_masks_pool_and_ldct_source(self):
  c=torch.zeros((1,1,512,512),dtype=torch.float32)
  c[:,:,:16,:16]=(0-(-1024))/4096
  r,_,_,_=masks(c)
  self.assertEqual(int(r["soft"].sum()),1)
  self.assertEqual(int(r["foreground"].sum()),1)
  self.assertEqual(int(r["all"].sum()),1024)
 def test_matching_correct_location_and_missing_far(self):
  if not torch.cuda.is_available():self.skipTest("GPU unavailable")
  f=torch.zeros((1,16,32,32),device="cuda")
  for k in range(16):f[0,k,k//4,k%4]=1
  valid=torch.zeros(1024,dtype=torch.bool,device="cuda")
  token=torch.tensor([y*32+x for y in range(4) for x in range(4)],device="cuda")
  valid[token]=True
  m=matching(f,f,valid,valid)
  self.assertEqual(m["queries"],16)
  self.assertEqual(m["top1"],1.0)
  self.assertEqual(m["top5"],1.0)
  self.assertEqual(m["far_valid"],0)
  self.assertIsNone(m["margin_far_mean"])
if __name__=="__main__":unittest.main()
