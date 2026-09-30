"""Throwaway 2-step readiness: never writes model weights."""
import copy
import json
import math
import time
from pathlib import Path
import torch
from context_2p5d.data import ContextPairs, ContextBatchStream, ContextDataGuard
from context_2p5d.model import neighbor_context
from context_2p5d.protocol import config as context_config
from knowledge_prior_v1.core import load_direct, matched_loss, DIRECT_SHA, MED_SHA, MED_COMMIT, sha
from knowledge_prior_v1.stream import make_manifest, digest, assert_batch, timestep_noise
from mayo2016_smallset_v1.runtime import ROOT, seed_everything, write_json
from src.rddm_rleb import normalize_to_neg_one_to_one
from fulltoken_ca_v1.model import GlobalPatchCrossAttention, load_candidate

OUT = ROOT / "experiments/meddino_fulltoken_ca_v1"
SOURCE_SHA = "e8ed93223fbdf4a0e2521f73c7a4f213561f6a191a1a7b2a5f95934b85ebf574"

def positive_finite(t):
    return t is not None and bool(torch.isfinite(t).all()) and float(t.detach().abs().sum()) > 0

def test_chunk_equivalence():
    torch.manual_seed(123)
    a = GlobalPatchCrossAttention(128, 0).eval()
    b = copy.deepcopy(a)
    b.query_chunk = 16
    with torch.no_grad():
        a.gate.fill_(1)
        b.gate.fill_(1)
        x = torch.randn(1, 128, 8, 8)
        tokens = torch.randn(1, 1024, 768)
        delta = float((a(x,tokens)-b(x,tokens)).abs().max())
    assert delta <= 1e-6, delta
    return delta

def run(query_chunk=0):
    OUT.mkdir(parents=True, exist_ok=True)
    seed_everything(20260829)
    torch.set_num_threads(2)
    cfg, base = context_config()
    assert sha(ROOT / "src/rddm_rleb.py") == SOURCE_SHA
    device = torch.device("cuda:0")
    train = ContextPairs(base, "train")
    assert len(train) == 1145
    manifest = make_manifest(train, steps=8000)
    assert digest(manifest[:6000]) == "ca176a324db9ce0547f0a5ad863108028e2b49dde2a554fc952cd059c3ca4768"
    guard = ContextDataGuard(base, "FAST")
    with guard:
        train.bind_guard(guard)
        stream = ContextBatchStream(train, 20260829)
        prev,center,next_,target,meta = stream.next(2)
        assert_batch(meta,manifest[0])
        assert prev.shape == center.shape == next_.shape == target.shape == (2,1,512,512)
        batch = [torch.from_numpy(x).to(device) if not isinstance(x,torch.Tensor) else x.to(device)
                 for x in (prev,center,next_,target)]
        prev,center,next_,target = batch
        t,noise = timestep_noise(manifest[0],target.shape,device)
        control = load_direct(device)
        candidate = load_candidate(device,query_chunk)
        direct = control.state_dict()
        cand = candidate.state_dict()
        assert all(k in cand and torch.equal(v,cand[k]) for k,v in direct.items())
        assert all(not p.requires_grad for p in candidate.model.unet0.med.parameters())
        assert not candidate.model.unet0.med.training
        assert float(candidate.model.unet0.ca128.gate)==float(candidate.model.unet0.ca64.gate)==0.0
        with torch.no_grad():
            _,ref,_,_ = matched_loss(control,prev,center,next_,target,t,noise)
            _,out,_,_ = matched_loss(candidate,prev,center,next_,target,t,noise)
            delta = float((ref-out).abs().max())
        assert ref.shape == out.shape == (2,1,512,512) and delta <= 1e-6, delta
        del control,ref,out
        torch.cuda.empty_cache()
        seed_everything(20260829)
        candidate.train()
        opt = torch.optim.Adam((p for p in candidate.parameters() if p.requires_grad),
                               lr=2e-4,betas=(.9,.99),weight_decay=1e-9)
        torch.cuda.reset_peak_memory_stats(device)
        times=[]
        gate_grad=[];proj_grad=[]
        for step in range(2):
            if step:
                prev,center,next_,target,meta=stream.next(2)
                assert_batch(meta,manifest[1])
                prev,center,next_,target=[x.to(device) for x in (prev,center,next_,target)]
                t,noise=timestep_noise(manifest[1],target.shape,device)
            started=time.perf_counter()
            opt.zero_grad(set_to_none=True)
            loss,pred,_,_=matched_loss(candidate,prev,center,next_,target,t,noise)
            assert math.isfinite(float(loss)) and bool(torch.isfinite(pred).all())
            loss.backward()
            u=candidate.model.unet0
            g=[u.ca128.gate.grad,u.ca64.gate.grad]
            gate_grad.append([float(x.detach()) for x in g])
            assert all(positive_finite(x) for x in g),gate_grad
            modules=[u.ca128,u.ca64]
            qkv=[m.q.weight.grad for m in modules]+[m.k.weight.grad for m in modules]+[m.v.weight.grad for m in modules]+[m.out.weight.grad for m in modules]
            assert all(x is not None and bool(torch.isfinite(x).all()) for x in qkv)
            proj_grad.append([float(x.detach().norm()) for x in qkv])
            if step==0:assert all(x==0 for x in proj_grad[-1])
            if step==1:assert all(x>0 for x in proj_grad[-1]),proj_grad[-1]
            assert all(p.grad is None for p in u.med.parameters())
            norm=float(torch.nn.utils.clip_grad_norm_((p for p in candidate.parameters() if p.requires_grad),1.0))
            assert math.isfinite(norm)
            opt.step()
            torch.cuda.synchronize(device)
            times.append(time.perf_counter()-started)
        access=guard.report()
    assert access["L506_opens"]==access["external_memory_opens"]==access["neighbor_target_opens"]==access["blocked_attempts"]==0
    eq=test_chunk_equivalence()
    report={
        "status":"PASS","branch":"experiment/rddm-meddino-fulltoken-ca-v1",
        "checkpoint_sha256":DIRECT_SHA,"meddino_sha256":MED_SHA,"official_meddino_commit":MED_COMMIT,
        "formal_source_sha256":SOURCE_SHA,"train_manifest_6k_sha256":digest(manifest[:6000]),
        "same_direct_state_bitwise":True,"step0_max_abs_output_diff":delta,
        "patch_tokens_shape":[2,1024,768],
        "ca128":{"query_tokens":16384,"kv_tokens":1024,"dim":256,"heads":4,"dim_head":64},
        "ca64":{"query_tokens":4096,"kv_tokens":1024,"dim":256,"heads":4,"dim_head":64},
        "gate_init":[0.0,0.0],"gate_gradients_by_step":gate_grad,
        "qkv_and_out_gradient_norms_by_step":proj_grad,
        "meddino_frozen_eval":True,"query_chunk":query_chunk,
        "small_unchunked_vs_chunked_max_abs_diff":eq,
        "control_trainable_params":sum(p.numel() for p in direct.values()),
        "candidate_trainable_params":sum(p.numel() for p in candidate.parameters() if p.requires_grad),
        "peak_allocated_bytes":torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes":torch.cuda.max_memory_reserved(device),
        "seconds_per_step":times,"data_access":access,
    }
    write_json(OUT/"READINESS_AUDIT.json",report)
    lines=["# Full-token CA readiness","","Status: PASS",
           f"Direct 12k EMA checkpoint SHA256: {DIRECT_SHA}",
           f"MedDINO checkpoint SHA256: {MED_SHA}; official source commit: {MED_COMMIT}",
           f"Direct shared weights bitwise identical; step-0 max abs difference: {delta:.9g}",
           "MedDINO normalized final-layer patch tokens: [2,1024,768], CLS/register excluded.",
           "CA128: 16384 queries x 1024 keys; CA64: 4096 queries x 1024 keys; 4 heads x 64.",
           f"Gate gradients: {gate_grad}; projection gradient norms: {proj_grad}",
           f"Q chunk: {query_chunk}; small unchunked/chunked max diff: {eq:.9g}",
           f"Peak allocated/reserved: {report['peak_allocated_bytes']}/{report['peak_reserved_bytes']} bytes",
           f"Two real optimizer-step times: {times}",
           f"Data access: {access}"]
    (OUT/"READINESS_REPORT.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({k:report[k] for k in ("status","step0_max_abs_output_diff","peak_allocated_bytes","peak_reserved_bytes","seconds_per_step","query_chunk")}))
if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--query-chunk",type=int,default=0)
    args=parser.parse_args()
    run(args.query_chunk)
