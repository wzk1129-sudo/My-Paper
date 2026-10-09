"""MLSF readiness: exact parent/stream audit and two isolated real batch=2 GPU steps."""
import hashlib
import json
import math
import sys
import time
import traceback
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent
V1 = ROOT.parent / "rddm_official_2p5d_meddino_condition_v1"
for path in (str(ROOT), str(V1)):
    if path not in sys.path:
        sys.path.insert(0, path)
from mlsf_model import build_model, parameter_hash, CORRECTED
from model import native_attention_grad, state_without_med
from context_data import ContextPairs, ContextBatchStream, ContextDataGuard
from fast_dev.protocol import CONFIG, check_config
from mayo2016_smallset_v1.runtime import seed_everything

EXPECTED_SHARED_SHA = "16d20db7edb42cafc5c5aee755218337f0aeefa45f300b41001002bc2be631f2"
NEW_PREFIXES = (".local_encoder.", ".med_l8.", ".med_l11.", ".cross_attention.",
                ".up_projection.", ".merge.", ".film_corrections.")


def norm(module):
    grads = [p.grad.detach().float() for p in module.parameters() if p.requires_grad and p.grad is not None]
    assert all(torch.isfinite(g).all() for g in grads)
    return math.sqrt(sum(float(g.square().sum()) for g in grads)) if grads else 0.0


def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed_everything(20260829)
    cfg = json.loads(CONFIG.read_text())
    check_config(cfg)
    started = time.perf_counter()
    with ContextDataGuard(cfg) as guard:
        data = ContextPairs(cfg, "train")
        data.bind_guard(guard)
        stream = ContextBatchStream(data, 20260829)
        model = build_model()
        u = model.model.unet0
        shared = {n: t for n, t in state_without_med(model).items()
                  if not any(x in n for x in (".layer_adapters.", ".fusion.", ".conditions.", *NEW_PREFIXES))}
        shared_sha = parameter_hash(shared)
        assert shared_sha == EXPECTED_SHARED_SHA, shared_sha
        assert len(u.conditions) == 9
        assert all(not p.requires_grad for p in u.med.parameters())
        for stage in CORRECTED:
            projection = u.film_corrections[stage]
            assert torch.count_nonzero(projection.weight) == torch.count_nonzero(projection.bias) == 0
        med_sha_before = parameter_hash(u.med.state_dict())
        model = model.to("cuda:0").train()
        u = model.model.unet0
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=2e-4, betas=(.9, .99), weight_decay=1e-9)
        assert all(id(p) in {id(q) for q in params} for stage in CORRECTED
                   for p in u.film_corrections[stage].parameters())
        torch.cuda.reset_peak_memory_stats()
        steps = []
        for step in (1, 2):
            previous, center, next_, target, metadata = stream.next(2)
            assert all(m["patient_id"] in ("L067", "L143") for m in metadata)
            previous, center, next_, target = [x.to("cuda:0") for x in (previous, center, next_, target)]
            optimizer.zero_grad(set_to_none=True)
            t = torch.randint(0, model.num_timesteps, (2,), device="cuda:0").long()
            noise = torch.randn_like(target)
            u.set_context(previous, center, next_)
            assert u.last_shapes == {"L256": [2, 32, 256, 256], "L128": [2, 64, 128, 128],
                                     "M32": [2, 64, 32, 32], "F128": [2, 64, 128, 128],
                                     "F256": [2, 32, 256, 256], "neighbors_per_query": 9}
            loss = model.p_losses([2 * target - 1, 2 * center - 1], t, noise=noise)[0]
            assert torch.isfinite(loss)
            loss.backward()
            u.clear_context()
            attention = native_attention_grad(model)
            group_norms = {
                "local_encoder": norm(u.local_encoder),
                "med_l8": norm(u.med_l8), "med_l11": norm(u.med_l11),
                "cross_attention": norm(u.cross_attention),
                "up_projection": norm(u.up_projection), "merge": norm(u.merge),
                **{stage: norm(u.film_corrections[stage]) for stage in CORRECTED},
            }
            assert all(group_norms[s] > 0 for s in CORRECTED), group_norms
            if step == 2:
                assert all(value > 0 for value in group_norms.values()), group_norms
            assert all(p.grad is None for p in u.med.parameters())
            grad = float(torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True))
            optimizer.step()
            assert all(torch.isfinite(p).all() for p in params)
            torch.cuda.synchronize()
            steps.append({"step": step, "loss": float(loss), "gradient_before_clip": grad,
                          "native_attention_gradients": attention, "new_module_gradients": group_norms,
                          "samples": metadata})
        med_sha_after = parameter_hash(u.med.state_dict())
        assert med_sha_before == med_sha_after
        report = guard.report()
        assert report["L506_opens"] == report["external_memory_opens"] == 0
        assert report["neighbor_target_opens"] == report["blocked_attempts"] == 0
        result = {"status": "PASS", "shared_initial_parameters_sha256": shared_sha,
                  "meddino_frozen_hash_unchanged": True,
                  "batch": 2, "accum": 1, "training_steps_in_readiness": 2,
                  "isolated_model_discarded": True, "steps": steps,
                  "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                  "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                  "elapsed_seconds": time.perf_counter() - started,
                  "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
                  "access": report}
        (ROOT / "READINESS.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print("READINESS_PASS", result["peak_allocated_gib"], result["peak_reserved_gib"], flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        (ROOT / "READINESS_FAILURE.txt").write_text(traceback.format_exc())
        raise
