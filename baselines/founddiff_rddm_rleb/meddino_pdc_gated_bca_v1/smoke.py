"""Twenty independent PDC/BCA optimizer steps before formal training."""
import json
import math
import torch

from context_2p5d.data import ContextDataGuard
from mayo2016_smallset_v1.runtime import write_json
from .model import numeric_activity
from .protocol import HERE, config
from .runtime import PDCTrainState


def numeric(block):
    return {k: float(v) for k, v in block.last_activity.items()}


def main():
    cfg = config()
    assert not (HERE / "SMOKE_RESULT.json").exists()
    guard = ContextDataGuard(cfg, "FAST", allow_training=True)
    with guard:
        state = PDCTrainState("cuda:0", guard)
        u = state.model.model.unet0
        main_initial = u.init_conv.weight.detach().clone()
        rows = []
        for _ in range(20):
            row = state.update()
            rows.append(row)
            if state.step % 5 == 0:
                print("SMOKE", state.step, row["train_loss"], row["peak_vram_bytes"], flush=True)
        s = u.structure
        access = guard.report()
        assert access["L506_opens"] == access["external_memory_opens"] == 0
        assert access["neighbor_target_opens"] == access["blocked_attempts"] == 0
        assert state.nonfinite_count == 0 and all(v > 0 for v in state.grad_usage.values())
        assert all(math.isfinite(row["train_loss"]) for row in rows)
        assert bool((u.init_conv.weight.detach() - main_initial).abs().max() > 0)
        assert all(not p.requires_grad and p.grad is None for p in s.anatomy.med.parameters())
        assert not s.anatomy.med.training
        assert not hasattr(u, "control")
        shapes = s.last_shapes
        expected = {
            "x1": [2,20,512,512], "x2": [2,40,256,256],
            "x3": [2,80,128,128], "x4": [2,80,64,64],
            "H128": [2,64,128,128], "H64": [2,128,64,64],
            "S128": [2,64,128,128], "S64": [2,128,64,64],
        }
        assert shapes == expected, (shapes, expected)
        activity = numeric_activity(s)
        for stage in ("128", "64"):
            a = activity[stage]
            assert all(math.isfinite(v) for v in a.values())
            assert a["gate_spatial_std"] > 0
            assert a["raw_shape_rms"] > 0 and a["weighted_shape_rms"] > 0
        for bca in (u.bca128, u.bca64):
            assert float(bca.last_activity["weighted_residual_rms"]) > 0
        assert state.peak_vram < 20 * 2**30
        result = {
            "status": "PASS", "steps": 20, "initialization": state.init_audit,
            "train_stream_sha256": state.stream_digest.hex(),
            "mean_seconds_per_step": state.train_seconds / state.step,
            "peak_vram_bytes": state.peak_vram,
            "loss_first_last": [rows[0]["train_loss"], rows[-1]["train_loss"]],
            "max_gradient_norm_pre_clip": max(r["gradient_norm_pre_clip"] for r in rows),
            "gradient_usage": state.grad_usage,
            "main_parameter_changed": True,
            "shapes": shapes, "structure_activity": activity,
            "beta128": float(s.beta128.detach()), "beta64": float(s.beta64.detach()),
            "alpha128": float(u.bca128.alpha.detach()), "alpha64": float(u.bca64.alpha.detach()),
            "attention128": numeric(u.bca128), "attention64": numeric(u.bca64),
            "access": access,
        }
        write_json(HERE / "SMOKE_RESULT.json", result)
        print(json.dumps({"status": "PASS", "seconds_per_step": result["mean_seconds_per_step"],
                          "peak_vram_bytes": result["peak_vram_bytes"]}), flush=True)


if __name__ == "__main__":
    main()
