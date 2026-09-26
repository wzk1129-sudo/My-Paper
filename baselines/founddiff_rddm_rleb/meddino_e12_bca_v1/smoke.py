"""Twenty-step isolated structural and gradient smoke; no formal checkpoint."""
import json
import math
import time
import torch

from context_2p5d.data import ContextDataGuard
from mayo2016_smallset_v1.runtime import write_json
from .protocol import HERE, config
from .runtime import BCATrainState


def main():
    cfg = config()
    assert not (HERE / "SMOKE_RESULT.json").exists()
    guard = ContextDataGuard(cfg, "FAST", allow_training=True)
    with guard:
        state = BCATrainState("cuda:0", guard)
        u = state.model.model.unet0
        initial_main = u.init_conv.weight.detach().clone()
        alpha_initial = (float(u.bca128.alpha), float(u.bca64.alpha))
        rows = []
        for _ in range(20):
            row = state.update()
            rows.append(row)
            if state.step % 5 == 0:
                print("SMOKE", state.step, row["train_loss"], row["peak_vram_bytes"], flush=True)
        def activity(block):
            return {k: float(v) for k, v in block.last_activity.items()}
        report = guard.report()
        assert report["L506_opens"] == report["external_memory_opens"] == 0
        assert report["neighbor_target_opens"] == report["blocked_attempts"] == 0
        assert state.nonfinite_count == 0 and all(v > 0 for v in state.grad_usage.values())
        assert all(math.isfinite(row["train_loss"]) for row in rows)
        assert bool((u.init_conv.weight.detach() - initial_main).abs().max() > 0)
        assert all(float(v) > 0 for v in (
            u.bca128.last_activity["output_rms"], u.bca64.last_activity["output_rms"],
            u.bca128.last_activity["weighted_residual_rms"],
            u.bca64.last_activity["weighted_residual_rms"]))
        assert all(not p.requires_grad and p.grad is None for p in u.structure.med.parameters())
        assert not u.structure.med.training
        assert all(not hasattr(u, name) for name in ("control", "local256", "cross128", "cross64"))
        result = {
            "status": "PASS", "steps": 20, "initialization": state.init_audit,
            "train_stream_sha256": state.stream_digest.hex(),
            "mean_seconds_per_step": state.train_seconds / state.step,
            "peak_vram_bytes": state.peak_vram,
            "loss_first_last": [rows[0]["train_loss"], rows[-1]["train_loss"]],
            "max_gradient_norm_pre_clip": max(r["gradient_norm_pre_clip"] for r in rows),
            "gradient_usage": state.grad_usage,
            "main_parameter_changed": True,
            "alpha_initial": alpha_initial,
            "alpha_final": (float(u.bca128.alpha), float(u.bca64.alpha)),
            "attention128": activity(u.bca128), "attention64": activity(u.bca64),
            "structure_activity": {
                "s128_rms": float(u.structure.last_hierarchy["H128_rms"]),
                "s64_rms": float(u.structure.last_hierarchy["H64_rms"]),
            } if "H128_rms" in u.structure.last_hierarchy else
            {k: float(v) for k, v in u.structure.last_hierarchy.items()
             if isinstance(v, (float, int))},
            "access": report,
        }
        write_json(HERE / "SMOKE_RESULT.json", result)
        print(json.dumps({"status": "PASS", "seconds_per_step": result["mean_seconds_per_step"],
                          "peak_vram_bytes": result["peak_vram_bytes"],
                          "grad_usage": result["gradient_usage"]}), flush=True)


if __name__ == "__main__":
    main()
