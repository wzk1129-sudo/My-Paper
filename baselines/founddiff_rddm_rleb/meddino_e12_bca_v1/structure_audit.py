"""Read-only shape and Q/K/V provenance audit on allowed fast-train input."""
import json
import torch

from context_2p5d.data import ContextDataGuard
from context_2p5d.model import training_loss
from mayo2016_smallset_v1.runtime import write_json
from .protocol import HERE, config
from .runtime import BCATrainState


def main():
    cfg = config()
    guard = ContextDataGuard(cfg, "FAST", allow_training=True)
    with guard:
        state = BCATrainState("cuda:0", guard)
        captures = {}
        u = state.model.model.unet0
        hooks = []
        for name in ("bca128", "bca64"):
            def hook(module, args, output, name=name):
                captures[name] = {
                    "H": list(args[0].shape), "S": list(args[1].shape),
                    "out": list(output.shape),
                    "window": module.window_size, "heads": module.heads,
                    "dim_head": module.dim_head,
                }
            hooks.append(getattr(u, name).register_forward_hook(hook))
        prev, center, next_, target, meta = state.stream.next(2)
        tensors = [v.to(state.device) for v in (prev, center, next_, target)]
        with torch.no_grad():
            loss = training_loss(state.model, tensors[3], tensors[1], tensors[0], tensors[2])
        for handle in hooks:
            handle.remove()
        assert bool(torch.isfinite(loss))
        assert captures["bca128"]["H"] == [2, 128, 128, 128]
        assert captures["bca128"]["S"] == [2, 64, 128, 128]
        assert captures["bca64"]["H"] == [2, 256, 64, 64]
        assert captures["bca64"]["S"] == [2, 128, 64, 64]
        access = guard.report()
        assert access["L506_opens"] == access["external_memory_opens"] == 0
        assert access["neighbor_target_opens"] == access["blocked_attempts"] == 0
        result = {
            "status": "PASS", "shapes": captures,
            "Q_source": "RDDM encoder main feature, argument 0 to each BCA module",
            "K_V_source": "E12 HierarchicalSALFProvider hint, argument 1",
            "step_zero_training_loss_finite": True,
            "meddino_frozen": all(not p.requires_grad for p in u.structure.med.parameters()),
            "controlnet_absent": not hasattr(u, "control"),
            "access": access,
        }
        write_json(HERE / "STRUCTURE_AUDIT.json", result)
        print(json.dumps({"status": "PASS", "shapes": captures}), flush=True)


if __name__ == "__main__":
    main()
