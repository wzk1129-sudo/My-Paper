"""Import-only compatibility for unused official-RDDM Trainer dependencies in flexict.

The M0/M1 path uses RDDMCTBackbone and RDDMCTDiffusion, never official Trainer.
Any accidental use of these placeholders fails immediately.
"""
import importlib.util
import sys
import types


class ForbiddenTrainerDependency:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("Unused official-RDDM Trainer dependency was invoked")


for module_name, symbol in (
    ("Augmentor", None),
    ("accelerate", "Accelerator"),
    ("ema_pytorch", "EMA"),
):
    if importlib.util.find_spec(module_name) is None:
        module = types.ModuleType(module_name)
        if symbol:
            setattr(module, symbol, ForbiddenTrainerDependency)
        sys.modules[module_name] = module

# Official image-dataset factory is referenced only inside the unused Trainer.
factory = types.ModuleType("datasets.get_dataset")
factory.dataset = ForbiddenTrainerDependency
sys.modules["datasets.get_dataset"] = factory
