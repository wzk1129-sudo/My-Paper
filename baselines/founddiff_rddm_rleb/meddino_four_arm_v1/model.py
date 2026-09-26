"""Unchanged Control Encoder/ZeroConv with an arm-specific MedDINO structure provider."""
from anatomy_controlnet_v1.model import ControlledUnet, MAIN_NAMES, ENCODER_NAMES, parameter_counts
from .structure import MultiProvider


class FourArmControlledUnet(ControlledUnet):
    def __init__(self, trained_main, arm):
        super().__init__(trained_main, "C1_CAPACITY")
        self.structure = MultiProvider(arm)
        self.arm = arm

    def forward(self, x, time, x_self_cond=None):
        self.structure.collect_diagnostics = self.collect_activity
        out = super().forward(x, time, x_self_cond)
        if self.collect_activity:
            self.activity.update(self.structure.last_activity)
            self.activity["edge_feature_rms"] = (
                [self.structure.last_activity[k + "_edge_rms"] for k in ("s256", "s128", "s64")]
                if self.structure.edge_enabled else None)
        return out
