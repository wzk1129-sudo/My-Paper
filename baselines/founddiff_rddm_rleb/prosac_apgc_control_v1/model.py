"""The V1 Control Encoder and four ZeroConv taps with only the provider swapped."""
from anatomy_controlnet_v1.model import ControlledUnet, MAIN_NAMES, ENCODER_NAMES, parameter_counts
from .structure import AnatomyProvider


class ProsacControlledUnet(ControlledUnet):
    def __init__(self, trained_main, arm, input_size):
        super().__init__(trained_main, "C1_CAPACITY")
        self.structure = AnatomyProvider(arm, input_size)
        self.arm = arm

    def forward(self, x, time, x_self_cond=None):
        out = super().forward(x, time, x_self_cond)
        if self.collect_activity:
            self.activity["anatomy_rms"] = self.structure.last_activity["afm_rms"]
            self.activity["edge_feature_rms"] = (
                None if self.arm == "C3_ANATOMY" else
                [self.structure.last_activity[k + "_efm_rms"]
                 for k in ("s256", "s128", "s64")])
        return out

