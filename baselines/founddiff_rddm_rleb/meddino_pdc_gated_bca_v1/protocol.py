"""Frozen E12-BCA Fast6k protocol with PiDiNet PDC anatomy-gated shape stream."""
import copy
import json
from mayo2016_smallset_v1.runtime import ROOT, sha256
from meddino_e12_bca_v1.protocol import config as bca_config, verify as verify_bca

HERE = ROOT / "experiments/meddino_pdc_gated_bca_v1"
ARM = "PDC_GATED_BCA"
STEPS = (0, 500, 1000, 2000, 4000, 6000)
PARENT_COMMIT = "48a78da568be3bb35ec33d9ca864ccb9b795c4c5"
FORMAL_SOURCE_SHA = "e8ed93223fbdf4a0e2521f73c7a4f213561f6a191a1a7b2a5f95934b85ebf574"
FAST64_SPLIT_SHA = "0d3170fd2462c7b19d1cc0f187b12e03bac2c9b15babca7c1d881568adde6895"
EVALUATOR_SHA = "94ddc8b2d9e9bae08457bb0f4e57c20f2d282eb80ef5b77c50304818f3f29366"
UPSTREAM_COMMIT = "d21aa881ed9c628571636fad39acfe1fad517ebd"


def verify():
    verify_bca()
    assert sha256(ROOT / "src/rddm_rleb.py") == FORMAL_SOURCE_SHA
    assert sha256(ROOT / "splits/L333_fast64.txt") == FAST64_SPLIT_SHA
    assert sha256(ROOT / "meddino_full_l333_v1/evaluate.py") == EVALUATOR_SHA
    audit = json.loads((HERE / "PIDINET_UPSTREAM_AUDIT.json").read_text())
    assert audit["upstream_commit"] == UPSTREAM_COMMIT
    assert json.loads((HERE / "OFFICIAL_PDC_EQUIVALENCE.json").read_text())["status"] == "PASS"
    return True


def config():
    verify()
    cfg = copy.deepcopy(bca_config())
    cfg["experiment_name"] = "MedDINO-PDC-Gated-BCA-V1"
    cfg["description"] = "Canonical Direct + unchanged E12 anatomy/BCA + trainable PiDiNet-Tiny PDC shape stream gated by anatomy"
    cfg["arm"] = ARM
    cfg["parent_commit"] = PARENT_COMMIT
    cfg["initialization"] = "same canonical Direct 12k EMA as E12-BCA; no E12-BCA 6k warm start"
    cfg["pdc_shape_stream"] = {
        "source": "https://github.com/hellozhuo/pidinet",
        "upstream_commit": UPSTREAM_COMMIT,
        "input": "center LDCT only, CT grayscale 1-channel",
        "preprocessing": "center in [-1,1] -> HU=(center+1)*2048-1024 -> clamp[-1000,1000] -> /1000",
        "backbone": "official PiDiNet-Tiny 20/40/80/80, carv4 PDC cd/ad/rd/cv per stage",
        "pretrained": False, "edge_heads": False, "CDCM": False, "CSAM": False,
        "x3_shape": [80,128,128], "x4_shape": [80,64,64],
    }
    cfg["shape_fusion"] = {
        "projection": "1x1 Conv + GELU, x3 80->64, x4 80->128",
        "gate": "MedDINO H -> 1x1 Conv -> concat projected PDC -> GroupNorm(1,C+1) -> 1x1 Conv -> GELU -> 1x1 Conv -> Sigmoid",
        "residual_gating": "projected_shape * (1 + gate)",
        "fused_structure": "S=H+beta*gated_shape",
        "beta128_init": 0.1, "beta64_init": 0.1,
    }
    cfg["backbone_cross_attention"] = copy.deepcopy(bca_config()["backbone_cross_attention"])
    assert cfg["training"]["max_optimizer_steps"] == 6000
    assert cfg["training"]["batch_size"] == 2
    assert cfg["training"]["learning_rate"] == 2e-4
    assert cfg["training"]["amp"] is False
    assert cfg["ema"]["decay"] == .995
    assert cfg["diffusion"]["sampling_timesteps"] == 2
    assert cfg["diffusion"]["sum_scale"] == .01
    return cfg
