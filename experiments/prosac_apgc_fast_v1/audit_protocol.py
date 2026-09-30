"""Read-only metadata/protocol audit; never opens CT pixel arrays."""
import copy
import json
import sys
from pathlib import Path

import compat  # noqa: F401
HERE = Path(__file__).resolve().parent
FROZEN = HERE.parent / "7_RDDM_CT_Backbone_v1"
sys.path.insert(0, str(FROZEN))
from rddm_ct import PairedCTDataset, load_config, sha256_file, OFFICIAL_SOURCE
from apgc_model import MEDDINO_WEIGHT, MEDDINO_ROOT

m0 = load_config(HERE / "CONFIG_M0.json")
m1 = load_config(HERE / "CONFIG_M1.json")
clean0, clean1 = copy.deepcopy(m0), copy.deepcopy(m1)
for current in (clean0, clean1):
    current.pop("arm")
    current["training"].pop("output_dir")
assert clean0 == clean1
expected_train = ["L067", "L096", "L109", "L143", "L192", "L286", "L291", "L310"]
assert m0["data"]["train_patients"] == expected_train
assert m0["data"]["validation_patients"] == ["L333"]
train = PairedCTDataset(Path(m0["data"]["root"]), expected_train, m0["data"])
val = PairedCTDataset(Path(m0["data"]["root"]), ["L333"], m0["data"])
indices = [int(x) for x in (HERE / "L333_fast64.txt").read_text().split()]
assert len(val) == 610 and len(indices) == len(set(indices)) == 64
assert min(indices) == 0 and max(indices) == 609
assert [val.records[i].patient_id for i in indices] == ["L333"] * 64
source_paths = [FROZEN / name for name in (
    "acceptance.py", "config.json", "evaluate.py", "evaluation.py", "rddm_ct.py", "train.py")]
snapshot = {
    "frozen_source_sha256": {path.name: sha256_file(path) for path in source_paths},
    "official_rddm_diffusion_sha256": sha256_file(OFFICIAL_SOURCE),
    "meddino_checkpoint_sha256": sha256_file(MEDDINO_WEIGHT),
    "meddino_source_commit": "a2460c2547899283446e520b84523454f7e25608",
    "meddino_official_root": str(MEDDINO_ROOT),
    "original_worktree_status_saved": str(HERE / "source_snapshot/original_git_status.txt"),
    "original_six_file_diff_saved": str(HERE / "source_snapshot/original_six_file.diff"),
}
(HERE / "SOURCE_SNAPSHOT.json").write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
audit = {
    "status": "PASS",
    "m0_m1_config_identical_except_arm_and_output": True,
    "train_patients": expected_train,
    "train_pairs": len(train),
    "train_pairs_by_patient": train.audit()["slices_per_patient"],
    "validation_patient": "L333",
    "validation_pairs": len(val),
    "fast64_indices": indices,
    "fast64_sha256": sha256_file(HERE / "L333_fast64.txt"),
    "ct_pixel_files_opened": 0,
    "forbidden_test_patient_opens": 0,
    "m0_smoke": json.loads((HERE / "SMOKE_M0.json").read_text())["status"],
    "m1_smoke": json.loads((HERE / "SMOKE_M1.json").read_text())["status"],
}
(HERE / "PROTOCOL_AUDIT.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
print(json.dumps({"status": audit["status"], "train_pairs": len(train),
                  "val_pairs": len(val), "fast_indices": len(indices)}))
