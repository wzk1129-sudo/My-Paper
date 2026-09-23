"""Final consistency edit of numerical-only report; no model or data access."""
import hashlib
import json
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent
OUT=HERE.parents[1]/"experiments/diagnostics/rddm_60k_bottleneck_v1"
summary_path=OUT/"DIAGNOSTIC_SUMMARY.json"
report_path=OUT/"DIAGNOSTIC_REPORT.md"
prov_path=OUT/"RUN_PROVENANCE.json"
summary=json.loads(summary_path.read_text())
assert summary["specificity_review"]["actual_sampler_timesteps_uniquely_weak"] is False
assert summary["mechanism_categories"]==["Timestep"]
summary["flag_only_categories"]=summary.pop("mechanism_categories")
summary["mechanism_categories"]=["none-clear"]
summary_path.write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
report=report_path.read_text()
old="Mechanism categories worth testing: ['Timestep']."
new="Flag-only candidate category: Timestep; the specificity review below does not support a unique weakness at t=499 or t=999."
assert report.count(old)==1
report_path.write_text(report.replace(old,new))
prov=json.loads(prov_path.read_text())
prov["report_finalizer"]={"code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          "completed_utc":datetime.now(timezone.utc).isoformat(),
                          "opened_medical_images":0,"model_forwards":0}
prov_path.write_text(json.dumps(prov,indent=2,allow_nan=False)+"\n")
print("REPORT_CONSISTENCY_PASS")
