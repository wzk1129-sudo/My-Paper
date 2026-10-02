"""Offline stdlib-only verification; no image, checkpoint, GPU, network or training access."""
import csv,json,hashlib,math
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def read(path):return json.loads(path.read_text())
def check():
    manifest=read(ROOT/"PUBLICATION_MANIFEST.json")
    for name,item in manifest["files"].items():
        path=ROOT/name
        assert path.is_file() and not path.is_symlink(),name
        raw=path.read_bytes()
        assert len(raw)==item["bytes"] and hashlib.sha256(raw).hexdigest()==item["sha256"],name
        assert hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()==item["git_blob_sha1"],name
    summary=read(ROOT/"FINAL_SUMMARY.json")
    arms=summary["arms"]
    source=read(ROOT/"audit_artifacts/TRAINING_SOURCE_MANIFEST.json")
    for arm,files in source["actual_source_files_by_arm"].items():
        for name,item in files.items():
            assert hashlib.sha256((ROOT/item["published_path"]).read_bytes()).hexdigest()==item["sha256"],(arm,name)
    init=read(ROOT/"audit_artifacts/INITIALIZATION_AUDIT.json")
    for key in init["matched_fields"]:
        assert init["initial_by_arm"][arms[0]][key]==init["initial_by_arm"][arms[1]][key],key
    with (ROOT/"TRAINING_DIAGNOSTICS.csv").open() as f:rows=list(csv.DictReader(f))
    trains={a:[r for r in rows if r["arm"]==a] for a in arms}
    for arm,records in trains.items():
        assert [int(r["optimizer_step"]) for r in records]==list(range(1,8001))
        assert all(math.isfinite(float(r["attention_gradient_norm"])) and float(r["attention_gradient_norm"])>0 for r in records)
        finite=[r for r in records if r["parameters_finite"]]
        assert len(finite)==81 and all(r["parameters_finite"]=="True" for r in finite)
    assert all(x["train_stream_sha256"]==y["train_stream_sha256"] for x,y in zip(trains[arms[0]],trains[arms[1]]))
    random=read(ROOT/"audit_artifacts/RANDOM_STREAM_AUDIT.json")
    assert random["status"]=="PASS" and random["first_batch_by_arm"][arms[0]]==random["first_batch_by_arm"][arms[1]]
    assert trains[arms[0]][-1]["train_stream_sha256"]==random["final_stream_sha256"]
    with (ROOT/"VALIDATION_HISTORY.csv").open() as f:history=list(csv.DictReader(f))
    keys=("psnr","ssim","rmse_hu","mae_hu")
    indices=list(map(int,(ROOT/"splits/L333_fast64.txt").read_text().splitlines()))
    assert len(indices)==64
    for row in history:
        step=int(row["step"]);per={}
        for role,arm in zip(("control","candidate"),arms):
            per[role]=read(ROOT/"audit_artifacts"/arm/("validation_step_%d.json"%step))
            assert [r["slice_index"] for r in per[role]]==indices
            for key in keys:
                average=sum(r[key] for r in per[role])/64
                assert abs(average-float(row[role+"_"+key]))<=1e-9,(role,step,key)
        for key in keys:
            assert abs(float(row["candidate_"+key])-float(row["control_"+key])-float(row["delta_"+key]))<=1e-12
    for key in keys:
        assert float(history[-1]["candidate_"+key])==summary["candidate_8k"][key]
        assert float(history[-1]["control_"+key])==summary["control_8k"][key]
    frozen=read(ROOT/"audit_artifacts/MEDDINO_FROZEN_AUDIT.json")
    assert frozen["status"]=="PASS" and frozen["final_training_unchanged_hash_assertion_passed"]
    assert frozen["initial_state_sha256"]==frozen["checkpoint_reload_frozen_state_sha256"]
    original=read(ROOT/"audit_artifacts"/arms[1]/"validation_step_8000.json")
    reload=read(ROOT/"DIAGNOSTIC_NORMAL_PER_SLICE.json")
    assert len(original)==len(reload)==64
    assert all(all(x[k]==y[k] for k in (*keys,"validation_loss","sampling_seed","slice_index")) for x,y in zip(original,reload))
    access=read(ROOT/"DATA_ACCESS_AUDIT.json")
    for scope in ("arms","inference_diagnostics"):
        for audit in access[scope].values():
            assert all(audit[k]==0 for k in ("L506_opens","external_memory_opens","neighbor_target_opens","blocked_attempts"))
    assert summary["matched_all_8000_streams"] and not summary["full_l333_executed"]
    for mode in ("NORMAL","CONDITION_OFF","SHUFFLED_MEDDINO"):
        result=read(ROOT/("DIAGNOSTIC_"+mode+".json"))["metrics"]
        assert result["slices"]==64 and result["condition_mode"]==mode
        assert all(result[k]==summary["inference_diagnostics"][mode][k] for k in keys)
    print("OFFLINE_AUDIT_PASS: %d files; both 8000-step streams; 6 Fast64 nodes; initialization, frozen MedDINO, checkpoint reload, diagnoses and data boundary"%len(manifest["files"]))
if __name__=="__main__":check()
