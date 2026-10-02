"""Verify CPU metric reduction drift using one fixed approved Fast64 slice."""
import gc,json,hashlib,datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import torch
from model import build_model,load_state_without_med
from postprocess import run_paths
from train import ARMS,tensor_state_hash
from provenance import ROOT,write_json
from fast_dev.protocol import CONFIG,check_config
from context_data import ContextPairs,ContextDataGuard
from mayo2016_smallset_v1.runtime import seed_everything
from mayo2016_smallset_v1.frozen_protocol import slice_metrics

def main():
    cfg=json.loads(CONFIG.read_text());check_config(cfg)
    torch.set_num_threads(cfg["training"]["cpu_threads"]);seed_everything(20260829)
    device=torch.device("cuda:0");output=run_paths()[ARMS[1]]
    ck=torch.load(output/"checkpoints/step_8000.pt",map_location="cpu",weights_only=False)
    assert ck["optimizer_step"]==ck["ema_updates"]==8000
    model=build_model(True).to(device).eval()
    load_state_without_med(model,ck["ema_state_dict"]);del ck;gc.collect()
    initial=json.loads((output/"INITIAL_STATE.json").read_text())
    frozen_hash=tensor_state_hash(model.model.unet0.med.state_dict())
    assert frozen_hash==initial["meddino_initial_weights_sha256"]
    original=json.loads((output/"validation_step_8000.json").read_text())
    prior=json.loads((ROOT/"runs/postprocess_attempt1_metadata_20261002/DIAGNOSTIC_NORMAL_PER_SLICE.json").read_text())
    with ContextDataGuard(cfg,"FAST",allow_training=False) as guard:
        dataset=ContextPairs(cfg,"validation","FAST");dataset.bind_guard(guard)
        prev,center,next_,target,meta=dataset.load(1)
        assert meta["slice_index"]==10
        prev,center,next_,target=[torch.from_numpy(x).unsqueeze(0).to(device) for x in (prev,center,next_,target)]
        with torch.no_grad():
            u=model.model.unet0;u.set_context(prev,center,next_)
            seed=cfg["evaluation"]["validation_seed"]+meta["slice_index"]
            torch.manual_seed(seed);torch.cuda.manual_seed(seed)
            prediction=model.sample([center],last=True)[-1]
            pred_hash=hashlib.sha256(prediction.detach().cpu().numpy().tobytes()).hexdigest()
            metrics={}
            for threads in (2,4,2):
                torch.set_num_threads(threads)
                metrics[str(threads)]=slice_metrics(prediction,target,cfg)
            assert pred_hash==hashlib.sha256(prediction.detach().cpu().numpy().tobytes()).hexdigest()
            u.clear_context()
        access=guard.report()
    assert all(access[k]==0 for k in ("L506_opens","external_memory_opens","neighbor_target_opens","blocked_attempts"))
    keys=("psnr","ssim","rmse_hu","mae_hu")
    assert all(metrics["2"][k]==original[1][k] for k in keys)
    assert all(metrics["4"][k]==prior[1][k] for k in keys)
    assert metrics["2"]["psnr"]!=metrics["4"]["psnr"]
    summary={"status":"PASS","at":datetime.datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "cause":"CPU float32 mean reduction uses a different reduction order with 2 vs 4 threads; same unchanged prediction reproduces original training and failed postprocess metrics exactly",
        "training_cpu_threads":2,"failed_postprocess_cpu_threads":4,"corrected_postprocess_cpu_threads":cfg["training"]["cpu_threads"],
        "slice_index":10,"prediction_sha256":pred_hash,"prediction_unchanged":True,
        "metrics_threads_2":{k:metrics["2"][k] for k in keys},"metrics_threads_4":{k:metrics["4"][k] for k in keys},
        "metrics_match_original_and_failed_diagnostic_exactly":True,
        "frozen_meddino_hash_matches_training":True,"meddino_weights_sha256":frozen_hash,
        "prior_normal_aggregate_psnr":json.loads((ROOT/"runs/postprocess_attempt1_metadata_20261002/DIAGNOSTIC_NORMAL.json").read_text())["metrics"]["psnr"],
        "original_normal_aggregate_psnr":float(sum(r["psnr"] for r in original)/64),
        "data_access":access,"training_restarted":False}
    write_json(ROOT/"POSTPROCESS_THREAD_AUDIT.json",summary)
    print("THREAD_AUDIT_PASS "+json.dumps({k:summary[k] for k in ("status","metrics_threads_2","metrics_threads_4","prediction_unchanged","frozen_meddino_hash_matches_training")}))
if __name__=="__main__":main()
