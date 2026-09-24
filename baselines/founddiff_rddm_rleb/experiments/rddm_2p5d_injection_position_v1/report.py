"""Lightweight cross-arm result summary; no images or model weights."""
import csv
import json
from .model import ARMS
from .protocol import HERE, STEPS, references
from mayo2016_smallset_v1.runtime import write_json


def write_csv(path,rows):
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)


def build():
    ref=references();summaries={};rows=[]
    for arm in ARMS:
        summary=json.loads((HERE/"runs"/arm/"final_summary.json").read_text())
        assert summary["arm"]==arm
        assert summary["status"]=="COMPLETE"
        summaries[arm]=summary
        for record in summary["history"]:
            step=int(record["step"])
            rows.append({
              "arm":arm,"step":step,"status":summary["status"],
              "psnr":record["psnr"],"baseline_psnr":ref[step]["psnr"],
              "delta_psnr":record["delta_psnr"],"ssim":record["ssim"],
              "rmse_hu":record["rmse_hu"],"mae_hu":record["mae_hu"],
              "validation_loss":record["validation_loss"],
              "train_loss":record["train_loss"],
              "gradient_norm_pre_clip":record["gradient_norm_pre_clip"],
              "learning_rate":record["learning_rate"],
              "parameter_count":record["parameter_count"],
              "peak_vram_bytes":record["peak_vram_bytes"],
              "seconds_per_step":record["seconds_per_step"],
              "validation_seconds":record["validation_seconds"],
              "train_stream_sha256":record["train_stream_sha256"],
            })
    write_csv(HERE/"POSITION_SCREEN_RESULTS.csv",rows)
    out={
      "arms":summaries,"baseline_reference":{str(k):v for k,v in ref.items()},
      "comparison":"matched 2P5D_DIRECT_CONCAT Fast64; 2k/4k/6k/8k/10k/12k",
      "promotion_rule":"delta 8k,10k,12k >0 and delta12k >=0.05 dB",
      "test_patient_access":sum(s["data_access"]["L506_opens"] for s in summaries.values()),
      "external_memory_access":sum(s["data_access"]["external_memory_opens"] for s in summaries.values()),
      "neighbor_target_access":sum(s["data_access"]["neighbor_target_opens"] for s in summaries.values()),
    }
    write_json(HERE/"POSITION_SCREEN_RESULTS.json",out)
    lines=[
      "# RDDM 2.5D Context Injection Position Screen V1","",
      "Three independent context-injection positions, each initialized from the canonical RDDM step-0 weights. L067/L143 optimization; fixed L333 Fast64 EMA. The Direct Concat reference is the matched 12k 2.5D run.",
      "",
      "| Arm | Step | PSNR | Delta PSNR | SSIM | RMSE HU | MAE HU | Params | Peak VRAM GiB | s/step |",
      "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(f"| {r['arm']} | {r['step']} | {r['psnr']:.6f} | {r['delta_psnr']:+.6f} | "
          f"{r['ssim']:.6f} | {r['rmse_hu']:.4f} | {r['mae_hu']:.4f} | "
          f"{r['parameter_count']} | {r['peak_vram_bytes']/2**30:.3f} | {r['seconds_per_step']:.4f} |")
    for arm,s in summaries.items():
        lines += ["",f"## {arm}","",
          f"Status: {s['status']}; steps: {s['optimizer_steps']}; promotion: {s['flags']['promotion']}.",
          f"Training seconds: {s['training_seconds']:.1f}; nonfinite: {s['nonfinite_count']}.",
          f"Context parameter usage: {json.dumps(s['parameter_usage'])}."]
        if s["full_l333"]:
            f=s["full_l333"]
            lines.append("Full L333 paired delta: "+json.dumps(f["delta_mean"]))
    lines += ["","L506, external-memory, and neighbor-target opens: "
              +f"{out['test_patient_access']}, {out['external_memory_access']}, {out['neighbor_target_access']}.",
              "Decoder C64 stage is instantiated but unused because decoder injections are at 128/256/512. Engineering screening only; no patient-level statistical significance claim.",""]
    (HERE/"POSITION_SCREEN_REPORT.md").write_text("\n".join(lines))
    print("REPORT_COMPLETE",json.dumps({
      "arms":{a:s["status"] for a,s in summaries.items()},
      "rows":len(rows)}),flush=True)
    return out

if __name__=="__main__":build()
