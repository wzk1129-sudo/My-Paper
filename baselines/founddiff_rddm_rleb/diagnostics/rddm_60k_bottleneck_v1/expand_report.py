"""Append complete distribution tables from saved numerical summaries only."""
import hashlib,json
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent
OUT=HERE.parents[1]/"experiments/diagnostics/rddm_60k_bottleneck_v1"
load=lambda n:json.loads((OUT/n).read_text())
f=load("FREQUENCY_SUMMARY.json")["bands"]
t=load("TIMESTEP_SUMMARY.json")["timesteps"]
p=OUT/"DIAGNOSTIC_REPORT.md"
text=p.read_text()
assert "## Complete aggregate distributions" not in text
lines=["## Complete aggregate distributions","",
       "Per-slice frequency power ratios (mean, median, IQR, contiguous-block 95% CI for median):","",
       "| Band | Ratio | Mean | Median | IQR | 95% CI |",
       "|---|---|---:|---:|---:|---:|"]
for band in ("LOW","MID","HIGH"):
    for key,label in (("remaining_error_ratio_band","Epred/Eld"),
                      ("normalized_error_pred_band","Epred/Egt")):
        s=f[band][key]
        lines.append(f"| {band} | {label} | {s['mean']:.6f} | {s['median']:.6f} | "
                     f"{s['iqr']:.6f} | {s['block_bootstrap_95ci_median']} |")
lines += ["","Timestep probe distributions across 128 slices × 3 noise replicates "
          "(descriptive, not independent-patient intervals):","",
          "| t | Quantity | Mean | Median | SD | IQR | Mean within-slice replicate SD |",
          "|---:|---|---:|---:|---:|---:|---:|"]
for step in (0,100,250,499,750,999):
    v=t[str(step)]
    for key in ("residual_nmse","psnr"):
        s=v[key]
        replicate=v["noise_replicate_variability"][key+"_within_slice_std_mean"]
        lines.append(f"| {step} | {key} | {s['mean']:.6f} | {s['median']:.6f} | "
                     f"{s['std']:.6f} | {s['iqr']:.6f} | {replicate:.6f} |")
lines += ["","Full numerical per-slice rows and all four image metrics are in TIMESTEP_PROBE.csv; "
          "all band powers and ratios are in FREQUENCY_BANDS.csv.",""]
p.write_text(text+"\n".join(lines))
prov_path=OUT/"RUN_PROVENANCE.json"
prov=json.loads(prov_path.read_text())
prov["distribution_report_pass"]={"code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                  "completed_utc":datetime.now(timezone.utc).isoformat(),
                                  "opened_medical_images":0,"model_forwards":0}
prov_path.write_text(json.dumps(prov,indent=2,allow_nan=False)+"\n")
print("DISTRIBUTION_TABLES_PASS")
