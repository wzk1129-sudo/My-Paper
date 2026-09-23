"""Numerical-only interpretation pass; opens no CT, model, or checkpoint."""
import json
from pathlib import Path
from datetime import datetime, timezone
import hashlib

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "experiments/diagnostics/rddm_60k_bottleneck_v1"


def load(name):
    return json.loads((OUT/name).read_text())


def save(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def main():
    control = load("CONTROL.json")
    assert control["pass"]
    s = load("STRUCTURE_REGION_SUMMARY.json")["metrics"]
    f = load("FREQUENCY_SUMMARY.json")["bands"]
    t = load("TIMESTEP_SUMMARY.json")["timesteps"]
    h = load("HARD_SLICE_SUMMARY.json")
    result = load("DIAGNOSTIC_SUMMARY.json")
    audit = load("DATA_ACCESS_AUDIT.json")
    assert audit["L506_opens"] == audit["external_memory_opens"] == audit["optimizer_steps"] == 0
    assert result["full_validation_slices"] == 610 and result["timestep_probe_rows"] == 2304
    assert "specificity_review" not in result
    med = lambda obj, key: obj[key]["median"]
    other = (100, 250, 499, 750, 999)
    other_psnr = [med(t[str(x)], "psnr") for x in other]
    other_nmse = [med(t[str(x)], "residual_nmse") for x in other]
    psnr_spread = max(other_psnr)-min(other_psnr)
    nmse_spread = max(other_nmse)-min(other_nmse)
    assert psnr_spread < .02 and nmse_spread < .005
    result["specificity_review"] = {
        "timestep_flag_trigger": "comparison to unusually easy t=0",
        "t_100_to_999_median_psnr_spread_db": psnr_spread,
        "t_100_to_999_median_nmse_spread": nmse_spread,
        "actual_sampler_timesteps_uniquely_weak": False,
        "interpretation": "The preregistered flag triggers, but t=499/999 are not materially worse than t=100/250/750.",
        "supported_specific_bottleneck": "none-clear",
        "exploratory_mechanism_categories": ["Structure", "Frequency"],
        "exploratory_caveat": "Neither structure nor frequency preregistered signal threshold passed; no module gain is implied.",
        "coupling_note": "high_frequency_remaining_ratio shares prediction error with PSNR gain, so their Spearman rho is not an independent causal predictor.",
        "duplicate_predictor_note": "raw_ldct_rmse and input_residual_rms are identical under this evaluation-window definition."
    }
    save("DIAGNOSTIC_SUMMARY.json", result)
    report = OUT/"DIAGNOSTIC_REPORT.md"
    text = report.read_text()
    assert "## Direct answers and flag specificity" not in text
    a=s["edge_error_concentration"]["median"]
    b=s["edge_flat_mae_ratio_pred"]["median"]
    ledge=s["edge_rmse_remaining_ratio"]["median"]
    lflat=s["flat_rmse_remaining_ratio"]["median"]
    gl=s["gradient_magnitude_ratio_ldct"]["median"]
    gp=s["gradient_magnitude_ratio_pred"]["median"]
    ol=s["orientation_cos_ldct"]["median"]
    op=s["orientation_cos_pred"]["median"]
    rho=h["spearman"]["prediction_rmse"]
    lines = [
        "## Direct answers and flag specificity", "",
        f"**A. Error location.** Edge error concentration is {a:.3f} and edge/flat MAE ratio is {b:.3f}; the preregistered structure flag does not pass. Edge RMSE remaining is {ledge:.3f} versus {lflat:.3f} in flat regions, so edge errors are repaired less completely. Under the fixed unequal-width frequency bands, MID has the largest median absolute residual power. Hard-slice error is not uniform along this single stack. No uniquely weak sampler timestep is established.", "",
        f"**B. Boundary preservation.** Raw LDCT edge orientation cosine is {ol:.3f}; prediction increases it to {op:.3f}. LDCT gradient magnitude ratio is {gl:.3f} and prediction is {gp:.3f}, indicating modest amplitude attenuation despite improved orientation. The oversmoothing heuristic (<0.9) does not pass.", "",
        "**C. Frequency.** Median residual power E_pred is "
        + ", ".join(f"{band} {f[band]['e_pred_band']['median']:.0f}" for band in ("LOW","MID","HIGH"))
        + ". Median E_pred/E_ld is "
        + ", ".join(f"{band} {f[band]['remaining_error_ratio_band']['median']:.3f}" for band in ("LOW","MID","HIGH"))
        + ". HIGH has the largest GT-relative normalized error, but the smallest remaining fraction versus raw LDCT. Band energies are sums across unequal numbers of Fourier bins, so interpret absolute values under these fixed band definitions.", "",
        f"**D. Timestep.** The flags for t=499 and t=999 trigger because the best t=0 is about {med(t['0'],'psnr')-max(other_psnr):.3f} dB easier. Across t=100/250/499/750/999, median one-pass PSNR spans only {psnr_spread:.4f} dB and median residual NMSE spans {nmse_spread:.6f}; 499 and 999 are not selectively worse. A specific sampler-timestep bottleneck is therefore not supported by this probe.", "",
        f"**E. Hard slices.** Prediction RMSE correlates with raw LDCT RMSE (rho={rho['raw_ldct_rmse']['rho']:.3f}) and GT high-frequency energy (rho={rho['gt_high_frequency_energy']['rho']:.3f}); GT gradient energy (rho={rho['gt_gradient_energy']['rho']:.3f}) and body fraction (rho={rho['body_fraction']['rho']:.3f}) are weaker. Raw LDCT RMSE and input residual RMS are the same variable here. The very strong PSNR-gain versus high-frequency-remaining-ratio correlation reuses prediction error on both sides and is not independent causal evidence.", "",
        "**F. Worth validating.** Structure-sensitive and frequency-aware mechanism categories are exploratory leads from incomplete edge recovery and frequency content, respectively; neither preregistered structure nor frequency threshold passed. There is no clear timestep-specific target. This diagnostic does not test 2.5D context or global modeling and predicts no module gain.", "",
        "The pre-registered flags remain recorded exactly as triggered; this specificity review limits their interpretation. Correlations and contiguous-block intervals describe one L333 stack only.", ""
    ]
    report.write_text(text+"\n".join(lines))
    provenance=load("RUN_PROVENANCE.json")
    code=Path(__file__)
    provenance["interpretation_pass"]={
        "code_sha256": hashlib.sha256(code.read_bytes()).hexdigest(),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "opened_medical_images": 0, "model_forwards": 0,
        "input_files": ["CONTROL.json","STRUCTURE_REGION_SUMMARY.json","FREQUENCY_SUMMARY.json",
                        "TIMESTEP_SUMMARY.json","HARD_SLICE_SUMMARY.json","DIAGNOSTIC_SUMMARY.json",
                        "DATA_ACCESS_AUDIT.json"]}
    save("RUN_PROVENANCE.json",provenance)
    print("INTERPRETATION_FINALIZED", psnr_spread, nmse_spread)


if __name__ == "__main__":
    main()
