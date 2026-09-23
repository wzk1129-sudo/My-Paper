# RDDM 60k Bottleneck Diagnostic V1

FoundDiff RDDM+RLEB backbone adapted to our fixed Mayo-2016 protocol.
Only L333 was opened. step_60000.pt EMA, 2-step / 0.01 control. No training.

Control reproduction: PASS. psnr=40.65805039 (delta +6.6320638e-10), ssim=0.94613206 (delta -9.0546415e-10), rmse_hu=18.64887882 (delta +1.8959692e-08), mae_hu=12.18173656 (delta -3.6927171e-08)

## Structure-region

| Measure | Median | Block-bootstrap 95% CI |
|---|---:|---:|
| edge_error_concentration | 1.499148 | [1.4186849540014133, 1.5486093893871522] |
| edge_flat_mae_ratio_pred | 1.351290 | [1.2951779292825667, 1.3967738883992176] |
| edge_rmse_remaining_ratio | 0.590566 | [0.5769579986457136, 0.6111099004303213] |
| flat_rmse_remaining_ratio | 0.471615 | [0.460729630962475, 0.4786095484221604] |
| gradient_magnitude_ratio_ldct | 1.010436 | [1.0070148979123494, 1.0139214751644734] |
| gradient_magnitude_ratio_pred | 0.949905 | [0.9408099480854095, 0.9551739705278393] |
| orientation_cos_ldct | 0.947020 | [0.9374779955563428, 0.9592057920865413] |
| orientation_cos_pred | 0.978176 | [0.9743662441879379, 0.9805305533579762] |

Invalid structure indices: [].

## Frequency

| Band | Median remaining Epred/Eld | 95% CI | Median normalized Epred/Egt | 95% CI |
|---|---:|---:|---:|---:|
| LOW | 0.496094 | [0.4705919666566969, 0.531222761944763] | 0.000261 | [0.00025228436321547595, 0.0002809926667756002] |
| MID | 0.312666 | [0.2888832079963313, 0.347801943289652] | 0.098131 | [0.07802393528161451, 0.12026589500413047] |
| HIGH | 0.213270 | [0.20683756541873535, 0.22299356152269587] | 0.561230 | [0.5306538488902516, 0.6225903698104887] |

## Timestep direct probe (not sampler sweep)

| t | 2-step visit | Median residual NMSE | Median one-pass x0 PSNR | Mean within-slice replicate PSNR SD |
|---:|:---:|---:|---:|---:|
| 0 | no | 0.137503 | 43.628957 | 0.006915 |
| 100 | no | 0.255589 | 40.732389 | 0.004744 |
| 250 | no | 0.256510 | 40.725369 | 0.002080 |
| 499 | yes | 0.256773 | 40.725687 | 0.000689 |
| 750 | no | 0.256782 | 40.724501 | 0.000642 |
| 999 | yes | 0.256808 | 40.724865 | 0.000670 |

## Hard slices

Hardest 20 indices: [129, 123, 138, 126, 120, 135, 132, 144, 141, 117, 150, 130, 118, 127, 119, 147, 124, 108, 125, 131].
Easiest 20 indices: [47, 457, 563, 445, 455, 50, 46, 571, 454, 596, 602, 607, 448, 601, 446, 49, 451, 452, 605, 604].

| Predictor | rho(prediction RMSE) | rho(PSNR gain) |
|---|---:|---:|
| raw_ldct_rmse | 0.8237 | 0.2591 |
| input_residual_rms | 0.8237 | 0.2591 |
| gt_gradient_energy | -0.1147 | -0.4696 |
| gt_high_frequency_energy | 0.8401 | -0.4023 |
| body_fraction | 0.2761 | 0.2797 |
| edge_error_concentration | -0.0518 | -0.3023 |
| high_frequency_remaining_ratio | 0.3920 | -0.9614 |

## Interpretation

Engineering flags: ['SAMPLER_TIMESTEP_WEAKNESS_SIGNAL_t499', 'SAMPLER_TIMESTEP_WEAKNESS_SIGNAL_t999', 'STRONG_DIAGNOSTIC_CORRELATION'].
Flag-only candidate category: Timestep; the specificity review below does not support a unique weakness at t=499 or t=999.
Compare prediction and raw LDCT columns in the structure and frequency tables before choosing a mechanism.
The direct timestep probe diagnoses residual prediction at fixed q_sample states. It is not a new sampler result.
No flag proves a future model improvement, and absence of a flag does not rule one out.
Block bootstrap: contiguous 10-slice blocks, 1000 repetitions, seed 20260923. Intervals reflect stability within this single L333 stack, not patient-population significance.
Elapsed time: 449.9 s. No optimizer step or Git push.
## Direct answers and flag specificity

**A. Error location.** Edge error concentration is 1.499 and edge/flat MAE ratio is 1.351; the preregistered structure flag does not pass. Edge RMSE remaining is 0.591 versus 0.472 in flat regions, so edge errors are repaired less completely. Under the fixed unequal-width frequency bands, MID has the largest median absolute residual power. Hard-slice error is not uniform along this single stack. No uniquely weak sampler timestep is established.

**B. Boundary preservation.** Raw LDCT edge orientation cosine is 0.947; prediction increases it to 0.978. LDCT gradient magnitude ratio is 1.010 and prediction is 0.950, indicating modest amplitude attenuation despite improved orientation. The oversmoothing heuristic (<0.9) does not pass.

**C. Frequency.** Median residual power E_pred is LOW 3458887, MID 15571573, HIGH 10505160. Median E_pred/E_ld is LOW 0.496, MID 0.313, HIGH 0.213. HIGH has the largest GT-relative normalized error, but the smallest remaining fraction versus raw LDCT. Band energies are sums across unequal numbers of Fourier bins, so interpret absolute values under these fixed band definitions.

**D. Timestep.** The flags for t=499 and t=999 trigger because the best t=0 is about 2.897 dB easier. Across t=100/250/499/750/999, median one-pass PSNR spans only 0.0079 dB and median residual NMSE spans 0.001218; 499 and 999 are not selectively worse. A specific sampler-timestep bottleneck is therefore not supported by this probe.

**E. Hard slices.** Prediction RMSE correlates with raw LDCT RMSE (rho=0.824) and GT high-frequency energy (rho=0.840); GT gradient energy (rho=-0.115) and body fraction (rho=0.276) are weaker. Raw LDCT RMSE and input residual RMS are the same variable here. The very strong PSNR-gain versus high-frequency-remaining-ratio correlation reuses prediction error on both sides and is not independent causal evidence.

**F. Worth validating.** Structure-sensitive and frequency-aware mechanism categories are exploratory leads from incomplete edge recovery and frequency content, respectively; neither preregistered structure nor frequency threshold passed. There is no clear timestep-specific target. This diagnostic does not test 2.5D context or global modeling and predicts no module gain.

The pre-registered flags remain recorded exactly as triggered; this specificity review limits their interpretation. Correlations and contiguous-block intervals describe one L333 stack only.
## Complete aggregate distributions

Per-slice frequency power ratios (mean, median, IQR, contiguous-block 95% CI for median):

| Band | Ratio | Mean | Median | IQR | 95% CI |
|---|---|---:|---:|---:|---:|
| LOW | Epred/Eld | 0.486218 | 0.496094 | 0.166932 | [0.4705919666566969, 0.531222761944763] |
| LOW | Epred/Egt | 0.000275 | 0.000261 | 0.000071 | [0.00025228436321547595, 0.0002809926667756002] |
| MID | Epred/Eld | 0.319484 | 0.312666 | 0.099355 | [0.2888832079963313, 0.347801943289652] |
| MID | Epred/Egt | 0.113394 | 0.098131 | 0.071164 | [0.07802393528161451, 0.12026589500413047] |
| HIGH | Epred/Eld | 0.213692 | 0.213270 | 0.042677 | [0.20683756541873535, 0.22299356152269587] |
| HIGH | Epred/Egt | 0.585956 | 0.561230 | 0.162652 | [0.5306538488902516, 0.6225903698104887] |

Timestep probe distributions across 128 slices × 3 noise replicates (descriptive, not independent-patient intervals):

| t | Quantity | Mean | Median | SD | IQR | Mean within-slice replicate SD |
|---:|---|---:|---:|---:|---:|---:|
| 0 | residual_nmse | 0.141533 | 0.137503 | 0.015873 | 0.020803 | 0.000209 |
| 0 | psnr | 43.499514 | 43.628957 | 0.838777 | 1.201124 | 0.006915 |
| 100 | residual_nmse | 0.260440 | 0.255589 | 0.025014 | 0.041733 | 0.000279 |
| 100 | psnr | 40.704645 | 40.732389 | 0.941546 | 1.291000 | 0.004744 |
| 250 | residual_nmse | 0.261267 | 0.256510 | 0.024991 | 0.041923 | 0.000125 |
| 250 | psnr | 40.691972 | 40.725369 | 0.942012 | 1.285603 | 0.002080 |
| 499 | residual_nmse | 0.261367 | 0.256773 | 0.024954 | 0.041944 | 0.000043 |
| 499 | psnr | 40.690336 | 40.725687 | 0.941355 | 1.280603 | 0.000689 |
| 750 | residual_nmse | 0.261407 | 0.256782 | 0.024973 | 0.041910 | 0.000040 |
| 750 | psnr | 40.689290 | 40.724501 | 0.941713 | 1.281539 | 0.000642 |
| 999 | residual_nmse | 0.261424 | 0.256808 | 0.024969 | 0.041919 | 0.000042 |
| 999 | psnr | 40.689526 | 40.724865 | 0.941558 | 1.281373 | 0.000670 |

Full numerical per-slice rows and all four image metrics are in TIMESTEP_PROBE.csv; all band powers and ratios are in FREQUENCY_BANDS.csv.
