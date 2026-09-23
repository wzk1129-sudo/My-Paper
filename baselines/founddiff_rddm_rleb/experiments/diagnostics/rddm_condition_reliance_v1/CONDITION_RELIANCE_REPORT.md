# RDDM Condition Reliance Probe V1

Forward-only, EMA step_60000.pt, L333 fixed 128 indices; no DDIM loop or optimizer update.
Control path matches all 1152 archived Bottleneck V1 direct-forward rows within configured tolerances.
Reconstruction always uses the original LDCT slice; donor data only replaces one model-input channel.

## 3×3 time matrix

| true t | feed t | N | median PSNR drop (dB) | median NMSE ratio | fraction PSNR worse |
|---:|---:|---:|---:|---:|---:|
| 100 | 100 | 384 | 0.000000 | 1.000000 | 0.000 |
| 100 | 499 | 384 | 0.005488 | 1.001110 | 0.997 |
| 100 | 999 | 384 | 0.007767 | 1.001661 | 1.000 |
| 499 | 100 | 384 | -0.003754 | 0.999319 | 0.143 |
| 499 | 499 | 384 | 0.000000 | 1.000000 | 0.000 |
| 499 | 999 | 384 | 0.001036 | 1.000216 | 0.872 |
| 999 | 100 | 384 | -0.004694 | 0.999133 | 0.143 |
| 999 | 499 | 384 | -0.001052 | 0.999781 | 0.112 |
| 999 | 999 | 384 | 0.000000 | 1.000000 | 0.000 |

## Paired dependence

| Intervention | N | Median PSNR drop | Mean | IQR | P05–P95 | Fraction worse | Median NMSE ratio |
|---|---:|---:|---:|---:|---|---:|---:|
| Wrong time (all off-diagonal) | 2304 | 0.000352 | 0.000872 | 0.005391 | -0.008722 to 0.011797 | 0.545 | 1.000107 |
| Shuffle x_t | 1152 | 0.002650 | 0.020180 | 0.042488 | -0.000385 to 0.075076 | 0.910 | 1.000937 |
| Shuffle LDCT condition | 1152 | 8.060289 | 8.361562 | 0.853843 | 7.538929 to 10.279352 | 1.000 | 6.414554 |

condition_to_xt_drop_ratio = 806.028920.

## Questions answered

1. Wrong time embedding: median paired PSNR drop 0.000352 dB; median NMSE ratio 1.000107.
2. True-t sensitivity: t=100: 0.006620 dB, t=499: 0.000031 dB, t=999: -0.001455 dB. Compare all six off-diagonal cells above.
3. Shuffled legal donor x_t: median PSNR drop 0.002650 dB, NMSE ratio 1.000937.
4. Shuffled donor LDCT condition: median PSNR drop 8.060289 dB, NMSE ratio 6.414554.
5. Relative reliance: LDCT condition has the larger median paired PSNR effect.
6. Time embedding contribution: small by preregistered engineering thresholds.
7. LDCT-conditioned residual-regression hypothesis: **supported** under this probe's pre-registered heuristic; this is not a mathematical proof.

Engineering flags: ['TIME_CONDITION_WEAK_SIGNAL', 'XT_SAMPLE_SPECIFIC_WEAK_SIGNAL', 'LDCT_CONDITION_DOMINANT_SIGNAL'].
All comparisons pair the same slice, true_t and replicate. The 128 slices come from one L333 patient; no patient-population statistical significance is claimed.
No training, model parameter modification, new sampler optimization, L506 access, external memory access, or Git push of this branch.
