# Knowledge-Prior Fast Screen V1

Matched flexict continuation from Direct Concat 12k EMA; no A+B experiment.

| Arm | Step | PSNR | Δ vs Control | SSIM | RMSE HU | MAE HU |
|---|---:|---:|---:|---:|---:|---:|
| CONTROL_CONTINUE | 0 | 40.767627 | +0.000000 | 0.947724 | 18.4085 | 12.1283 |
| CONTROL_CONTINUE | 1000 | 40.777333 | +0.000000 | 0.947871 | 18.3879 | 12.1121 |
| CONTROL_CONTINUE | 2000 | 40.792144 | +0.000000 | 0.947956 | 18.3579 | 12.1009 |
| CONTROL_CONTINUE | 4000 | 40.809092 | +0.000000 | 0.948117 | 18.3228 | 12.0675 |
| CONTROL_CONTINUE | 6000 | 40.818550 | +0.000000 | 0.948176 | 18.3031 | 12.0584 |
| A_MEDDINO_BCA | 0 | 40.767627 | +0.000000 | 0.947724 | 18.4085 | 12.1283 |
| A_MEDDINO_BCA | 1000 | 40.781973 | +0.004639 | 0.947910 | 18.3786 | 12.1088 |
| A_MEDDINO_BCA | 2000 | 40.800666 | +0.008522 | 0.948070 | 18.3407 | 12.0897 |
| A_MEDDINO_BCA | 4000 | 40.820040 | +0.010948 | 0.948275 | 18.2997 | 12.0448 |
| A_MEDDINO_BCA | 6000 | 40.840992 | +0.022442 | 0.948405 | 18.2579 | 12.0357 |
| B_NDCT_DETAIL_PRIOR | 0 | 40.767627 | +0.000000 | 0.947724 | 18.4085 | 12.1283 |
| B_NDCT_DETAIL_PRIOR | 1000 | 40.777444 | +0.000110 | 0.947923 | 18.3890 | 12.1470 |
| B_NDCT_DETAIL_PRIOR | 2000 | 40.790059 | -0.002085 | 0.948025 | 18.3627 | 12.1137 |
| B_NDCT_DETAIL_PRIOR | 4000 | 40.808286 | -0.000806 | 0.948168 | 18.3250 | 12.0875 |

## Gates

A: {'promote': False, 'reason': 'FAST_ONLY', 'delta': 0.022441885135421558, 'positive_slice_fraction': 0.96875}

B: {'promote': False, 'reason': 'STOP_AT_4K'}

## Full L333

{}

## Questions answered from observed data

Control continuation gain: +0.050923 dB.
A 6k increment: +0.022442 dB; gate FAST_ONLY.
B 6k increment: -0.000806 dB; gate STOP_AT_4K.
B prior-loss trend is in B_NDCT_DETAIL_PRIOR/train_history.csv.
A and B stability is limited to the matched 0/1k/2k/4k/6k nodes and any promoted Full-L333 results.
Neither prior is a paper result at this stage; no L506 access.
A+B combination is a candidate next experiment only if both have positive matched gains. It was not run.

## Post-run integrity audit

PASS. First queue failed before A optimizer step in a BF16-only state-hash audit; the dtype-safe retry preserved Control 6k and completed the serial phases. The three RDDM arms share the same Direct 12k EMA main state and manifest SHA256 `ca176a324db9ce0547f0a5ad863108028e2b49dde2a554fc952cd059c3ca4768`.

Control continuation: +0.050923 dB. A at 6k: ΔPSNR +0.022442 dB, positive slice fraction 0.9688; below the +0.03 dB Full-L333 gate. B stopped at 4k: ΔPSNR -0.000806 dB. B prior loss median across first/last ten logged points: 0.019379 → 0.021857; this is a training diagnostic, not a validation gain. Teacher trained 8k on disjoint donor NDCT only. No arm promoted to Full-L333. L506, unauthorized external-memory, neighbor-target and blocked opens were all zero. No A+B experiment was run.
