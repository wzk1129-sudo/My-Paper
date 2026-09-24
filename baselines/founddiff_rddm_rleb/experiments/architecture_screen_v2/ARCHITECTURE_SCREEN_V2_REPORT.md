# RDDM Architecture Fast Screen V2

Four independent additions to the frozen FoundDiff RDDM+RLEB backbone. All Fast results use L067/L143 optimization and the same fixed L333 Fast64 EMA protocol. No candidate combinations were tested.

| Arm | 2k PSNR | 4k | 6k | 8k | 10k | 12k | Delta 12k | Promotion | Params | Peak VRAM GiB | s/step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| GATED_SKIP | 38.11878 | 40.11738 | 40.38895 | 40.47434 | 40.52601 | 40.56867 | -0.00784 | NO_PROMOTION | 13317313 | 4.71 | 0.359 |
| BOTTLENECK_CONTEXT | 37.15195 | 40.13356 | 40.38616 | 40.47442 | 40.52800 | 40.56899 | -0.00752 | NO_PROMOTION | 14014145 | 4.33 | 0.338 |
| LDCT_CONTENT_MODULATION | 38.85413 | 40.16300 | 40.39004 | 40.46509 | 40.51813 | 40.55286 | -0.02365 | NO_PROMOTION | 13183169 | 5.04 | 0.352 |
| PARTIAL_NAF_REFINEMENT | 37.31079 | 40.18033 | 40.39184 | 40.47764 | 40.53379 | 40.56900 | -0.00751 | NO_PROMOTION | 15520961 | 4.64 | 0.353 |

Full metric curves, losses, gradient norms, and matched baseline deltas are in ARCHITECTURE_SCREEN_V2.csv.
Promotion thresholds are engineering screening rules, not statistical significance.
Full-L333 paired comparisons refer only to one patient volume.

## Access and integrity

- GATED_SKIP: status COMPLETE; test opens 0; external-memory opens 0; blocked attempts 0; nonfinite count 0.
- BOTTLENECK_CONTEXT: status COMPLETE; test opens 0; external-memory opens 0; blocked attempts 0; nonfinite count 0.
- LDCT_CONTENT_MODULATION: status COMPLETE; test opens 0; external-memory opens 0; blocked attempts 0; nonfinite count 0.
- PARTIAL_NAF_REFINEMENT: status COMPLETE; test opens 0; external-memory opens 0; blocked attempts 0; nonfinite count 0.

Formal baseline source, checkpoint, frozen protocol, and split remain unchanged. No 60k formal training or candidate combination is part of this screen.
