# Frozen RDDM Fast Development Protocol

- [Frozen protocol and usage](../../baselines/founddiff_rddm_rleb/experiments/fast_dev/README.md)
- [Completed unique 8k reference results](../../baselines/founddiff_rddm_rleb/experiments/fast_dev/baseline_8k/RESULTS.md)
- [Verification report](../../baselines/founddiff_rddm_rleb/experiments/fast_dev/baseline_8k/VERIFICATION.json)
- [Frozen configuration](../../baselines/founddiff_rddm_rleb/configs/rddm_fast_dev.json)
- [Fixed L333-fast64 indices](../../baselines/founddiff_rddm_rleb/splits/L333_fast64.txt)

Train: L067 + L143, 1145 pairs. The other six formal training patients are external
memory only. L333/L506 never enter external memory; L506 remains untouched during
development. The completed reference stops at 8000 optimizer steps; 2k/4k/6k/8k
metrics use fixed L333-fast64, EMA and two-step deterministic sampling.

Best fast64 PSNR: **40.484568 dB at 8k**. Compare methods only against this fast
reference at matched steps, not the formal 8-patient/60k score. Candidate promotion
requires positive 6k/8k Delta_PSNR, 8k Delta_PSNR >=0.05 dB, and review of other
metrics. These are engineering heuristics, not statistical significance.
Same-checkpoint full-L333 verification precedes fresh formal 8-patient training.
Fast-dev values must never serve as final paper performance.

No datasets, images, weights, full training logs, caches or credentials are included.
The formal baseline commit remains d961c30f311911b893e190b526603e9369ea45e1.
