# Official RDDM restoration backbone → Mayo CT (matched fast-dev)

## What to read

- [Full implementation and result report](REPORT.md): official commit, backbone differences, CT protocol, readiness, data access audit and the complete 0/2k/4k/6k/8k table.
- [Exact metric CSV](results/fast_dev_metrics.csv) and [per-step diagnostics](results/training_diagnostics.csv).
- [Control source](src/rddm_rleb.py), [official candidate source](src/official_rddm_ct.py), [matched trainer](matched_train.py), [readiness audit](readiness.py) and [run queue](run_queue.sh).

At 8k, EMA L333-fast64 PSNR was **40.6234 dB** (official) versus **40.4846 dB** (simplified), Δ **+0.1388 dB**. The candidate improved all four metrics on all fixed 64 slices. This is a single-seed short-line result.

Official source: https://github.com/nachifur/RDDM at commit
`524caea34cedb3733aa5bf6c27dc1cd6e0469216`.
The official clone at `/home/xujian/projects/RDDM_CVPR2024/RDDM_Official` was only read.
`src/official_rddm_ct.py` contains the official restoration U-Net helpers and
topology, followed by the byte-identical diffusion/wrapper tail of
`src/rddm_rleb.py`. `SOURCE_PROVENANCE.json` records SHA-256 hashes.

Control and candidate use the same frozen CT data, normalization, optimizer,
pred_res residual diffusion, two-step sampler, EMA, Fast64 evaluator, and
seed 20260829. Only the U-Net backbone differs. The model initialization RNG
is restored before each run; the first-batch audit checks sample order,
timestep, and noise SHA-256 equality.

Readiness uses synthetic tensors only. Real data access is constrained by
`DataAccessGuard` to L067/L143 training and fixed L333-fast64 validation;
external memory and L506 are disallowed. Checkpoints, CT data, and run logs
stay outside Git under ignored `runs/` and `logs/`.

Run from this directory with `CUBLAS_WORKSPACE_CONFIG=:4096:8` and
`/home/xujian/miniconda3/envs/corediff/bin/python`.

