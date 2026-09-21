# Source attribution and snapshot identity

The reconstructed `src/rddm_rleb.py` derives from the official FoundDiff repository
by hao1635, commit `c85a3a0e87ae2e4587650f0fb7efc15f3311d8e2`:
https://github.com/hao1635/FoundDiff/tree/c85a3a0e87ae2e4587650f0fb7efc15f3311d8e2

Original model: `src/DADiff.py`. The standalone residual diffusion retains its
equations and removes the unused perceptual-network construction. The U-Net keeps
the released convolution, skip and resize topology; commented time-only MLPs in
the residual blocks were restored with explicit experiment-owner approval.
The paper-to-code discrepancies and exact forward graph are documented in
`experiments/founddiff_rddm_baseline/historical/ARCHITECTURE_AUDIT.md`.

The official source audit did not identify a repository-wide license. The license
under its unrelated dataset-generation directory is not applied to this code.
This snapshot does not assert a new license over third-party code.

Normalization and metric functions were copied from the owner's fixed RDDM-CT
protocol. Their original hashes and function names are recorded in
`baselines/founddiff_rddm_rleb/mayo2016_smallset_v1/protocol_provenance.json`.

The original run started from an uncommitted reconstruction on the official
source ancestor. Therefore the ancestor Git commit alone does not identify the
trained implementation. `source_manifest.json` records all seven training source
hashes; those files match both the current remote source and the run's archived
source byte for byte. The local publication commit adds packaging and documentation
only. `export_manifest.json` records hashes of every exported original file.

This publication checkout was initialized separately because GitHub authentication
was unavailable at export time. It is not claimed to be based on the target
repository's existing branch/history. Publishing requires an authenticated fetch
and integration into the target repository if it already has content; never force
push or replace existing repository history.
