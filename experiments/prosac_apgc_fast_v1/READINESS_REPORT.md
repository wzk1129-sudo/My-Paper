# ProSAC-style APGC matched fast reproduction: pre-training readiness

This is an implementation and smoke-test report. No M0/M1 optimizer step or
2k/4k/8k run has started. It is not an official ProSAC-CT reproduction.

## Source snapshot

The parent repository's branch was `rddm-ct-backbone-v1` at
`bf1e4eb458ec382830cda4d978f5e88b02183a37`, with exactly six tracked
modifications in `experiments/7_RDDM_CT_Backbone_v1`. Their status, raw
`git diff`, and SHA256 list were saved under `source_snapshot/`. The
six modified source files were copied byte-for-byte into the independent
worktree and committed as `8a159791d98c4d61b8b3c9e9f598d9df9a8aeeca`.
The original worktree was not overwritten.

- `rddm_ct.py`: FP32-stable convolutions/heads and attention limited to
  128/64. The FP32 path is a numerical stability fix; the attention pruning
  is an efficiency/architecture choice, not a pure bugfix.
- `train.py`: AMP scaler/overflow and provenance logging. This experimental
  fast runner does not inherit its old long-run training loop.
- `config.json`: an efficiency configuration with batch 2 and lr 1e-4,
  not the requested M0 protocol. The two new configs explicitly override it.
- `acceptance.py`: GPU and model acceptance diagnostics.
- `evaluate.py`, `evaluation.py`: stricter metric definitions, provenance,
  fixed visualization support. Both arms use the same frozen
  `evaluation.slice_metrics`.

Full hashes are in `SOURCE_SNAPSHOT.json`. The official RDDM diffusion
source SHA256 is `b7c41d56313a5073a88c85cedd9ae2b65167515f6294f42288bba52831ea5c3c`.
The official MedDINO teacher checkpoint SHA256 is
`492b2b96c00e9d3b93ba3b07a8d387aa2196b5e04698378fcc23049b447966d2`.

## Frozen M0/M1 fast protocol

- Train: existing 8 Mayo-2016 patients, 4,800 paired 512x512 slices.
  Validation: L333, 610 indexed pairs; fixed 64 indices from the previously
  frozen L333-fast64 file for screening. The forbidden test patient's images
  are not enumerated or opened.
- Stored NPY values are offset to HU, clipped to [-160,240], then normalized
  to [0,1]. RDDM `training_losses` and the new runner transform both NDCT
  and three LDCT slices to [-1,1] before `q_sample`. `DATA_DOMAIN_AUDIT.json` confirms actual
  four-pair LDCT/NDCT ranges of [0,1] before diffusion and [-1,1]
  entering `q_sample`.
- Shared RDDM-CT encoder-decoder with separate residual/noise 1x1 heads;
  center NDCT target. `x_t=x_0+A_t r+B_t eps`; T=1000,
  sum_scale=1, deterministic 10-step residual/noise sampler.
- Equal-weight `L1(pred_res,r)+L1(pred_noise,eps)` is
  `RECONSTRUCTION_CHOICE_NOT_EXPLICIT_IN_PAPER`.
- Seed 20260829, Adam default betas, lr 2e-4, physical batch 4,
  BF16 autocast, no accumulation, gradient clip 1, EMA 0.995.
  Both arms use the same stateless step-indexed sample, transform, timestep,
  and Gaussian-noise generator. M1's base RDDM weights have exactly the
  same initialization hash as M0.
- Checkpoints/validation are planned at 2k/4k/8k. The runner supports a
  4k gate and resumption to 8k without repeating optimizer steps. Candidate
  comparison is paired on fixed L333-fast64. The edge-region metric uses
  the top quartile of target NDCT Sobel magnitude, a reconstruction choice.

## M1 APGC reconstruction choices

Official MedDINOv3 ViT-B/16 teacher is frozen. Its final normalized
32x32 patch-token map is reshaped to [B,768,32,32], projected to 512
channels, then resized to 64x64. The center LDCT is converted from the
RDDM [-1,1] input back to HU [-160,240], standardized with the official
MedDINO mean/std, and repeated to three channels. Sobel magnitude plus
signed Laplacian of center LDCT [0,1] is projected to 512 channels and
resized to 64x64. Channel-wise learnable alpha/gamma modulate AFM and
EFM, and their product is added once to the shared encoder output
[B,512,64,64] before the mid block. Projection, edge magnitude, injection
location, and additive injection are not specified by the paper and are
explicit reconstruction choices. No RFDDS, TD3, BCA, PDC, ControlNet,
or auxiliary structure loss is present.

The `flexict` environment (Python 3.11.16, PyTorch 2.8.0+cu128) is
required by the official MedDINO source. The older `rddm_cvpr2024`
environment (Python 3.7/PyTorch 1.13.1) cannot import that source. The
new `compat.py` supplies fail-fast import placeholders for the official
RDDM file's unused `Augmentor`, `accelerate`, `ema_pytorch`, and
`datasets.get_dataset` Trainer dependencies. It never replaces the
RDDM diffusion, schedule, or model math.

## Unit/smoke results

Both arms passed shape/range/finite forward and backward checks on
four real L067 training pairs, with zero optimizer steps. The shared
backbone state SHA256 at initialization was identical:
`d429070140ac394c654af01945a49485b2139882a3d1d8c3ad0e283f5a709d4a`.
At t=999, A=1 and B=1, and synthetic `q_sample` matched
`x_0+A_t r+B_t eps` and sampler terminal initialization to <2e-6.
A one-slice deterministic 10-step sample completed with finite output
for each arm. M1 MedDINO weights had no gradient; alpha and gamma
gradients were finite and nonzero, as was AEFM. The edge metric unit test
passed. Index-only audit found 4,800 training pairs, L333 610 pairs,
and fixed Fast64 64 indices. No forbidden test image or external memory
was opened.

The physical batch-4 GPU check held full Adam moment and EMA-shadow
buffers resident during forward/backward without stepping the optimizer:

| Arm | Trainable parameters | Peak allocated | Peak reserved | Result |
| --- | ---: | ---: | ---: | --- |
| M0 | 36,116,098 | 22.268 GiB | 22.500 GiB | PASS |
| M1 | 36,511,874 | 21.769 GiB | 22.137 GiB | PASS |

The 3090 reports 24,576 MiB total and about 771 MiB in use before the
test. CUDA backend/allocator choices cause the M0/M1 peak ordering to
vary, and the memory margin is small. Recheck GPU occupancy before
launch; do not change batch size without user approval.

The planned run outputs are isolated under `runs/M0` and `runs/M1`;
neither directory exists. No checkpoint, optimizer state, or training
metric has been produced by this preparation phase.
