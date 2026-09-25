# ProSAC-CT APGC structure probe on frozen RDDM ControlNet

The official MedDINOv3 ViT-B/16 CT-3M teacher was frozen. This is an APGC transfer probe, not a ProSAC-CT reproduction.
C3 uses its final normalized dense patch tokens; C4 adds explicit Sobel/Laplacian modulation.
The original RDDM Direct main and Control Encoder topology, ZeroConv taps, loss, sampler, and Fast64 evaluator were unchanged.

| Arm | step | PSNR | delta vs C1 | soft PSNR | RMSE HU | positive slices vs C1 | trainable params | peak GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C3_ANATOMY | 0 | 40.76762736 | +0.00000005 | 28.45385731 | 18.40853109 | 0.297 | 6260640 | 4.850 |
| C3_ANATOMY | 500 | 40.77394810 | +0.00426065 | 28.46146082 | 18.39557060 | 0.953 | 6260640 | 4.850 |
| C3_ANATOMY | 1000 | 40.77986161 | +0.00939762 | 28.46918486 | 18.38352157 | 0.969 | 6260640 | 4.850 |
| C3_ANATOMY | 2000 | 40.78820919 | +0.01674104 | 28.47784415 | 18.36643241 | 0.984 | 6260640 | 4.850 |
| C3_ANATOMY | 4000 | 40.79749591 | +0.02244954 | 28.48629579 | 18.34719042 | 0.984 | 6260640 | 4.850 |
| C3_ANATOMY | 6000 | 40.80220121 | +0.02355979 | 28.49143501 | 18.33751649 | 0.984 | 6260640 | 4.850 |
| C4_APGC | 0 | 40.76762736 | +0.00000005 | 28.45385731 | 18.40853109 | 0.297 | 6261760 | 4.958 |
| C4_APGC | 500 | 40.77079853 | +0.00111108 | 28.45790567 | 18.40211312 | 0.672 | 6261760 | 4.958 |
| C4_APGC | 1000 | 40.77533963 | +0.00487564 | 28.46364860 | 18.39290554 | 0.844 | 6261760 | 4.958 |
| C4_APGC | 2000 | 40.78218357 | +0.01071542 | 28.47121836 | 18.37897835 | 0.953 | 6261760 | 4.958 |
| C4_APGC | 4000 | 40.79195415 | +0.01690778 | 28.48074682 | 18.35868012 | 0.984 | 6261760 | 4.958 |
| C4_APGC | 6000 | 40.79787761 | +0.01923619 | 28.48752568 | 18.34656574 | 0.984 | 6261760 | 4.958 |

## Gate and interpretation

- C3_ANATOMY: 2k delta vs C1 +0.01674104 dB; positive slice fraction 0.984; continue to 6k: True.
- C4_APGC: 2k delta vs C1 +0.01071542 dB; positive slice fraction 0.953; continue to 6k: True.

C3 anatomy and C4 APGC claims are limited to observed Fast64 deltas at matched steps. Small positive differences do not establish a robust benefit.
C4 minus C3 at their final common step: see PER_SLICE_DELTA.csv and FINAL_AUDIT.json.

## Provenance and limits

- MedDINO input is 512x512 native Mayo resolution, chosen explicitly for this experiment. The official notebook example sets 2048; weights, HU clip/normalization, channel repeat, constructor, and final patch-token convention follow that source.
- Official paper Eq.(7) writes Sobel + Laplacian. This probe instead uses concat(Sobel,Laplacian) followed by trainable Conv1x1 as specified by the experiment request: RECONSTRUCTION_CHOICE_NOT_EXPLICIT_IN_PAPER.
- C0/C1/C2 controls were reused only after protocol, Fast64 split and evaluator source hashes matched. C0 was separately reproduced in the flexict runtime within 1e-7 dB.
- PyTorch/skimage runtime versions differ from the archived C1/C2 run; this is a cross-runtime comparison. Source and evaluation formulas are unchanged, and the C0 numeric reproduction bounds the observed mismatch.
- L506, external memory, and neighbor NDCT access were zero; no formal 8-patient training was run.
