# ProSAC Eq.(7) add-fusion probe

This isolated Fast64 experiment starts from commit
6595ca325b204769400653836ac447d919aaa40e on
experiment/rddm-prosac-apgc-control-v1. C4_ADD differs from the archived
wide-window C4_APGC only by applying Sobel + Laplacian before a 1-to-C
Conv1x1 projection instead of concatenating them before a 2-to-C
projection. No parameter-matching layer is added.

The Direct 12k EMA main, frozen MedDINOv3 ViT-B/16 CT-3M, HU window
[-1000,1000], APGC multiplication, Fast64 L333 evaluator, training
L067/L143 split, optimizer, seed, augmentation, sampler and EMA are fixed.
C1, C3 and old C4 are read-only archived references. L506, external
memory and neighbor NDCT are prohibited. No Full-L333, 8-patient or test
evaluation is scheduled.

The arm independently runs a 20-step smoke and a 2k first stage.
Only the predeclared gate may extend it to 4k/6k. Checkpoints, MedDINO
weights, medical images and large logs are excluded from Git.
