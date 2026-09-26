# NDCT clean structure target audit

The target is computed only from the paired center NDCT of the current training batch. It is never a model input and does not use neighbor NDCT, L333 as optimization data, L506 or external memory. The frozen Fast64/Full-L333 evaluators remain unchanged and do not generate auxiliary targets.

Input: original paired NDCT float32 in [0,1], full 512x512. Recover HU = 4096*NDCT - 1024, clamp to the established evaluation window [-1000,1000] HU, divide by 1000 to [-1,1]. This uses the same HU storage and window protocol as the parent experiment.

Gaussian: separable 5x5, sigma=1.0 pixel, coefficients proportional to exp(-i^2/2) for i=-2..2, normalized to sum1 on each axis; reflect padding. Scharr X kernel [[-3,0,3],[-10,0,10],[-3,0,3]]/32 and Scharr Y kernel [[-3,-10,-3],[0,0,0],[3,10,3]]/32; reflect padding. Magnitude = sqrt(gx^2+gy^2+1e-12), then target_full = clamp(magnitude/sqrt(2),0,1). No per-slice or dataset-dependent maximum is used. Downsample full to 128x128 and 64x64 by area interpolation. The same deterministic function applies to every training batch, before any loss computation. Model side heads take gated PDC shape at 128/64, not a reconstruction output.

The training objective uses L1(pred_struct128,target128) and L1(pred_struct64,target64), each with a single fixed lambda chosen by the 20-step smoke calibration. No output-gradient, Sobel, segmentation, curvature or image auxiliary loss is added.
