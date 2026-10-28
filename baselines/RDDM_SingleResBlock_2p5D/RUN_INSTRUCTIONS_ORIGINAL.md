# RDDM_SingleResBlock_2p5D_8P_8K
Remote-only independent baseline; immutable official snapshot and SOURCE.json record upstream provenance.
Configuration is fixed in config.json. Data is read only from /home/xujian/data/mayo_2016_npy using explicit nine-patient allowlist.
No L506 image path is requested. Patient boundaries duplicate center, and all four image transforms are paired.
Native full ResnetBlock includes two internal convolution Blocks; the removed block is the second full ResnetBlock at each U-Net scale.

From /home/xujian/research/My-Paper/baselines/RDDM_SingleResBlock_2p5D:
- Preflight: CUDA_VISIBLE_DEVICES=0 /home/xujian/miniforge3/bin/conda run --no-capture-output -n founddiff python -B -u preflight.py
- Train: CUDA_VISIBLE_DEVICES=0 /home/xujian/miniforge3/bin/conda run --no-capture-output -n founddiff python -B -u train.py
- Explicit recovery only: same train command with --resume ABSOLUTE_CHECKPOINT_PATH.
Do not restart a running tmux session. No automatic resume or training beyond 8000 steps.
New runs refuse existing checkpoints; checkpoint/validation/report files use exclusive creation.
Resume restores RNGs and the exact shuffle order and cursor; already written later checkpoints prohibit rewind.
If a partial validation CSV exists after interruption, inspect it before recovery; this script refuses to overwrite it.

Model weights use ResidualDiffusion state_dict; EMA weights are checkpoint key ema_state_dict.
Metrics, reports and logs are under experiments/RDDM_SingleResBlock_2p5D_8P_8K/.
No predictions or patient pixels are exported.
