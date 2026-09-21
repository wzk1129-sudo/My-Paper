# FoundDiff RDDM + RLEB reconstruction audit

Date: 2026-09-21. Status: implemented and synthetic acceptance passed after explicit user approval of the timestep reconstruction. This is **not an exact reproduction of the authors' Table V executable or metrics**.

## 1. Provenance and scope

- Official source: https://github.com/hao1635/FoundDiff
- Frozen source commit: `c85a3a0e87ae2e4587650f0fb7efc15f3311d8e2` (2026-02-07).
- Paper targeted: https://arxiv.org/html/2508.17299v1 , Table V and Sec. III-E2. The corresponding ablation is Table X in v2, https://arxiv.org/html/2508.17299v2 .
- The paper describes its first ablation as RDDM with only RLEB. Neither current source nor examined DADiff history identifies a verified Table-V-first-row configuration. Older code includes different attention/skip topology and does not establish the desired baseline.
- Remote source clone: `/home/xujian/projects/CoreDiff/experiments/founddiff_table_v_20260921/official`.
- Independent worktree: `/home/xujian/projects/CoreDiff/experiments/founddiff_table_v_20260921/rddm_rleb`.
- Branch: `audit/rddm-rleb-table-v-20260921`, based on the official commit, no new commit/push.
- Source clone and all original tracked code remain unchanged. Four new Python files implement and verify a separate executable path; upstream full-model files are archival reference, never imported by the baseline.
- No private medical images were read. Tests use fresh random synthetic tensors. No packages installed; no pretrained weights fetched; no long training or full-dataset evaluation.

## 2. Read-only audit: actual official entry and call graph

References below use original `src/DADiff.py` line numbers at the frozen commit.

```text
main.sh / README command -> train.py --name ... --is_train
  original_ddim_ddpm=False; condition=True; input_condition=False
  num_unet=1; objective=pred_res; test_res_or_noise=res
  dim=64; dim_mults=(1,2,4,8); image_size=512; T=1000; sampling=2
  -> UnetRes.__init__ (743) -> Unet.__init__ (530)
       -> load RN50 into clip_model (unused copy in Unet.forward)
       -> CLIPIQA(clipiqa+) -> RN50 + learned text prompts + two heads
       -> torch.load('Dose-CLIP.pth') into dose_encoder; frozen
  -> ResidualDiffusion.__init__ (908)
       -> allocate frozen LPIPS(AlexNet), although perceptual loss is commented out
  -> Trainer.__init__ (1506) -> PDFDataset -> paired [NDCT, LDCT]
  -> Trainer.train (1673) -> self.model(data) (1699)
  -> ResidualDiffusion.forward (1484): [0,1] -> [-1,1], random integer t
  -> p_losses (1399): residual = LDCT - NDCT
       -> q_sample (1382): NDCT + Abar[t]*residual + Bbar[t]*epsilon
       -> cat([x_t, LDCT], channel dimension)
       -> UnetRes.forward (817), selects time[0] = Abar[t]*T
       -> Unet.forward (685)
       -> residual prediction -> MSE(residual_pred, LDCT-NDCT)
       -> backward -> gradient clip 1.0 -> Adam -> EMA

Unet.forward:
  LDCT channel -> repeat RGB -> dose_encoder / CLIPIQA.forward
    -> RN50.encode_image -> head1: 1024-dimensional dose embedding
                        -> head2: 256-dimensional normalized anatomy embedding
    -> learned prompt similarity also computes scalar dose score (unused by Unet)
  time -> sinusoidal embedding -> Linear/GELU/Linear -> 256-dimensional t
  dose -> text_mlp -> softmax * learned prompt -> prompt_mlp -> add to t
  concat(x_t, LDCT) -> stem Conv7x7(2,64); save stem skip r
  encoder x4: Mamba_block(x, anatomy, fused_t) -> ResnetBlock(x) -> save skip -> resize
  middle: ResnetBlock -> Mamba_block
  decoder x4: concatenate popped skip -> ResnetBlock -> Mamba_block -> resize
  concatenate stem skip -> final ResnetBlock -> Conv1x1(64,1)

Mamba_block (453), the code-level DACB equivalent:
  adaLN_modulation(fused_t) -> shift/scale/gate x2 (zero-initialized final Linear)
  x + gate1 * SS2D(modulated LayerNorm(x), anatomy)
  x + gate2 * TransposedAttention(modulated LayerNorm(x))
  every constructed block uses cross=False; CrossAttention is not selected

SS2D in src/emamba2.py:
  forward (713) -> projected anatomy = SiLU(Linear256toInner(c))
  in_proj -> x,z -> depthwise Conv3x3/SiLU(x)
  forward_corev2 (698) -> cross_selective_scan (295)
    -> EfficientScan -> image-derived delta/B/C -> SelectiveScan CUDA -> EfficientMerge
  output = out_proj(scan_output * SiLU(z) + projected_anatomy)
```

`train.py:134` passes an undefined variable `folder`: the released entry is not runnable as-is even if its external dependencies and weights were satisfied. We did not execute this full path. `Trainer` also hardcodes authors' dataset paths through PDFDataset. The advertised `DA-CLIP.pth` filename differs from actual `Dose-CLIP.pth`. No baseline/DA-CLIP training entry was inferred from unrelated code.

## 3. Paper / code differences and the authorized decision

1. The repository has no named `RLEB` class. `ResnetBlock` is the local residual block corresponding to its architectural role; that mapping is an inference, not an author-verified class alias. Its actual spatial path is one weight-standardized Conv3x3 + GroupNorm(8) + SiLU, with identity/Conv1x1 residual. The second convolution is commented out. We did not add one.
2. In `ResnetBlock:400-425`, the timestep MLP and scale/shift use are commented out. `Unet.forward` calls these blocks without t. **Deleting all DACBs without a repair removes all effective timestep conditioning**, even if `time_mlp` is retained. A CPU check of the exact original class gave identical outputs for zero and one time embeddings (maximum difference 0).
3. The user explicitly authorized restoring those commented timestep lines, passing `t` into all ten blocks, and describing the result as a reconstruction. The restored time-only SiLU/Linear predicts GroupNorm scale/shift. It is not dose conditioning and is not adaLN-Zero: no LayerNorm, zero initialization or residual gates are present in the baseline.
4. The paper's encoder order is RLEB then DACB; actual official encoder order is DACB then ResnetBlock. The baseline removes DACB and retains the actual remaining convolution/skip/resize topology.
5. The paper places anatomy conditioning into the SSM output matrix C. Current `SS2D` instead adds projected anatomy after scan and z gating, immediately before out_proj. Its scan function does not receive anatomy. We did not attempt to repair or reproduce CSSM.
6. Official dimensions `dims=[64,64,128,256,512]` put encoder blocks at channels 64,64,128,256, with a 512-channel middle. We preserve this concrete construction, not an assumed 64,128,256,512 at every encoder block.
7. Residual-only means a single residual prediction objective, not zero forward diffusion noise. Official q_sample includes noise. `convert_to_ddim=True` uses a linear beta schedule; sum_scale=0.01 still determines sampling initialization. This pre-existing schedule behavior is retained, not silently changed.

## 4. Actual implemented forward graph

Only `src/rddm_rleb.py` is imported by the new training/smoke entry points.

```text
train_rddm_rleb.py -> build_baseline()
  -> ResidualDiffusion.forward([NDCT, LDCT])
     -> normalize to [-1,1]; sample integer t in [0,999]
     -> r = LDCT - NDCT
     -> x_t = NDCT + Abar[t]*r + Bbar[t]*epsilon
     -> UnetRes(cat(x_t, LDCT), [Abar[t]*1000, Bbar[t]*1000])
        -> Unet(time=Abar[t]*1000)
           time_mlp -> t256 -> all 10 RLEB scale/shift MLPs
           stem -> encoder x4 -> middle -> decoder x4 -> final RLEB -> residual
     -> mean squared error(predicted residual, r)

sample([LDCT]) -> normalize -> ddim_sample
  -> x = LDCT + sqrt(0.01)*epsilon
  -> time pairs (999,499), (499,-1)
  -> model_predictions -> same UnetRes / Unet
  -> first update x -= (Abar[999]-Abar[499])*predicted_residual (eta=0)
  -> last step x0 = clamp(LDCT - predicted_residual, -1,1)
  -> unnormalize; return [initial noisy image, restored NDCT]
```

The class is retained from official code, including other dormant objective branches; the supported `build_baseline` configuration fixes one network, pred_res, LDCT condition=True, input_condition=False, no self-conditioning and no learned variance. `condition=True` here means LDCT image concatenation, not dose embeddings.

### Stage table: input H=W=512, batch B

All RLEBs receive the same 256-dimensional timestep embedding. Channels shown are the tensors actually passed to each block. `downs.i.0` and `ups.i.0` are RLEBs.

| Stage / RLEB name | Input channels | Output channels | Resolution | Following resize / saved skip |
|---|---:|---:|---:|---|
| stem init_conv | 2 | 64 | 512x512 | save stem skip r |
| downs.0.0 | 64 | 64 | 512x512 | save h0; Conv4 stride2: 64->64 at 256 |
| downs.1.0 | 64 | 64 | 256x256 | save h1; Conv4 stride2: 64->128 at 128 |
| downs.2.0 | 128 | 128 | 128x128 | save h2; Conv4 stride2: 128->256 at 64 |
| downs.3.0 | 256 | 256 | 64x64 | save h3; Conv3 stride1: 256->512 at 64 |
| mid_block | 512 | 512 | 64x64 | no spatial resize |
| ups.0.0 | 768 = 512+256(h3) | 512 | 64x64 | nearest x2 + Conv3: 512->256 at 128 |
| ups.1.0 | 384 = 256+128(h2) | 256 | 128x128 | nearest x2 + Conv3: 256->128 at 256 |
| ups.2.0 | 192 = 128+64(h1) | 128 | 256x256 | nearest x2 + Conv3: 128->64 at 512 |
| ups.3.0 | 128 = 64+64(h0) | 64 | 512x512 | Conv3 stride1: 64->64 |
| final_res_block | 128 = 64+64(r) | 64 | 512x512 | final Conv1: 64->1 |

There are **10 RLEBs**, **4 encoder-to-decoder concatenation skips**, **1 stem-to-final concatenation skip**, 3 spatial downsamplings and 3 spatial upsamplings. Each RLEB also has its own additive residual; those are separate from the five concatenation skips.

```text
[x_t, LDCT] Bx2xHxW
  stem Conv7: 2->64; save r
  E0 RLEB64; save h0 -> Down 64->64 (/2)
  E1 RLEB64; save h1 -> Down 64->128 (/2)
  E2 RLEB128; save h2 -> Down 128->256 (/2)
  E3 RLEB256; save h3 -> Conv3 256->512 (same size)
  middle RLEB512
  D0 cat(h3):768 -> RLEB512 -> Up 512->256 (x2)
  D1 cat(h2):384 -> RLEB256 -> Up 256->128 (x2)
  D2 cat(h1):192 -> RLEB128 -> Up 128->64 (x2)
  D3 cat(h0):128 -> RLEB64 -> Conv3 64->64 (same size)
  final cat(r):128 -> RLEB64 -> Conv1 64->1
[predicted residual] Bx1xHxW
shared time_mlp -> t256 -> scale/shift MLP in every RLEB
```

### RLEB internals and parameters

```text
feature -> WSConv3x3 -> GroupNorm(8) -> x*(1+scale(t))+shift(t) -> SiLU --+
feature -> Identity if channels unchanged, otherwise Conv1x1 ----------+ -> sum
             t256 -> SiLU -> Linear(256, 2*out_channels) -> scale, shift
```

Total trainable parameters: **13,130,945** across **94 parameter tensors**; no frozen encoders or perceptual networks exist in this module tree. Restoring time MLPs adds 1,052,672 parameters to the retained official component inventory of 12,078,273 (which includes the shared 82,432-parameter time_mlp). This is the actual reconstruction count, not a claimed author baseline count.

| RLEB | Parameters |
|---|---:|
| downs.0.0 | 69,952 |
| downs.1.0 | 69,952 |
| downs.2.0 | 213,632 |
| downs.3.0 | 722,176 |
| mid_block | 2,624,000 |
| ups.0.0 | 4,197,376 |
| ups.1.0 | 1,115,648 |
| ups.2.0 | 312,064 |
| ups.3.0 | 115,072 |
| final_res_block | 115,072 |

## 5. Removed from the executable baseline

- Both CLIP/RN50 objects, CLIPIQA/DA-CLIP heads/prompts, pretrained checkpoint loading and all encoder calls.
- Dose embedding extraction, text_mlp, prompt/prompt_mlp, dose-timestep fusion.
- Every adaLN_modulation / adaLN-Zero scale/shift/gate path from Mamba_block.
- Anatomy embedding, context path, all SS2D/selective scan/Mamba/CSSM code and dependencies.
- All 9 Mamba_block/DACB instances (four encoder, one middle, four decoder).
- All 9 transposed-attention instances. No replacement attention or Mamba module was added.
- Unused LPIPS allocation and full-model training dependencies such as accelerate, W&B and external dataset loaders from the standalone import path.

Frozen upstream files containing these modules remain unchanged as provenance. They are not reachable from either new entry point; the local export contains only the baseline package and executable Python entry points.

## 6. Actual environment and acceptance

Remote user/host/project: `xujian / DESKTOP-0044 / /home/xujian/projects/CoreDiff`.
Python: `/home/xujian/miniconda3/envs/corediff/bin/python`, 3.8.13.
PyTorch 1.10.1+cu111, compiled CUDA 11.1; torchvision 0.11.2+cu111; numpy 1.23.1; einops 0.4.1.
GPU: RTX 3090, 24 GiB. No GPU compute process was present before the test. Existing tmux sessions were left untouched.
`accelerate`, `timm`, `lpips`, `mamba-ssm` are absent in this environment and are not required by the baseline.

Executed from the worktree (all through SSH):

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 /home/xujian/miniconda3/envs/corediff/bin/python smoke_rddm_rleb.py --device cuda:0 --full-resolution --report acceptance/smoke_r2.json
PYTHONDONTWRITEBYTECODE=1 /home/xujian/miniconda3/envs/corediff/bin/python smoke_rddm_rleb.py --device cpu --upstream-source src/DADiff.py --report acceptance/smoke_final_cpu.json
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 /home/xujian/miniconda3/envs/corediff/bin/python train_rddm_rleb.py --ndct acceptance/synthetic_ndct.npy --ldct acceptance/synthetic_ldct.npy --output acceptance/train_one_step --steps 1 --device cpu
```

GPU synthetic B=2, 1x64x64, full production width:
- scalar loss 0.22437141835689545, finite;
- backward global gradient norm 6.906728267669678, finite;
- all 94 parameter tensors have non-missing, finite, not-all-zero gradients;
- changing timestep gives max output difference 0.1332111805677414;
- changing LDCT gives max output difference 0.5248737335205078;
- 2-step sample shape [2,1,64,64], finite, restored output within [0,1];
- full-size B=1 forward shape [1,1,512,512], finite;
- PyTorch peak allocated GPU memory 627,267,584 bytes (not whole-process reserved memory).

Final CPU suite: PASS; loss 0.22876709699630737, gradient norm 4.846174716949463. One synthetic training optimizer step also passed: loss 0.22470371425151825 and pre-clip gradient norm 5.489029884338379. The CPU/GPU random streams differ; these are integrity smoke checks, not denoising quality metrics.

Tests verify the residual q_sample equation, actual MSE objective, topology/forward-hook tensor shapes, timestep/LDCT sensitivity, all parameter gradients and sampling. Import guards reject upstream full-model modules, CLIP, LPIPS, Mamba, selective scan and W&B. Weight-loading functions are replaced by fail-fast guards. No such call was attempted. Actual model module names/types are also checked for forbidden modules.

AST comparison proves the entire ResidualDiffusion class matches official source after deleting only its unused LPIPS construction. First GPU test attempt failed before model execution because peak memory reset preceded CUDA initialization in PyTorch 1.10. The harness was corrected with `torch.cuda.set_device`; the original failure log is retained as `acceptance/smoke.log`.

Detailed output/trace: `acceptance/smoke_r2.json`, `acceptance/smoke_r2.log`, `acceptance/smoke_final_cpu.json`, `acceptance/smoke_final_cpu.log`.
Synthetic training log/checkpoint: `acceptance/train_one_step/train.jsonl`, `acceptance/train_one_step/final.pt`. This checkpoint is random synthetic smoke-test output, not a trained restoration model.

## 7. Fidelity verdict and limits

**Acceptable name:** "FoundDiff-official-code reconstruction of RDDM + RLEB (timestep MLP restored)".
**Not established:** identity with the unpublished Table V first-row implementation, its training recipe, PSNR/SSIM, or author checkpoint compatibility.

The module content and residual objective satisfy the user-authorized reconstruction. The exact original timestep injection and the code-level RLEB naming remain unverified by the authors. This report does not infer Mamba/SS2D belongs in the baseline. The minimal new training entry accepts paired normalized .npy arrays, uses official Adam rate/betas, accumulation=2 and gradient clipping; it deliberately makes no claim to reproduce the authors' augmentation, dataset split or EMA protocol.

Only synthetic minimum acceptance and one optimizer step were executed. Full-resolution backward, long training, full-image dataset sampling, clinical accuracy and generalization are not verified. Recommended next step before any reproduction claim: obtain the authors' ablation configuration and timestep implementation; authorize a dataset-specific training plan separately if needed.

## 8. Files and Git safety

New implementation files: `src/__init__.py`, `src/rddm_rleb.py`, `smoke_rddm_rleb.py`, `train_rddm_rleb.py`.
New remote documentation: this file. New acceptance outputs: `acceptance/` (synthetic only).
No tracked source modification, no staged changes, no commit/push. Thus `git diff --stat`, `git diff` and `git diff --cached` are empty; new files are explicitly untracked and must be reviewed separately. A final manifest records every untracked filename and source hashes. Existing CoreDiff work remains unchanged except the newly created `experiments/founddiff_table_v_20260921/` subtree.

Only four .py files are to be copied locally. No audit documents, logs, synthetic images, checkpoints, original full-model dependencies, Git metadata or medical data are part of the local export.
