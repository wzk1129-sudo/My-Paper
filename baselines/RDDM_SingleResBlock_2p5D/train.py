from common import *
import argparse,copy,csv,time,traceback,hashlib
from datetime import datetime,timezone
from mayo_data import Mayo,Stream,TRAIN_PATIENTS
from src.residual_diffusion import ResidualDiffusion
from evaluate import evaluate

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--resume");args=parser.parse_args()
    assert json.loads((OUT/"reports/PREFLIGHT.json").read_text())["passed"]
    setup()
    if (OUT/"COMPLETE.json").exists(): raise RuntimeError("Run already complete")
    stream=Stream(Mayo(CFG["data_root"],TRAIN_PATIENTS),CFG["seed"])
    model=ResidualDiffusion().cuda(); ema=copy.deepcopy(model).eval()
    for param in ema.parameters():param.requires_grad_(False)
    optimizer=torch.optim.Adam(model.parameters(),lr=2e-4,betas=(0.9,0.99),weight_decay=1e-9)
    start_step=0; ema_updates=0; elapsed_prior=0.0; train_seconds=0.0
    if args.resume:
        ck=torch.load(args.resume,map_location="cpu",weights_only=False)
        assert ck["training_config"]==CFG and ck["source_commit"]==SOURCE["commit"]
        start_step=ck["optimizer_step"]; assert 0<=start_step<=8000
        model.load_state_dict(ck["model_state_dict"]);ema.load_state_dict(ck["ema_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"]);ema_updates=ck["ema_updates"]
        stream.restore(ck)
        random.setstate(ck["python_rng_state"]);np.random.set_state(ck["numpy_rng_state"])
        torch.set_rng_state(ck["torch_rng_state"]);torch.cuda.set_rng_state_all(ck["cuda_rng_state"])
        elapsed_prior=ck["elapsed_seconds"];train_seconds=ck["training_seconds"]
        assert not any(int(q.stem.split("_")[-1])>start_step for q in (OUT/"checkpoints").glob("step_*.pt"))
    else:
        assert not list((OUT/"checkpoints").glob("*.pt")),"Existing checkpoints: explicit resume required"
    start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    def checkpoint(step):
        state=dict(optimizer_step=step,model_state_dict=model.state_dict(),ema_state_dict=ema.state_dict(),
                   optimizer_state_dict=optimizer.state_dict(),ema_updates=ema_updates,
                   python_rng_state=random.getstate(),numpy_rng_state=np.random.get_state(),
                   torch_rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state_all(),
                   training_config=CFG,source_commit=SOURCE["commit"],source_metadata=SOURCE,
                   elapsed_seconds=elapsed_prior+time.monotonic()-start,training_seconds=train_seconds,
                   **stream.state())
        with (OUT/"checkpoints"/f"step_{step}.pt").open("xb") as f:torch.save(state,f)
    if not args.resume:checkpoint(0)
    summary_path=OUT/"validation_summary.csv"
    summaries=[]
    if summary_path.exists():
        with summary_path.open() as f:
            summaries=[{k:(int(v) if k=="step" else float(v)) for k,v in r.items()} for r in csv.DictReader(f)]
    def validation(step):
        # Snapshot all training RNGs to verify validation cannot perturb training.
        cpu=torch.get_rng_state().clone();gpu=[s.clone() for s in torch.cuda.get_rng_state_all()]
        py=random.getstate();np_state=np.random.get_state()
        row=evaluate(ema,step)
        assert torch.equal(cpu,torch.get_rng_state())
        assert all(torch.equal(a,b) for a,b in zip(gpu,torch.cuda.get_rng_state_all()))
        assert py==random.getstate()
        after=np.random.get_state()
        assert np_state[0]==after[0] and np.array_equal(np_state[1],after[1]) and np_state[2:]==after[2:]
        summaries.append(row)
        exists=summary_path.exists()
        with summary_path.open("a" if exists else "x",newline="") as f:
            w=csv.DictWriter(f,fieldnames=row.keys())
            if not exists:w.writeheader()
            w.writerow(row)
        print("VALIDATION "+json.dumps(row),flush=True)
    if args.resume and start_step in (2000,4000,6000,8000) and start_step not in [r["step"] for r in summaries]:
        validation(start_step)
    logfile=OUT/"logs/train.jsonl"
    with logfile.open("a" if args.resume else "x") as log:
        for step in range(start_step+1,8001):
            tick=time.monotonic();model.train()
            c,y=stream.batch();optimizer.zero_grad(set_to_none=True)
            loss=model.loss(c,y)
            if not torch.isfinite(loss):raise FloatingPointError(f"Nonfinite loss at {step}")
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                for ep,mp in zip(ema.parameters(),model.parameters()):ep.mul_(0.995).add_(mp,alpha=0.005)
                for eb,mb in zip(ema.buffers(),model.buffers()):eb.copy_(mb)
            ema_updates+=1;torch.cuda.synchronize();seconds=time.monotonic()-tick;train_seconds+=seconds
            row=dict(step=step,loss=float(loss),gradient_norm=float(norm),learning_rate=optimizer.param_groups[0]["lr"],
                     seconds_per_step=seconds,peak_gpu_memory=torch.cuda.max_memory_allocated())
            log.write(json.dumps(row,allow_nan=False)+chr(10));log.flush()
            if step==1 or step%20==0:print("TRAIN "+json.dumps(row),flush=True)
            if step in (2000,4000,6000,8000):
                checkpoint(step);validation(step)
    assert [r["step"] for r in summaries]==[2000,4000,6000,8000] and ema_updates==8000
    assert all(torch.isfinite(q).all() for q in model.parameters())
    best=max(summaries,key=lambda r:r["psnr"])
    elapsed=elapsed_prior+time.monotonic()-start
    pre=json.loads((OUT/"reports/PREFLIGHT.json").read_text())
    complete=dict(optimizer_steps=8000,ema_updates=ema_updates,validation_steps=[r["step"] for r in summaries],
                  best_ema_checkpoint=str(OUT/"checkpoints"/f"step_{best['step']}.pt"),
                  best_psnr=best["psnr"],parameters=pre["parameters"],training_seconds=train_seconds,
                  total_elapsed_seconds=elapsed,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),L506_accesses=0,
                  completed_utc=datetime.now(timezone.utc).isoformat(),source_commit=SOURCE["commit"])
    table="| Step | PSNR | SSIM | RMSE HU | MAE HU | Soft PSNR | Soft SSIM | Soft RMSE HU | Soft MAE HU |"+chr(10)+"|---|---|---|---|---|---|---|---|---|"+chr(10)
    for r in summaries:
        table+="| "+str(r["step"])+" | "+" | ".join(f'{r[k]:.6f}' for k in ["psnr","ssim","rmse_hu","mae_hu","soft_psnr","soft_ssim","soft_rmse_hu","soft_mae_hu"])+" |"+chr(10)
    report=f"""# RDDM SingleResBlock 2.5D 8P 8K
Completed 8000 optimizer steps and four fixed EMA validations.

## Provenance and architecture
Official repository: {SOURCE['repository']}
Official commit: {SOURCE['commit']}
Official source: {SOURCE['path']}
Snapshot SHA256: {SOURCE['sha256']}
Project commit at creation: {SOURCE['project_commit']}
Official building blocks are copied verbatim and AST-verified against the immutable snapshot.
Original single U-Net: 19 full ResBlocks (8 encoder + 2 bottleneck + 8 decoder + 1 final), 8 LinearAttention, 1 Global Attention, 8 encoder-decoder skips, 1 stem skip.
New single U-Net: 10 full ResBlocks (4+1+4+1), 8 LinearAttention, 1 Global Attention, 4 encoder-decoder skips, 1 stem skip.
Total/trainable parameters: {pre['parameters']}/{pre['trainable_parameters']}.
Input: [I_t, LDCT_prev, LDCT_center, LDCT_next], output: center residual [B,1,512,512].
Only the new experiment directory was created. No pretrained weights, extra losses, or extra modules.
Each complete ResBlock retains its internal block1 and block2 convolution subblocks; there is no second ResBlock per scale.
MLSF can subsequently be integrated at explicit encoder/decoder interfaces, but none is present in this experiment.

## Protocol
Stored values -> float32 HU = stored - 1024; clip [-1024,3072]; u=(HU+1024)/4096; diffusion maps u to 2u-1 once.
Linear beta: 0.0001 to 0.02 over 1000 steps; P=cumprod(1-beta); A=1-sqrt(P); B=sqrt(1-P), float32 buffers.
Training: R=Y-X0; X_t=X0+A_t R+B_t epsilon; only residual MSE. t uniform integer [0,999].
Time input always A_t*1000 to native time MLP.
Sampling: X999=Y+0.1 epsilon; clip predicted residual to [-1,1]; X499=X999-(A999-A499)R999; final clamp(Y-R499,-1,1). Model times 999,499. No predicted-noise update.
Training B999={pre['B999']}; inference initial standard deviation=0.1. This mismatch is retained as requested.
Adam lr=2e-4, betas=(0.9,0.99), weight_decay=1e-9; batch=2, accumulation=1, FP32, AMP/TF32 off; gradient clip=1; no scheduler.
EMA decay=.995 every optimizer step, zero warmup. Seed=20260829. Augmentation uses isolated seed=10 RNG, paired horizontal flip p=.5 and uniform quarter rotations. No crop.
Training patients: {CFG['train_patients']}; {pre['train_pairs']} pairs. Validation L333 Fast64, exact saved order, batch=1, isolated CUDA noise generator seed=20260829+slice_index.
L506 accesses: 0 (allowlisted enumeration and reads only).
Checkpoint contains model/EMA/optimizer, all requested RNG states, shuffle permutation and cursor, configuration and source metadata.
Environment: {json.dumps(pre['environment'])}

## Metrics
Each slice is independently windowed and evaluated, then 64 slice metrics are arithmetically averaged.
Main window [-1000,1000], data_range=2000; soft window [-160,240], data_range=400.
skimage SSIM win_size=7, gaussian_weights=False, use_sample_covariance=True, channel_axis=None, K1=.01,K2=.03.
{table}
Best EMA checkpoint: {complete['best_ema_checkpoint']} (ema_state_dict).
Training-step time: {train_seconds:.2f} seconds; total including saves/validation: {elapsed:.2f} seconds.
Peak allocated GPU memory: {complete['peak_gpu_allocated_bytes']/2**30:.3f} GiB; peak reserved: {complete['peak_gpu_reserved_bytes']/2**30:.3f} GiB.
Reference supplied by user: FoundDiff TimeOnly 40.846046 dB; difference of best score: {best['psnr']-40.846046:+.6f} dB.
That historical score is not independently re-evaluated here: differences in architecture, initialization, training schedule, fixed evaluation noise or preprocessing limit causal comparison.
This is one seed and one validation patient; validation checkpoint selection is not a held-out test result.
No additional ablations or noise settings were tested. Inspect loss trend and soft-tissue scores before adding MLSF/detail modules.
"""
    with (OUT/"REPORT.md").open("x") as f:f.write(report)
    write_json(OUT/"COMPLETE.json",complete)
    print("COMPLETE "+json.dumps(complete),flush=True)

if __name__=="__main__":
    try:main()
    except Exception:
        path=OUT/"reports"/("FAILURE_"+str(time.time_ns())+".json")
        write_json(path,{"traceback":traceback.format_exc()})
        raise
