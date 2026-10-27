"""Report only actual EMA metrics for the new eight-patient augmented MedCNN run."""
from pathlib import Path
import csv,json
OUT=Path(__file__).resolve().parent
OLD=Path('/home/xujian/research/FoundDiff_TimeOnly_PairedFastDev/experiments/Fast2P_TimeOnly_MedCNN_8k')
KEYS=('psnr','ssim','rmse_hu','mae_hu','soft_psnr','soft_ssim','soft_rmse_hu')
def write_report(history,params,peak,reserved):
 with (OLD/'validation_summary.csv').open() as f:old={int(x['step']):x for x in csv.DictReader(f)}
 comparisons=[{'step':r['step'],**{k:r[k] for k in KEYS},**{'old_2patient_noaug_'+k:float(old[r['step']][k]) for k in KEYS},'delta_psnr':r['psnr']-float(old[r['step']]['psnr'])} for r in history]
 with (OUT/'comparison.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(comparisons[0]));w.writeheader();w.writerows(comparisons)
 best=max(history,key=lambda x:x['psnr'])
 (OUT/'BEST_CHECKPOINT.json').write_text(json.dumps({'step':best['step'],'psnr':best['psnr'],'path':str(OUT/f"checkpoints/step_{best['step']}.pt"),'weights_key':'ema_state_dict'},indent=2))
 text='# MedCNN eight-patient synchronized augmentation 8k\n\n'
 text+=f"Completed {history[-1]['step']}/8000 optimizer steps, {len(history)}/4 EMA validations.\n\n"
 text+='|Step|New 8patient+aug PSNR|Old 2patient noaug PSNR|Delta PSNR|SSIM|RMSE HU|MAE HU|Soft PSNR|Soft SSIM|Soft RMSE HU|\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
 for x in comparisons:
  text+='|'+str(x['step'])+'|'+'|'.join(f'{x[k]:.8f}' for k in ['psnr','old_2patient_noaug_psnr','delta_psnr','ssim','rmse_hu','mae_hu','soft_psnr','soft_ssim','soft_rmse_hu'])+'|\n'
 pre=json.loads((OUT.parents[1]/'PREFLIGHT_DETAILS.json').read_text())
 text+=f"\nParameters {pre['parameters']['B']}; peak allocated {peak/2**30:.3f} GiB; reserved {reserved/2**30:.3f} GiB. Best EMA {OUT}/checkpoints/step_{best['step']}.pt, key ema_state_dict.\n"
 if (OUT/'COMPLETE.json').exists():
  done=json.loads((OUT/'COMPLETE.json').read_text());text+=f"\nTraining plus validation elapsed {done['seconds']:.2f} seconds.\n"
 text+='\nTrain L067/L096/L109/L143/L192/L286/L291/L310,4800 centers; original L333-Fast64 order; no L506. Seed10,batch2,accum2,Adam2e-4 betas(.9,.99),wd0,FP32,clip1,no scheduler,EMA.995 every optimizer step,original residualMSE,RDDM1000diffusion/2sampling,sumscale.01. Stored-1024 HU,clip[-1000,1000],dataset[0,1],model[-1,1]. Main PSNRrange2000,SSIMwin7;soft[-160,240].\n'
 text+='\nModel and computational train.py copied byte-for-byte from the actual archived MedCNN run. Common backbone loaded only from original A untrained step0; no trained A/B weights used. New modules freshly initialized with original seed and verified equal to original B step0. MedDINO official weight SHA256 '+pre['meddino']['sha256']+' strictly loaded, frozen/eval. Original CNN,local cross-attention,Fusion,nine independent spatial/FiLM/time gates,TimeOnlyDACBs and RDDM unchanged. See src/DADiff.py and src/medcnn.py.\n'
 text+='\nAugmentation only in training Stream.next: independent private RandomState10 chooses horizontal flip probability.5 and uniform rotation0/90/180/270. Same transform applied jointly to all3 LDCT channels and center NDCT; contiguous index permutations, no interpolation,crop,resize,filtering. Validation uses unaugmented Pairs.load. Aug RNG saved/restored in checkpoints; independent of shuffle and diffusion RNG. Actual transforms logged per sample.\n'
 text+='\nHistorical comparison changes BOTH training patients and augmentation. It cannot isolate the augmentation effect. No extra ablation or architecture change was run. Preflight real batch2accum2,all8 transformations,synchronization,resume RNG,initialization,gradients,frozen MedDINO and2-step sampling passed.\n'
 (OUT/'REPORT.md').write_text(text)
