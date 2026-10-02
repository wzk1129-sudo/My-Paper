from pathlib import Path
import subprocess,json,os,time
root=Path('/home/xujian/projects/CoreDiff')
out=root/'experiments/corediff_official_checkpoint_eval'
args=['/home/xujian/miniconda3/envs/corediff/bin/python','-u',str(root/'evaluate_official_checkpoint_dual_protocol.py'),'--checkpoint',str(root/'output/corediff_dose25_mayo_2016/save_models/ema_model-150000'),'--output',str(out)]
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0')
with open(out/'run.log','x') as log:
    result=subprocess.run(args,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
with open(out/'EXIT_STATUS.json','x') as f:json.dump({'exit_code':result.returncode,'finished_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')},f,indent=2)
raise SystemExit(result.returncode)
