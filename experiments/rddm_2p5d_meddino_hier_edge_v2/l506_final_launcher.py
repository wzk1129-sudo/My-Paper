"""Wait for V2 final audit to pass, then run one authorized, read-only L506 EMA evaluation."""
from pathlib import Path
import hashlib,json,os,subprocess,time,traceback
r=Path('/home/xujian/projects/CoreDiff_meddino_hier_edge_v2/experiments/rddm_2p5d_meddino_hier_edge_v2')
source=r/'evaluate_l506_final_v2.py';expected_sha='ff72002a799629b216e599fd0f8938e019259ef698429a11ea6e84f898359a9b'
command=['/home/xujian/miniconda3/envs/flexict/bin/python', '-u', '/home/xujian/projects/CoreDiff_meddino_hier_edge_v2/experiments/rddm_2p5d_meddino_hier_edge_v2/evaluate_l506_final_v2.py']
log=r/'logs/l506_final_8k.log'
def write(name,obj):(r/name).write_text(json.dumps(obj,indent=2)+'\n')
try:
 assert hashlib.sha256(source.read_bytes()).hexdigest()==expected_sha
 began=time.monotonic()
 while True:
  state=json.loads((r/'PIPELINE_STATUS.json').read_text())
  if state['status']=='COMPLETE':break
  if state['status']=='FAIL':raise RuntimeError('V2 training/finalize pipeline failed; L506 evaluation cancelled')
  if time.monotonic()-began>6*3600:raise TimeoutError('V2 8k completion gate timed out')
  time.sleep(30)
 while True:
  gpu=subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],text=True).strip()
  used,total=(int(x.strip()) for x in gpu.split(',')[:2])
  if used<4096 and total-used>18000:break
  if time.monotonic()-began>7*3600:raise TimeoutError('GPU never became idle for final L506 evaluation')
  time.sleep(30)
 assert not (r/'L506_FINAL_EVAL_8K').exists() and not log.exists()
 write('L506_PRE_INFERENCE_AUDIT.json',{'training_pipeline_status':state['status'],'gpu_memory_used_mib':used,'gpu_memory_total_mib':total,'evaluator_sha256':expected_sha,'command':command,'checkpoint':str(r/'runs/2P5D_MEDDINO_HIER_EDGE_V2_8K/checkpoints/step_8000.pt'),'output':str(r/'L506_FINAL_EVAL_8K'),'started_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'gpu_index':0,'conda_environment':'flexict'})
 write('L506_PIPELINE_STATUS.json',{'status':'RUNNING','started_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')})
 with log.open('x') as f:
  result=subprocess.run(command,cwd=r,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8'),stdout=f,stderr=subprocess.STDOUT)
 write('L506_PIPELINE_EXIT_STATUS.json',{'exit_code':result.returncode,'finished_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')})
 if result.returncode:raise RuntimeError('L506 evaluator exited '+str(result.returncode))
 summary=json.loads((r/'L506_FINAL_EVAL_8K/SUMMARY.json').read_text())
 assert summary['status']=='COMPLETE' and summary['slice_count']==524
 write('L506_PIPELINE_STATUS.json',{'status':'COMPLETE','finished_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'summary':str(r/'L506_FINAL_EVAL_8K/SUMMARY.json')})
except Exception:
 write('L506_PIPELINE_STATUS.json',{'status':'FAIL','traceback':traceback.format_exc(),'finished_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')})
 raise
