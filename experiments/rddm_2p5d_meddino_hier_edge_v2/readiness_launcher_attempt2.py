from pathlib import Path
import os,subprocess,json,time
root=Path('/home/xujian/projects/CoreDiff_meddino_hier_edge_v2/experiments/rddm_2p5d_meddino_hier_edge_v2')
with open(root/'logs/readiness_attempt2.log','x') as log:
    r=subprocess.run(['/home/xujian/miniconda3/envs/flexict/bin/python','-u',str(root/'readiness_v2.py')],cwd=root,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8'),stdout=log,stderr=subprocess.STDOUT)
(root/'READINESS_EXIT_STATUS_ATTEMPT2.json').write_text(json.dumps({'exit_code':r.returncode,'finished_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')},indent=2))
raise SystemExit(r.returncode)
