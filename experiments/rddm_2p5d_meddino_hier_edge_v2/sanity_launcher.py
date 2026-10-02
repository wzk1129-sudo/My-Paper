from pathlib import Path
import subprocess,os,time,json
r=Path('/home/xujian/projects/CoreDiff_meddino_hier_edge_v2/experiments/rddm_2p5d_meddino_hier_edge_v2')
cmd=['/home/xujian/miniconda3/envs/flexict/bin/python', '-u', '/home/xujian/projects/CoreDiff_meddino_hier_edge_v2/experiments/rddm_2p5d_meddino_hier_edge_v2/train_v2.py', '--phase', 'sanity', '--output', '/home/xujian/projects/CoreDiff_meddino_hier_edge_v2/experiments/rddm_2p5d_meddino_hier_edge_v2/runs/SANITY_200']
with open(r/"logs/sanity_200.log","x") as log:
 result=subprocess.run(cmd,cwd=r,env=dict(os.environ,CUDA_VISIBLE_DEVICES="0",CUBLAS_WORKSPACE_CONFIG=":4096:8"),stdout=log,stderr=subprocess.STDOUT)
(r/"SANITY_EXIT_STATUS.json").write_text(json.dumps({"exit_code":result.returncode,"finished_at":time.strftime("%Y-%m-%dT%H:%M:%S%z")},indent=2))
raise SystemExit(result.returncode)
