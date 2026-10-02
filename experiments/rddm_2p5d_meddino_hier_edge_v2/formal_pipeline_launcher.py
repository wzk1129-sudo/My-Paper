from pathlib import Path
import subprocess,os,time,json,hashlib
r=Path('/home/xujian/projects/CoreDiff_meddino_hier_edge_v2/experiments/rddm_2p5d_meddino_hier_edge_v2')
commands=[['/home/xujian/miniconda3/envs/flexict/bin/python', '-u', '/home/xujian/projects/CoreDiff_meddino_hier_edge_v2/experiments/rddm_2p5d_meddino_hier_edge_v2/train_v2.py', '--phase', 'formal', '--output', '/home/xujian/projects/CoreDiff_meddino_hier_edge_v2/experiments/rddm_2p5d_meddino_hier_edge_v2/runs/2P5D_MEDDINO_HIER_EDGE_V2_8K'], ['/home/xujian/miniconda3/envs/flexict/bin/python', '-u', '/home/xujian/projects/CoreDiff_meddino_hier_edge_v2/experiments/rddm_2p5d_meddino_hier_edge_v2/finalize_v2.py']]
source_sha={'/home/xujian/projects/CoreDiff_meddino_hier_edge_v2/experiments/rddm_2p5d_meddino_hier_edge_v2/train_v2.py': '7bfc1da59a0d99e34a4e5a704154941254bbb0d5283a0669dafa0a8b904ec2a2', '/home/xujian/projects/CoreDiff_meddino_hier_edge_v2/experiments/rddm_2p5d_meddino_hier_edge_v2/model_v2.py': 'e5e133bc9d3a0876e60dedf66d1eb98383a47058d909e7748e2332a3fe50ed90', '/home/xujian/projects/CoreDiff_meddino_hier_edge_v2/experiments/rddm_2p5d_meddino_hier_edge_v2/finalize_v2.py': 'c234b8a6bbeb07b9e299cc3ee1d0c40faaa6ae0fbdc6b5c10e0edebc24e3c379', '/home/xujian/projects/CoreDiff_meddino_hier_edge_v2/experiments/rddm_2p5d_meddino_hier_edge_v2/CONFIG.json': 'cc44c6f18a74e052d8a38d841cebbbe1027a34a5b32084347db40fd7afe5d6e4'}
for p,h in source_sha.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
results=[]
for stage,cmd,name in zip(('training','region_diagnostics_report'),commands,('formal_8k.log','finalize_8k.log')):
 (r/'PIPELINE_STATUS.json').write_text(json.dumps({'stage':stage,'status':'RUNNING','time':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'commands':commands},indent=2))
 with open(r/'logs'/name,'x') as log:
  result=subprocess.run(cmd,cwd=r,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8'),stdout=log,stderr=subprocess.STDOUT)
 results.append({'stage':stage,'exit_code':result.returncode,'finished_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')})
 (r/'PIPELINE_EXIT_STATUS.json').write_text(json.dumps(results,indent=2))
 if result.returncode:
  (r/'PIPELINE_STATUS.json').write_text(json.dumps({'stage':stage,'status':'FAIL','exit_code':result.returncode},indent=2));raise SystemExit(result.returncode)
 for p,h in source_sha.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
(r/'PIPELINE_STATUS.json').write_text(json.dumps({'status':'COMPLETE','finished_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'no_extension':True},indent=2))
