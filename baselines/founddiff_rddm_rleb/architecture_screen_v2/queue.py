"""One-process-at-a-time V2 queue. Failure never triggers retry or overlap."""
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path
from mayo2016_smallset_v1.runtime import ROOT, write_json
from .protocol import HERE, config, source_manifest
from .variants import ARMS

def timestamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def main():
    cfg,_=config();manifest=source_manifest()
    for arm in ARMS:
        record=json.loads((HERE/'smoke_pass_v4'/arm/'SMOKE_RESULT.json').read_text())
        assert record['status']=='SMOKE_PASS' and record['source_manifest']==manifest
    assert not (HERE/'queue_status.json').exists(), 'Queue has already been launched'
    state={'status':'RUNNING','started_utc':timestamp(),'source_manifest':manifest,
      'order':list(ARMS),'arms':{a:{'status':'PENDING'} for a in ARMS}}
    write_json(HERE/'queue_status.json',state)
    log=HERE/'queue.log';process_log=HERE/'queue_process.log'
    with log.open('x') as summary,process_log.open('x') as output:
        for arm in ARMS:
            state['arms'][arm]={'status':'RUNNING','started_utc':timestamp()}
            write_json(HERE/'queue_status.json',state)
            summary.write(arm+' START '+timestamp()+'\n');summary.flush()
            command=[sys.executable,'-u','-m','architecture_screen_v2.train','train','--arm',arm,'--device','cuda:0']
            env=os.environ.copy();env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
            result=subprocess.run(command,cwd=str(ROOT),env=env,stdout=output,stderr=subprocess.STDOUT)
            output.flush()
            final_path=HERE/'runs'/arm/'final_summary.json'
            if result.returncode==0 and final_path.is_file():
                final=json.loads(final_path.read_text())
                status=final['status']
            else:status='FAILED'
            state['arms'][arm]={'status':status,'exit_code':result.returncode,'ended_utc':timestamp()}
            write_json(HERE/'queue_status.json',state)
            summary.write(arm+' '+status+' exit='+str(result.returncode)+' '+timestamp()+'\n');summary.flush()
        state['status']='COMPLETE_WITH_FAILURES' if any(x['status']=='FAILED' for x in state['arms'].values()) else 'COMPLETE'
        state['ended_utc']=timestamp();write_json(HERE/'queue_status.json',state)
    from .report import build_report
    build_report()

if __name__=='__main__':main()
