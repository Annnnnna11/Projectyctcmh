#!/usr/bin/env python
"""Actual two-worker smoke on one host, not proof of Mac execution."""
import os,sys,json,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'experiments/two_machine_exchange_verified';OUT.mkdir(parents=True,exist_ok=True)
def call(*args):
    with (OUT/'integration.log').open('a') as log:
        subprocess.run([sys.executable,'-u',str(HERE/'collaborate.py'),*args,'--smoke'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
def exchange(stage,phase,sender,receiver):
    bundle=OUT/f'{stage}_{phase}_{sender}'
    call('export','--node',sender,'--stage',stage,'--phase',phase,'--output',str(bundle))
    call('import','--node',receiver,'--bundle',str(bundle))
started=time.time()
try:
    # Direct 8-thread prefix probe with the same tiny cutoff/model setup.
    env=os.environ.copy();env.update(M5_COLLAB_NODE='windows',M5_COLLAB_SMOKE='1',M5_PYTHON=sys.executable)
    call('show','--node','windows')
    with (OUT/'prefix_8.log').open('w') as log:
        subprocess.run([sys.executable,'-u',str(HERE/'run.py'),'prepare','--cutoff','1857','--store','CA_1'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        subprocess.run([sys.executable,'-u',str(HERE/'run.py'),'equivalence_check','--cutoff','1857','--store','CA_1'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    for stage in ['development','test','final']:
        if stage!='development':
            call('gate','--node','windows','--stage',stage)
            exchange(stage,'gate','windows','mac')
        if stage!='final':
            for node in ['windows','mac']:call('worker','--node',node,'--stage',stage,'--phase','candidates')
            exchange(stage,'candidates','mac','windows')
            call('select','--node','windows','--stage',stage)
            exchange(stage,'selection','windows','mac')
        for node in ['windows','mac']:call('worker','--node',node,'--stage',stage,'--phase','retrain')
        exchange(stage,'retrain','mac','windows')
        call('score','--node','windows','--stage',stage)
        print('SMOKE_STAGE_COMPLETE',stage,flush=True)
    status=dict(state='complete',seconds=time.time()-started,scope='single-host two-worker smoke; CA_1+CA_3 x 48 items, all three stages')
except BaseException as e:
    status=dict(state='failed',seconds=time.time()-started,error=repr(e));raise
finally:
    (OUT/'status.json').write_text(json.dumps(status,indent=2))
