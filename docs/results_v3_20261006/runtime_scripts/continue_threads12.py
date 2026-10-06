"""Verified 8->12 transition, preserving active training and producer provenance."""
import os,sys,json,time,pathlib,subprocess,signal,shutil,fcntl,traceback
ROOT=pathlib.Path('/home/cyangcj/projects/m5-two-machine');HERE=ROOT/'src/time_validation'
sys.path.insert(0,str(HERE))
from common import config,ensure_run,run_root,atomic_json,artifact_id,complete,checkpoint,sha,FILES,features_root,selection_root
from handoff import verify_store,export_bundle,import_bundle,contract,plan
os.environ['M5_COLLAB_NODE']='windows';os.environ['M5_PYTHON']=sys.executable
old=config();assert old['threads']==8;ensure_run(old)
runtime=pathlib.Path(__file__).parent/'thread12_runtime';adapter=runtime/'sitecustomize.py';adapter_sha=sha(adapter)
new=dict(old,threads=12,experiment='time_validation_v3_two_machine_windows_threads12');ensure_run(new)
dest=run_root(new);source=run_root(old);assert contract(old)==contract(new)
state=dict(state='running',threads=12,started=time.time(),completed_steps=[],current='waiting_for_active_8_thread_task',source_root=str(source))
def save():atomic_json(state,dest/'local_continuation.json')
def child(name,*args):
    assert sha(adapter)==adapter_sha,'Execution adapter changed'
    state.update(current=name);save()
    log=dest/'continuation_logs'/f'{name}.log';log.parent.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env.update(M5_RUNTIME_THREADS='12',PYTHONPATH=str(runtime),OMP_NUM_THREADS='12',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with log.open('a') as stream:
        subprocess.run([sys.executable,'-u',str(HERE/'collaborate.py'),*args,'--node','windows'],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    state['completed_steps'].append(name);save()
def migrate_caches():
    records=[]
    for store in old['stores']:
        d=features_root(old,1857)/store
        if not (d/'complete.json').exists():continue
        assert complete(d/'complete.json',artifact_id(old,'features',1857,store,'cache'),FILES)
        t=features_root(new,1857)/store;t.mkdir(parents=True,exist_ok=True)
        for name in [*FILES,'categories.json']:
            if not (t/name).exists():shutil.copy2(d/name,t/name)
            assert sha(t/name)==sha(d/name)
        receipt=json.loads((d/'complete.json').read_text());receipt['fingerprint']=artifact_id(new,'features',1857,store,'cache')
        atomic_json(receipt,t/'complete.json')
        records.append(dict(kind='thread_independent_features',source=str(d),target=str(t),source_checkpoint_sha256=sha(d/'complete.json')))
    # Verified CA_1 binary data establish both ordered schemas for normal bundle import.
    # Source construction used 8 threads; subsequent dataset consumers use 12.
    for mode in old['modes']:
        d=selection_root(old,1857)/mode/'CA_1'/'dataset'
        assert complete(d/'complete.json',artifact_id(old,'dataset',1857,'CA_1',kind='train',mode=mode),['train.bin','schema.json'])
        t=selection_root(new,1857)/mode/'CA_1'/'dataset';t.mkdir(parents=True,exist_ok=True)
        if not (t/'train.bin').exists():shutil.copy2(d/'train.bin',t/'train.bin')
        assert sha(t/'train.bin')==sha(d/'train.bin')
        schema=json.loads((d/'schema.json').read_text());schema['params']['num_threads']=12
        atomic_json(schema,t/'schema.json')
        checkpoint(t/'complete.json',artifact_id(new,'dataset',1857,'CA_1',kind='train',mode=mode),['train.bin','schema.json'])
        records.append(dict(kind='binary_dataset',source=str(d),source_construction_threads=8,target_training_threads=12,source_checkpoint_sha256=sha(d/'complete.json'),binary_sha256=sha(d/'train.bin')))
    atomic_json(dict(source_manifest_sha256=sha(source/'manifest.json'),target_manifest_sha256=sha(dest/'manifest.json'),semantic_contract=contract(new),records=records),dest/'cache_migration.json')
def import_finished():
    for store in plan()['nodes']['windows']['stores']:
        modes=[]
        for mode in old['modes']:
            d=selection_root(old,1857)/mode/store
            if (d/'complete.json').exists():verify_store(old,'development','candidates',mode,store);modes.append(mode)
        if not modes:continue
        bundle=ROOT/'transfers'/f'threads8_to12_dev_{store}'
        if not bundle.exists():export_bundle(old,'windows','development','candidates',bundle,[store],modes)
        import_bundle(new,bundle)
    import_bundle(new,ROOT/'transfers/incoming_mac_CA_3_20261006/mac_dev_candidates_CA_3')
def alive_training(pid):
    p=pathlib.Path(f'/proc/{pid}/stat')
    return p.exists() and p.read_text().split(') ',1)[1].split()[0]!='Z'
if '--validate-only' in sys.argv:
    migrate_caches();import_finished()
    print('Verified thread-independent caches, all completed local results and Mac CA_3; target threads=12',flush=True)
    sys.exit(0)
with (dest/'continuation.lock').open('a') as guard:
    fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        save();boundary=json.loads((source/'thread_transition.json').read_text())
        while any(alive_training(pid) for pid in boundary['active_children']):time.sleep(10)
        parent=boundary['worker_pid']
        if pathlib.Path(f'/proc/{parent}').exists():
            cmd=pathlib.Path(f'/proc/{parent}/cmdline').read_text().replace('\x00',' ')
            assert 'collaborate.py worker --node windows --stage development --phase candidates' in cmd
            os.kill(parent,signal.SIGTERM);os.kill(parent,signal.SIGCONT)
        time.sleep(2)
        with (source/'worker.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        atomic_json(dict(state='superseded',target_root=str(dest),reason='user requested 12 threads; active 8-thread training completed first'),source/'local_continuation.json')
        atomic_json(dict(threads=12,adapter=str(adapter),adapter_sha256=adapter_sha,supervisor_sha256=sha(__file__),source_root=str(source),policy='Serial; previous predictions retain producer provenance; new models use 12 threads; unchanged computational code/data/parameters'),dest/'runtime_profile.json')
        state.update(current='verify_and_import_completed_results');save();migrate_caches();import_finished()
        for store in new['stores']:
            for mode in new['modes']:
                d=selection_root(new,1857)/mode/store
                if (d/'complete.json').exists() or (d/'received.json').exists():verify_store(new,'development','candidates',mode,store);continue
                args=['worker','--stage','development','--phase','candidates','--stores',store,'--modes',mode]
                if store in plan()['nodes']['mac']['stores']:args.append('--takeover')
                child(f'development_candidates_{store}_{mode}',*args)
        child('development_select','select','--stage','development')
        for stage in ['development','test','final']:
            if stage!='development':child(stage+'_gate','gate','--stage',stage)
            if stage=='test':
                child('test_candidates_windows','worker','--stage','test','--phase','candidates')
                child('test_candidates_takeover','worker','--stage','test','--phase','candidates','--stores','CA_3','CA_4','--takeover')
                child('test_select','select','--stage','test')
            child(stage+'_retrain_windows','worker','--stage',stage,'--phase','retrain')
            child(stage+'_retrain_takeover','worker','--stage',stage,'--phase','retrain','--stores','CA_3','CA_4','--takeover')
            child(stage+'_score','score','--stage',stage)
        state.update(state='complete',finished=time.time());save()
    except BaseException:
        state.update(state='failed',error=traceback.format_exc(),finished=time.time());save();raise
