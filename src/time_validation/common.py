from pathlib import Path
import hashlib
import json
import os
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
META = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
F = [f'F{i}' for i in range(1, 29)]
FILES = ['grid_part_1.pkl','grid_part_2.pkl','grid_part_3.pkl','lags_df_28.pkl','mean_encoding_df.pkl']

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

def atomic_json(value, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def atomic_csv(value, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    value.to_csv(tmp, index=False); os.replace(tmp, path)

def event(name, **kwargs):
    print(json.dumps(dict(event=name, time=time.strftime('%Y-%m-%dT%H:%M:%S%z'), **kwargs), default=str), flush=True)

def config(smoke=False):
    c = json.loads((HERE/'config.json').read_text())
    if smoke:
        c.update(experiment='smoke_v2', rounds=5, stores=['CA_1'], smoke_items=48)
    return c

def run_root(c):
    return ROOT/'experiments'/c['experiment']

def stage_root(c, stage):
    return run_root(c)/f'{stage}_d{c["stages"][stage]}'

def identity(c):
    code = {str(p.relative_to(ROOT)): sha(p) for p in sorted(HERE.glob('*.py'))}
    code['src/1_preprocessing_by_store.py'] = sha(ROOT/'src/1_preprocessing_by_store.py')
    raw = {name: sha(ROOT/'data'/name) for name in ['calendar.csv','sell_prices.csv','sales_train_evaluation.csv','sales_train_validation.csv','sample_submission.csv']}
    deps = subprocess.check_output([str(ROOT/'.venv/bin/python'), '-m','pip','freeze'], text=True)
    return dict(config=c, code=code, inputs=raw, dependencies=deps,
                git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())

def ensure_run(c):
    p = run_root(c)/'manifest.json'; actual = identity(c)
    # Commit ID is provenance; content hashes determine compatibility across doc-only commits.
    key = digest({k:v for k,v in actual.items() if k!='git_commit'})
    actual['fingerprint'] = key
    if p.exists():
        old = json.loads(p.read_text())
        if old['fingerprint'] != key:
            raise RuntimeError('Run input/config/code/dependencies changed. Use a new experiment name; no silent reuse.')
    else: atomic_json(actual,p)
    return key

def manifest(c):
    return json.loads((run_root(c)/'manifest.json').read_text())['fingerprint']

def complete(path, fingerprint, files):
    path = Path(path)
    if not path.exists(): return False
    value=json.loads(path.read_text())
    if value['fingerprint'] != fingerprint: raise RuntimeError(f'Incompatible checkpoint: {path}')
    for name in files:
        p=path.parent/name
        if not p.exists() or value['artifacts'].get(name)!=sha(p): return False
    return True

def checkpoint(path, fingerprint, files, **extra):
    path=Path(path)
    atomic_json(dict(fingerprint=fingerprint, artifacts={name:sha(path.parent/name) for name in files}, **extra), path)
