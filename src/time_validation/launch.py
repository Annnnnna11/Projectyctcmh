#!/usr/bin/env python
"""Keep a WSL client alive, detach training from terminal, preserve log and PID."""
import os
import subprocess
import sys
from common import ROOT, HERE, config, run_root, atomic_json

root=run_root(config()); root.mkdir(parents=True,exist_ok=True)
with (root/'runner.log').open('a') as log:
    process=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u',str(HERE/'run.py'),'all'],
                             cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                             start_new_session=True)
    atomic_json(dict(launcher_pid=os.getpid(),runner_pid=process.pid),root/'launcher.json')
    sys.exit(process.wait())
