#!/usr/bin/env python
"""Cutoff-safe training and inference. Original implementation: baseline commit c75d089.

Use src/time_validation/run.py all for the complete authorized workflow.
This compatibility entry point requires --stage and --store and a prepared run/cache.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"time_validation"))
from run import main
if __name__ == "__main__":
    sys.argv[1:1] = ["model", "--mode", "nonrecursive"]
    main()
