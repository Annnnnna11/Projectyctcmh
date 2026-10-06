"""Opt-in execution adapter; shared computational source remains immutable."""
import os,sys
if os.environ.get('M5_RUNTIME_THREADS')=='12':
    sys.path.insert(0,'/home/cyangcj/projects/m5-two-machine/src/time_validation')
    import common
    original_config=common.config
    def runtime_config(smoke=False):
        c=original_config(smoke)
        if os.environ.get('M5_COLLAB_NODE')=='windows' and not smoke and os.environ.get('M5_COLLAB_SMOKE')!='1':
            c.update(threads=12,experiment='time_validation_v3_two_machine_windows_threads12')
        return c
    common.config=runtime_config
