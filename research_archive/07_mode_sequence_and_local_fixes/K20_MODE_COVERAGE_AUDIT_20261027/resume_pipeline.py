from common import *
import subprocess,traceback
if __name__=='__main__':
    assert not (HERE/'resume_state.json').exists()
    state=dict(status='RUNNING',started=now(),pid=os.getpid(),no_old_worker_restart=True,no_training=True)
    write(HERE/'resume_state.json',state)
    try:
        for name in ('repair_source_manifest.py','preflight.py','e1_modes.py','e2_oracle.py','aggregate.py','plot_diagnostics.py'):
            mark(name);p=subprocess.run([sys.executable,'-B','-u',str(HERE/name)],cwd=HERE);assert p.returncode==0,name
        verify_sources();state.update(status='SUCCESS_REVIEW_REQUIRED',finished=now());write(HERE/'resume_state.json',state);write(HERE/'resume_exit.json',state)
    except BaseException as e:
        state.update(status='FAILURE',finished=now(),error=str(e),trace=traceback.format_exc());write(HERE/'resume_state.json',state);write(HERE/'resume_exit.json',state);raise
