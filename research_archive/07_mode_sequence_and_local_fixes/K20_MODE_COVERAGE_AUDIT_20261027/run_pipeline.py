from common import *
import subprocess,traceback
def main():
    if (HERE/'worker_state.json').exists():raise RuntimeError('Existing execution preserved, do not duplicate')
    state=dict(status='RUNNING',started=now(),pid=os.getpid(),scope='finite CPU native-only E1/E2;0newmodels');write(HERE/'worker_state.json',state)
    try:
        for script in ('prepare_inputs.py','preflight.py','e1_modes.py','e2_oracle.py','aggregate.py','plot_diagnostics.py'):
            mark(script);p=subprocess.run([sys.executable,'-B','-u',str(HERE/script)],cwd=HERE);assert p.returncode==0,script+' failed; preserve all artifacts'
        verify_sources();state.update(status='SUCCESS_REVIEW_REQUIRED',finished=now());write(HERE/'worker_state.json',state);write(HERE/'exit_status.json',state)
    except BaseException as e:
        state.update(status='FAILURE',finished=now(),error=str(e),trace=traceback.format_exc());write(HERE/'FAILURE.json',state);write(HERE/'worker_state.json',state);write(HERE/'exit_status.json',state);raise
if __name__=='__main__':main()
