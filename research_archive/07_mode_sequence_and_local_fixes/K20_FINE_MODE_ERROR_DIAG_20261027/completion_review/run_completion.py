from pathlib import Path
import sys,subprocess,traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
if __name__=='__main__':
    state=dict(status='RUNNING',at=now(),no_model_or_training=True)
    write(HERE/'completion_review/completion_state.json',state)
    try:
        for name in ('junction_and_phase.py','review_saved_v2.py','plot_figures.py'):
            print('COMPLETION',name,flush=True)
            result=subprocess.run([sys.executable,'-B','-u',str(HERE/'completion_review'/name)])
            assert result.returncode==0,name
        state.update(status='SUCCESS_REQUIRES_VISUAL_AND_FINAL_REPORT',finished=now())
        write(HERE/'completion_review/completion_state.json',state)
    except BaseException as e:
        state.update(status='FAILURE',finished=now(),error=str(e),trace=traceback.format_exc())
        write(HERE/'completion_review/completion_state.json',state);write(HERE/'completion_review/FAILURE.json',state);raise
