"""Admit one CUDA subprocess at a time using the existing resource policy."""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parents[1]
OLD = WORK/'qoder_work/NEXT_PHASE_EXEC_20261114'
RECOVERY = WORK/'qoder_work/FIVEFOLD_B0_PARALLEL_20261115'
sys.path[:0] = [str(OLD),str(RECOVERY/'code'),str(RECOVERY/'recovery_20261116')]
from resource_guard import snapshot,decide
from job_control import ChildJob
from safe_status import replace_json
from datetime import datetime


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def report(phase, **fields):
    replace_json(ROOT/'CURRENT.json',dict(at=datetime.now().astimezone().isoformat(),phase=phase,**fields))


def main():
    estimated_peak = read(OLD/'PREFLIGHT_E1.json')['estimated_peak_bytes']
    owner = ChildJob()
    try:
        for step in ('preflight','run'):
            while True:
                if (ROOT/'PAUSE.request').exists():
                    report('PAUSED',pending=step)
                    return 0
                admission = decide(snapshot(),estimated_peak,'startup')
                if admission['status'] != 'PASS':
                    report('WAITING_RESOURCES',pending=step,admission=admission)
                    time.sleep(15)
                    continue
                # In case the worker was paused and restarted after preflight.
                if step=='preflight' and (ROOT/'PREFLIGHT_PASS.json').exists():
                    break
                report('PREFLIGHT' if step=='preflight' else 'STARTING_TRAINING',admission=admission)
                cmd = [sys.executable,'-B','-X','utf8',str(ROOT/'run_pilot.py'),step]
                code = owner.popen(cmd,cwd=str(WORK)).wait()
                if code==75:
                    report('WAITING_RESOURCES',pending=step,last_child_exit=code,
                           details_file=str(ROOT/f'{step}_FAILURE.txt'))
                    time.sleep(30)
                    continue
                if code:
                    report('FAILED',step=step,exit_code=code)
                    return code
                break
    finally:
        owner.close()
    return 0


if __name__=='__main__':
    sys.exit(main())
