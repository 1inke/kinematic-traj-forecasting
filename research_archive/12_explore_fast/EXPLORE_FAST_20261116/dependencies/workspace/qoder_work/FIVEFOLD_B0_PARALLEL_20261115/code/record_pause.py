"""Record a user-requested stop only after the caller verifies child processes exited."""
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parent.parent
OLD = ROOT.parent / 'FIVEFOLD_B0_EXEC_20261115'
sys.path.insert(0, str(OLD / 'code'))
import runtime as rt
from fivefold_common import now, read_json, sha256

event = ROOT / 'standalone/pause_events' / str(time.time_ns())
event.mkdir(parents=True, exist_ok=False)
for name in ('CURRENT.json', 'ACTIVE_WORKER.json', 'CHILD_STATE.json'):
    path = ROOT / 'standalone' / name
    if path.exists():
        (event / ('prior_' + name)).write_bytes(path.read_bytes())
checkpoints = {}
done = []
for directory in (OLD / 'FIVEFOLD_RUNS/runs').iterdir():
    if (directory / 'DONE.json').exists():
        rt.validate_done(directory); done.append(directory.name)
    else:
        committed = []
        for pt in sorted(directory.glob('ckpt_epoch*.pt')):
            side = pt.with_suffix('.json')
            if side.exists():
                receipt = read_json(side)
                if sha256(pt) != receipt['sha256']:
                    raise ValueError('CHECKPOINT_SHA_MISMATCH')
                committed.append(receipt)
        if committed:
            checkpoints[directory.name] = committed[-1]
value = dict(at=now(), status='PAUSED_BY_USER', stage='PAUSED_BY_USER', controller='PARALLEL',
             exit_code=None, controller_action_exit_code=0, automatic_resume=False,
             note='Windows user stop; worker normal exit code not captured; incomplete epochs repeat on resume.',
             durable_checkpoints=checkpoints, completed_runs=done, pause_event=str(event))
rt.atomic(event / 'PAUSE_COMPLETE.json', value)
for base in (ROOT / 'standalone', OLD / 'standalone'):
    rt.atomic(base / 'CURRENT.json', value)
    rt.atomic(base / 'WORKER_EXIT.json', value)
    rt.atomic(base / 'CHILD_STATE.json', dict(at=now(), state='PAUSED_BY_USER', active=[]))
print('PAUSED; committed epoch checkpoints verified.')
