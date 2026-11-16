"""Record only committed epoch checkpoints after the experiment's processes stop."""
from eth_exec_common import *

verify_source()
checkpoints={}
for run in (ROOT/'runs').glob('*'):
    sidecars=sorted(run.glob('ckpt_epoch*.json'))
    if sidecars:
        receipt=read(sidecars[-1]);assert sha(receipt['file'])==receipt['sha256']
        checkpoints[run.name]=receipt
    for name in ('TRAIN_SEAL.json','EVALUATION_SEAL.json'):
        if (run/name).exists():validate_seal(run,name)
record=dict(at=now(),status='PAUSED_BY_USER',automatic_resume=False,complete_epoch_checkpoints_preserved=True,durable_checkpoints=checkpoints)
replace_json(ROOT/'standalone/CURRENT.json',record)
replace_json(ROOT/'standalone/WORKER_EXIT.json',dict(at=now(),status='PAUSED_BY_USER',exit_code=None,controller_action_exit_code=0))
print('PAUSED; committed epoch checkpoint hashes verified.')
