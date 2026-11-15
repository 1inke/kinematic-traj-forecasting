"""Seal a reviewed recovery adapter without changing any original locked source."""
import ast
import hashlib
from pathlib import Path
import sys
import numpy as np
REC=Path(__file__).resolve().parent;BASE=REC.parent
sys.path.insert(0,str(BASE))
from common import *
import resource_guard as rg


def main():
    verify(read_json(BASE/'SOURCE_MANIFEST.json')['files'])
    if read_json(REC/'PREFLIGHT_RECOVERY.json')['status']!='PASS':raise ValueError('ACTUAL_RECOVERY_PREFLIGHT_REQUIRED')
    for p in REC.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
    fit=dict(np.load(BASE/'C0/fit.npz'))
    tracks=np.array(['|'.join(str(x).split('|')[:2]) for x in fit['ids']])
    ordered=sorted(set(tracks),key=lambda x:hashlib.sha256(f'{x}|20261114'.encode()).hexdigest())
    chosen=set(ordered[:int(np.ceil(.15*len(ordered)))]);mask=np.array([x in chosen for x in tracks])
    with np.load(BASE/'E1/calib_split.npz') as split:
        if not np.array_equal(split['calib_index'],np.flatnonzero(mask)) or not np.array_equal(split['fit85_index'],np.flatnonzero(~mask)):raise ValueError('EXISTING_CALIB_SPLIT_NOT_EXACT')
    before=[BASE/'SOURCE_MANIFEST.json',BASE/'WORKER_EXIT.json',BASE/'FAILURE.json',
            BASE/'logs/E1A_s43_train.stdout.log',BASE/'logs/E1A_s43_train.stderr.log',BASE/'E1/calib_split.npz',
            *[p for p in (BASE/'runs/E1A_s43').iterdir() if p.is_file()]]
    source=[*REC.glob('*.py'),REC/'START.ps1',REC/'START.cmd',REC/'STATUS.cmd',REC/'README.md',REC/'ORIGINAL_TASK.xml',REC/'PREFLIGHT_RECOVERY.json']
    files=seal(before+source)
    write_json(REC/'RECOVERY_MANIFEST.json',dict(at=now(),files=files,
        original_scientific_source_manifest_sha=sha(BASE/'SOURCE_MANIFEST.json'),
        root_cause='Local variable seal in original train shadows imported seal function, raises after final.pt save',
        recovery_scope='No new scientific arm; reuse saved full seed43; same finite sequence and Trainer.step; add epoch checkpoints and scheduler-owned lifecycle',
        source_changes='New recovery files only; original125 source/inputs unchanged',
        expected_reused_final_sha=sha(BASE/'runs/E1A_s43/final.pt')))
    estimate=read_json(BASE/'PREFLIGHT_E1.json')['estimated_peak_bytes']
    admission=rg.decide(rg.snapshot(),estimate,'startup');write_json(REC/'PRELAUNCH_ADMISSION.json',dict(at=now(),**admission))
    if admission['status']!='PASS':raise MemoryError('CURRENT_RESOURCE_ADMISSION')
    print('RECOVERY_SEALED_ADMITTED',len(files),flush=True)


if __name__=='__main__':main()
