"""Small execution safeguards for a finite exploratory study."""
from pathlib import Path
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta

ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = ROOT.parents[1]
EXEC = WORKSPACE / 'qoder_work/NEXT_PHASE_EXEC_20261114'
FIVE = WORKSPACE / 'qoder_work/FIVEFOLD_B0_EXEC_20261115'
PARALLEL = WORKSPACE / 'qoder_work/FIVEFOLD_B0_PARALLEL_20261115'
sys.path.insert(0, str(PARALLEL / 'recovery_20261116'))
from safe_status import replace_json, transient


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def verify(files):
    for path, expected in files.items():
        if sha(path) != expected:
            raise ValueError('SOURCE_OR_OUTPUT_CHANGED: ' + str(path))


def verify_source():
    verify(read(ROOT / 'SOURCE_MANIFEST.json')['files'])


def status(path, value):
    value = dict(at=now(), **value)
    try:
        replace_json(path, value)
    except OSError as exc:
        if not transient(exc):
            raise
        mirror = ROOT / 'standalone/status_mirror' / (sha_text(str(path))[:16] + '.json')
        try:
            replace_json(mirror, dict(original_path=str(path), value=value, error=str(exc)))
        except OSError:
            pass


def sha_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def save_npz(path, **arrays):
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + str(os.getpid()) + '.pending')
    with temp.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def load_training():
    sys.path.insert(0, str(EXEC))
    sys.path.insert(0, str(FIVE / 'code'))
    import torch
    torch.set_num_threads(1)
    import train_core as tc
    import run_fivefold as rf
    import resource_guard as rg
    estimate = read(EXEC / 'PREFLIGHT_E1.json')['estimated_peak_bytes']
    return tc, rf, rg.Guard(estimate)


def validate_result(directory):
    result = read(Path(directory) / 'result.json')
    verify(result['files'])
    if result['contract']['source'] != sha(ROOT / 'SOURCE_MANIFEST.json'):
        raise ValueError('EXPLORATION_RESULT_SOURCE_MISMATCH')
    return result


def train_contract(arm, epochs, seed, count):
    return dict(study='EXPLORE_FAST_20261116', phase='explore', arm=arm,
                epochs=epochs, seed=seed, n_train=count, train_set='C0_full_fit',
                primary='change_coverage', final_weights='EMA_0.995',
                interim_stopping=False, source=sha(ROOT / 'SOURCE_MANIFEST.json'),
                fit_sha256=sha(EXEC / 'C0/fit.npz'), dev_sha256=sha(EXEC / 'C0/dev.npz'))

