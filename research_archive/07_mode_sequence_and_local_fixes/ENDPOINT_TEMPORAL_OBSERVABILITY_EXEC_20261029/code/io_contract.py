"""Explicit inputs, immutable writes, hashes. Imports have no write side effects."""
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def code_digest():
    root = Path(__file__).resolve().parent
    files = {p.name: sha(p) for p in sorted(root.glob('*.py'))}
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def json_read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def json_new(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation: failure evidence or an existing run is never overwritten.
    with p.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)


def npz_new(path, **data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('xb') as f:
        np.savez_compressed(f, **data)


def load_npz(path, keys=None):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in (z.files if keys is None else keys)}


def validate_observed(x):
    for key in ('sample_id', 'past', 'native', 'track_key', 'group'):
        if key not in x:
            raise ValueError('missing observed field: '+key)
    n = len(x['sample_id'])
    if n == 0 or len(np.unique(x['sample_id'])) != n:
        raise ValueError('empty/duplicate sample IDs')
    if x['past'].shape != (n, 8, 2) or x['native'].shape != (n, 20, 12, 2):
        raise ValueError('observed shape mismatch')
    if x['group'].shape != (n,) or not np.isin(x['group'], [0, 1, 2]).all():
        raise ValueError('group must be original stable=0/transition=1/change=2')
    if x['track_key'].shape != (n,) or x['track_key'].dtype.kind not in 'US':
        raise ValueError('explicit sequence|target string track_key required')
    if not np.isfinite(x['past']).all() or not np.isfinite(x['native']).all():
        raise ValueError('nonfinite input')
    if not np.all(x['past'][:, -1] == 0):
        raise ValueError('history origin is not exact zero')
    return n


def aligned_gt(x, y):
    if not np.array_equal(x['sample_id'], y['sample_id']):
        raise ValueError('GT sample ID/order mismatch; no implicit reordering')
    gt = np.asarray(y['gt'], dtype=np.float64)
    if gt.shape != (len(x['sample_id']), 12, 2) or not np.isfinite(gt).all():
        raise ValueError('invalid GT')
    return gt


def assert_disjoint(a, b):
    if np.intersect1d(a['sample_id'], b['sample_id']).size:
        raise ValueError('sample overlap')
    if np.intersect1d(a['track_key'], b['track_key']).size:
        raise ValueError('target-track overlap')


def verify_manifest(path):
    p = Path(path).resolve()
    m = json_read(p)
    for rel, expected in m['files'].items():
        target = Path(rel) if Path(rel).is_absolute() else p.parent / rel
        if sha(target) != expected:
            raise ValueError('source/output SHA changed: '+str(target))
    return m


def seal(folder, files, **metadata):
    root = Path(folder).resolve()
    listing = {str(Path(p).resolve().relative_to(root)): sha(p) for p in files}
    json_new(root/'done.json', dict(files=listing, code_digest=code_digest(), **metadata))


def checked_seal(folder):
    if (Path(folder)/'FAILURE.json').exists():
        raise ValueError('failed stage cannot be reused merely because done.json exists')
    terminal = verify_manifest(Path(folder)/'STAGE_COMPLETE.json')
    if (terminal.get('status') != 'STAGE_OUTPUTS_COMPLETE_NOT_SCIENTIFIC_PASS' or
            terminal.get('code_digest') != code_digest() or
            terminal['files'].get('done.json') != sha(Path(folder)/'done.json')):
        raise ValueError('missing/mismatched terminal stage seal')
    m = verify_manifest(Path(folder)/'done.json')
    if m['code_digest'] != code_digest():
        raise ValueError('code digest mismatch; version recovery needs a new namespace')
    return m


def finish_stage(folder, resource_file):
    """Called only AFTER final source/resource checks; killed partial stages cannot skip."""
    root = Path(folder)
    verify_manifest(root/'done.json')
    json_new(root/'STAGE_COMPLETE.json', dict(
        status='STAGE_OUTPUTS_COMPLETE_NOT_SCIENTIFIC_PASS', code_digest=code_digest(),
        files={'done.json': sha(root/'done.json'), resource_file: sha(root/resource_file)}))
