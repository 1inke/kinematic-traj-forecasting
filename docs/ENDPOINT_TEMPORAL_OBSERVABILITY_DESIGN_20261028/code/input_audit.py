"""Prepare canonical train/val from the EXISTING split, never redraw it."""
from pathlib import Path
import numpy as np
from io_contract import (load_npz, validate_observed, aligned_gt, assert_disjoint,
                         npz_new, json_new, sha, seal)
from features import fit_statistics


def prepare(opt_observed, opt_gt, split_path, out):
    x, y, split = map(load_npz, (opt_observed, opt_gt, split_path))
    n = validate_observed(x)
    gt = aligned_gt(x, y)
    if n != 4096 or not np.array_equal(split['sample_id'], x['sample_id']):
        raise ValueError('not the registered opt population/split order')
    if not np.array_equal(split['track_key'], x['track_key']):
        raise ValueError('registered split track mismatch')
    train, val = split['train'], split['val']
    if train.dtype.kind not in 'iu' or val.dtype.kind not in 'iu':
        raise ValueError('split indices must be integers')
    if len(train) != 3103 or len(val) != 993 or not np.array_equal(np.sort(np.r_[train, val]), np.arange(n)):
        raise ValueError('split not an exact registered partition')
    keys = ('sample_id', 'past', 'native', 'track_key', 'group', 'scene', 'sequence')
    subsets = [{k: x[k][idx] for k in keys if k in x} for idx in (train, val)]
    assert_disjoint(*subsets)
    if [int((s['group'] == 2).sum()) for s in subsets] != [721, 256]:
        raise ValueError('registered change counts mismatch')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    files = []
    for name, subset, idx in zip(('train', 'val'), subsets, (train, val)):
        npz_new(out/f'X_{name}.npz', **subset)
        npz_new(out/f'GT_{name}.npz', sample_id=subset['sample_id'], gt=gt[idx])
        files.extend((out/f'X_{name}.npz', out/f'GT_{name}.npz'))
    stats = fit_statistics(subsets[0]['past'], subsets[0]['native'], subsets[0]['sample_id'])
    npz_new(out/'statistics.npz', **stats)
    files.append(out/'statistics.npz')
    json_new(out/'INPUT_AUDIT.json', dict(status='PASS_REGISTERED_TRAIN_VAL_INPUTS',
        original_inputs={str(Path(p).resolve()): sha(p) for p in (opt_observed, opt_gt, split_path)},
        split_sha=sha(split_path),
        counts=[3103, 993], change=[721, 256], no_held_GT_access=True,
        limitation='Full parent source lock verification remains a production preflight requirement.'))
    files.append(out/'INPUT_AUDIT.json')
    seal(out, files, stage='TRAIN_VAL_PREPARED', no_training=True)
