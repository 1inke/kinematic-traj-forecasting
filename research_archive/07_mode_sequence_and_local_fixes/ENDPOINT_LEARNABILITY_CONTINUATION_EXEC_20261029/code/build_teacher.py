"""P0: seal GT-free masks first, then associate GT in a separate invocation."""
from pathlib import Path
import numpy as np
from scipy.spatial.distance import pdist
from action_geometry import space, enumerate_actions, choose
from motion_guard import inspect
from io_contract import (load_npz, validate_observed, aligned_gt, npz_new,
                         json_new, sha, seal, checked_seal)


def samepath_quality(paths, gt):
    d = np.linalg.norm(np.asarray(paths, dtype=np.float64)-gt, axis=-1)
    ade, fde = d.mean(-1), d[..., -1]
    return ade, fde, np.maximum(ade/.2, fde/.4), (ade <= .2) & (fde <= .4)


def masks_for_window(past, native, actions):
    legal, safe, failures = [], [], []
    for p in native:
        paths, geometry = enumerate_actions(p, actions)
        accepted, failed = inspect(past, p, paths)
        legal.append(geometry)
        safe.append(geometry & accepted)
        failures.append(failed)
    return dict(legal=np.array(legal), safe=np.array(safe), failed=np.array(failures))


def teacher_for_window(native, gt, actions, masks):
    n, m = len(native), len(actions.params)
    js = np.empty((n, m), np.float64)
    result = dict(action_id=np.empty(n, np.int32), target=np.empty((n, 2)),
                  best_J=np.empty(n), identity_J=np.empty(n),
                  unfiltered_J=np.empty(n), exact_ties=np.empty(n, np.int32),
                  near_ties=np.empty(n, np.int32), second_gap=np.empty(n),
                  diameter=np.empty(n), safe_count=masks['safe'].sum(-1))
    for k, path in enumerate(native):
        proposals, legal = enumerate_actions(path, actions)
        if not np.array_equal(legal, masks['legal'][k]):
            raise ValueError('sealed geometry mismatch')
        js[k] = samepath_quality(proposals, gt)[2]
        safe = masks['safe'][k]
        if not safe[actions.identity] or np.any(safe & ~legal):
            raise ValueError('invalid safe mask')
        j = choose(js[k], safe, actions)
        best = js[k, j]
        vals = np.sort(js[k, safe])
        equivalent = actions.normalized[safe & (js[k] <= best+.01)]
        result['action_id'][k], result['target'][k] = j, actions.normalized[j]
        result['best_J'][k], result['identity_J'][k] = best, js[k, actions.identity]
        result['unfiltered_J'][k] = js[k, legal].min()
        result['exact_ties'][k] = np.count_nonzero(safe & (js[k] == best))
        result['near_ties'][k] = np.count_nonzero(safe & (js[k] <= best+1e-6))
        result['second_gap'][k] = vals[1]-vals[0] if len(vals) > 1 else np.nan
        result['diameter'][k] = pdist(equivalent).max() if len(equivalent) > 1 else 0.
    result['J_matrix'] = js  # Invalid geometry rows are Native and MUST use legal mask.
    return result


def write_masks(observed_path, out, branch, check=lambda: None):
    x = load_npz(observed_path)
    n = validate_observed(x)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    actions = space(branch)
    npz_new(out/'actions.npz', params=actions.params, normalized=actions.normalized,
            identity=np.array(actions.identity), priority=actions.priority)
    files = [out/'actions.npz']
    for start in range(0, n, 32):
        rows = []
        for i in range(start, min(start+32, n)):
            check()
            rows.append(masks_for_window(x['past'][i], x['native'][i], actions))
        name = out/f'{start:06d}.npz'
        npz_new(name, sample_id=x['sample_id'][start:start+len(rows)],
                **{k: np.stack([r[k] for r in rows]) for k in rows[0]})
        files.append(name)
    seal(out, files, stage='GT_FREE_MASKS_SEALED', branch=branch,
         observed_sha=sha(observed_path), windows=n, gt_access=False)


def write_labels(observed_path, gt_path, mask_dir, out, check=lambda: None):
    receipt = checked_seal(mask_dir)
    if receipt['stage'] != 'GT_FREE_MASKS_SEALED' or receipt['observed_sha'] != sha(observed_path):
        raise ValueError('mask/input seal mismatch')
    x = load_npz(observed_path)
    n = validate_observed(x)
    gt = aligned_gt(x, load_npz(gt_path))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    actions = space(receipt['branch'])
    files, offset, all_best, all_base, all_raw = [], 0, [], [], []
    for file in sorted(Path(mask_dir).glob('[0-9]*.npz')):
        masks = load_npz(file)
        count = len(masks['sample_id'])
        if not np.array_equal(masks['sample_id'], x['sample_id'][offset:offset+count]):
            raise ValueError('mask shard ID mismatch')
        rows = []
        for local in range(count):
            check()
            i = offset+local
            rows.append(teacher_for_window(x['native'][i], gt[i], actions,
                        {k: masks[k][local] for k in ('legal', 'safe')}))
        z = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
        name = out/file.name
        npz_new(name, sample_id=masks['sample_id'], **z)
        files.append(name)
        all_best.extend(z['best_J'].min(1))
        all_base.extend(z['identity_J'].min(1))
        all_raw.extend(z['unfiltered_J'].min(1))
        offset += count
    if offset != n:
        raise ValueError('missing/duplicate mask shards')
    base, best, raw = (np.asarray(v) <= 1 for v in (all_base, all_best, all_raw))
    if np.any(base & ~best):
        raise AssertionError('safe J oracle lost an original success')
    change = x['group'] == 2
    rescue = change & ~base & best
    tracks = len(np.unique(x['track_key'][rescue]))
    threshold = int(np.ceil(.02*change.sum()))
    json_new(out/'opportunity.json', dict(
        change_N=int(change.sum()), change_native_good=int((change & base).sum()),
        change_safe_good=int((change & best).sum()), change_rescue=int(rescue.sum()),
        change_unfiltered_good=int((change & raw).sum()), rescue_tracks=tracks,
        required_rescue=threshold, opportunity_pass=bool(change.any() and
            rescue.sum() >= threshold and tracks >= 5),
        note='Only train population can authorize a training stage; oracle zero-lost is structural.'))
    files.append(out/'opportunity.json')
    seal(out, files, stage='TEACHERS_SEALED', branch=receipt['branch'],
         observed_sha=sha(observed_path), gt_sha=sha(gt_path), windows=n,
         mask_done_sha=sha(Path(mask_dir)/'done.json'))


def read_targets(folder, sample_ids):
    receipt = checked_seal(folder)
    if receipt['stage'] != 'TEACHERS_SEALED':
        raise ValueError('not a teacher seal')
    chunks = [load_npz(p, ('sample_id', 'target'))
              for p in sorted(Path(folder).glob('[0-9]*.npz'))]
    if not chunks or not np.array_equal(np.concatenate([z['sample_id'] for z in chunks]), sample_ids):
        raise ValueError('teacher IDs/order mismatch')
    return np.concatenate([z['target'] for z in chunks]).astype(np.float32)
