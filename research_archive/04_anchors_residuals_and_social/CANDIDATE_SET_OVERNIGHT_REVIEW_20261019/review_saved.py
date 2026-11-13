"""Read-only source-data review; write new review outputs only. No project imports/models."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
EX = ROOT / 'CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018'
CAL = ROOT / 'ORIGINAL_CAL_TRANSFER_EVAL_20261019'
CPV = ROOT / 'COMPLEMENTARY_POOL_VALIDATION_20261018'
SEL = ROOT / 'UNION_SELECTOR_K40_TO_K20_20261018'
used = {}

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

def track(p):
    p = Path(p)
    if p not in used:
        used[p] = sha(p)
    return p

def js(p):
    return json.loads(track(p).read_text(encoding='utf-8-sig'))

def arr(p):
    return np.load(track(p), allow_pickle=False)

def quality(p, gt):
    e = np.linalg.norm(p - gt[:, None], axis=-1)
    a, f = e.mean(-1), e[..., -1]
    j = np.maximum(a / .2, f / .4)
    return dict(covered=(j <= 1).any(1), minADE=a.min(1), minFDE=f.min(1), mean_minJ=j.min(1))

def motion(p):
    v = np.diff(np.concatenate([np.zeros((*p.shape[:2], 1, 2)), p], axis=2), axis=2) / .4
    a = np.diff(v, axis=2) / .4
    j = np.diff(a, axis=2) / .4
    s, a, j = [np.linalg.norm(x, axis=-1) for x in (v, a, j)]
    return dict(speed_p95=float(np.quantile(s, .95)), accel_mean=float(a.mean()),
                accel_p95=float(np.quantile(a, .95)), jerk_mean=float(j.mean()), jerk_p95=float(np.quantile(j, .95)))

def evaluate(p, gt, native, groups):
    q, n = quality(p, gt), quality(native, gt)
    result = {}
    for g, mask in [('overall', np.ones(len(p), bool)), ('stable', groups == 0), ('transition', groups == 1), ('change', groups == 2)]:
        x = dict(N=int(mask.sum()), good_count=int(q['covered'][mask].sum()), native_good_count=int(n['covered'][mask].sum()),
                 rescue=int((q['covered'] & ~n['covered'] & mask).sum()), lost=int((~q['covered'] & n['covered'] & mask).sum()))
        for k in ('minADE', 'minFDE', 'mean_minJ'):
            x[k] = float(q[k][mask].mean())
            x['native_' + k] = float(n[k][mask].mean())
        x['motion'], x['native_motion'] = motion(p[mask]), motion(native[mask])
        x['motion_fails'] = [k for k in x['motion'] if x['motion'][k] > 1.01 * x['native_motion'][k]]
        x['windows_minADE_harmed'] = int(((q['minADE'] > n['minADE'] + 1e-12) & mask).sum())
        result[g] = x
    ch = result['change']
    good = ch['good_count'] >= ch['native_good_count'] + int(np.ceil(.02 * ch['N']))
    good &= ch['minADE'] <= .95 * ch['native_minADE'] and ch['minFDE'] <= .95 * ch['native_minFDE']
    for g, x in result.items():
        good &= not x['motion_fails'] and x['lost'] <= .01 * x['native_good_count'] and x['mean_minJ'] <= x['native_mean_minJ']
        if g != 'change':
            good &= x['good_count'] >= x['native_good_count'] and x['minADE'] <= 1.01 * x['native_minADE'] and x['minFDE'] <= 1.01 * x['native_minFDE']
    return result, bool(good)

with arr(ROOT / 'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz') as z:
    ids = z['sample_id'].copy()
    native_all = z['candidate'][..., :24].reshape(-1, 20, 12, 2).astype(np.float64)
    groups_all = z['change_group'].copy()
with arr(ROOT / 'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/labels/train.npz') as z:
    assert np.array_equal(ids, z['sample_id'])
    gt_all = z['future'].astype(np.float64)
with arr(ROOT / 'GENERATION_GROUPED_TRANSFER_SERIES_20261006/identity_audit/splits.npz') as z:
    assert np.array_equal(ids[:25912], z['sample_id'])
    splits = dict(outer=z['outer_eval_index'].copy(), inner=z['inner_cal_index'].copy(), original_CAL=np.arange(25912, 28791))
    fit = z['grouped_train_index'].copy()
    tracks = np.array([str(s) + '|' + str(int(p)) for s, p in zip(z['sequence'], z['pedestrian_id'])])
rec = js(CPV / 'results/v1/FIT_CAL_SPLIT.json')
u = np.unique(tracks[fit]); np.random.default_rng(rec['seed']).shuffle(u)
caltracks = u[:max(1, round(len(u) * rec['frac']))]
fit_local = np.flatnonzero(np.isin(tracks[fit], caltracks))
splits['FIT_cal'] = fit[fit_local]
assert len(fit_local) == rec['n_windows_cal']
records, max_diff, mismatches = [], 0., []
for seed in (42, 43, 44):
    for split, ix in splits.items():
        if split == 'outer':
            kin = arr(SEL / f'results/cache_pred/kin_outer_seed{seed}.npy').astype(np.float64)
        elif split == 'inner':
            kin = arr(CPV / f'artifacts/v1/kin_inner_seed{seed}.npy').astype(np.float64)
        elif split == 'FIT_cal':
            kin = arr(SEL / f'results/cache_pred/kin_FIT_seed{seed}.npy')[fit_local].astype(np.float64)
        else:
            kin = arr(CPV / f'artifacts/v1/kin_original_CAL_seed{seed}.npy').astype(np.float64)
        nat, gt, groups = native_all[ix], gt_all[ix], groups_all[ix]
        pool = np.concatenate([nat, kin], axis=1)
        if split == 'original_CAL':
            ports = js(CAL / f'results/seed{seed}/done.json')['ports']
            specs = [('K0_C', CAL / f'indices/k0c_seed{seed}_indices.npy', ports['K0_C']), ('S', CAL / f'indices/k1s_seed{seed}_indices.npy', ports['K1_S']), ('V1', CAL / f'indices/v1_seed{seed}_indices.npy', ports['v1_cond_boost_11_9'])]
        else:
            pdir = EX / f'k1_exec/seed{seed}/predictions/{split}'
            ports = js(pdir / 'done.json')['ports']
            specs = [(a, pdir / f'{a}_indices.npy', ports[a]) for a in ('I', 'S')]
            kp = EX / f'results/seed{seed}/{split}'
            specs.append(('K0_C', kp / 'k0c_indices.npy', js(kp / 'portfolios.json')['k0c']))
            if split == 'outer':
                assert np.array_equal(arr(pdir / 'sample_id.npy'), ids[ix])
                assert np.array_equal(arr(pdir / 'groups.npy'), groups)
        for arm, ip, stored in specs:
            idx = arr(ip).astype(np.int64)  # Recorded CAL V1 hash is explicitly over evaluation int64, not storage int16.
            assert idx.shape == (len(ix), 20) and (idx >= 0).all() and (idx < 40).all()
            assert all(len(np.unique(row)) == 20 for row in idx)
            if 'indices_sha' in stored:
                h = hashlib.sha256(); h.update(np.ascontiguousarray(idx).tobytes()); h.update(str(idx.shape).encode()); h.update(str(idx.dtype).encode())
                assert h.hexdigest() == stored['indices_sha']
            p = pool[np.arange(len(ix))[:, None], idx]
            ev, W = evaluate(p, gt, nat, groups)
            for g, x in ev.items():
                old = stored['eval'][g]
                for k in ('N', 'good_count', 'native_good_count', 'rescue', 'lost', 'minADE', 'minFDE', 'mean_minJ'):
                    if k not in old:
                        continue  # CAL compact JSON omits counts; recomputed counts remain in this review.
                    diff = abs(x[k] - old[k]); max_diff = max(max_diff, diff)
                    if diff > 1e-12: mismatches.append([seed, split, arm, g, k, diff])
                for k, v in x['motion'].items():
                    actual = v if 'motion' in old else v / x['native_motion'][k]
                    saved = old['motion'][k] if 'motion' in old else old['motion_gate']['ratios'][k]
                    diff = abs(actual - saved); max_diff = max(max_diff, diff)
                    if diff > 1e-12: mismatches.append([seed, split, arm, g, k, diff])
            if W != stored['W_pass']: mismatches.append([seed, split, arm, 'W', W, stored['W_pass']])
            records.append(dict(seed=seed, split=split, arm=arm, W_literal=W, swaps_mean=float((idx >= 20).sum(1).mean()), groups=ev))
        print(seed, split, 'reviewed', flush=True)
unchanged = all(sha(p) == h for p, h in used.items())
out = dict(scope='36 saved K20 selections; 144 group metrics; independent quality/motion/literal W; no model execution or CI/cluster replay',
           max_numeric_diff=max_diff, mismatches=mismatches, inputs_unchanged=unchanged, input_count=len(used), records=records,
           input_sha={str(p): h for p, h in used.items()})
(OUT / 'SAVED_REVIEW.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({k:v for k,v in out.items() if k not in ('records','input_sha')}, ensure_ascii=False), flush=True)
