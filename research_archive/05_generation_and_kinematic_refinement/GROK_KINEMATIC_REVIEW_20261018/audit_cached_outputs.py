"""Finite saved-array review; no legacy imports, Torch, model forward or training."""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SEL = ROOT / 'UNION_SELECTOR_K40_TO_K20_20261018'
CACHE = ROOT / 'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz'
LABELS = ROOT / 'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/labels/train.npz'
SPLITS = ROOT / 'GENERATION_GROUPED_TRANSFER_SERIES_20261006/identity_audit/splits.npz'
KEYS = ('speed_p95', 'accel_mean', 'accel_p95', 'jerk_mean', 'jerk_p95')

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def motion(p):
    start = np.zeros((*p.shape[:2], 1, 2), dtype=np.float64)
    pos = np.concatenate((start, p.astype(np.float64)), axis=2)
    vel = (pos[:, :, 1:] - pos[:, :, :-1]) / .4
    acc = (vel[:, :, 1:] - vel[:, :, :-1]) / .4
    jerk = (acc[:, :, 1:] - acc[:, :, :-1]) / .4
    s, a, j = [np.sqrt(np.sum(x*x, axis=-1)) for x in (vel, acc, jerk)]
    vals = (np.quantile(s, .95), a.mean(), np.quantile(a, .95), j.mean(), np.quantile(j, .95))
    return dict(zip(KEYS, map(float, vals)))

def per_window(p, gt):
    e = np.sqrt(np.sum((p.astype(np.float64) - gt[:, None])**2, axis=-1))
    ade, fde = e.mean(axis=2), e[:, :, -1]
    good = (ade <= .2) & (fde <= .4)
    distinct, effective = [], []
    i, k = np.triu_indices(p.shape[1], 1)
    for row, g in zip(p, good):
        d = np.sqrt(np.sum((row[i] - row[k])**2, axis=-1))
        condensed = np.maximum(d.mean(axis=1)/.10, d[:, -1]/.20)
        lab = fcluster(linkage(condensed, method='complete'), 1., criterion='distance')
        distinct.append(len(np.unique(lab)))
        effective.append(len(np.unique(lab[g])))
    return good.any(axis=1), ade.min(axis=1), fde.min(axis=1), np.array(distinct), np.array(effective)

def main():
    sources = [CACHE, LABELS, SPLITS, Path(__file__).resolve()]
    sources += [SEL / f'results/cache_pred/kin_outer_seed{s}.npy' for s in (42,43,44)]
    sources += [SEL / f'results/per_seed/seed{s}_outer.json' for s in (42,43,44)]
    before = {str(p): sha(p) for p in sources}
    with np.load(SPLITS, allow_pickle=False) as z:
        outer = z['outer_eval_index'].copy()
        fit_ids = z['sample_id'].copy()
    with np.load(CACHE, allow_pickle=False) as z:
        ids = z['sample_id'].copy()
        assert np.array_equal(fit_ids, ids[:25912])
        n = z['candidate'][outer, :, :24].reshape(-1,20,12,2).astype(np.float64)
        groups = z['change_group'][outer].copy()
    with np.load(LABELS, allow_pickle=False) as z:
        assert np.array_equal(ids, z['sample_id'])
        gt = z['future'][outer].astype(np.float64)
    assert len(outer) == 1035 and len(np.unique(ids[outer])) == 1035
    assert np.isfinite(n).all() and np.isfinite(gt).all()
    base = per_window(n, gt)
    masks = {'overall': np.ones(len(outer), dtype=bool)}
    masks.update({name: groups == g for g, name in enumerate(('stable','transition','change'))})
    rows, comparisons = [], []
    for seed in (42,43,44):
        kin = np.load(SEL / f'results/cache_pred/kin_outer_seed{seed}.npy', allow_pickle=False).astype(np.float64)
        assert kin.shape == n.shape and np.isfinite(kin).all()
        reported = json.loads((SEL / f'results/per_seed/seed{seed}_outer.json').read_text(encoding='utf-8-sig'))
        for port, paths in [('native', n), ('kin', kin), ('union40', np.concatenate((n,kin), axis=1))]:
            stats = base if port == 'native' else per_window(paths, gt)
            for group, mask in masks.items():
                good, ade, fde, distinct, effective = stats
                nm, pm = motion(n[mask]), motion(paths[mask])
                fails = [k for k in KEYS if pm[k] > 1.01 * nm[k]]
                row = dict(seed=seed, port=port, group=group, N=int(mask.sum()),
                    good=int(good[mask].sum()), native_good=int(base[0][mask].sum()),
                    rescue=int((good & ~base[0] & mask).sum()), lost=int((~good & base[0] & mask).sum()),
                    minADE=float(ade[mask].mean()), minFDE=float(fde[mask].mean()),
                    distinct=float(distinct[mask].mean()), effective_good=float(effective[mask].mean()),
                    motion=pm, native_motion=nm, motion_fails=fails,
                    ratios={k: pm[k]/nm[k] if nm[k] else None for k in KEYS})
                rows.append(row)
                saved = reported['methods'][port]
                expected_cov = saved['coverage'] if group == 'overall' else saved['by_group'][group]['coverage']
                assert abs(row['good']/row['N'] - expected_cov) < 1e-14
                if group == 'overall':
                    for key in ('minADE','minFDE'):
                        comparisons.append(abs(row[key] - saved[key]))
                    assert row['rescue'] == saved['rescue'] and row['lost'] == saved['lost']
        sm = reported['methods']['cond_boost_11_9']
        comparisons.append(abs(sm['coverage']*1035 - (625+sm['rescue']-sm['lost'])))
    after = {str(p): sha(p) for p in sources}
    assert before == after
    result = dict(status='PASS_SAVED_CACHE_REVIEW_ONLY', no_torch_no_training=True,
        provenance_limit='KIN npy has no embedded IDs/source checkpoint binding; assumed loader order, report values match, historical generation provenance not certified.',
        selector_limit='No selected indices, fitted HGB or scores found in results; selector paths/group motion/CI not replayed.',
        rows=rows, max_report_numeric_difference=max(comparisons), sha_before=before, sha_after=after)
    (HERE/'SAVED_CACHE_REVIEW.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k,v in result.items() if k not in ('rows','sha_before','sha_after')}, ensure_ascii=False))
    for r in rows:
        if r['port'] != 'native':
            print(r['seed'], r['port'], r['group'], r['good'], r['rescue'], r['lost'], r['motion_fails'], r['ratios'])

if __name__ == '__main__':
    main()
