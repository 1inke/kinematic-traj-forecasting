"""Read-only source review: nine saved V1 bundles, no training or model forward.

Writes only this review's JSON; does not rerun CI or full clustering/W0 arithmetic.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
V = ROOT / 'COMPLEMENTARY_POOL_VALIDATION_20261018'
tracked = {}

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

def take(p):
    p = Path(p)
    tracked[str(p)] = sha(p)
    return p

def quality(p, gt):
    e = np.linalg.norm(p.astype('float64') - gt[:, None], axis=-1)
    a, f = e.mean(-1), e[..., -1]
    return ((a <= .2) & (f <= .4)).any(1), a.min(1), f.min(1), np.maximum(a/.2, f/.4).min(1)

def motion(p):
    p = p.astype('float64')
    z = np.zeros((*p.shape[:2], 1, 2))
    v = np.diff(np.concatenate([z, p], axis=2), axis=2)/.4
    a = np.diff(v, axis=2)/.4
    j = np.diff(a, axis=2)/.4
    v, a, j = [np.linalg.norm(x, axis=-1) for x in (v, a, j)]
    return dict(speed_p95=float(np.quantile(v,.95)), accel_mean=float(a.mean()),
                accel_p95=float(np.quantile(a,.95)), jerk_mean=float(j.mean()), jerk_p95=float(np.quantile(j,.95)))

tracked[str(Path(__file__).resolve())] = sha(Path(__file__).resolve())
with np.load(take(ROOT/'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz'), allow_pickle=False) as z:
    ids = z['sample_id'].copy()
    native = z['candidate'][..., :24].reshape(-1,20,12,2).copy()
    groups = z['change_group'].copy()
with np.load(take(ROOT/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/labels/train.npz'), allow_pickle=False) as z:
    assert np.array_equal(ids, z['sample_id'])
    gt = z['future'].astype('float64')
with np.load(take(ROOT/'GENERATION_GROUPED_TRANSFER_SERIES_20261006/identity_audit/splits.npz'), allow_pickle=False) as z:
    fit = z['grouped_train_index'].copy()
idrow = {int(k): i for i,k in enumerate(ids)}
fitrow = {int(k): i for i,k in enumerate(fit)}
rows = []
maxerr = 0.
for split in ['outer','inner','FIT_cal']:
    for seed in [42,43,44]:
        report = json.loads(take(V/f'results/v1/seed{seed}_{split}.json').read_text(encoding='utf-8-sig'))
        with np.load(take(V/f'artifacts/v1/{split}_seed{seed}_bundle.npz'), allow_pickle=False) as z:
            ix = np.array([idrow[int(k)] for k in z['sample_id']])
            assert len(np.unique(ix)) == len(ix)
            assert np.array_equal(groups[ix], z['groups'])
            pred = z['selected_paths'].astype('float64')
            pick = z['selected_pool_indices'].copy()
        kin = np.load(take(Path(report['kin_source'])), allow_pickle=False)
        if split == 'FIT_cal':
            kin = kin[[fitrow[int(i)] for i in ix]]
        pool = np.concatenate([native[ix], kin], axis=1)
        assert np.all((pick>=0)&(pick<40))
        assert all(len(np.unique(r)) == 20 for r in pick)
        assert np.array_equal(pred, pool[np.arange(len(ix))[:,None], pick].astype('float64'))
        pn, pa, pf, pj = quality(pred, gt[ix])
        nn, na, nf, nj = quality(native[ix], gt[ix])
        for gi,g in enumerate(['overall','stable','transition','change']):
            mask = np.ones(len(ix),bool) if gi==0 else groups[ix]==gi-1
            pm,nm = motion(pred[mask]),motion(native[ix][mask])
            expected = report['portfolios']['cond_boost_11_9'][g]
            vals = dict(good_count=int(pn[mask].sum()), native_good_count=int(nn[mask].sum()),
                        rescue=int((pn&~nn&mask).sum()),lost=int((~pn&nn&mask).sum()),
                        minADE=float(pa[mask].mean()),minFDE=float(pf[mask].mean()),mean_minJ=float(pj[mask].mean()))
            for k,v in vals.items(): maxerr=max(maxerr,abs(v-expected[k]))
            for k in pm:
                maxerr=max(maxerr,abs(pm[k]-expected['motion'][k]),abs(nm[k]-expected['native_motion'][k]))
            fails=[k for k in pm if pm[k]>1.01*nm[k]]
            assert fails==expected['motion_gate']['fails']
            rows.append(dict(split=split,seed=seed,group=g,**vals,
                             literal_motion_fails=fails,ratios={k:pm[k]/nm[k] for k in pm}))
        print(split,seed,'IDs + exact pool reconstruction + quality + literal motion checked',flush=True)
assert maxerr<1e-12
assert all(sha(Path(p))==h for p,h in tracked.items())
out=dict(status='PASS_LIMITED_SAVED_RECOMPUTE',bundles=9,group_rows=len(rows),max_metric_abs_error=maxerr,
         no_model_forward_or_training=True,scope='Pool reconstruction, same-path coverage, minADE/minFDE/J, rescue/lost and literal five-motion; no CI, clustering or W0 full arithmetic replay.',
         source_files_unchanged=True,source_sha256=tracked,rows=rows)
(HERE/'NEW_V1_SAVED_REVIEW.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print('PASS',len(rows),'rows; max error',maxerr,'; unchanged',len(tracked),'files')
