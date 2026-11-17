"""Self-tests (NumPy only). Run:  python test_kine_oracle.py"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np

import kine_common as kc
import kine_repr as kr
from kine_decide import decide


def straight(n, speed, heading, rng):
    t = np.arange(-7, 13)[None, :, None] * kc.DT
    d = np.stack([np.cos(heading), np.sin(heading)], -1)[:, None]
    xy = speed[:, None, None] * t * d + rng.uniform(-3, 3, (n, 1, 2))
    past = np.zeros((n, 9, 8, 2)); past[:, 0] = xy[:, :8]
    return past, xy[:, 8:]


def test_focal_frame():
    rng = np.random.default_rng(0)
    past, fut = straight(50, rng.uniform(.5, 2, 50), rng.uniform(-np.pi, np.pi, 50), rng)
    fl, vr, ok = kc.focal_frame(past, fut)
    assert ok.all() and np.allclose(fl[..., 1], 0, atol=1e-9) and (fl[..., 0] > 0).all()
    assert np.allclose(vr, np.linalg.norm(fut[:, 0] - past[:, 0, -1], axis=-1) / kc.DT)
    still = np.zeros((1, 9, 8, 2)); fl, vr, ok = kc.focal_frame(still, np.ones((1, 12, 2)))
    assert not ok[0] and np.allclose(fl[0], 1)          # default heading (1,0): identity rotation


def test_interpolation():
    w = kr.W_INTERP
    assert np.allclose(w[3], [1, 0, 0]) and np.allclose(w[7], [0, 1, 0]) and np.allclose(w[11], [0, 0, 1])
    assert np.allclose(w[0], [.25, 0, 0]) and np.allclose(w.sum(1)[4:], 1)


def test_maneuver_roundtrip():
    rng = np.random.default_rng(1)
    kp = np.cumsum(rng.normal([1.2, 0], .3, (200, 3, 2)), axis=1)
    ve = rng.uniform(.3, 2, 200)
    feat = kr.to_maneuver(kp, ve)
    rec = np.stack([kr.from_maneuver(feat[i:i + 1], ve[i:i + 1])[0, 0] for i in range(200)])
    assert np.allclose(rec, kp, atol=1e-9)


def test_speed_equivariance():
    rng = np.random.default_rng(2)
    templ = {f: rng.normal(0, 1, (20, 6)) for f in ('P', 'S', 'K')}
    ve = rng.uniform(.4, 1.0, 30)
    c = 1.7
    for f in ('S', 'K'):
        assert np.allclose(kr.candidates(f, templ[f], ve * c), c * kr.candidates(f, templ[f], ve))
    assert np.allclose(kr.candidates('P', templ['P'], ve * c), kr.candidates('P', templ['P'], ve))
    assert np.allclose(kr.v_eff([.1, .5]), [kr.V0, .5])


def test_kmeans():
    rng = np.random.default_rng(3)
    centers = rng.uniform(-50, 50, (20, 6))
    x = np.repeat(centers, 30, 0) + rng.normal(0, .1, (600, 6))
    c1, c2 = kr.kmeans(x, seed=5), kr.kmeans(x, seed=5)
    assert np.array_equal(c1, c2)
    d = np.sqrt(((centers[:, None] - c1[None]) ** 2).sum(-1)).min(1)
    assert (d < 1).sum() >= 18                          # k-means++ recovers almost all clusters


def test_best_of():
    fut = np.cumsum(np.ones((4, 12, 2)), 1)
    cand = np.broadcast_to(kr.keypoints(fut)[:, None], (4, 20, 3, 2)).copy()
    cand[:, 1:] += 5
    a, f = kr.best_of(cand, fut)
    assert np.allclose(a, 0) and np.allclose(f, 0)


def fake_summary(k_gain_eth, k_gain_other=0.0, dec_gain=0.02):
    folds = {}
    for fold in kc.FOLDS:
        g = k_gain_eth if fold == 'eth' else k_gain_other
        arms = {'P': dict(minADE=[.5] * 3, minFDE=[1.] * 3),
                'P_AUG': dict(minADE=[.5] * 3, minFDE=[.99] * 3),
                'S': dict(minADE=[.5] * 3, minFDE=[1.] * 3),
                'K': dict(minADE=[.5 * (1 - g)] * 3, minFDE=[1. * (1 - g)] * 3)}
        folds[fold] = dict(val=arms, test=arms)
    pooled = {a: dict(decel_stop=dict(minADE=[.5] * 3, minFDE=[1.] * 3)) for a in ('P', 'P_AUG', 'S')}
    pooled['K'] = dict(decel_stop=dict(minADE=[.5] * 3, minFDE=[1 - dec_gain] * 3))
    return dict(folds=folds, pooled_test=pooled)


def test_decide():
    assert decide(fake_summary(.08))['verdict'] == 'SUPPORT'
    assert decide(fake_summary(.03))['verdict'] == 'WEAK'
    assert decide(fake_summary(.08, k_gain_other=-.05))['verdict'] == 'NOT_SUPPORTED'
    assert decide(fake_summary(0.0))['verdict'] == 'NOT_SUPPORTED'
    att = decide(fake_summary(.08))['attribution']['K_vs_S']
    assert att['label'] == 'K_BETTER'


def test_smoke_end_to_end():
    import run_kine_oracle as run
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / 'smoke'
        result = run.main(['--smoke', '--out', str(out), '--seeds', '0', '1'])
        assert result['decision']['verdict'] in ('SUPPORT', 'WEAK', 'NOT_SUPPORTED')
        saved = json.loads((out / 'RESULT.json').read_text(encoding='utf-8'))
        assert saved['training_started'] is False and (out / 'REPORT.md').exists()
        assert set(saved['folds']) == set(kc.FOLDS)
        try:
            run.main(['--smoke', '--out', str(out)])
        except FileExistsError:
            pass
        else:
            raise AssertionError('output directory must be write-once')
        print('smoke verdict:', result['decision']['verdict'], 'eth gFDE',
              round(result['decision']['numbers']['eth_test_g_FDE'], 3))


def test_reference_predictions():
    import run_kine_oracle as run
    with tempfile.TemporaryDirectory() as tmp:
        data = run.make_smoke(Path(tmp) / 'data')
        test = kc.load_split(data, 'eth', 'test')
        paths = np.repeat(test['future'][:, None], 20, 1).astype(np.float32)
        paths[:, 1:] += 1.0
        order = np.random.default_rng(0).permutation(len(paths))      # saved order may differ
        np.savez(Path(tmp) / 'pred.npz', ids=test['ids'][order], paths=paths[order], future=test['future'][order])
        r = run.main(['--data-dir', str(data), '--out', str(Path(tmp) / 'o'), '--seeds', '0',
                      '--reference', f"eth:PERFECT:{Path(tmp) / 'pred.npz'}"])
        ref = r['folds']['eth']['splits']['test']['REF_PERFECT'][0]['overall']
        assert abs(ref['minADE']) < 1e-6 and abs(ref['minFDE']) < 1e-6


def test_smoke_negative_control():
    import run_kine_oracle as run
    with tempfile.TemporaryDirectory() as tmp:
        result = run.main(['--smoke-negative', '--out', str(Path(tmp) / 'neg'), '--seeds', '0', '1'])
        assert result['decision']['verdict'] == 'NOT_SUPPORTED', result['decision']['verdict']
        print('negative control verdict:', result['decision']['verdict'])


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn(); print('ok', name)
    print('ALL TESTS PASSED')
