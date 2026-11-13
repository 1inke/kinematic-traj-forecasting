"""Synthetic CPU unit tests only. No project datasets, old models or workers."""
import io
import unittest
import numpy as np
from action_geometry import space, legal_actions, apply_action, enumerate_actions, choose, decode
from motion_guard import inspect, candidate_metrics
from build_teacher import masks_for_window, teacher_for_window, samepath_quality
from features import fit_statistics, transform
from inference import predict_paths, combine
from evaluate import sample_metrics, evaluate_population
from bootstrap import make_draws, paired_ci
from audit_saved import scalar_apply, scalar_metrics, check_saved
from io_contract import validate_observed, assert_disjoint, aligned_gt


def fixture(n=3):
    rng = np.random.default_rng(54)
    past = np.zeros((n, 8, 2))
    past[:, :, 0] = np.arange(-7, 1)*.12
    paths = []
    for _ in range(n):
        velocities = rng.uniform(.04, .18, (20, 12, 2))
        velocities[..., 1] -= .10
        paths.append(np.cumsum(velocities, axis=1))
    native = np.array(paths)
    return dict(sample_id=np.arange(n), past=past, native=native,
                track_key=np.array(['s|'+str(i//2) for i in range(n)]),
                group=np.arange(n) % 3)


class GeometryTests(unittest.TestCase):
    def test_counts_bounds_and_identity(self):
        e, t = space('END'), space('TIME')
        self.assertEqual(len(e.params), 49)
        self.assertEqual(len(t.params), 441)
        self.assertTrue(np.all(np.linalg.norm(e.params, axis=1) <= .4))
        p = fixture(1)['native'][0, 0]
        for a in (e, t):
            np.testing.assert_array_equal(apply_action(p, a, a.identity), p)
            self.assertEqual(decode(p, [0., 0.], a), a.identity)

    def test_all_grid_against_scalar_reference(self):
        p = fixture(1)['native'][0, 0]
        for branch in ('END', 'TIME'):
            a = space(branch)
            paths, valid = enumerate_actions(p, a)
            for k in np.flatnonzero(valid):
                np.testing.assert_allclose(paths[k], scalar_apply(p, branch, k), rtol=0, atol=2e-12)
                if branch == 'TIME':
                    np.testing.assert_array_equal(paths[k, -1], p[-1])

    def test_stationary_and_zero_segment_time(self):
        a = space('TIME')
        p = np.zeros((12, 2))
        self.assertEqual(legal_actions(p, a).sum(), 1)
        p = np.c_[np.arange(1, 13)*.1, np.zeros(12)]
        p[4] = p[3]
        self.assertEqual(legal_actions(p, a).sum(), 1)
        self.assertEqual(decode(p, [1, -1], a), a.identity)

    def test_invalid_actions_and_nan_fail(self):
        a = space('TIME')
        p = np.zeros((12, 2))
        with self.assertRaises(ValueError):
            apply_action(p, a, 0)
        with self.assertRaises(ValueError):
            decode(p, [np.nan, 0], a)
        with self.assertRaises(ValueError):
            apply_action(p, a, -1)

    def test_ties_identity_then_norm_then_index(self):
        a = space('END')
        values = np.zeros(49)
        mask = np.ones(49, bool)
        self.assertEqual(choose(values, mask, a), a.identity)
        mask[a.identity] = False
        want = min(np.flatnonzero(mask), key=lambda k: ((a.normalized[k]**2).sum(), k))
        self.assertEqual(choose(values, mask, a), want)


class SafetyTeacherTests(unittest.TestCase):
    def test_literal_zero_guard(self):
        n, h = np.zeros((12, 2)), np.zeros((8, 2))
        accepted, fail = inspect(h, n, n)
        self.assertTrue(accepted)
        self.assertFalse(fail.any())
        p = n.copy()
        p[:, 0] = np.arange(1, 13)*.01
        accepted, fail = inspect(h, n, p)
        self.assertFalse(accepted)
        self.assertTrue(fail[0])
        self.assertTrue(fail[5])

    def test_motion_against_scalar(self):
        x = fixture(1)
        for p in x['native'][0]:
            np.testing.assert_allclose(candidate_metrics(p), scalar_metrics(p, x['past'][0])[:5],
                                       rtol=0, atol=2e-12)

    def test_teacher_identity_and_no_loss(self):
        x = fixture(1)
        for branch in ('END', 'TIME'):
            a = space(branch)
            m = masks_for_window(x['past'][0], x['native'][0], a)
            self.assertTrue(m['safe'][:, a.identity].all())
            teacher = teacher_for_window(x['native'][0], x['native'][0, 0], a, m)
            self.assertTrue(np.all(teacher['best_J'] <= teacher['identity_J']))
            self.assertEqual(teacher['action_id'][0], a.identity)
            self.assertEqual(teacher['best_J'][0], 0.)

    def test_same_candidate_not_disjoint_minima(self):
        gt = np.zeros((12, 2))
        p = np.zeros((2, 12, 2))
        p[0, -1, 0] = .6   # ADE passes, FDE fails
        p[1, :11, 0] = .3  # FDE passes, ADE fails
        a, f, _, good = samepath_quality(p, gt)
        self.assertLess(a.min(), .2)
        self.assertLess(f.min(), .4)
        self.assertFalse(good.any())

    def test_inference_audit_and_GT_free_signature(self):
        x = fixture(2)
        for branch in ('END', 'TIME'):
            u = np.random.default_rng(17).uniform(-1, 1, (2, 20, 2))
            saved = predict_paths(x['past'], x['native'], u, branch)
            saved['sample_id'] = x['sample_id']
            review = check_saved(x, saved, branch)
            self.assertEqual(review['status'], 'PASS_SCALAR_GEOMETRY_AND_GUARD_ONLY')
            self.assertFalse(review['full_quality_clusters_bootstrap_audit'])
            saved['guarded_paths'][0, 0, 0, 0] += 1.
            with self.assertRaises(AssertionError):
                check_saved(x, saved, branch)

    def test_combine_identity(self):
        x = fixture(2)
        e = np.full((2, 20), space('END').identity)
        t = np.full((2, 20), space('TIME').identity)
        z = combine(x['past'], x['native'], e, t)
        np.testing.assert_array_equal(z['guarded_paths'], x['native'])

    def test_saved_parameter_and_reason_tampering_rejected(self):
        x = fixture(1)
        saved = predict_paths(x['past'], x['native'], np.zeros((1, 20, 2)), 'END')
        saved['sample_id'] = x['sample_id']
        saved['continuous_param'][0, 0] = [1., 0.]
        with self.assertRaises(AssertionError):
            check_saved(x, saved, 'END')
        saved['continuous_param'][0, 0] = 0.
        saved['rejection_reason'][0, 0, 3] = True
        with self.assertRaises(AssertionError):
            check_saved(x, saved, 'END')
        saved['rejection_reason'][0, 0, 3] = False
        saved['raw_paths'][0, 0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            check_saved(x, saved, 'END')


class DataMetricsTests(unittest.TestCase):
    def test_terminal_seal_required_and_failed_stage_refused(self):
        from unittest.mock import patch
        from io_contract import checked_seal
        with patch('io_contract.Path.exists', return_value=True):
            with self.assertRaises(ValueError):
                checked_seal('not-read')
        terminal = dict(status='STAGE_OUTPUTS_COMPLETE_NOT_SCIENTIFIC_PASS',
                        code_digest='current', files={'done.json': 'sha'})
        done = dict(code_digest='current', files={'fixture': 'hash'})
        with patch('io_contract.Path.exists', return_value=False), \
                patch('io_contract.code_digest', return_value='current'), \
                patch('io_contract.sha', return_value='sha'), \
                patch('io_contract.verify_manifest', side_effect=[terminal, done]):
            self.assertEqual(checked_seal('not-read'), done)
        with patch('io_contract.Path.exists', return_value=False), \
                patch('io_contract.verify_manifest', side_effect=FileNotFoundError('terminal absent')):
            with self.assertRaises(FileNotFoundError):
                checked_seal('not-read')

    def test_population_and_preflight_gates(self):
        from unittest.mock import patch
        from run_stage import population_contract, preflight_receipt, require_release
        held = dict(sample_id=np.arange(3825), group=np.r_[np.full(1023, 2), np.zeros(2802)])
        with patch('run_stage.load_npz', return_value=held):
            population_contract('synthetic-not-read.npz', 'hold')
            with self.assertRaises(ValueError):
                population_contract('synthetic-not-read.npz', 'train')
        with patch('run_stage.verify_manifest', return_value={'status': 'SYNTHETIC_PASS'}):
            with self.assertRaises(ValueError):
                preflight_receipt('not-read.json', {}, 123, 'cuda:0')
            with self.assertRaises(ValueError):
                require_release('not-read.json')
        incomplete = dict(status='PASS_REAL_DATA_GPU_RESUME_AND_TINY', binding={},
                          inventory_bytes=123, device='cuda:0', checks={})
        with patch('run_stage.verify_manifest', return_value=incomplete):
            with self.assertRaises(ValueError):
                preflight_receipt('not-read.json', {}, 123, 'cuda:0')
            with self.assertRaises(ValueError):
                preflight_receipt('not-read.json', {}, 456, 'cuda:0')

    def test_data_rejection(self):
        x = fixture(3)
        self.assertEqual(validate_observed(x), 3)
        with self.assertRaises(ValueError):
            assert_disjoint(x, x)
        with self.assertRaises(ValueError):
            aligned_gt(x, dict(sample_id=x['sample_id'][::-1], gt=np.zeros((3, 12, 2))))
        x['sample_id'][1] = x['sample_id'][0]
        with self.assertRaises(ValueError):
            validate_observed(x)

    def test_statistics_and_mask_after_normalization(self):
        x = fixture(3)
        stats = fit_statistics(x['past'], x['native'], x['sample_id'])
        g = transform(x['past'], x['native'], stats, 'G')
        h = transform(x['past'], x['native'], stats, 'H')
        self.assertTrue(np.all(g[..., :16] == 0))
        np.testing.assert_array_equal(g[..., 16:], h[..., 16:])
        self.assertTrue(np.all(stats['std'] > 0))

    def test_native_metrics_and_clusters(self):
        x = fixture(3)
        gt = x['native'][:, 0].copy()
        samples, rows = evaluate_population(x, gt, x['native'])
        self.assertTrue(samples['covered'].all())
        for row in rows:
            self.assertEqual(row['lost'], 0)
            self.assertEqual(row['rescue'], 0)
            self.assertTrue(row['motion_literal_pass'])
        repeated = np.repeat(gt[:, None], 20, axis=1)
        sm = sample_metrics(repeated, gt)
        np.testing.assert_array_equal(sm['distinct'], 1)
        np.testing.assert_array_equal(sm['effective_good'], 1)

    def test_bootstrap_cluster_sampling_matches_expansion(self):
        track = np.array(['a', 'a', 'b', 'c'])
        d = np.arange(12).reshape(3, 4).astype(float)
        mask = np.array([True, False, True, False])
        draws = make_draws(track, draws=64)
        report, values = paired_ci(d, track, mask, draws)
        for i, idx in enumerate(draws['track_draw']):
            windows = np.concatenate([np.flatnonzero((track == draws['tracks'][k]) & mask) for k in idx])
            if not len(windows):
                self.assertTrue(np.isnan(values['paired'][i]))
            else:
                want = d[draws['seed_draw'][i]][:, windows].mean()
                self.assertAlmostEqual(values['paired'][i], want)
        self.assertEqual(report['family'], 6)
        self.assertEqual(report['status'], 'INSUFFICIENT_PRODUCTION_DRAWS')


def tree_equal(a, b):
    import torch
    if isinstance(a, torch.Tensor):
        return torch.equal(a.cpu(), b.cpu())
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(tree_equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(tree_equal(x, y) for x, y in zip(a, b))
    return a == b


class TorchSyntheticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def test_parameter_count_and_zero_initial(self):
        import torch
        from model import ParameterRegressor
        m = ParameterRegressor()
        self.assertEqual(sum(p.numel() for p in m.parameters()), 4770)
        self.assertTrue(torch.equal(m(torch.randn(2, 20, 40)), torch.zeros(2, 20, 2)))

    def test_exact_serialized_cpu_mid_epoch_resume(self):
        import torch
        from train import Engine
        rng = np.random.default_rng(94)
        x = rng.normal(size=(7, 20, 40)).astype(np.float32)
        y = rng.uniform(-1, 1, (7, 20, 2)).astype(np.float32)
        contract = {'synthetic_only': True, 'stats': 'fixture'}
        full = Engine(x, y, contract, epochs=2, batch=3)
        while not full.complete:
            full.step()
        expected = full.snapshot()
        self.assertEqual(expected['cuda_rng'], [])
        partial = Engine(x, y, contract, epochs=2, batch=3)
        partial.step()
        partial.step()
        stream = io.BytesIO()
        torch.save(partial.snapshot(), stream)
        stream.seek(0)
        saved = torch.load(stream, map_location='cpu', weights_only=False)
        resumed = Engine(x, y, contract, epochs=2, batch=3)
        resumed.restore(saved)
        while not resumed.complete:
            resumed.step()
        self.assertTrue(tree_equal(expected, resumed.snapshot()))
        bad = Engine(x, y, {'stats': 'changed'}, epochs=2, batch=3)
        with self.assertRaises(ValueError):
            bad.restore(saved)
        wrong = x.copy()
        wrong[0, 0, 0] += 1
        bad_data = Engine(wrong, y, contract, epochs=2, batch=3)
        with self.assertRaises(ValueError):
            bad_data.restore(saved)

    def test_G_masked_gradient_zero(self):
        from train import Engine
        x = np.ones((3, 20, 40), np.float32)
        x[..., :16] = 0
        y = np.ones((3, 20, 2), np.float32)*.5
        e = Engine(x, y, {'synthetic_only': True}, epochs=2, batch=3)
        e.step()
        e.step()
        gradient = e.model.layers[0].weight.grad.cpu().numpy()
        self.assertTrue(np.all(gradient[:, :16] == 0))
        self.assertTrue(np.any(gradient[:, 16:] != 0))

    def test_independent_numpy_all_gradients_and_fd(self):
        from preflight_checks import gradient_check
        rng = np.random.default_rng(105)
        x = rng.normal(size=(8, 20, 40)).astype(np.float32)
        y = rng.uniform(-1, 1, size=(8, 20, 2)).astype(np.float32)
        for arm in ('G', 'H'):
            features = x.copy()
            if arm == 'G':
                features[..., :16] = 0
            report = gradient_check(features, y, device='cpu')
            self.assertLess(report['max_gradient_error'], 1e-10)


if __name__ == '__main__':
    unittest.main(verbosity=2)
