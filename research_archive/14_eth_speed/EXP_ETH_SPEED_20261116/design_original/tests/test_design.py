"""Meaningful CPU synthetic checks only; no real-data model training/inference."""
from __future__ import annotations

import ast
import copy
import inspect
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
import eth_speed_math as em
from speed_aug_adapter import wide_speed, trainer_type
from speed_io import config
from audit_eth_speed import raw_index, signature


def observations(n=12):
    rng = np.random.default_rng(74)
    p = rng.normal(size=(n, 9, 8, 2)).astype(np.float32)
    p[:, 0, :, 0] = np.arange(8)[None]*np.linspace(.01, 1., n)[:, None]
    p[:, 0, :, 1] = 2
    nums = np.arange(n) % 8 + 1
    for i, num in enumerate(nums):
        p[i, num:] = 0
    return p, nums


class TransformTests(unittest.TestCase):
    def test_identity_padding_and_last_geometry(self):
        p, n = observations(); s = np.linspace(.5, 1., len(p))
        self.assertTrue(np.array_equal(em.transform_past(p, n, np.ones(len(p))), p))
        q = em.transform_past(p, n, s)
        for i, num in enumerate(n):
            np.testing.assert_array_equal(q[i, :num, -1], p[i, :num, -1])
            np.testing.assert_array_equal(q[i, num:], p[i, num:])
            np.testing.assert_allclose(np.diff(q[i, :num], axis=1),
                                       s[i]*np.diff(p[i, :num], axis=1), atol=8e-7)

    def test_scene_control_changes_neighbor_distances(self):
        p, n = observations(); s = np.full(len(p), .5)
        q = em.transform_past(p, n, s, 'scene')
        for i, num in enumerate(n):
            np.testing.assert_allclose(q[i, :num, -1]-q[i, 0, -1],
                      .5*(p[i, :num, -1]-p[i, 0, -1]), atol=5e-7)

    def test_world_inverse_and_translation(self):
        p, n = observations(); s = np.linspace(.5, 1., len(p))
        paths = np.random.default_rng(5).normal(size=(len(p), 20, 12, 2))
        anchor = p[:, 0, -1, None, None]
        transformed = anchor + s[:, None, None, None]*(paths-anchor)
        np.testing.assert_allclose(em.inverse_paths(transformed, p, s), paths, atol=3e-15)
        shifted = p + np.array([10., -20.], np.float32)
        q = em.transform_past(shifted, np.full(len(p), 9), s)
        ref = em.transform_past(p, np.full(len(p), 9), s)
        np.testing.assert_allclose(q-ref, np.broadcast_to([10., -20.], q.shape), atol=5e-6)

    def test_scales_use_observed_only_and_shared_train_edges(self):
        p, n = observations(); profile = em.fit_profile(p[:7])
        before = em.scales(p, profile)
        np.testing.assert_allclose(em.observed_speed(p), np.linspace(.01, 1., len(p))/.4, rtol=2e-6)
        self.assertTrue(np.all(before[em.speed_bins(p, profile) != 2] == 1))
        self.assertGreaterEqual(before.min(), .5)
        self.assertEqual(list(inspect.signature(em.scales).parameters), ['past', 'profile'])
        self.assertEqual(list(inspect.signature(em.fit_profile).parameters), ['past'])
        self.assertEqual(sum(np.sum(em.speed_bins(p, profile) == i) for i in range(3)), len(p))

    def test_invalid_scales_rejected(self):
        p, n = observations()
        for bad in (0., np.nan, -1.):
            with self.assertRaises(ValueError):
                em.transform_past(p, n, np.full(len(p), bad))

    def test_maneuver_partition(self):
        p, n = observations()
        f = p[:, 0, -1, None]+np.arange(1, 13)[None, :, None]*np.array([.2, 0.])
        labels = em.maneuver_labels(p, f)
        self.assertEqual(len(labels), len(p)); self.assertTrue(set(labels) <= set(em.MANEUVERS))


class AuditTests(unittest.TestCase):
    def test_exact_raw_lineage_detects_scale_and_frame_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp)/'toy_train.txt'
            points = np.arange(40, dtype=np.float32).reshape(20, 2)
            rows = []
            for i, xy in enumerate(points):
                row = ['-1']*17; row[0] = str(i); row[1] = '7'; row[2] = 'Pedestrian'
                row[13], row[15] = map(str, xy); rows.append(' '.join(row))
            file.write_text('\n'.join(rows), encoding='utf-8')
            index, report = raw_index([file])
            self.assertIn(signature(points), index)
            self.assertNotIn(signature(points*10), index)
            # Missing an intermediate label frame must not be accepted as 20 equal steps.
            file.write_text('\n'.join(rows[:9]+rows[10:]), encoding='utf-8')
            index, _ = raw_index([file]); self.assertNotIn(signature(points), index)


class DecisionTests(unittest.TestCase):
    def synthetic(self):
        def case(a, f):
            return dict(overall=dict(minADE=a, minFDE=f),
                        speed=dict(high=dict(n=50, minADE=a*2)))
        cases = dict(identity=case(1., 2.), high_motion=case(.85, 1.9), reverse_motion=case(1.1, 2.2))
        return {f: dict(cases=copy.deepcopy(cases), baseline_replay_passed=True,
                       low_speed_identity_passed=True) for f in ('eth', 'hotel')}

    def test_gate_is_conjunctive_and_no_training_action(self):
        r = self.synthetic(); a = dict(pipeline_consistent=True)
        out = em.diagnostic_decision(r, a)
        self.assertEqual(out['status'], 'ELIGIBLE_EXPLORATORY')
        self.assertFalse(out['training_started']); self.assertEqual(out['causal_conclusion'], 'NOT_ESTABLISHED')
        r['eth']['cases']['high_motion']['overall']['minFDE'] = 2.1
        self.assertEqual(em.diagnostic_decision(r, a)['status'], 'DO_NOT_TRAIN_YET')
        r = self.synthetic(); a['pipeline_consistent'] = False
        self.assertEqual(em.diagnostic_decision(r, a)['status'], 'DO_NOT_TRAIN_YET')

    def test_empty_high_speed_group_is_not_evidence(self):
        r = self.synthetic()
        for case in r['eth']['cases'].values():
            case['speed']['high'] = dict(n=0, minADE=None)
        self.assertEqual(em.diagnostic_decision(r, dict(pipeline_consistent=True))['status'], 'DO_NOT_TRAIN_YET')


class PipelineTests(unittest.TestCase):
    def test_complete_synthetic_diag_reuses_done_and_never_trains(self):
        import run_eth_speed as run
        sys.path.insert(0, config()['fivefold_code'])
        import fivefold_metrics as fm
        p, nums = observations(65)
        v = p[:, 0, -1]-p[:, 0, -2]
        future = p[:, 0, -1, None]+np.arange(1, 13)[None, :, None]*v[:, None]
        data = dict(past=p, num_valid=nums, future=future, ids=np.arange(65),
                    group=np.arange(65)%3, scene=np.zeros(65, np.int64))
        calls = []
        def infer(model, inputs, batch):
            self.assertEqual(set(inputs), {'past', 'num_valid', 'ids'})
            calls.append(len(inputs['past']))
            past = inputs['past'][:, 0]
            pred = past[:, -1, None]+np.arange(1, 13)[None, :, None]*(past[:, -1, None]-past[:, -2, None])
            return np.repeat(pred[:, None], 20, axis=1), np.zeros((len(past), 20))
        fd = SimpleNamespace(load_split=lambda *a: data)
        tc = SimpleNamespace(infer=infer)
        population = (fd, fm, data, np.arange(65), np.array([1, 2]), em.fit_profile(p))
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(run, 'prepare', return_value=dict(pipeline_consistent=True)), \
                 patch.object(run, 'load_population', return_value=population), \
                 patch.object(run, 'model_from_checkpoint', return_value=(tc, object())):
                import time
                run.diag(dict(data_dir='unused'), tmp, 'synthetic_contract', time.monotonic()+60)
                from speed_io import read
                decision = read(Path(tmp)/'DIAG/DECISION.json')
                self.assertFalse(decision['training_started'])
                self.assertTrue((Path(tmp)/'DIAG/STRATIFIED_RESULTS.csv').is_file())
                count = len(calls)
                run.diag(dict(data_dir='unused'), tmp, 'synthetic_contract', time.monotonic()+60)
                self.assertEqual(len(calls), count)


class RealAugmentationFunctionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        source = Path(config()['workspace'])/'FRONT_ENCODER_SERIES/train_front_variant.py'
        tree = ast.parse(source.read_text(encoding='utf-8-sig'))
        fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'augment_trajectories')
        # Run the real existing augmentation function on synthetic tensors,
        # without importing the training launcher or loading actual trajectories.
        ns = dict(torch=torch, math=math, argparse=__import__('argparse'))
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), 'exec'), ns)
        cls.augment = staticmethod(ns['augment_trajectories'])

    def test_wide_range_replaces_and_does_not_stack(self):
        t = self.torch
        original = SimpleNamespace(augment_speed_min=.9, augment_speed_max=1.1,
            augment_probability=1., augment_rotation=False, augment_reflection=False, augment_curvature_max=0.)
        factory = SimpleNamespace(AUGMENT=original)
        p = t.zeros(4000, 9, 8, 2); p[..., 0] = t.arange(8)
        f = t.zeros(4000, 9, 12, 2); f[..., 0] = t.arange(8, 20)
        t.manual_seed(42)
        with wide_speed(factory):
            q, y = self.augment(p, f, factory.AUGMENT)
        self.assertIs(factory.AUGMENT, original)
        ratios = q[:, 0, -1, 0]-q[:, 0, -2, 0]
        self.assertGreaterEqual(float(ratios.min()), .5-1e-6)
        self.assertLessEqual(float(ratios.max()), 2.+1e-6)
        self.assertLess(float(ratios.min()), .51); self.assertGreater(float(ratios.max()), 1.99)
        t.testing.assert_close(q[:, :, -1], p[:, :, -1], rtol=0, atol=0)
        t.testing.assert_close(y[:, 0, 0, 0]-p[:, 0, -1, 0], ratios, rtol=3e-6, atol=3e-6)

    def test_draw_count_and_rng_resume_preserved(self):
        t = self.torch
        base = SimpleNamespace(augment_speed_min=.9, augment_speed_max=1.1,
            augment_probability=.5, augment_rotation=False, augment_reflection=True, augment_curvature_max=0.)
        p = t.arange(9*8*2, dtype=t.float32).reshape(1, 9, 8, 2).repeat(4, 1, 1, 1)
        f = t.zeros(4, 9, 12, 2)
        t.manual_seed(77); self.augment(p, f, base); rng_base = t.get_rng_state()
        t.manual_seed(77); factory = SimpleNamespace(AUGMENT=base)
        with wide_speed(factory): self.augment(p, f, factory.AUGMENT)
        self.assertTrue(t.equal(rng_base, t.get_rng_state()))
        state = t.get_rng_state()
        with wide_speed(factory): a, b = self.augment(p, f, factory.AUGMENT)
        t.set_rng_state(state)
        with wide_speed(factory): c, d = self.augment(p, f, factory.AUGMENT)
        self.assertTrue(t.equal(a, c) and t.equal(b, d))

    def test_exception_restores_original_configuration(self):
        factory = SimpleNamespace(AUGMENT=SimpleNamespace(augment_speed_min=.9, augment_speed_max=1.1))
        original = factory.AUGMENT
        with self.assertRaises(RuntimeError):
            with wide_speed(factory): raise RuntimeError('synthetic failure')
        self.assertIs(factory.AUGMENT, original)


if __name__ == '__main__':
    unittest.main(verbosity=2)
