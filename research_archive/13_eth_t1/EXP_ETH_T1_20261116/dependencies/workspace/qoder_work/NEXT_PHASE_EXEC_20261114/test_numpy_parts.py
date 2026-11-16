"""CPU checks for the numpy parts of this package. Run: python -I test_numpy_parts.py

Checks only maths/contract properties on synthetic numbers; they say nothing
about prediction quality.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spline_basis import basis, apply_basis, KNOTS, STEPS  # noqa: E402
from exchange_null import pair_exchange, motion_stats, gt_ratio  # noqa: E402

rng = np.random.default_rng(20261114)


def test_spline_matches_scipy_and_passes_anchors():
    anchors = rng.normal(size=(50, 3, 2))
    v0 = rng.normal(scale=0.3, size=(50, 2))
    for variant in ("natural", "clamped"):
        rel = apply_basis(anchors, v0, variant)
        # passes anchors exactly at steps 4, 8, 12
        assert np.allclose(rel[:, [3, 7, 11]], anchors, atol=1e-12), variant
        for i in range(5):
            for c in range(2):
                y = np.concatenate([[0.0], anchors[i, :, c]])
                bc = ("natural", "natural") if variant == "natural" else ((1, v0[i, c]), (2, 0.0))
                ref = CubicSpline(KNOTS, y, bc_type=bc)(STEPS)
                assert np.allclose(rel[i, :, c], ref, atol=1e-12)


def test_clamped_start_slope_and_linear_matches_h3():
    W, w_v = basis("clamped")
    # derivative at t=0 equals v0 for any anchors
    anchors = rng.normal(size=(3, 2))
    v0 = np.array([0.37, -0.11])
    for c in range(2):
        y = np.concatenate([[0.0], anchors[:, c]])
        s = CubicSpline(KNOTS, y, bc_type=((1, v0[c]), (2, 0.0)))
        assert abs(s(0.0, 1) - v0[c]) < 1e-12
    Wl, wl = basis("linear")
    expected = np.zeros((12, 3))
    for t in range(1, 13):
        if t <= 4:
            expected[t - 1, 0] = t / 4
        elif t <= 8:
            expected[t - 1, 0] = 1 - (t - 4) / 4; expected[t - 1, 1] = (t - 4) / 4
        else:
            expected[t - 1, 1] = 1 - (t - 8) / 4; expected[t - 1, 2] = (t - 8) / 4
    assert np.allclose(Wl, expected) and not wl.any()


def test_constant_velocity_is_reproduced_exactly_by_clamped():
    # If anchors lie on the CV line and v0 is the CV step, clamped spline == CV path.
    v0 = np.array([0.5, 0.1])
    anchors = np.stack([4 * v0, 8 * v0, 12 * v0])
    rel = apply_basis(anchors, v0, "clamped")
    assert np.allclose(rel, STEPS[:, None] * v0, atol=1e-12)


def test_junction_acceleration_drops_on_cv_like_case():
    # Linear h3 with anchors slightly slower than observed speed -> velocity jump at junction;
    # clamped spline removes most of the jump.
    dt = 0.4
    v0 = np.array([0.5, 0.0])                 # m/step = 1.25 m/s
    anchors = np.stack([4 * 0.4 * np.array([1, 0.]), 8 * 0.4 * np.array([1, 0.]), 12 * 0.4 * np.array([1, 0.])])
    for variant, bound in (("linear", None), ("clamped", None)):
        rel = apply_basis(anchors, v0, variant)
        v1 = rel[0] / dt
        jun = np.linalg.norm(v1 - v0 / dt) / dt
        if variant == "linear":
            lin = jun
        else:
            assert jun < 0.5 * lin, (jun, lin)


def test_exchange_counts():
    good_a = np.array([[True, False], [False, False], [True, True], [False, True]])
    good_b = np.array([[False, False], [True, False], [True, False], [False, False]])
    r = pair_exchange(good_a.any(1), good_b.any(1), np.zeros(4, int))
    o = r["overall"]
    assert (o["ref_good"], o["rescue"], o["lost"]) == (3, 1, 2)


def test_motion_and_gt_ratio_on_cv_paths():
    past = np.zeros((10, 8, 2)); past[:, :, 0] = np.arange(-7, 1) * 0.5
    fut = np.zeros((10, 12, 2)); fut[:, :, 0] = np.arange(1, 13) * 0.5
    paths = np.repeat(fut[:, None], 20, 1)
    m = motion_stats(paths, past)
    assert m["accel_mean"] < 1e-12 and m["junction_accel_mean"] < 1e-12
    ratio = gt_ratio({"accel_mean": 0.2}, {"accel_mean": 0.1})
    assert abs(ratio["accel_mean"]["ratio"] - 2.0) < 1e-12


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"{len(tests)} numpy checks passed")
