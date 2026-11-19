"""Sealed input alignment and fit-only observation-bin reference."""

from pathlib import Path
import hashlib
import sys

import numpy as np
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from calib_metrics import SEED, hpd_mc, log_density

ROOT = Path(__file__).resolve().parent
RUNS = ROOT.parent
PREC = RUNS / "EXP_PREC_DIAG_EXEC_20261119"
HISTORY = RUNS / "EXP_PREC_HISTORY_DIAG_EXEC_20261119"
FAST_RUN = RUNS / "EXPLORE_FAST_EXEC_20261116/runs/B0_E16_s42"
CV_REFERENCE = FAST_RUN / "dev_window_metrics.npz"
sys.path[:0] = [str(PREC), str(HISTORY)]
from run_prec_diag import (ARMS, LABEL, OLD, arrays, groups, load_prediction, maneuver_labels, quality,
                           read, run_folder, sha)
from run_history_diag import observation_masks

OBS = ("stationary", "steady", "other")
BINS = ("stationary", "steady_low", "steady_mid", "steady_high", "other_low", "other_mid", "other_high")
STEPS = (12, 6)


def window_save(path, values):
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **values)
    tmp.replace(path)


def inputs():
    prediction_seal = read(PREC / "INPUT_MANIFEST.json")["files"]
    prec_seal = read(PREC / "DONE.json")["files"]
    assert sha(PREC / "INPUT_MANIFEST.json") == prec_seal[str(PREC / "INPUT_MANIFEST.json")]
    saved = [LABEL, OLD / "exchange_null.py", OLD / "C0/fit.npz", OLD / "C0/dev.npz",
             *(run_folder(model, seed) / split / "predictions.npz"
               for model, seeds in ARMS.items() for seed in seeds for split in ("fit", "dev"))]
    baselines = [PREC / f"windows/B0_s{seed}.npz" for seed in ARMS["B0"]]
    seals = {**prediction_seal, **prec_seal, **read(FAST_RUN / "result.json")["files"]}
    checked = {}
    for path in saved + baselines + [PREC / "run_prec_diag.py", CV_REFERENCE]:
        digest = sha(path)
        assert digest == seals[str(path)], path
        checked[str(path)] = digest
    pinned_now = [FAST_RUN / "result.json", HISTORY / "run_history_diag.py", *ROOT.glob("*.py"),
                  ROOT / "PROTOCOL.md", ROOT / "DESIGN.md", *ROOT.glob("standalone/*.ps1")]
    for path in pinned_now:
        checked[str(path)] = sha(path)
    return checked


def history_groups(past):
    m = observation_masks(past)
    return np.select([m["stationary"], m["steady"]], OBS[:2], default="other")


def observed_speed(past):
    delta = np.diff(past, axis=1)
    return np.hypot(delta[..., 0], delta[..., 1]).mean(1) / .4


def coordinates(past):
    past = past.astype(np.float64)
    end = past[:, -1]
    cv = end[:, None] + np.arange(1, 13)[None, :, None] * (end - past[:, -2])[:, None]
    heading = end - past[:, -4]
    length = np.linalg.norm(heading, axis=1)
    use_direction = length > .03
    heading[use_direction] /= length[use_direction, None]
    heading[~use_direction] = (1., 0.)
    cos, sin = heading.T
    rotation = np.stack((cos, -sin, sin, cos), axis=1).reshape((-1, 2, 2))
    return cv, heading, rotation


def split_halves(ids):
    parts = [str(v).split("|") for v in ids]
    assert all(len(v) == 3 and all(v) for v in parts)
    tracks = ["|".join(v[:2]) for v in parts]
    track_half = {v: hashlib.sha256(f"{SEED}|{v}".encode()).digest()[0] % 2 for v in set(tracks)}
    half = np.array([track_half[v] for v in tracks], dtype=np.int8)
    halves = [set(v for v, h in zip(tracks, half) if h == a) for a in (0, 1)]
    assert all(halves) and not halves[0] & halves[1]
    return half, {name: dict(windows=int((half == i).sum()), tracks=len(halves[i]))
                  for i, name in enumerate("AB")}


def aligned_reference(path, data_ids, keys):
    saved = arrays(path)
    positions = {item: i for i, item in enumerate(saved["ids"])}
    assert len(positions) == len(data_ids) == len(saved["ids"]) and set(positions) == set(data_ids)
    order = [positions[item] for item in data_ids]
    return [saved[key][order] for key in keys]


def quantizer(fit, dev, fit_obs, dev_obs, fit_cv, dev_cv, fit_R, dev_R):
    fit_speed, dev_speed = [observed_speed(d["past"][:, 0]) for d in (fit, dev)]
    cuts = {name: np.quantile(fit_speed[fit_obs == name], [1/3, 2/3]) for name in OBS[1:]}

    def assign(obs, speed):
        assigned = np.zeros(len(obs), dtype=np.int8)
        for offset, name in ((1, "steady"), (4, "other")):
            m = obs == name
            assigned[m] = offset + np.searchsorted(cuts[name], speed[m], side="right")
        return assigned

    fit_bin, dev_bin = assign(fit_obs, fit_speed), assign(dev_obs, dev_speed)
    residual = np.einsum("nij,nti->ntj", fit_R, fit["future"] - fit_cv)
    reference = np.empty((len(dev_bin), 20, 12, 2))
    fit_rows = {}
    with threadpool_limits(limits=1):
        for b, name in enumerate(BINS):
            train, test = residual[fit_bin == b], dev_bin == b
            vector = train.reshape((-1, 24))

            def fit_centers():
                estimator = KMeans(n_clusters=20, init="k-means++", n_init=10, max_iter=300,
                                   random_state=SEED, algorithm="lloyd", tol=1e-4)
                estimator.fit(vector)
                return estimator.cluster_centers_, estimator.n_iter_

            a, iterations = fit_centers()
            b_centers, _ = fit_centers()
            np.testing.assert_array_equal(a, b_centers)
            templates = a.reshape((20, 12, 2))
            reference[test] = dev_cv[test, None] + np.einsum("nij,ktj->nkti", dev_R[test], templates)
            row = dict(fit_n=int(len(train)), dev_n=int(test.sum()), kmeans_iterations=int(iterations))
            for step in STEPS:
                end = train[:, step - 1]
                row[f"t{step}_empirical_spread_m"] = float(np.sqrt(np.square(end - end.mean(0)).sum(1).mean()))
            fit_rows[name] = row
    dist = np.linalg.norm(reference - dev["future"][:, None], axis=-1)
    ref_metrics = {}
    for step in STEPS:
        ref_metrics[step] = (dist[..., :step].mean(-1).min(1), dist[..., step-1].min(1))
    return dev_bin, fit_rows, {name: values.tolist() for name, values in cuts.items()}, ref_metrics


def numerical_check():
    rng = np.random.default_rng(SEED)
    # Each test row is an ordinary single 2D Gaussian duplicated into 20 components.
    centers = np.zeros((6, 20, 2))
    h = np.linspace(.03, .3, len(centers))
    p = rng.dirichlet(np.ones(20), len(h))
    radius = np.linspace(.3, 2.6, len(h))
    truth = log_density(np.broadcast_to((radius*h)[:, None] ** 2, p.shape), np.log(p), h)
    draws = 20000
    u, area = hpd_mc(centers, p, h, truth, rng.random((len(h), draws)), rng.standard_normal((len(h), draws, 2)))
    np.testing.assert_allclose(u, -np.expm1(-radius**2/2), rtol=0, atol=.012)
    np.testing.assert_allclose(area, -2*np.pi*h**2*np.log(.1), rtol=.05, atol=0)


def prepare(on_stage):
    inputs_manifest = inputs()
    fit, dev = [arrays(OLD / "C0" / f"{name}.npz") for name in ("fit", "dev")]
    fit_past, dev_past = [data["past"][:, 0].astype(np.float64) for data in (fit, dev)]
    fit_obs, dev_obs = [history_groups(p) for p in (fit_past, dev_past)]
    counts = {name: int((dev_obs == name).sum()) for name in OBS}
    assert list(counts.values()) == [1109, 1991, 1653], counts
    half, half_stats = split_halves(dev["ids"])
    fit_cv, _, fit_R = coordinates(fit_past)
    dev_cv, heading, dev_R = coordinates(dev_past)
    old_fde, = aligned_reference(CV_REFERENCE, dev["ids"], ["cv_FDE"])
    cv_fde = np.linalg.norm(dev_cv[:, -1] - dev["future"][:, -1], axis=-1)
    np.testing.assert_allclose(cv_fde, old_fde, atol=1e-9, rtol=0)
    matches = {}
    for seed in ARMS["B0"]:
        paths, _ = load_prediction(run_folder("B0", seed), "dev", dev)
        ref = aligned_reference(PREC / f"windows/B0_s{seed}.npz", dev["ids"], ["base__minADE", "base__minFDE"])
        q = quality(paths, dev["future"])
        delta = [float(np.max(np.abs(q[key].min(1) - expected))) for key, expected in zip(("ADE", "FDE"), ref)]
        assert max(delta) <= 1e-9, (seed, delta)
        matches[str(seed)] = dict(zip(("minADE", "minFDE"), delta))
    numerical_check()
    on_stage("quantizer_fit_preflight")
    dev_bin, fit_rows, cuts, ref = quantizer(fit, dev, fit_obs, dev_obs, fit_cv, dev_cv, fit_R, dev_R)
    labels = maneuver_labels(dev["past"], dev["future"])
    masks = groups(dev, labels)
    masks.update({f"obs_{v}": dev_obs == v for v in OBS})
    masks.update({f"bin_{v}": dev_bin == i for i, v in enumerate(BINS)})
    masks = {name: m for name, m in masks.items() if m.any()}
    preflight = dict(windows=dict(fit=len(fit["ids"]), dev=len(dev["ids"])), history_counts=counts,
                     cross_fit=half_stats, track_overlap=0, kmeans_exact_repeat=True,
                     cv_FDE_max_abs_error=float(np.max(np.abs(cv_fde - old_fde))),
                     B0_max_abs_error=matches, HPD_gaussian_mc_check=True)
    return dict(fit=fit, dev=dev, heading=heading, half=half, dev_bin=dev_bin, fit_bins=fit_rows,
                cuts=cuts, ref=ref, labels=labels, obs=dev_obs, masks=masks, preflight=preflight,
                inputs=inputs_manifest)
