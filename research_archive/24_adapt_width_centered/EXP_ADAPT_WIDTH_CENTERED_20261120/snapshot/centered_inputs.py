"""Read pinned forecasts and verify width interventions before the formal grids."""
from datetime import datetime
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
PREV = ROOT.parent / "EXP_ADAPT_ES_PRECHECK_EXEC_20261120"
sys.path.insert(0, str(PREV))

from adapt_data import (BINS, CAL, OBS, OLD, SEEDS, arrays, groups, old_paths, prepare_data,
                        read, sha, split_halves, verify, write)
from adapt_math import ES_NAMES, errors, probabilities, scatter
from calib_metrics import DRAW_COUNT, SEED
from coupling_analysis import LABELS
from centered_metrics import AXIS, ISO, all_grids, from_paths
from centered_scaling_ref import center_scale, center_scale_aniso, weighted_mean

RUNS = tuple(("B0", s) for s in SEEDS) + (("E1B", 42), ("E1B", 43))


def status(phase, **details):
    write(ROOT / "CURRENT.json",
          dict(at=datetime.now().astimezone().isoformat(), pid=os.getpid(), phase=phase, **details))


def init():
    files = {}
    data, archive, cuts = prepare_data(files)
    replay = read(ROOT / "HELDOUT_REPLAY.json")
    assert replay["status"] == "PASS"
    for p, digest in replay["inputs"].items():
        verify(Path(p), digest, files)
    for name, digest in replay["outputs"].items():
        verify(ROOT / name, digest, files)
    files[str(ROOT / "HELDOUT_REPLAY.json")] = sha(ROOT / "HELDOUT_REPLAY.json")
    split = arrays(OLD / "E1/calib_split.npz")
    fit = data["fit"]
    calib = {key: value[split["calib_index"]] for key, value in fit.items()}
    train_tracks = set(fit["track"][split["fit85_index"]])
    assert len(calib["ids"]) == 3054 and train_tracks.isdisjoint(calib["track"])
    calib_cuts = {}
    for offset, name in ((1, "steady"), (4, "other")):
        mask = calib["obs"] == name
        bounds = np.quantile(calib["speed"][mask], [1/3, 2/3])
        calib["bin"][mask] = np.asarray(BINS)[offset + np.searchsorted(bounds, calib["speed"][mask], side="right")]
        calib_cuts[name] = bounds.tolist()
    data["CALIB"] = calib
    return data, archive, dict(fit=cuts, CALIB=calib_cuts), replay, files


def dev_groups(data, paths, weights):
    return {**groups(data), **{f"future_{label}": data["label"] == label for label in LABELS},
            "long_mean": np.linalg.norm(weighted_mean(paths, weights)[:, -1], axis=-1) >= .5}


def draws_for(n):
    rng = np.random.default_rng(SEED)
    return rng.random((n, DRAW_COUNT)), rng.standard_normal((n, DRAW_COUNT, 2))


def widths(archive, seed, d):
    source = archive["runs"]["B0"][str(seed)]["t12"]["hpd"]["equal"]["fitted_on_other_half"]
    half, _ = split_halves(d["ids"])
    return np.array([source[side]["global"]["chosen"] for side in "AB"])[half]


def load_pair(arm, seed, data, files):
    selection = "fit" if arm == "B0" else "CALIB"
    by_split = {}
    for name in (selection, "dev"):
        d = data[name]
        if arm == "B0":
            paths, weights = old_paths(f"E1C_E64_s{seed}", name, d, files)
        else:
            p = (ROOT / f"E1B_dev_s{seed}.npz" if name == "dev" else
                 OLD / f"runs/E1B_E64_s{seed}/calib/predictions.npz")
            original = arrays(p)
            np.testing.assert_array_equal(original["ids"], d["ids"])
            paths = original["paths"].astype(np.float64) - d["last"][:, None]
            weights = probabilities(original["logits"])
        assert paths.shape == (len(d["ids"]), 20, 12, 2) and np.isfinite(paths).all()
        np.testing.assert_allclose(weights.sum(1), 1, atol=1e-14, rtol=0)
        by_split[name] = paths, weights
    return selection, by_split


def max_error(a, b):
    return float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))


def manipulation(x, w, heading):
    """Deterministic real windows, every isotropic scale and anisotropic pair."""
    i, j = np.triu_indices(x.shape[1], 1)
    mean = weighted_mean(x, w)
    pair = np.linalg.norm(x[:, i] - x[:, j], axis=-1)
    spread = errors(x, mean)["spread"]
    base_scatter = scatter(x, w)
    checks = {}
    for a in ISO:
        iso = center_scale(x, w, a)
        diagonal = center_scale_aniso(x, w, a, a, heading)
        row = dict(mean_m=max_error(weighted_mean(iso, w), mean),
                   pair_m=max_error(np.linalg.norm(iso[:, i] - iso[:, j], axis=-1), a * pair),
                   spread_m=max_error(errors(iso, mean)["spread"], a * spread),
                   scatter=max_error(scatter(iso, w), a * base_scatter),
                   diag_m=max_error(diagonal, iso))
        assert max(row.values()) <= 1e-9, (a, row)
        checks[str(a)] = row
    shift = max(max_error(weighted_mean(center_scale_aniso(x, w, a, b, heading), w), mean)
                for a in AXIS for b in AXIS)
    assert shift <= 1e-9
    return dict(iso=checks, aniso_mean_max_m=shift, windows=len(x))


def fast_step_grid(x, w):
    center = weighted_mean(x, w)[:, None]
    step = np.diff(center, prepend=0, axis=2)
    deviation = np.diff(x - center, prepend=0, axis=2)
    return {str(a): float((np.square(step + a * deviation).sum(-1) > 1.).mean()) for a in ISO}


def preflight(data, archive, cuts, replay, files):
    prev_seal = read(PREV / "DONE.json")["files"]
    cal_seal = read(CAL / "DONE.json")["files"]
    old_grid = PREV / "P1_alpha_curves.json"
    verify(old_grid, prev_seal[old_grid.name], files)
    old_curves = read(old_grid)
    checks = dict(status="PASS", PF1={}, PF2_PF3_PF4={}, PF5={}, PF6=replay["checks"], PF7={},
                  optimized_grid_check={}, speed_cuts_mps=cuts, files=files)
    d = data["dev"]
    dev_draws = draws_for(len(d["ids"]))
    for arm, seed in RUNS:
        status("PREFLIGHT", arm=arm, seed=seed)
        key = f"{arm}_s{seed}"
        select, forecasts = load_pair(arm, seed, data, files)
        x, w = forecasts["dev"]
        h = widths(archive, seed, d)
        if arm == "B0":
            base = from_paths(center_scale(x, w, 1.), d["target_rel"], w, h, dev_draws)
            previous = PREV / f"P2_windows_B0_s{seed}.npz"
            verify(previous, prev_seal[previous.name], files)
            saved = arrays(previous)
            np.testing.assert_array_equal(d["ids"], saved["ids"])
            matching = [k for k in base if k not in ("HPD90", "mean_bias", "fast_steps")]
            distance = {field: max_error(base[field], saved[field]) for field in matching}
            assert max(distance.values()) <= 1e-9, distance
            hp = f"windows/B0_s{seed}.npz"
            verify(CAL / hp, cal_seal[hp], files)
            raw = arrays(CAL / hp)
            np.testing.assert_array_equal(d["ids"], raw["ids"])
            np.testing.assert_array_equal(base["HPD90"], raw["t12__equal__u__global"] <= .9)
            checks["PF1"][key] = dict(max_difference=distance, windows=len(x), archived_HPD90_match=True)
            fx, fw = forecasts[select]
            fd = data[select]
            steady = fd["obs"] == "steady"
            fx, fw, fy = fx[steady], fw[steady], fd["target_rel"][steady]
            denom = errors(fx, fy)["minFDE"].mean()
            old = float(errors(fx * .9, fy)["minFDE"].mean() / denom)
            old_rows = old_curves[str(seed)]["fit"]
            old_sealed = old_rows["0.9"]["all"]["steady"]["minFDE"] / old_rows["1.0"]["all"]["steady"]["minFDE"]
            assert abs(old - old_sealed) <= 1e-12
            new = float(errors(center_scale(fx, fw, .9), fy)["minFDE"].mean() / denom)
            checks["PF5"][key] = dict(steady_n=len(fx), old_0_9_over_1_0=old, new_0_9_over_1_0=new)
        checks["PF2_PF3_PF4"][key] = {}
        checks["PF7"][key] = {}
        for split, (paths, weights) in forecasts.items():
            meta = data[split]
            idx = np.linspace(0, len(paths) - 1, 256, dtype=int)
            checks["PF2_PF3_PF4"][key][split] = manipulation(paths[idx], weights[idx], meta["rotation"][idx, :, 0])
            checks["PF7"][key][split] = fast_step_grid(paths, weights)
        idx = np.linspace(0, len(x) - 1, 32, dtype=int)
        y, heading = d["target_rel"][idx], d["rotation"][idx, :, 0]
        x, w, h = x[idx], w[idx], h[idx]
        draws = tuple(a[idx] for a in dev_draws)
        iso, aniso = all_grids(x, y, w, heading, h, draws, lambda *_: None)
        found = {}
        for lon, lat in ((.4, .4), (.5, 1.6), (.9, 1.3), (1., 1.), (1.6, .5), (1.6, 1.6)):
            p = (center_scale(x, w, lon) if lon == lat else center_scale_aniso(x, w, lon, lat, heading))
            exact = from_paths(p, y, w, h, draws)
            fast = iso[lon] if lon == lat else aniso[lon, lat]
            error = {k: max_error(exact[k], fast[k]) for k in exact}
            assert max(error.values()) <= 1e-9, (lon, lat, error)
            found[f"{lon},{lat}"] = error
        checks["optimized_grid_check"][key] = found
        print(f"Preflight passed for {key}", flush=True)
    return checks


if __name__ == "__main__":
    result = preflight(*init())
    write(ROOT / "PREFLIGHT.json", result)
    status("PREFLIGHT_PASSED")

