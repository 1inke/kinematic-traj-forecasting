"""Load sealed C0 forecasts and derive strictly observation-only features."""
from datetime import datetime
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
CENTERED = ROOT.parent / "EXP_ADAPT_WIDTH_CENTERED_EXEC_20261120"
DECEL = ROOT.parent / "EXP_DECEL_DIAG_EXEC_20261117"
sys.path.insert(0, str(CENTERED))

from centered_inputs import CAL, RUNS, arrays, draws_for, init, load_pair, read, sha, split_halves, verify, write

GRID = np.round(np.arange(-.6, .6001, .05), 2)
TILT = np.round(np.arange(-3., 3.01, .25), 2)
ALPHA = (.1, 1., 10., 100.)
FEATURES = ("K", "K+N", "S", "K+S")
OBS = ("v_obs", "hist_ratio", "curv", "lat_drift")
TAU = np.arange(1, 13) / 12
FOLD_SEED = 20261121


def status(phase, **details):
    write(ROOT / "CURRENT.json",
          dict(at=datetime.now().astimezone().isoformat(), pid=os.getpid(), phase=phase, **details))


def features(past, counts):
    p = past[:, 0].astype(float)
    steps = np.linalg.norm(np.diff(p, axis=1), axis=-1) / .4
    early = (p[:, 4] - p[:, 0]) / 4
    late = (p[:, -1] - p[:, -4]) / 3
    length = np.linalg.norm(late, axis=-1)
    valid = length > .03
    heading = np.divide(late, np.maximum(length[:, None], 1e-12))
    heading[~valid] = [1., 0.]
    normal = heading[:, ::-1] * [-1., 1.]
    curvature = np.arctan2(early[:, 0]*late[:, 1] - early[:, 1]*late[:, 0], (early*late).sum(1))
    good_curv = valid & (np.linalg.norm(early, axis=1) > .03)
    curvature[~good_curv] = 0.
    time = np.arange(8) - 3.5
    lateral = ((p - p[:, -1, None]) * normal[:, None]).sum(-1)
    count = np.sum((np.linalg.norm(past[:, 1:, -1].astype(float)-p[:, -1, None], axis=-1) <= 2.) &
                   (np.arange(1, past.shape[1])[None] < counts.reshape(-1, 1)), axis=1)
    v_last = steps[:, 4:].mean(1)
    return dict(v_obs=steps.mean(1), v_last=v_last, hist_ratio=v_last/np.maximum(steps[:, :4].mean(1), .1),
                curv=curvature, curvature_valid=good_curv, lat_drift=lateral@time/(time@time),
                n_nb2=count, direction_valid=valid, frame=np.stack((heading, normal), axis=-1))


def observed(dataset):
    return features(dataset["past"], dataset["num_valid"])


def quantiles(meta, mask):
    return dict(v_obs=np.quantile(meta["v_obs"][mask], [.2,.4,.6,.8]).tolist(),
                lat_drift=np.quantile(meta["lat_drift"][mask], [1/3,2/3]).tolist())


def bins(meta, cuts):
    def three(values, lo, hi):
        return np.where(values <= lo, 0, np.where(values >= hi, 2, 1))

    return {"v_obs": np.searchsorted(cuts["v_obs"], meta["v_obs"], side="right"),
            "hist_ratio": three(meta["hist_ratio"], .85, 1.15),
            "curv": three(meta["curv"], -.15, .15),
            "lat_drift": np.searchsorted(cuts["lat_drift"], meta["lat_drift"], side="right"),
            "n_nb2": three(meta["n_nb2"], 0, 3), "scene": meta["scene"]}


def cells(columns, names):
    values = [columns[k].astype(str) for k in names]
    key = values[0]
    for v in values[1:]:
        key = np.char.add(np.char.add(key, "/"), v)
    return key


def cell_features(name):
    return {"K": OBS, "K+N": (*OBS, "n_nb2"), "S": ("scene",), "K+S": (*OBS, "scene")}[name]


def fold_ids(track, k, seed=FOLD_SEED):
    unique, inverse = np.unique(track, return_inverse=True)
    assignment = np.empty(len(unique), int)
    assignment[np.random.default_rng(seed).permutation(len(unique))] = np.arange(len(unique)) % k
    return assignment[inverse]


def data_model(meta, paths, weights):
    frame = meta["frame"]
    y = meta["target_rel"]
    err = np.einsum("nkti,nij->nktj", paths - y[:, None], frame)
    speed_ratio = np.linalg.norm(np.diff(paths, axis=2, prepend=0), axis=-1).mean(2)/.4
    speed_ratio /= np.maximum(meta["v_obs"], .1)[:, None]
    lat = np.einsum("nki,ni->nk", paths[:, :, -1], frame[:, :, 1])
    tilt = np.stack((speed_ratio, lat), axis=-1)
    tilt -= np.median(tilt, axis=1, keepdims=True)
    return dict(paths=paths, y=y, w=weights, error=err, tilt=tilt)


def weighted_widths(archive, seed, d):
    fit = archive["runs"]["B0"][str(seed)]["t12"]["hpd"]["weighted"]["fitted_on_other_half"]
    half, _ = split_halves(d["ids"])
    return np.array([fit[k]["global"]["chosen"] for k in "AB"])[half]


def decel_reference(seed, files):
    p = DECEL / f"populations/C0_dev/B0_s{seed}/decel_window_metrics.npz"
    verify(p, read(DECEL/"FINAL_OUTPUT_SEAL.json")["files"][str(p)], files)
    return arrays(p)


def ready():
    data, archive, _, replay, files = init()
    half = draws_for(len(data["dev"]["ids"]))
    for d in data.values():
        d.update(observed(d))
        d["valid"] = d["direction_valid"]
    assert len(data["dev"]["ids"]) == 4753 and int(data["dev"]["valid"].sum()) == 3550
    assert set(data["fit"]["track"]).isdisjoint(data["dev"]["track"])
    for p in (ROOT/"DESIGN.md", ROOT/"PROTOCOL.md", *ROOT.glob("*.py"), *ROOT.glob("standalone/*.ps1"),
              *ROOT.glob("reference/*.py"), *ROOT.glob("reference/*.log"), *ROOT.glob("reference/*.json")):
        files[str(p)] = sha(p)
    return data, archive, replay, files, half

