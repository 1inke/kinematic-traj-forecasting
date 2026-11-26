"""Fit pair-distribution energy from each fold's original training recordings."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, sosfiltfilt


HELD_OUT = {
    "eth": {"biwi_eth"}, "hotel": {"biwi_hotel"},
    "univ": {"students001", "students003"}, "zara01": {"crowds_zara01"},
    "zara02": {"crowds_zara02"},
}
EDGES = np.linspace(0, 8, 801)
RADII = (0.2, 0.6)
N_SHUFFLES = 64


def instantaneous(path):
    data = np.loadtxt(path, usecols=(0, 1, 13, 15))
    frames, peds = data[:, :2].astype(np.int64).T
    steps, counts = np.unique(np.concatenate([
        np.diff(np.sort(frames[peds == ped])) for ped in np.unique(peds)
    ]), return_counts=True)
    step = steps[steps > 0][counts[steps > 0].argmax()]
    sos = butter(2, 0.8, output="sos")
    series = []
    for ped in np.unique(peds):
        rows = data[peds == ped]
        rows = rows[rows[:, 0].argsort()]
        gaps = np.flatnonzero(np.diff(rows[:, 0]) != step) + 1
        for part in np.split(rows, gaps):
            if len(part) < 2:
                continue
            xy = part[:, 2:]
            if len(xy) > 9:
                xy = sosfiltfilt(sos, xy, axis=0)
            vel = np.gradient(xy, 0.4, axis=0)
            series.append(np.column_stack((part[:, :2], xy, vel)))
    out = np.concatenate(series)
    return out[out[:, 0].argsort(kind="stable")]


def pair_indices(rows):
    _, starts, counts = np.unique(rows[:, 0], return_index=True, return_counts=True)
    pairs = [np.array(np.triu_indices(n, 1)) + start for start, n in zip(starts, counts) if n > 1]
    return np.concatenate(pairs, axis=1)


def histograms(rows, left, right):
    other = rows[left, 1] != rows[right, 1]
    r = rows[left, 2:4] - rows[right, 2:4]
    v = rows[left, 4:6] - rows[right, 4:6]
    a = np.einsum("ij,ij->i", v, v)
    b = -np.einsum("ij,ij->i", r, v)
    rr = np.einsum("ij,ij->i", r, r)
    counts = []
    for radius in RADII:
        c = rr - radius**2
        disc = b * b - a * c
        valid = other & (c > 0) & (a > 1e-12) & (b > 0) & (disc >= 0)
        tau = c[valid] / (b[valid] + np.sqrt(disc[valid]))
        # Overlaps have tau=0 and are outside the fitted interval.
        hist = np.histogram(tau, EDGES)[0]
        hist[0] += np.count_nonzero(other & (c <= 0))
        counts.append(hist)
    return np.stack(counts), int(other.sum())


def recording(path, rng):
    states = instantaneous(path)
    left, right = pair_indices(states)
    observed, observed_pairs = histograms(states, left, right)
    reference = np.zeros_like(observed)
    reference_pairs = 0
    for _ in range(N_SHUFFLES):
        # Assign paired position/velocity samples to shuffled frame slots.
        # Frame populations are held fixed and same-person pairs are excluded.
        hist, count = histograms(states[rng.permutation(len(states))], left, right)
        reference += hist
        reference_pairs += count
    return observed, reference, observed_pairs, reference_pairs


def fit(observed, reference, observed_pairs, reference_pairs):
    # At the benchmark's lower sampling rate, smooth the 0.01 s histogram
    # by 0.03 s to limit zero-bin noise; retain both original histograms.
    x = (EDGES[:-1] + EDGES[1:]) / 2
    signal = gaussian_filter1d(observed.astype(float), sigma=3) / observed_pairs
    null = gaussian_filter1d(reference.astype(float), sigma=3) / reference_pairs
    far = (x >= 4.8) & (x < 8)
    scale = signal[far].sum() / null[far].sum() if null[far].sum() else 0
    if scale <= 0:
        return dict(exponent=None, reason="NO_TAIL_REFERENCE")
    g = np.divide(signal, null * scale, out=np.full_like(signal, np.nan), where=null > 0)
    usable = (x >= 0.4) & (x <= 2.4) & (g > 0) & (g < 1)
    if usable.sum() < 100:
        return dict(exponent=None, used_bins=int(usable.sum()), reason="INSUFFICIENT_POSITIVE_ENERGY")
    design = np.column_stack((np.ones(usable.sum()), -np.log(x[usable])))
    target = np.log(-np.log(g[usable]))
    coeff = np.linalg.lstsq(design, target, rcond=None)[0]
    for _ in range(10):
        errors = target - design @ coeff
        mad = np.median(np.abs(errors - np.median(errors)))
        if mad <= 1e-12:
            break
        weight = np.maximum(1 - (errors / (4.685 * 1.4826 * mad))**2, 0)
        if np.count_nonzero(weight) < 2:
            break
        coeff = np.linalg.lstsq(design * weight[:, None], target * weight, rcond=None)[0]
    return dict(exponent=float(coeff[1]), used_bins=int(usable.sum()), tail_scale=float(scale))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = {"method": dict(shuffles=N_SHUFFLES, bin_seconds=.01, smoothing_seconds=.03,
                             fit_seconds=[.4, 2.4], tail_seconds=[4.8, 8.0],
                             radii=RADII, C0_power=2.0), "folds": {}}
    for fold, held in HELD_OUT.items():
        sources = sorted((args.labels / fold).glob("*_train.txt"))
        assert {p.stem.removesuffix("_train") for p in sources}.isdisjoint(held)
        assert len(sources) == (6 if fold == "univ" else 7), (fold, sources)
        rows = []
        for path in sources:
            seed = int(sha256(path.name.encode()).hexdigest()[:8], 16)
            rows.append(recording(path, np.random.default_rng(seed)))
        total = [sum(group) for group in zip(*rows)]
        observed, reference, n_observed, n_reference = total
        estimates = {}
        for k, radius in enumerate(RADII):
            result = fit(observed[k], reference[k], n_observed, n_reference)
            leave_out = [
                fit(observed[k] - own[0][k], reference[k] - own[1][k],
                    n_observed - own[2], n_reference - own[3])["exponent"]
                for own in rows
            ]
            result.update(leave_one_recording_out=dict(zip([p.stem for p in sources], leave_out)),
                          observed=observed[k].tolist(), reference=reference[k].tolist())
            estimates[str(radius)] = result
        estimate = estimates["0.6"]["exponent"]
        deploy = estimate if estimate is not None and estimate > 0 and not 1.6 <= estimate <= 2.4 else 2.0
        output["folds"][fold] = dict(recordings={p.name: sha256(p.read_bytes()).hexdigest() for p in sources},
                                     held_out=sorted(held), observed_pairs=n_observed,
                                     shuffled_pairs=n_reference, fits=estimates, power=deploy)
        print(f"LAW {fold} p={estimate} deployed={deploy}", flush=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
