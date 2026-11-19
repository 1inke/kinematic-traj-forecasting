"""Verify the Phase 20 archive and recompute summaries from derived windows."""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CAL = ROOT / "snapshot/SET_CALIB_DIAG"
REF = ROOT / "snapshot/REFINE"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load(path, matrices=False):
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files}
    assert len(np.unique(out["ids"])) == len(out["ids"]) == 4753
    assert all(x.shape == (4753,) or (matrices and x.shape == (4753, 20)) for x in out.values())
    return out


def close(a, b):
    np.testing.assert_allclose(a, b, atol=1e-12, rtol=1e-12)


def masks(w):
    labels = w["maneuver"]
    out = {"overall": np.ones(len(labels), dtype=bool),
           "turn_any": np.isin(labels, ["turn", "turn_and_speed"]),
           "decel_stop": np.isin(labels, ["deceleration", "stop"])}
    out.update({k: w["group"] == i for i, k in enumerate(("stable", "transition", "change"))})
    out.update({k: labels == k for k in np.unique(labels)})
    out.update({f"obs_{k}": w["observation"] == k for k in np.unique(w["observation"])})
    if "reference_bin" in w:
        out.update({f"bin_{k}": w["reference_bin"] == k for k in np.unique(w["reference_bin"])})
    return {k: v for k, v in out.items() if v.any()}


def calibration():
    r = read(CAL / "RESULT.json")
    n, b0, betas, coverage = 0, [], [], []
    for model, population in r["runs"].items():
        for seed, run in population.items():
            w = load(CAL / f"windows/{model}_s{seed}.npz")
            tracks = np.array([v.rsplit("|", 1)[0] for v in w["ids"]])
            assert not set(tracks[w["half"] == 0]) & set(tracks[w["half"] == 1])
            groups = masks(w)
            for t, row in run.items():
                for mode, skill in row["spread_skill"].items():
                    prefix = f"{t}__{mode}__"
                    s, err = w[prefix+"s"], w[prefix+"r"]
                    idx = np.searchsorted(skill["cuts_m"], s, side="right") + 1
                    ss, rr = [], []
                    for b in skill["bins"]:
                        g = idx == b["bin"]
                        assert int(g.sum()) == b["n"]
                        ss.append(np.sqrt(np.mean(s[g] ** 2)))
                        rr.append(np.sqrt(np.mean(err[g] ** 2)))
                        close([ss[-1], rr[-1]], [b["S_m"], b["R_m"]])
                    beta, intercept = np.polyfit(np.log(ss), np.log(rr), 1)
                    close([beta, intercept], [skill["beta"], skill["intercept"]])
                    if model == "B0" and t == "t12" and mode == "equal":
                        betas.append(beta)
                    for variant, result in row["hpd"][mode]["variants"].items():
                        u = w[prefix+"u__"+variant]
                        area = w[prefix+"area90__"+variant]
                        assert np.all((0 <= u) & (u <= 1)) and np.all(area > 0)
                        for name, g in groups.items():
                            stored = result["groups"][name]
                            selected = (u[g, None] <= r["levels"]).mean(0)
                            close(selected, stored["coverage"])
                            close(area[g].mean(), stored["mean_HPD90_area_m2"])
                            if (model, t, mode, variant, name) == ("B0", "t12", "equal", "global", "overall"):
                                coverage.append(selected[2])
                for name, g in groups.items():
                    q = row["quantizer"][name]
                    assert q["n"] == int(g.sum())
                    new = [w[f"{t}__{k}"][g].mean()
                           for k in ("minADE", "minFDE", "reference_minADE", "reference_minFDE")]
                    close(new, [q[k] for k in ("model_minADE_m", "model_minFDE_m",
                                              "reference_minADE_m", "reference_minFDE_m")])
                    if model == "B0" and t == "t12" and name == "bin_stationary":
                        b0.append(new[1]/new[3]-1)
                n += len(w["ids"])
    assert len(b0) == len(betas) == len(coverage) == 5
    assert .8 <= np.mean(betas) <= 1.25 and sum(.8 <= x <= 1.25 for x in betas) >= 4
    assert .85 <= np.mean(coverage) <= .95 and sum(.85 <= x <= .95 for x in coverage) >= 4
    assert np.mean(b0) >= .05 and sum(x >= .05 for x in b0) >= 4
    assert r["decisions"]["conclusion"] == "RANGE_OK_PLACEMENT_BAD"
    return n


def mechanism():
    r = read(REF / "MECH_DIAG/RESULT.json")
    n, markers = 0, {name: 0 for name in r["markers"]}
    for seed, row in r["seeds"].items():
        w = load(REF / f"MECH_DIAG/windows/E3_self_s{seed}.npz", matrices=True)
        first, second = w["first_FDE_winner"], w["second_FDE_winner"]
        ix = np.arange(len(first))
        gain = w["first_minFDE"] - w["second_minFDE"]
        precision = np.where(first == second, gain, 0)
        close(precision, w["precision_gain_m"])
        close(gain - precision, w["redistribution_gain_m"])
        delta = w["endpoint_shift_m"]
        close(delta, np.hypot(w["longitudinal_shift_m"], w["lateral_shift_m"]))
        gap_pre = np.abs(w["path_shape_first_m"][ix, first] - w["path_shape_truth_m"])
        gap_post = np.abs(w["path_shape_second_m"][ix, second] - w["path_shape_truth_m"])
        mg = masks(w)
        for name, g in mg.items():
            actual = row["groups"][name]
            assert actual["n"] == int(g.sum())
            for key in ("first_minADE", "second_minADE", "B0_minADE", "first_minFDE", "second_minFDE", "B0_minFDE"):
                close(actual[key], w[key][g].mean())
            close(actual["same_winner_fraction"], np.mean((first == second)[g]))
            close(actual["shape"]["first_mean_gap_m"], gap_pre[g].mean())
            close(actual["shape"]["second_mean_gap_m"], gap_post[g].mean())
            close(actual["candidates"]["median_norm_m"], np.median(delta[g]))
            close(actual["candidates"]["saturation"], w["saturation_fraction"][g].mean())
        straight = mg["steady"]
        static = mg["obs_stationary"]
        turn = mg["turn"]
        harmed = turn & (gain < 0)
        outer = turn[:, None] & (w["radial_quartile"] == 4)
        inward = w["first_winner_outer_and_inward"]
        s_gain, p_gain = gain[straight].sum(), precision[straight].sum()
        markers["STEADY_PRECISION"] += (s_gain > 0 and p_gain > 0 and
            p_gain / s_gain >= .60 and w["precision_longitudinal_m"][straight].sum() / p_gain >= .60)
        markers["SHAPE_REGULARIZED"] += 1 - gap_post[straight].sum() / gap_pre[straight].sum() >= .20
        markers["LOWSPEED_OVERSHIFT"] += np.median(delta[static]) >= np.median(w["B0_minFDE"][static])
        markers["TAIL_CONTRACTION"] += (w["radial_shift_in_scale"][outer].mean() <= -.05
                                         and harmed.any() and inward[harmed].mean() >= .50)
        markers["STRAIGHTENING"] += w["abs_lateral_change_m"][turn].mean() <= -.03
        markers["DIVISION_OF_LABOR"] += w["first_minFDE"].mean()/w["B0_minFDE"].mean() - 1 >= .02
        n += len(first)
    for name, v in markers.items():
        assert v == r["markers"][name]["agreeing_seeds"]
        assert bool(v >= 2) == r["markers"][name]["active"]
    return n


def ablations():
    r = read(REF / "ABL/RESULT.json")
    n, epochs, updates = 0, 0, 0
    for arm, runs in r["runs"].items():
        for seed, result in runs.items():
            folder = REF / f"ABL/runs/{arm}_s{seed}"
            assert result == read(folder / "RESULT.json")
            hist = read(folder / "TRAINING.json")["history"]
            old = read(ROOT / f"analysis_inputs/A1_s{seed}_archive18.json")["history"]
            assert len(hist) == len(old) == 20
            for i, (a, b) in enumerate(zip(hist, old), 1):
                assert a["epoch"] == i and a["updates"] == i*85 and a["order_hash"] == b["order_hash"]
                assert read(folder / f"checkpoints/epoch{i:02d}.json")["epoch"] == i
                if arm == "A1":
                    assert a["loss"] == b["loss"]
            w = load(folder / "window_metrics.npz")
            for name, g in masks(w).items():
                record = result["clean"][name]
                assert record["n"] == int(g.sum())
                for key in ("minFDE", "minADE", "top1FDE", "saturation_fraction", "mean_shift", "uncovered"):
                    close(record[key], w[key][g].mean())
            n += len(w["ids"])
            epochs += len(hist)
            updates += hist[-1]["updates"]
    for arm, bygroup in r["paired"].items():
        for name, comparison in bygroup.items():
            for metric, recorded in comparison.items():
                base = r["runs"]["A1"]
                ratios = [1 - row["clean"][name][metric]/base[s]["clean"][name][metric]
                          for s, row in r["runs"][arm].items()]
                close(ratios, recorded["values"])
                close(np.mean(ratios), recorded["mean"])
    assert epochs == 300 and updates == 25500
    decision = r["decision"]
    assert decision["mode_and_geometry_needed"] and decision["centroid_within_band"] and not decision["S1_lowspeed"]
    return n, epochs, updates


def main():
    files = read(ROOT / "ARTIFACT_MANIFEST.json")["files"]
    assert set(files) == {p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*")
                          if p.is_file() and p.name != "ARTIFACT_MANIFEST.json"}
    for name, digest in files.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    cal = calibration()
    mech = mechanism()
    abl, epochs, updates = ablations()
    print(json.dumps(dict(status="PASS", manifest_files=len(files), horizon_window_records=cal,
                          mechanism_windows=mech, ablation_windows=abl, paired_epochs=epochs,
                          optimizer_updates=updates)))


if __name__ == "__main__":
    main()
