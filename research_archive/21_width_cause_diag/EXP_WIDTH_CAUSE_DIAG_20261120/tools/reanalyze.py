"""Recompute grouped gradients and the registered checks from archived evidence."""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "snapshot"
SEEDS = ("42", "43", "44", "45", "46")
GROUPS = ("remain", "start", "steady", "other")


def read(name):
    return json.loads((SNAPSHOT / name).read_text(encoding="utf-8-sig"))


def close(a, b, atol=1e-9):
    np.testing.assert_allclose(a, b, rtol=2e-6, atol=atol)


def read_gradients(seed, audit, saved_a, saved_t, saved_d):
    with np.load(SNAPSHOT / f"A_gradients_s{seed}.npz", allow_pickle=False) as z:
        x = {key: z[key] for key in z.files}
    assert len(x["ids"]) == len(np.unique(x["ids"])) == 9216
    np.testing.assert_array_equal(x["fit_index"], audit["fit_sample_indices"])
    gs = x["group"]
    near = np.isin(gs, ("remain", "start"))
    assert int(near.sum()) == 3216
    close(x["winner_S"] + x["loser_S"], x["S"], atol=3e-7)
    close(x["S"][:, :6].sum(axis=1), x["S"][:, -1], atol=2e-6)
    rho = {}
    for group in (*GROUPS, "stationary"):
        m = near if group == "stationary" else gs == group
        assert int(m.sum()) == saved_a[group]["n"] == saved_d[group]["n"]
        norm = x["anchor_grad_norm"][m].astype(np.float64)
        win = norm[np.arange(len(norm))[:, None], np.arange(7)[None],
                   x["winner"][m, None]]
        loser = norm.sum(2) - win
        for i, term in enumerate(x["terms"]):
            want = saved_a[group]["weighted"][term]
            s = x["S"][m, i]
            close([s.mean(), s.sum(), np.abs(s).mean()],
                  [want["mean_S"], want["sum_S"], want["mean_abs_S"]])
            close(x["template_radial"][m].sum(0)[i], saved_t[group][term], atol=5e-6)
            total = norm[:, i].sum()
            if total:
                value = loser[:, i].sum() / total
                # The original grouped sums were reduced in float32.
                close(value, want["pooled_loser_gradient_fraction"], atol=1e-5)
                if term == "total":
                    rho[group] = float(value)
            else:
                assert want["pooled_loser_gradient_fraction"] is None
        close(np.median(x["effective_candidates"][m]), saved_a[group]["effective_coverage_candidates_median"])
        widening = np.maximum(-x["S"][m, :6], 0).sum(axis=0)
        if widening.sum():
            close(widening[3]/widening.sum(), saved_d[group]["widening_share"])
        close(x["active_pairs"][m].mean(), saved_d[group]["active_pairs"])
    rare_rates = np.bincount(x["winner"][near], minlength=20) / int(near.sum())
    rare = rare_rates < 1/40
    inward = x["inward_saturated"][near][:, rare].sum() / (int(near.sum()) * rare.sum() * 6)
    return rho["remain"], float(np.median(x["effective_candidates"][gs=="remain"])), inward, rare_rates


def reanalyze():
    proof = json.loads((ROOT / "ARTIFACT_MANIFEST.json").read_text(encoding="utf-8"))["files"]
    assert set(proof) == {p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*")
                          if p.is_file() and p.name != "ARTIFACT_MANIFEST.json"}
    for name, h in proof.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == h, name
    audit, a, t, b, c, d, e, f, final = (read(name) for name in (
        "PREFLIGHT.json", "A_summary.json", "A_templates.json", "B_scale_slice.json",
        "C_reallocation.json", "D_diversity.json", "E_frame.json", "F_existing_runs.json", "RESULT.json"))
    votes, rows = {}, {}
    for seed in SEEDS:
        loser, eff, inward, win_rate = read_gradients(seed, audit, a[seed]["groups"], t[seed], d[seed])
        close(win_rate, c[seed]["fit_win_rates"])
        close(inward, a[seed]["saturation"]["1/40"]["inward_all"])
        fit, dev = b[seed]["fit"], b[seed]["dev"]
        loss = {float(alpha): row["stationary"]["total"] for alpha, row in fit.items()}
        nodiv = {float(alpha): row["stationary"]["no_diversity"] for alpha, row in fit.items()}
        for row in fit.values():
            for group in ("remain", "start", "stationary"):
                values = row[group]
                close(sum(values["weighted"].values()), values["total"])
                close(values["total"] - values["weighted"]["diversity"], values["no_diversity"])
        optimum, optimum_nodiv = min(loss, key=loss.get), min(nodiv, key=nodiv.get)
        assert (optimum,optimum_nodiv) == (b[seed]["alpha_star"], b[seed]["alpha_star_no_diversity"])
        c1, c5 = [c[seed]["variants"]["1/40"][alpha] for alpha in ("1.0","0.5")]
        assert c5["rare_slots"] == np.flatnonzero(win_rate < 1/40).tolist()
        gain = {}
        for g in ("remain", "start", "stationary"):
            base, moved = (entry["groups"][g] for entry in (c1,c5))
            close(base["minFDE"], dev["1.0"][g]["minFDE"])
            gain[g] = 1 - moved["minFDE"] / base["minFDE"]
            close(gain[g], moved["relative_minFDE_gain"])
        dd = d[seed]["stationary"]
        delta = {g: fit["0.8"][g]["geometric_unweighted"] - fit["1.0"][g]["geometric_unweighted"]
                 for g in ("start","remain")}
        votes[seed] = dict(
            M1=loser <= .10 and eff <= 1.5,
            M2=all(loss[alpha] >= loss[1.] for alpha in (.7,.8,.9)),
            M2_prime=min(v for x,v in loss.items() if x <= .9) <= .99*loss[1.],
            M3=delta["start"] > 0 and delta["remain"] <= 0,
            M4=gain["stationary"] >= .03 and gain["start"] >= -.05,
            M5=c5["reachable"] <= .5 and inward >= .30,
            M6=(dd["fit_active_pairs_all_windows"] >= .05 and dd["widening_share"] >= .30)
                or optimum-optimum_nodiv >= .1-1e-12)
        for k, v in votes[seed].items():
            assert final["per_seed"][seed][k]["passes"] == v, (seed,k)
        rows[seed] = dict(loser_gradient=loser, best_fit_alpha=optimum, c_fde_gain=gain["stationary"],
                          dev_fde_gain_at_08=1-dev["0.8"]["stationary"]["minFDE"]/dev["1.0"]["stationary"]["minFDE"],
                          fit_loss_rise_at_08=loss[.8]/loss[1.]-1)
    for k in votes["42"]:
        hits = int(sum(v[k] for v in votes.values()))
        direction = k != "M4" or all(v["c_fde_gain"] > 0 for v in rows.values())
        assert final["markers"][k]["seeds_passing"] == hits
        assert final["markers"][k]["active"] == (hits >= 4 and direction)
    assert sum(e["dev"]["angle_histogram_45_degree"]) == e["dev"]["start_windows"] - e["dev"]["zero_net_start"]
    assert len(f) == 29 and sum("archived_calib_replay" in v for v in f.values()) == 8
    for seed in ("42","43"):
        ratio = f[f"E1B_E64_s{seed}"]["stationary"]["median_spread"] / f[f"E1B_E8_s{seed}"]["stationary"]["median_spread"] - 1
        close(ratio, final["markers"]["M7"]["relative_change"][seed])
    assert final["markers"]["M7"]["active"] == all(abs(v) <= .05 for v in final["markers"]["M7"]["relative_change"].values())
    print(json.dumps(dict(status="PASS", derived_windows=len(SEEDS)*9216, markers_checked=8,
                          budget_replays=8, saved_variants=21, seeds=rows)))


if __name__ == "__main__":
    reanalyze()
