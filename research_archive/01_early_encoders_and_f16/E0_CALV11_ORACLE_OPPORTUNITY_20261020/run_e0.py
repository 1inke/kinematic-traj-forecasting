# -*- coding: utf-8 -*-
"""E0: cal_v11 clean-population oracle opportunity re-estimation. ZERO training.

Question (Gate 1 of the preserve+complement distillation direction):
  After removing CAL windows that share tracks/time with FIT/inner/outer
  (author-frozen v1.1 partition, qoder_work/v1_1/SPLIT_V1_1.json,
  state=FROZEN_BY_AUTHOR), does the native∪KIN oracle K20 combination
  (K0-C, ORACLE_DIAGNOSTIC_ONLY) still show a coverage opportunity over
  native K20?

Populations:
  clean   = cal_v11 (1620 windows, 443 change, 77 tracks)
  dropped = original_CAL \ cal_v11 (1259 windows, 257 change) -- contrast only

Arms per seed (42/43/44): native20 (frozen, seed-independent), K0-C oracle.
No GT-free learner arms in E0 (Gate 1 only asks whether the opportunity exists).

Inference: paired cluster bootstrap over change tracks (PCG64 20261020,
20000 draws), per-seed and seed-averaged. cal_v11 supplies real track labels,
so the BLOCKED_IDENTITY limitation of the original_CAL eval does not apply here.

Read-only on every project input; SHA-256 recorded before and after the run.
Writes only inside this directory.
"""
from __future__ import annotations
import hashlib
import json
import sys
import time
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    import os
    os.environ.setdefault(_k, "2")

ROOT = Path(r"<WORKSPACE_ROOT>")
EXEC = ROOT / "CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018"
V11 = ROOT / "qoder_work" / "v1_1"
OUT = Path(__file__).resolve().parent

sys.path.insert(0, str(EXEC))

from k0.data_io import load_bundle, load_kin, array_sha, file_sha  # noqa: E402
from k0.metrics import evaluate_set, check_W, gather  # noqa: E402
from k0.k0c_greedy import run_k0c  # noqa: E402
from k0.config import SEEDS, N_NAT, ORACLE_TAG  # noqa: E402

INPUTS = [
    ROOT / "REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz",
    ROOT / "F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/labels/train.npz",
    ROOT / "GENERATION_GROUPED_TRANSFER_SERIES_20261006/identity_audit/splits.npz",
    ROOT / "GENERATION_NATIVE_DECODER_SERIES/train_encoder.npz",
    V11 / "cal_v11_index.npy",
    V11 / "cal_v11_windows.csv",
    V11 / "SPLIT_V1_1.json",
] + [ROOT / f"UNION_SELECTOR_K40_TO_K20_20261018/results/cache_pred/kin_original_CAL_seed{s}.npy"
     for s in SEEDS]

RNG_SEED = 20261020
N_BOOT = 20000


def now():
    return datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%dT%H:%M:%S+09:00")


def load_v11():
    idx = np.load(V11 / "cal_v11_index.npy").astype(np.int64)  # absolute registry ids
    import csv
    tracks = []
    with (V11 / "cal_v11_windows.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tracks.append(row["track"])
    return idx, np.array(tracks, dtype=object)


def slice_pack(data, registry_ix):
    return dict(
        gt=data["gt"][registry_ix],
        native=data["native"][registry_ix],
        groups=data["groups"][registry_ix],
    )


def boot_delta(delta_by_seed, track_by_seed, rng, n_boot=N_BOOT):
    """Cluster bootstrap over tracks of mean delta; returns per-seed and
    seed-averaged 95% percentile CIs. delta_by_seed[s]: (N_change,) window
    deltas; track_by_seed[s]: (N_change,) track labels."""
    per_seed = {}
    n_seeds = len(delta_by_seed)
    acc = [[] for _ in range(n_seeds)]
    avg = []
    uniq_per_seed = [np.unique(t) for t in track_by_seed]
    for _ in range(n_boot):
        tot = 0.0
        for s in range(n_seeds):
            uniq = uniq_per_seed[s]
            draw = rng.choice(uniq, size=len(uniq), replace=True)
            d, t = delta_by_seed[s], track_by_seed[s]
            vals = np.concatenate([d[t == u] for u in draw])
            m = float(vals.mean())
            acc[s].append(m)
            tot += m
        avg.append(tot / n_seeds)
    def ci(x):
        x = np.asarray(x)
        return dict(mean=float(x.mean()), lo=float(np.percentile(x, 2.5)),
                    hi=float(np.percentile(x, 97.5)))
    return dict(per_seed={str(s): ci(acc[s]) for s in range(n_seeds)},
                seed_avg=ci(avg))


def main():
    t_start = time.time()
    print(f"START E0 cal_v11 oracle opportunity {now()}", flush=True)
    sha_before = {str(p.relative_to(ROOT)): file_sha(p) for p in INPUTS}

    idx_abs, tracks_all = load_v11()
    cal_registry = np.arange(25912, 28791)
    rel = idx_abs - 25912
    assert np.all(cal_registry[rel] == idx_abs), "cal_v11 index alignment failed"
    dropped_abs = np.setdiff1d(cal_registry, idx_abs)

    data = load_bundle()
    cal_ix = data["split"]["original_CAL"]
    assert len(cal_ix) == 2879

    pops = dict(
        clean=dict(registry=idx_abs, rel=rel, tracks=tracks_all),
        dropped=dict(registry=dropped_abs, rel=dropped_abs - 25912, tracks=None),
    )
    import os as _os
    smoke = int(_os.environ.get("E0_SMOKE", "0"))
    if smoke:
        for p in pops.values():
            p["registry"] = p["registry"][:smoke]
            p["rel"] = p["rel"][:smoke]
            if p["tracks"] is not None:
                p["tracks"] = p["tracks"][:smoke]
        print(f"SMOKE mode: {smoke} windows per population", flush=True)
    packs = {name: slice_pack(data, p["registry"]) for name, p in pops.items()}
    for name, pk in packs.items():
        n_ch = int((pk["groups"] == 2).sum())
        print(f"pop {name}: N={len(pk['groups'])} change={n_ch}", flush=True)

    results = {}
    for seed in SEEDS:
        print(f"==== seed {seed} ====", flush=True)
        kin_full, kin_src = load_kin("original_CAL", seed, data)
        assert kin_full is not None and kin_full.shape == (2879, 20, 12, 2)
        seed_rec = {"kin_src": kin_src, "kin_sha": array_sha(kin_full)}
        for name, pk in packs.items():
            rel_ix = pops[name]["rel"]
            kin = kin_full[rel_ix].astype(np.float64)
            native, gt, groups = pk["native"], pk["gt"], pk["groups"]
            pool40 = np.concatenate([native, kin], axis=1)

            idx_n = np.tile(np.arange(N_NAT), (len(native), 1))
            ev_n = evaluate_set(gather(pool40, idx_n), gt, native, groups)
            if "change" not in ev_n:
                seed_rec[name] = dict(status="SMOKE_NO_CHANGE_WINDOWS")
                continue

            t0 = time.time()
            k0c = run_k0c(pool40, gt, native, groups, do_cluster=False)
            np.save(OUT / f"k0c_indices_{name}_seed{seed}.npy", k0c["indices"])

            ev_o, w_o = k0c["eval"], k0c["W"]
            ch_n, ch_o = ev_n["change"], ev_o["change"]
            seed_rec[name] = dict(
                N=int(len(groups)),
                change_N=int(ch_n["N"]),
                native=dict(change_good=ch_n["native_good_count"],
                            change_cov=ch_n["native_coverage"],
                            change_minADE=ch_n["native_minADE"],
                            change_minFDE=ch_n["native_minFDE"]),
                k0c=dict(change_good=ch_o["good_count"],
                         change_cov=ch_o["coverage"],
                         change_minADE=ch_o["minADE"],
                         change_minFDE=ch_o["minFDE"],
                         change_rescue=ch_o["rescue"],
                         change_lost=ch_o["lost"],
                         n_swaps_mean=k0c["n_swaps_mean"],
                         reject_totals=k0c["reject_totals"],
                         W_pass=bool(w_o["pass_"]),
                         wall_s=time.time() - t0),
                delta_change_good=int(ch_o["good_count"] - ch_n["native_good_count"]),
                tag=ORACLE_TAG,
            )
            # per-window paired deltas on change windows (for bootstrap)
            pw_n, pw_o = ev_n["_per_window"], ev_o["_per_window"]
            ch_mask = groups == 2
            d = (pw_o["covered"][ch_mask].astype(float)
                 - pw_n["native_covered"][ch_mask].astype(float))
            np.save(OUT / f"delta_change_{name}_seed{seed}.npy", d)
            print(f"[seed{seed}][{name}] native cg={ch_n['native_good_count']} "
                  f"k0c cg={ch_o['good_count']} Δ={seed_rec[name]['delta_change_good']} "
                  f"W={w_o['pass_']} ({time.time()-t0:.0f}s)", flush=True)
        results[str(seed)] = seed_rec

    # ---- track-clustered bootstrap on the clean population -----------------
    rng = np.random.default_rng(RNG_SEED)
    delta_by_seed, track_by_seed = [], []
    boot = None
    ch_tracks = np.array([], dtype=object)
    if all((OUT / f"delta_change_clean_seed{s}.npy").exists() for s in SEEDS):
        for seed in SEEDS:
            d = np.load(OUT / f"delta_change_clean_seed{seed}.npy")
            delta_by_seed.append(d)
            pk = packs["clean"]
            tr = pops["clean"]["tracks"][pk["groups"] == 2]
            track_by_seed.append(tr)
        ch_tracks = np.unique(track_by_seed[0])
        boot = boot_delta(delta_by_seed, track_by_seed, rng)
        print(f"bootstrap change-track clusters={len(ch_tracks)} "
              f"seed_avg mean={boot['seed_avg']['mean']*100:.2f}pp "
              f"CI=[{boot['seed_avg']['lo']*100:.2f},{boot['seed_avg']['hi']*100:.2f}]pp",
              flush=True)

    # same bootstrap on dropped subset as exposure contrast (exploratory)
    # tracks unknown for dropped (registry placeholders) -> use track labels
    # from v1.1 excluded list is impossible per-window; skip CI, point only.

    sha_after = {str(p.relative_to(ROOT)): file_sha(p) for p in INPUTS}
    unchanged = sha_before == sha_after

    rec = dict(
        experiment="E0_calv11_oracle_opportunity",
        created_at=now(), wall_s=time.time() - t_start,
        split_ref="qoder_work/v1_1/SPLIT_V1_1.json (FROZEN_BY_AUTHOR, CAL scope)",
        rng_seed=RNG_SEED, n_boot=N_BOOT,
        n_change_tracks_clean=int(len(ch_tracks)),
        results=results,
        bootstrap_clean_change=boot,
        inputs_unchanged=unchanged,
        inputs_sha_before=sha_before, inputs_sha_after=sha_after,
        note="K0-C uses GT: ORACLE_DIAGNOSTIC_ONLY. No training performed.",
    )
    (OUT / "RESULT.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"DONE {now()} inputs_unchanged={unchanged}", flush=True)
    return rec


if __name__ == "__main__":
    try:
        main()
    except Exception:
        err = traceback.format_exc()
        (OUT / "FATAL.txt").write_text(err, encoding="utf-8")
        print(err, flush=True)
        raise
