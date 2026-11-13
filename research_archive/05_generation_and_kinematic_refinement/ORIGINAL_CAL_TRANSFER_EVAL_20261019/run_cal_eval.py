# -*- coding: utf-8 -*-
"""original_CAL transfer eval: native20 / V1 / K1-S / K0-C(oracle). Quiet, THREADS=2.
Identity/track CI: BLOCKED (all track_key are '?|-1'); do not forge.
K0-C uses GT -> ORACLE_DIAGNOSTIC_ONLY upper bound only.
"""
from __future__ import annotations
import os, sys, json, time, traceback
from datetime import datetime
from pathlib import Path
import numpy as np

for _k, _v in (
    ("OMP_NUM_THREADS", "2"), ("MKL_NUM_THREADS", "2"),
    ("OPENBLAS_NUM_THREADS", "2"), ("NUMEXPR_NUM_THREADS", "2"),
    ("VECLIB_MAXIMUM_THREADS", "2"),
):
    os.environ[_k] = _v

ROOT = Path(r"<WORKSPACE_ROOT>")
EXEC = ROOT / "CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018"
CPV = ROOT / "COMPLEMENTARY_POOL_VALIDATION_20261018"
OUT = ROOT / "ORIGINAL_CAL_TRANSFER_EVAL_20261019"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "results").mkdir(exist_ok=True)
(OUT / "indices").mkdir(exist_ok=True)
(OUT / "logs").mkdir(exist_ok=True)

sys.path.insert(0, str(EXEC))
sys.path.insert(0, str(CPV))

from k0.data_io import load_bundle, load_kin, array_sha, file_sha
from k0.metrics import evaluate_set, check_W, gather
from k0.serialize import dump, eval_public, jsonable
from k0.k0c_greedy import run_k0c
from k0.config import SEEDS, ORACLE_TAG, N_NAT
from k1.train_hgb import load_arm
from k1.features113 import build_pool_features_22, pairwise_D_matrix
from k1.select_set import select_batch
from kdec.features import build_pool_features
from kdec.hgb_locked import load_hgb, score_hgb
from kdec.select_locked import select_cond_boost_11_9


def now():
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S+09:00")


def slice_split(data, ix):
    return dict(
        sample_id=data["sample_id"][ix],
        past=data["past"][ix],
        gt=data["gt"][ix],
        native=data["native"][ix],
        groups=data["groups"][ix],
        track_key=data["track_key"][ix],
        sequence=data["sequence"][ix],
        pedestrian_id=data["pedestrian_id"][ix],
        ix=ix,
    )


def prepare_pool(native, kin):
    return np.concatenate([native, kin], axis=1)


def eval_indices(pool40, indices, gt, native, groups):
    paths = gather(pool40, indices)
    ev = evaluate_set(paths, gt, native, groups, do_cluster=False)
    w = check_W(ev, groups)
    return paths, ev, w


def compact_eval(ev):
    pub = eval_public(ev)
    # keep key scalars for digest
    out = {}
    for g in ("overall", "stable", "transition", "change"):
        if g not in pub:
            continue
        r = pub[g]
        out[g] = dict(
            N=r.get("N"),
            good_count=r.get("good_count"),
            native_good_count=r.get("native_good_count"),
            coverage=r.get("coverage"),
            native_coverage=r.get("native_coverage"),
            minADE=r.get("minADE"),
            minFDE=r.get("minFDE"),
            native_minADE=r.get("native_minADE"),
            native_minFDE=r.get("native_minFDE"),
            mean_minJ=r.get("mean_minJ"),
            native_mean_minJ=r.get("native_mean_minJ"),
            conditional_lost_frac=r.get("conditional_lost_frac"),
            motion_gate=r.get("motion_gate"),
        )
    return out


def identity_audit(pack):
    tracks = pack["track_key"].astype(str)
    seqs = pack["sequence"].astype(str)
    peds = pack["pedestrian_id"]
    nuniq_t = int(len(np.unique(tracks)))
    nuniq_s = int(len(np.unique(seqs)))
    nuniq_p = int(len(np.unique(peds)))
    all_placeholder = (nuniq_t == 1 and tracks[0] == "?|-1")
    return dict(
        N=int(len(tracks)),
        n_unique_track_key=nuniq_t,
        n_unique_sequence=nuniq_s,
        n_unique_pedestrian_id=nuniq_p,
        all_track_placeholder=bool(all_placeholder),
        track_ci_status="BLOCKED_IDENTITY" if all_placeholder else "AVAILABLE",
        note=(
            "original_CAL rows sit past n_id=25912; loader fills sequence='?', ped=-1. "
            "Cannot forge track/hierarchical CI. Point metrics still valid under GT."
            if all_placeholder else "real track keys present"
        ),
    )


def score_v1(past, native, kin, model):
    fn, fk = build_pool_features(past, native, kin)
    F = fn.shape[-1]
    B = native.shape[0]
    sn = score_hgb(fn.reshape(-1, F), model).reshape(B, -1)
    sk = score_hgb(fk.reshape(-1, F), model).reshape(B, -1)
    return sn, sk


def port_rec(ev, w, **extra):
    rec = dict(
        eval=compact_eval(ev),
        W=jsonable(w),
        W_pass=bool(w["pass_"]),
    )
    rec.update(extra)
    return rec


def run_seed(seed: int, data, pack, kin, kin_src: str):
    seed_dir = OUT / "results" / f"seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    marker = seed_dir / "done.json"
    if marker.exists():
        print(f"[seed{seed}] SKIP done.json present", flush=True)
        return json.loads(marker.read_text(encoding="utf-8"))

    native = pack["native"]
    gt = pack["gt"]
    groups = pack["groups"]
    past = pack["past"]
    N = len(native)
    pool40 = prepare_pool(native, kin.astype(np.float64))
    ports = {}
    id_audit = identity_audit(pack)
    print(f"[seed{seed}] identity track_ci={id_audit['track_ci_status']}", flush=True)

    # --- native20 ---
    t0 = time.time()
    idx_n = np.tile(np.arange(N_NAT), (N, 1))
    _, ev_n, w_n = eval_indices(pool40, idx_n, gt, native, groups)
    ports["native20"] = port_rec(ev_n, w_n, wall_s=time.time() - t0)
    print(f"[seed{seed}] native20 W={w_n['pass_']} change_good={ev_n['change']['good_count']}", flush=True)

    # --- V1 cond_boost_11_9 (indices from astra2 helper; sklearn mismatch in venv-k0) ---
    v1_idx_path = OUT / "indices" / f"v1_seed{seed}_indices.npy"
    v1_meta_path = OUT / "indices" / f"v1_seed{seed}_meta.json"
    if v1_idx_path.exists():
        t0 = time.time()
        pool_idx_b = np.load(v1_idx_path).astype(np.int64)
        assert pool_idx_b.shape == (N, 20), pool_idx_b.shape
        _, ev_v, w_v = eval_indices(pool40, pool_idx_b, gt, native, groups)
        meta = {}
        if v1_meta_path.exists():
            meta = json.loads(v1_meta_path.read_text(encoding="utf-8"))
        ports["v1_cond_boost_11_9"] = port_rec(
            ev_v, w_v, wall_s=time.time() - t0,
            src=str(v1_idx_path),
            indices_sha=array_sha(pool_idx_b),
            v1_meta=meta,
            note="V1 indices selected under astra2 (sklearn 1.4) then evaluated here; no GT in select",
        )
        print(f"[seed{seed}] V1 W={w_v['pass_']} change_good={ev_v['change']['good_count']}", flush=True)
    else:
        ports["v1_cond_boost_11_9"] = dict(
            status="BLOCKED_NO_V1_INDICES",
            path=str(v1_idx_path),
            note="run run_v1_select_astra2.py with astra2 first",
        )

    # --- K1-S (frozen models) ---
    models_dir = EXEC / "k1_exec" / f"seed{seed}" / "models"
    t0 = time.time()
    models_S = load_arm(models_dir, "S", seed)
    feat40 = build_pool_features_22(past, native, kin.astype(np.float64))
    # D on the fly to limit RAM at 2879
    print(f"[seed{seed}] K1-S select N={N} (D on-the-fly)", flush=True)
    indices_s, traces_s, rejects_s = select_batch(
        pool40, native, feat40, models_S, D_cache=None, progress_every=200,
    )
    _, ev_s, w_s = eval_indices(pool40, indices_s, gt, native, groups)
    np.save(OUT / "indices" / f"k1s_seed{seed}_indices.npy", indices_s)
    np.save(OUT / "indices" / f"k1s_seed{seed}_n_swaps.npy", np.array([len(t) for t in traces_s], dtype=np.int16))
    ports["K1_S"] = port_rec(
        ev_s, w_s, wall_s=time.time() - t0,
        models_dir=str(models_dir),
        indices_sha=array_sha(indices_s),
        rejects=rejects_s,
        n_swaps_mean=float(np.mean([len(t) for t in traces_s])),
        note="frozen K1-S arm; same deploy as outer contract",
    )
    print(f"[seed{seed}] K1-S W={w_s['pass_']} change_good={ev_s['change']['good_count']} swaps={ports['K1_S']['n_swaps_mean']:.2f}", flush=True)

    # --- K0-C oracle (GT) ---
    t0 = time.time()
    print(f"[seed{seed}] K0-C ORACLE_DIAGNOSTIC_ONLY", flush=True)
    k0c = run_k0c(pool40, gt, native, groups, do_cluster=False)
    np.save(OUT / "indices" / f"k0c_seed{seed}_indices.npy", k0c["indices"])
    ports["K0_C"] = port_rec(
        k0c["eval"], k0c["W"], wall_s=time.time() - t0,
        tag=ORACLE_TAG,
        n_swaps_mean=k0c["n_swaps_mean"],
        reject_totals=k0c["reject_totals"],
        indices_sha=array_sha(k0c["indices"]),
        note="GT-using greedy policy upper bound; NOT deployable; ORACLE_DIAGNOSTIC_ONLY",
    )
    print(f"[seed{seed}] K0-C W={k0c['W']['pass_']} change_good={k0c['eval']['change']['good_count']}", flush=True)

    rec = dict(
        seed=seed,
        split="original_CAL",
        N=N,
        kin_src=kin_src,
        kin_sha=array_sha(kin),
        identity=id_audit,
        ports=ports,
        finished_at=now(),
        BREAKTHROUGH=False,
        note="Transfer point-metrics only; track CI BLOCKED by identity placeholder",
    )
    dump(marker, rec)
    return rec


def write_analysis(all_seeds):
    lines = []
    lines.append("# original_CAL 迁移补评")
    lines.append("")
    lines.append(f"- time: {now()} (Asia/Tokyo)")
    lines.append(f"- dir: `{OUT}`")
    lines.append("- KIN cache: present `(2879,20,12,2)` per seed 42/43/44")
    lines.append("- **BREAKTHROUGH=false**（本任务为收尾补评，非突破门槛）")
    lines.append("")
    lines.append("## 身份 / track CI")
    lines.append("")
    id0 = all_seeds[0]["identity"]
    lines.append(f"- track_ci_status: **{id0['track_ci_status']}**")
    lines.append(f"- n_unique_track_key={id0['n_unique_track_key']}（期望 ≫1 才可做 track CI）")
    lines.append(f"- 说明：{id0['note']}")
    lines.append("- **未伪造** track / hierarchical bootstrap CI。")
    lines.append("")
    lines.append("## 主表（点估计；W 同 00 合同）")
    lines.append("")
    lines.append("| seed | arm | W | change_good | change_cov | change_ADE | change_FDE | overall_ADE | motion_change | swaps |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|---|---:|")
    for rec in all_seeds:
        seed = rec["seed"]
        for arm in ("native20", "v1_cond_boost_11_9", "K1_S", "K0_C"):
            p = rec["ports"].get(arm, {})
            if p.get("status"):
                lines.append(f"| {seed} | {arm} | BLOCKED | — | — | — | — | — | {p.get('status')} | — |")
                continue
            ch = p["eval"]["change"]
            ov = p["eval"]["overall"]
            mg = ch.get("motion_gate") or {}
            mstat = mg.get("status", "?")
            swaps = p.get("n_swaps_mean", "—")
            if isinstance(swaps, float):
                swaps = f"{swaps:.2f}"
            lines.append(
                f"| {seed} | {arm} | {p['W_pass']} | {ch['good_count']} | {ch.get('coverage', float('nan')):.4f} | "
                f"{ch['minADE']:.5f} | {ch['minFDE']:.5f} | {ov['minADE']:.5f} | {mstat} | {swaps} |"
            )
    lines.append("")
    lines.append("## 读法")
    lines.append("")
    lines.append("- **native20 / V1 / K1-S**：可部署或历史选择器对照（无 GT 选择）。")
    lines.append("- **K0-C**：`ORACLE_DIAGNOSTIC_ONLY`，用 GT 的贪心策略上界；不可当作迁移成功。")
    lines.append("- 若 K0-C 达 W 而 K1-S/V1 未达：迁移上仍是「池可达、选择器未达」，与 outer 同型。")
    lines.append("- track CI 缺失 ⇒ **不能**声称 CAL 上的稳健区间结论；仅报告点估计。")
    lines.append("")
    # W summary
    lines.append("## W 摘要")
    lines.append("")
    for arm in ("native20", "v1_cond_boost_11_9", "K1_S", "K0_C"):
        flags = []
        for rec in all_seeds:
            p = rec["ports"].get(arm, {})
            flags.append(str(p.get("W_pass", p.get("status", "?"))))
        lines.append(f"- {arm}: per-seed W = {', '.join(flags)}")
    lines.append("")
    lines.append("## 产物")
    lines.append("")
    lines.append(f"- `{OUT / 'RESULT_ANALYSIS.json'}`")
    lines.append(f"- `{OUT / 'RESULT_ANALYSIS.md'}`")
    lines.append(f"- `{OUT / 'STATUS.json'}`")
    lines.append(f"- indices under `{OUT / 'indices'}`")
    md = "\n".join(lines) + "\n"
    (OUT / "RESULT_ANALYSIS.md").write_text(md, encoding="utf-8")
    dump(OUT / "RESULT_ANALYSIS.json", dict(
        created_at=now(), timezone="Asia/Tokyo",
        BREAKTHROUGH=False, seeds=all_seeds,
    ))
    dump(OUT / "STATUS.json", dict(
        status="DONE", BREAKTHROUGH=False,
        finished_at=now(),
        track_ci="BLOCKED_IDENTITY",
        n_seeds=len(all_seeds),
    ))


def main():
    print(f"START original_CAL eval {now()}", flush=True)
    data = load_bundle()
    ix = data["split"]["original_CAL"]
    assert len(ix) == 2879, len(ix)
    pack = slice_split(data, ix)
    all_seeds = []
    for seed in SEEDS:
        print(f"==== seed {seed} ====", flush=True)
        kin, src = load_kin("original_CAL", seed, data)
        if kin is None:
            raise RuntimeError(f"missing KIN cache seed{seed}: {src}")
        assert kin.shape == (2879, 20, 12, 2), kin.shape
        rec = run_seed(seed, data, pack, kin, src)
        all_seeds.append(rec)
    write_analysis(all_seeds)
    # append overnight log
    logp = EXEC / "OVERNIGHT_LOG.md"
    with logp.open("a", encoding="utf-8") as f:
        f.write(f"\n- {now()}: original_CAL transfer eval DONE BREAKTHROUGH=false; track_ci=BLOCKED_IDENTITY; see {OUT}\n")
    print(f"DONE {now()}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        err = traceback.format_exc()
        (OUT / "FATAL.txt").write_text(err, encoding="utf-8")
        print(err, flush=True)
        raise
