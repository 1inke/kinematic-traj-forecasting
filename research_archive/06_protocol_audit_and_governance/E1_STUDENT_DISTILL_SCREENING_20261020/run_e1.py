# -*- coding: utf-8 -*-
"""E1: five-arm student distillation screening (retain + complement).

Gate: only run if E0 shows the oracle opportunity persists on cal_v11.

Student: kdec DirectPositionDecoder (SharedTrunk + K20 queries, ~57k params,
single forward pass). All arms share architecture, budget, and base loss.
Only the teacher supervision differs (A/B/C/D/E per DESIGN.md).

Teachers: frozen native/KIN caches on FIT windows; FIT GT decides "good"
(ADE<=0.2 & FDE<=0.4). GT is used for training supervision only, never for
selection or evaluation decisions.

Eval: inner split (track-disjoint from FIT), seed 42, point estimates.
Reads project inputs read-only; writes only inside this directory.
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
KDEC = ROOT / "HYBRID_NATIVE_KIN_UNION_20261018"
OUT = Path(__file__).resolve().parent

sys.path.insert(0, str(EXEC))
sys.path.insert(0, str(KDEC))

import torch  # noqa: E402

from k0.data_io import load_bundle, load_kin, file_sha, array_sha  # noqa: E402
from k0.metrics import evaluate_set  # noqa: E402
from kdec import models as M  # noqa: E402
from kdec.train import winner_loss, diversity_hinge, seed_all, predict  # noqa: E402

import os
SEED = int(os.environ.get("E1_SEED", "42"))
ARMS = os.environ.get("E1_ARMS", "ABCDE")
EPOCHS = 8
BATCH = 256
LR = 1e-3
LAMBDA = 1.0
QUALITY = (0.2, 0.4)

INPUTS = [
    ROOT / "REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz",
    ROOT / "F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/labels/train.npz",
    ROOT / "GENERATION_GROUPED_TRANSFER_SERIES_20261006/identity_audit/splits.npz",
    ROOT / "GENERATION_NATIVE_DECODER_SERIES/train_encoder.npz",
    ROOT / f"UNION_SELECTOR_K40_TO_K20_20261018/results/cache_pred/kin_FIT_seed{SEED}.npy",
]


def now():
    return datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%dT%H:%M:%S+09:00")


def path_dist(a, b):
    """a (B,T,H,2), b (B,K,H,2) -> (B,T,K): mean L2 + 0.5*final L2."""
    d = torch.linalg.vector_norm(a[:, :, None] - b[:, None], dim=-1)
    return d.mean(-1) + 0.5 * d[..., -1]


def cover_loss(paths, teachers, tmask):
    """One-directional Chamfer: each teacher path pulls nearest student slot.
    paths (B,K,H,2), teachers (B,T,H,2), tmask (B,T) bool."""
    if teachers.shape[1] == 0:
        return torch.zeros((), device=paths.device)
    d = path_dist(teachers, paths)  # (B,T,K)
    dmin = d.min(-1).values  # (B,T)
    denom = tmask.float().sum().clamp(min=1.0)
    return (dmin * tmask.float()).sum() / denom


def build_teachers(native, kin, gt):
    """Return padded teacher arrays + masks for B/C/D sets on FIT windows."""
    def good_mask(paths):
        e = np.linalg.norm(paths - gt[:, None], axis=-1)
        ade, fde = e.mean(-1), e[..., -1]
        return (ade <= QUALITY[0]) & (fde <= QUALITY[1])
    ng, kg = good_mask(native), good_mask(kin)
    native_covered = ng.any(1)
    B = len(native)
    T = 40
    t_all = np.zeros((B, T, 12, 2), np.float32)
    m_all = np.zeros((B, T), bool)
    t_ret = np.zeros_like(t_all); m_ret = np.zeros_like(m_all)
    t_cmp = np.zeros_like(t_all); m_cmp = np.zeros_like(m_all)
    both = np.concatenate([native, kin], 1)
    both_good = np.concatenate([ng, kg], 1)
    for i in range(B):
        g = both[i][both_good[i]]
        t_all[i, : len(g)] = g; m_all[i, : len(g)] = True
        if native_covered[i]:
            r = native[i][ng[i]]
            t_ret[i, : len(r)] = r; m_ret[i, : len(r)] = True
        else:
            c = g  # union-good on native-failed windows
            t_cmp[i, : len(c)] = c; m_cmp[i, : len(c)] = True
    stats = dict(
        native_covered_frac=float(native_covered.mean()),
        teachers_all_mean=float(m_all.sum(1).mean()),
        teachers_retain_mean=float(m_ret.sum(1).mean()),
        teachers_complement_mean=float(m_cmp.sum(1).mean()),
        complement_windows=int((~native_covered).sum()),
    )
    return (t_all, m_all), (t_ret, m_ret), (t_cmp, m_cmp), stats


def train_arm(arm, data, teachers, ix_train):
    seed_all(SEED)
    net = M.build("BASE").to("cuda:0")
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    (t_all, m_all), (t_ret, m_ret), (t_cmp, m_cmp) = teachers
    fit_pos = np.full(len(data["sample_id"]), -1, dtype=np.int64)
    fit_pos[ix_train] = np.arange(len(ix_train))
    def to_t(x):
        return torch.from_numpy(x).to("cuda:0")
    history = []
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        net.train()
        order = ix_train[np.random.permutation(len(ix_train))]
        agg = {}
        for s in range(0, len(order), BATCH):
            bi = order[s: s + BATCH]
            lb = fit_pos[bi]  # FIT-local rows for teacher arrays
            past = to_t(data["past"][bi].astype(np.float32))
            gt = to_t(data["gt"][bi].astype(np.float32))
            out = net(past)
            paths = out["paths"]
            loss = winner_loss(paths, gt) + 0.05 * diversity_hinge(paths)
            parts = {}
            if arm == "B":
                loss = loss + LAMBDA * cover_loss(paths, to_t(t_all[lb]), to_t(m_all[lb]))
            elif arm == "C":
                loss = loss + LAMBDA * cover_loss(paths, to_t(t_ret[lb]), to_t(m_ret[lb]))
            elif arm == "D":
                loss = loss + LAMBDA * cover_loss(paths, to_t(t_cmp[lb]), to_t(m_cmp[lb]))
            elif arm == "E":
                loss = (loss
                        + LAMBDA * cover_loss(paths, to_t(t_ret[lb]), to_t(m_ret[lb]))
                        + LAMBDA * cover_loss(paths, to_t(t_cmp[lb]), to_t(m_cmp[lb])))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            parts["loss"] = float(loss.detach())
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v * len(bi)
        history.append({"epoch": ep, **{k: v / len(order) for k, v in agg.items()}})
        print(f"[{arm}] epoch {ep}/{EPOCHS} loss={history[-1]['loss']:.4f}", flush=True)
    elapsed = time.time() - t0
    return net, history, elapsed


def main():
    t_start = time.time()
    print(f"START E1 screening {now()}", flush=True)
    sha_before = {str(p.relative_to(ROOT)): file_sha(p) for p in INPUTS}

    data = load_bundle()
    kin_fit, kin_src = load_kin("FIT", SEED, data)
    fit_ix = data["split"]["FIT"]
    inner_ix = data["split"]["inner"]

    teachers = build_teachers(data["native"][fit_ix], kin_fit, data["gt"][fit_ix])
    t_stats = teachers[3]
    print("teacher stats:", json.dumps(t_stats), flush=True)

    arms = list(ARMS)
    results = {}
    for arm in arms:
        net, hist, elapsed = train_arm(arm, data, teachers[:3], fit_ix)
        paths_inner = predict(net, data["past"][inner_ix])
        ev = evaluate_set(paths_inner, data["gt"][inner_ix],
                          data["native"][inner_ix], data["groups"][inner_ix])
        row = {}
        for g in ("overall", "stable", "transition", "change"):
            if g not in ev:
                continue
            r = ev[g]
            row[g] = dict(N=r["N"], good_count=r["good_count"],
                          coverage=r["coverage"],
                          native_good_count=r["native_good_count"],
                          native_coverage=r["native_coverage"],
                          minADE=r["minADE"], minFDE=r["minFDE"],
                          rescue=r["rescue"], lost=r["lost"],
                          motion_gate=r["motion_gate"]["status"])
        results[arm] = dict(eval=row, history=hist, elapsed_s=elapsed,
                            params=M.count_params(net))
        torch.save({"state_dict": net.state_dict(), "arm": arm, "seed": SEED},
                   OUT / f"student_{arm}_seed{SEED}.pt")
        ch = row.get("change", {})
        print(f"[{arm}] change cov={ch.get('coverage'):.4f} good={ch.get('good_count')} "
              f"ADE={ch.get('minADE'):.4f} FDE={ch.get('minFDE'):.4f} "
              f"motion={ch.get('motion_gate')}", flush=True)

    sha_after = {str(p.relative_to(ROOT)): file_sha(p) for p in INPUTS}
    rec = dict(
        experiment="E1_student_distill_screening",
        created_at=now(), wall_s=time.time() - t_start,
        seed=SEED, epochs=EPOCHS, batch=BATCH, lr=LR, lambda_teach=LAMBDA,
        teacher_stats=t_stats, kin_src=kin_src,
        results=results,
        inputs_unchanged=sha_before == sha_after,
        inputs_sha_before=sha_before, inputs_sha_after=sha_after,
        note="Screening on inner only; point estimates; no outer/CAL consumed.",
    )
    (OUT / f"RESULT_seed{SEED}.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"DONE {now()} inputs_unchanged={rec['inputs_unchanged']}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        err = traceback.format_exc()
        (OUT / "FATAL.txt").write_text(err, encoding="utf-8")
        print(err, flush=True)
        raise
