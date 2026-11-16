"""Exploration shortcut for the five-fold study: evaluate the 10 EXISTING budget models
(fold x E in {64,128}, seed 42, trained on scene_tail-train = 95%) on each fold's test set.
No training.  Same training-data protocol as EXP-156 (also 95% train + scene_tail val), so the
numbers are directly comparable with EXP-156 M2 at single seed.

Requires BUDGET_DECISION.json to exist (the test gate of the five-fold design is respected).
Writes test_explore_* files next to each budget run (never touches metrics.json / DONE.json)
and EXPLORE_BUDGET_TEST.md in --out.  Does not import anything from the five-fold code dir into
its source hash, so in-progress `final` runs can still resume.

usage: python eval_budget_on_test.py --fivefold-code <FIVEFOLD_B0_20261115/code> \
         --exec-dir <NEXT_PHASE_EXEC_20261114> --data-dir <.../processed_data_diverse> \
         --out FIVEFOLD_RUNS --device cuda:0 [--exp156 <EXP_156B.../summary.json>]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    for k in ("--fivefold-code", "--exec-dir", "--data-dir", "--out"):
        ap.add_argument(k, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--exp156")
    a = ap.parse_args()
    sys.path.insert(0, str(Path(a.fivefold_code).resolve()))
    import run_fivefold as rf                       # functions only; its source hash is not changed
    from fivefold_common import BUDGETS, BUDGET_SEED, FOLDS
    out = Path(a.out)
    if not (out / "BUDGET_DECISION.json").exists():
        raise RuntimeError("BUDGET_DECISION_MISSING: test stays closed")
    tc, _ = rf.import_exec(a.exec_dir)
    import torch
    table = {}
    for fold in FOLDS:
        train = rf.load_split(a.data_dir, fold, "train")
        tr, _ = rf.scene_tail_split(train["scene"])
        test = rf.load_split(a.data_dir, fold, "test")
        for E in BUDGETS:
            rd = out / "runs" / rf.run_name("budget", fold, E, BUDGET_SEED)
            target = rd / "test_explore_metrics.json"
            if target.exists():
                table[(fold, E)] = json.loads(target.read_text(encoding="utf-8"))["test"]["overall"]; continue
            if not (rd / "DONE.json").exists():
                raise FileNotFoundError(f"{rd} is not finished")
            ckpt = sorted(rd.glob("ckpt_epoch*.pt"))[-1]
            state = torch.load(ckpt, map_location="cpu", weights_only=False)
            if state["epoch"] != E or state["cursor"] != len(state["order"]):
                raise ValueError(f"{ckpt.name} is not the final epoch of a {E}-epoch run")
            trainer = tc.Trainer(train, tr, batch=64, epochs=E, device=a.device, seed=BUDGET_SEED, arm="B0")
            trainer.load(state, state["contract"])
            m = rf.evaluate(tc, trainer.ema.module, test, rd, "test_explore")
            target.write_text(json.dumps(dict(note="EXPLORATION: budget model (95% train, seed 42) on test",
                                              checkpoint=ckpt.name, test=m), ensure_ascii=False, indent=1), encoding="utf-8")
            table[(fold, E)] = m["overall"]
            print(f"{fold} E{E}: minADE {m['overall']['minADE']:.4f} minFDE {m['overall']['minFDE']:.4f}", flush=True)
    m2 = None
    if a.exp156:
        r = {x["fold"]: x for x in json.loads(Path(a.exp156).read_text(encoding="utf-8"))["rows"]}
        m2 = {f: (r[f]["min_ade_20"], r[f]["min_fde_20"]) for f in FOLDS}
    L = ["# 探索：预算模型直接测试（95% 训练、seed 42，单 seed）", "",
         "| 折 | B0 E64 ADE/FDE | B0 E128 ADE/FDE | EXP-156 M2 ADE/FDE |", "|---|---|---|---|"]
    for f in FOLDS:
        x, y = table[(f, 64)], table[(f, 128)]
        L.append(f"| {f} | {x['minADE']:.4f}/{x['minFDE']:.4f} | {y['minADE']:.4f}/{y['minFDE']:.4f} | "
                 + (f"{m2[f][0]:.4f}/{m2[f][1]:.4f} |" if m2 else "— |"))
    avg = lambda E, k: float(np.mean([table[(f, E)][k] for f in FOLDS]))
    L.append(f"| **等权** | **{avg(64,'minADE'):.4f}/{avg(64,'minFDE'):.4f}** | **{avg(128,'minADE'):.4f}/{avg(128,'minFDE'):.4f}** | "
             + (f"**{np.mean([m2[f][0] for f in FOLDS]):.4f}/{np.mean([m2[f][1] for f in FOLDS]):.4f}** |" if m2 else "— |"))
    text = "\n".join(L) + "\n"
    (out / "EXPLORE_BUDGET_TEST.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
