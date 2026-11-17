"""Orchestration test with a FAKE torch / train_core (no real model). Run: python test_orchestration_fake.py

Checks: phase order and gating (final refuses before decide; test arrays never opened in
budget), DONE-skip, per-epoch checkpoint + exact resume after a simulated crash (the batch-order
hash chain and final outputs equal an uninterrupted run), write-once outputs, summary end-to-end.
"""
import json
import os
import pickle
import random
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

FAKE_TORCH = '''
import pickle
__version__ = "fake"
class _V: cuda = None
version = _V()
class _C:
    @staticmethod
    def is_available(): return False
cuda = _C()
def load(path, map_location=None, weights_only=None):
    with open(path, "rb") as f: return pickle.load(f)
'''

FAKE_TRAIN_CORE = '''
import os, pickle, numpy as np
from pathlib import Path
OPENED = []
class _P:
    def __init__(s, n): s.n = n
    def numel(s): return s.n
class _M:
    def parameters(s): return [_P(295474)]
class _EMA:
    def __init__(s): s.module = _M()
class Trainer:
    def __init__(s, data, indices, batch=64, epochs=8, device="cpu", seed=42, order_seed=None, arm="B0"):
        s.data, s.indices, s.batch, s.epochs = data, np.asarray(indices), batch, epochs
        s.model, s.ema = _M(), _EMA()
        s.steps_per_epoch = -(-len(indices) // batch)
        s.rng = np.random.default_rng(seed); s.epoch = s.cursor = s.updates = 0
        s.order = None; s.history = []; s.batch_orders = []; s.connected = {"w"}
        s.crash_after = int(os.environ.get("FAKE_CRASH_AFTER", "-1"))
    def step(s):
        if s.order is None or s.cursor == len(s.order):
            if s.epoch >= s.epochs: raise StopIteration
            s.order = s.indices[s.rng.permutation(len(s.indices))]; s.cursor = 0; s.epoch += 1
        idx = s.order[s.cursor:s.cursor + s.batch]
        s.updates += 1; s.cursor += len(idx)
        s.batch_orders.append(s.data["ids"][idx].tolist())
        s.history.append(dict(loss=1.0 / s.updates, lr=1e-4))
        if s.updates == s.crash_after: raise RuntimeError("SIMULATED_CRASH")
    def snapshot(s, contract):
        return dict(contract=contract, rng=s.rng.bit_generator.state, epoch=s.epoch, cursor=s.cursor,
                    updates=s.updates, order=s.order, history=list(s.history))
    def load(s, state, contract):
        assert state["contract"] == contract, "CONTRACT_MISMATCH"
        s.rng.bit_generator.state = state["rng"]; s.epoch, s.cursor, s.updates = state["epoch"], state["cursor"], state["updates"]
        s.order = state["order"]; s.history = list(state["history"])
def save_checkpoint(path, state):
    with open(path, "xb") as f: pickle.dump(state, f)
def infer(model, data, batch=64, guard=None):
    r = np.random.default_rng(len(data["ids"]))
    fut = data["future"].astype(np.float64)
    return fut[:, None] + r.normal(0, .3, (len(fut), 20, 12, 2)), r.normal(size=(len(fut), 20))
'''


def run(args, env, ok=True):
    r = subprocess.run([sys.executable, str(HERE / "run_fivefold.py"), *args], capture_output=True, text=True, env=env)
    if ok and r.returncode:
        raise AssertionError(r.stdout + r.stderr)
    return r


def main():
    sys.path.insert(0, str(HERE))
    from test_fivefold_numpy import make_data
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "faketorch/torch").mkdir(parents=True); (tmp / "faketorch/torch/__init__.py").write_text(FAKE_TORCH)
        ex = tmp / "exec"; ex.mkdir()
        (ex / "train_core.py").write_text(FAKE_TRAIN_CORE)
        (ex / "baseline_factory.py").write_text("# fake\n"); (ex / "common.py").write_text("# fake\n")
        data = tmp / "data"; data.mkdir(); make_data(data)
        env = dict(os.environ, PYTHONPATH=str(tmp / "faketorch"))
        r = subprocess.run([sys.executable, "-I", str(HERE / "audit_folds.py"), "--data-dir", str(data),
                            "--out", str(tmp / "AUDIT.json")], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        common = ["--exec-dir", str(ex), "--data-dir", str(data), "--audit", str(tmp / "AUDIT.json"), "--device", "cpu"]

        # 1. final before decide must refuse (test stays closed)
        r = run(["final", "--out", str(tmp / "A"), *common], env, ok=False)
        assert r.returncode != 0 and "BUDGET_DECISION_MISSING" in (r.stdout + r.stderr)

        # 2. uninterrupted reference run of one budget cell
        run(["budget", "--out", str(tmp / "REF"), "--folds", "eth", *common], env)
        ref = json.loads((tmp / "REF/runs/budget_eth_E64_s42/metrics.json").read_text())

        # 3. crash mid-run (update 30 of 6*64... ) then resume -> identical order hash and metrics
        crash_env = dict(env, FAKE_CRASH_AFTER="30")
        r = run(["budget", "--out", str(tmp / "B"), "--folds", "eth", *common], crash_env, ok=False)
        assert "SIMULATED_CRASH" in r.stderr
        assert list((tmp / "B/runs/budget_eth_E64_s42").glob("ckpt_epoch*.pt")), "no checkpoint before crash"
        run(["budget", "--out", str(tmp / "B"), "--folds", "eth", *common], env)
        res = json.loads((tmp / "B/runs/budget_eth_E64_s42/metrics.json").read_text())
        assert res["train"]["order_hash"] == ref["train"]["order_hash"], "resume changed batch order"
        assert res["val"] == ref["val"] and res["train"]["updates"] == ref["train"]["updates"]
        assert len(list((tmp / "B/runs/budget_eth_E64_s42").glob("ckpt_epoch*.pt"))) == 2

        # 4. full pipeline; second invocation must skip everything (DONE) and not fail on write-once
        run(["all", "--out", str(tmp / "C"), *common], env)
        dec = json.loads((tmp / "C/BUDGET_DECISION.json").read_text())
        assert dec["E_star"] in (64, 128) and dec["test_accessed"] is False
        run(["all", "--out", str(tmp / "C"), *common], env)
        runs = sorted(p.name for p in (tmp / "C/runs").iterdir())
        assert len(runs) == 10 + 15, runs
        assert all((tmp / "C/runs" / n / "DONE.json").exists() for n in runs)
        assert all((tmp / "C/runs" / n / "val_predictions.npz").exists() for n in runs if n.startswith("budget"))
        assert all((tmp / "C/runs" / n / "test_predictions.npz").exists() for n in runs if n.startswith("final"))
        assert not any((tmp / "C/runs" / n / "test_predictions.npz").exists() for n in runs if n.startswith("budget"))

        # 4b. smoke mode is isolated and tiny
        r = run(["budget", "--smoke", "--out", str(tmp / "C"), "--folds", "hotel", *common], env, ok=False)
        assert r.returncode != 0 and "SMOKE" in r.stderr
        run(["budget", "--smoke", "--out", str(tmp / "SMOKE_1"), "--folds", "hotel", *common], env)
        sm = json.loads((tmp / "SMOKE_1/runs/smoke_hotel_E1_s42/metrics.json").read_text())
        assert sm["contract"]["n_train"] <= 512 and sm["val"]["overall"]["n"] <= 256

        # 4c. parallel processes on different folds into one --out: no race, same results
        procs = [subprocess.Popen([sys.executable, str(HERE / "run_fivefold.py"), "budget", "--out", str(tmp / "P"),
                                   "--folds", f, *common], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                 for f in ("eth", "hotel", "univ", "zara01", "zara02")]
        for p in procs:
            o, e = p.communicate(); assert p.returncode == 0, o + e
        run(["decide", "--out", str(tmp / "P")], env)
        a = json.loads((tmp / "P/BUDGET_DECISION.json").read_text()); b = json.loads((tmp / "C/BUDGET_DECISION.json").read_text())
        assert a["E_star"] == b["E_star"] and a["per_fold_relative_minADE_gain"] == b["per_fold_relative_minADE_gain"]

        # 5. summary end-to-end
        r = subprocess.run([sys.executable, "-I", str(HERE / "summarize_fivefold.py"), "--out", str(tmp / "C")],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        s = json.loads((tmp / "C/SUMMARY.json").read_text())
        assert s["parameters"] == 295474 and set(s["per_fold"]) == {"eth", "hotel", "univ", "zara01", "zara02"}
    print("orchestration checks passed (fake torch/train_core)")


if __name__ == "__main__":
    main()
