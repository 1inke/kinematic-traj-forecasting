"""Tests for fast_run / fast_compare / eval_budget_on_test with a FAKE torch + train_core.
Run: python test_explore_fake.py --fivefold-code <FIVEFOLD_B0_20261115/code>
Fake arms differ only by candidate noise (B0 .20, T1 .185, E3_self .18, BAD .30), so the
expected verdicts are known; this checks plumbing and decision logic, not models."""
import argparse
import json
import os
import subprocess
import sys
import tempfile
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
import pickle, numpy as np
NOISE = {"B0": .20, "T1": .185, "E3_self": .18, "BAD": .30}
class _P:
    def __init__(s, n): s.n = n
    def numel(s): return s.n
class _M:
    def __init__(s, arm, progress): s.arm, s.progress = arm, progress
    def parameters(s): return [_P(295474)]
class _EMA:
    def __init__(s, t): s.t = t
    @property
    def module(s): return _M(s.t.arm, s.t.updates / max(1, s.t.epochs * s.t.steps_per_epoch))
class Trainer:
    def __init__(s, data, indices, batch=64, epochs=8, device="cpu", seed=42, order_seed=None, arm="B0"):
        s.data, s.indices, s.batch, s.epochs, s.arm, s.seed = data, np.asarray(indices), batch, epochs, arm, seed
        s.model = _M(arm, 0); s.ema = _EMA(s)
        s.steps_per_epoch = -(-len(indices) // batch)
        s.rng = np.random.default_rng(seed); s.epoch = s.cursor = s.updates = 0
        s.order = None; s.history = []; s.batch_orders = []; s.connected = {"w"}
    def step(s):
        if s.order is None or s.cursor == len(s.order):
            if s.epoch >= s.epochs: raise StopIteration
            s.order = s.indices[s.rng.permutation(len(s.indices))]; s.cursor = 0; s.epoch += 1
        idx = s.order[s.cursor:s.cursor + s.batch]
        s.updates += 1; s.cursor += len(idx)
        s.batch_orders.append(s.data["ids"][idx].tolist()); s.history.append(dict(loss=1.0, lr=1e-4))
    def snapshot(s, contract):
        return dict(contract=contract, rng=s.rng.bit_generator.state, epoch=s.epoch, cursor=s.cursor,
                    updates=s.updates, order=s.order, history=list(s.history))
    def load(s, state, contract):
        assert state["contract"] == contract
        s.rng.bit_generator.state = state["rng"]; s.epoch, s.cursor, s.updates = state["epoch"], state["cursor"], state["updates"]
        s.order = state["order"]; s.history = list(state["history"])
def save_checkpoint(path, state):
    with open(path, "xb") as f: pickle.dump(state, f)
def infer(model, data, batch=64, guard=None):
    r = np.random.default_rng(len(data["ids"]) + 7)
    fut = data["future"].astype(np.float64)
    noise = NOISE[model.arm] * (1.5 - .5 * model.progress)      # improves during training
    return fut[:, None] + r.normal(0, noise, (len(fut), 20, 12, 2)), r.normal(size=(len(fut), 20))
'''


def make_c0(ex, rng):
    def window(n):
        past = np.zeros((n, 9, 8, 2), np.float32); v = rng.normal([.5, 0], [.1, .05], (n, 2))
        past[:, 0] = (np.arange(-7, 1)[None, :, None] * v[:, None]).astype(np.float32)
        fut = (np.arange(1, 13)[None, :, None] * v[:, None] + rng.normal(0, .05, (n, 12, 2))).astype(np.float32)
        g = rng.integers(0, 3, n).astype(np.int8)
        return dict(past=past, future=fut, num_valid=np.ones(n, np.int64), ids=np.array([f"r|{i}|0" for i in range(n)]), group=g)
    (ex / "C0").mkdir(); (ex / "E1").mkdir()
    fit, dev = window(600), window(300)
    np.savez(ex / "C0/fit.npz", **fit); np.savez(ex / "C0/dev.npz", **dev)
    idx = rng.permutation(600)
    np.savez(ex / "E1/calib_split.npz", fit85_index=np.sort(idx[90:]), calib_index=np.sort(idx[:90]))


def sh(args, env, ok=True):
    r = subprocess.run([sys.executable, *args], capture_output=True, text=True, env=env)
    if ok and r.returncode:
        raise AssertionError(r.stdout + r.stderr)
    return r


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--fivefold-code", required=True); a = ap.parse_args()
    ff = Path(a.fivefold_code).resolve()
    rng = np.random.default_rng(1)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "ft/torch").mkdir(parents=True); (tmp / "ft/torch/__init__.py").write_text(FAKE_TORCH)
        ex = tmp / "exec"; ex.mkdir(); (ex / "train_core.py").write_text(FAKE_TRAIN_CORE)
        (ex / "baseline_factory.py").write_text("#\n"); (ex / "common.py").write_text("#\n")
        make_c0(ex, rng)
        env = dict(os.environ, PYTHONPATH=str(tmp / "ft"))
        out = tmp / "EXPLORE"
        # 1. budget calibration runs (full fit) + verdicts
        sh([str(HERE / "fast_run.py"), "--exec-dir", str(ex), "--out", str(out), "--arms", "T1", "B0", "E3_self",
            "BAD", "--epochs", "4", "8", "--seeds", "42", "43", "--device", "cpu"], env)
        led = (out / "LEDGER.csv").read_text().splitlines()
        assert len(led) == 1 + 2 * 2 * 4, len(led)
        first_e4 = [l.split(",")[1] for l in led[1:5]]
        assert first_e4[0] == "B0", first_e4                         # B0 runs first
        r = sh(["-I", str(HERE / "fast_compare.py"), "--out", str(out), "--calibrate"], env)
        txt = (out / "COMPARE.md").read_text()
        assert "BAD" in txt and "dead" in txt and "promising" in txt, txt
        # rerun is a no-op (skips existing results)
        sh([str(HERE / "fast_run.py"), "--exec-dir", str(ex), "--out", str(out), "--arms", "B0", "--epochs", "4",
            "--seeds", "42", "--device", "cpu"], env)
        assert len((out / "LEDGER.csv").read_text().splitlines()) == len(led)
        # 2. interim futility on CALIB: BAD stops early, T1 completes; runs are tagged fit85
        sh([str(HERE / "fast_run.py"), "--exec-dir", str(ex), "--out", str(out), "--arms", "B0", "BAD", "T1",
            "--epochs", "8", "--seeds", "42", "--calib-every", "1", "--stop-if-behind", "coverage", "--device", "cpu"], env)
        bad = json.loads((out / "runs/BAD_E8_s42_fit85/result.json").read_text())
        t1 = json.loads((out / "runs/T1_E8_s42_fit85/result.json").read_text())
        assert bad["status"] == "STOPPED_FUTILE" and bad["stopped_at_epoch"] in (4, 5, 6, 7), bad["stopped_at_epoch"]
        assert t1["status"] == "DONE" and t1["train_set"] == "fit85"
        r = sh([str(HERE / "fast_run.py"), "--exec-dir", str(ex), "--out", str(out), "--arms", "B0", "--epochs", "4",
                "--stop-if-behind", "coverage", "--device", "cpu"], env, ok=False)
        assert r.returncode != 0
        # 3. eval_budget_on_test against a fake five-fold budget phase
        sys.path.insert(0, str(ff)); from test_fivefold_numpy import make_data  # noqa
        data = tmp / "data"; data.mkdir(); make_data(data)
        sh(["-I", str(ff / "audit_folds.py"), "--data-dir", str(data), "--out", str(tmp / "AUDIT.json")], env)
        common = ["--exec-dir", str(ex), "--data-dir", str(data), "--audit", str(tmp / "AUDIT.json"), "--device", "cpu"]
        ffout = tmp / "FF"
        sh([str(ff / "run_fivefold.py"), "budget", "--out", str(ffout), *common], env)
        r = sh([str(HERE / "eval_budget_on_test.py"), "--fivefold-code", str(ff), "--exec-dir", str(ex),
                "--data-dir", str(data), "--out", str(ffout), "--device", "cpu"], env, ok=False)
        assert r.returncode != 0 and "BUDGET_DECISION_MISSING" in r.stderr
        sh([str(ff / "run_fivefold.py"), "decide", "--out", str(ffout)], env)
        sh([str(HERE / "eval_budget_on_test.py"), "--fivefold-code", str(ff), "--exec-dir", str(ex),
            "--data-dir", str(data), "--out", str(ffout), "--device", "cpu"], env)
        md = (ffout / "EXPLORE_BUDGET_TEST.md").read_text()
        assert "等权" in md and md.count("|") > 30
        assert (ffout / "runs/budget_eth_E64_s42/test_explore_predictions.npz").exists()
        assert not (ffout / "runs/budget_eth_E64_s42/test_predictions.npz").exists()
        # final phase can still start/resume afterwards (source hash of the five-fold code unchanged)
        sh([str(ff / "run_fivefold.py"), "final", "--out", str(ffout), "--folds", "hotel", "--seeds", "42", *common], env)
    print("exploration tool checks passed (fake torch/train_core)")


if __name__ == "__main__":
    main()
