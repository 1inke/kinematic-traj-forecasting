"""Checks for 实验一 code. Run: python test_eth_t1.py --fivefold-code <FIVEFOLD_B0_20261115/code>
Stub torch / explore_common / run_explore / train_core; nothing trains a real model."""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_eth_t1 as A  # noqa: E402

STUB_TORCH = '''
import pickle
class _C:
    @staticmethod
    def is_available(): return False
    @staticmethod
    def empty_cache(): pass
cuda = _C()
def load(p, map_location=None, weights_only=None):
    with open(p, "rb") as f: return pickle.load(f)
'''
STUB_COMMON = '''
import hashlib, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
EXEC = ROOT / "exec"
PARALLEL = ROOT / "parallel"
def now(): return "2026-11-16T00:00:00+08:00"
def read(p): return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def replace_json(p, v):
    Path(p).parent.mkdir(parents=True, exist_ok=True); Path(p).write_text(json.dumps(v))
def save_npz(p, **a):
    Path(p).parent.mkdir(parents=True, exist_ok=True); np.savez(p, **a)
class _G:
    def check(self, force=False): pass
def load_training():
    sys.path.insert(0, str(ROOT / "code"))
    sys.path.insert(0, FIVE_CODE)
    import fake_tc as tc, fake_rf as rf
    return tc, rf, _G()
'''
STUB_RUN = '''
import pickle
from pathlib import Path
def training_loop(tc, rf, guard, data, indices, arm, epochs, seed, directory, contract, progress=None, stop_after=None):
    Path(directory).mkdir(parents=True, exist_ok=True)
    t = tc.Trainer(data, indices, epochs=epochs, seed=seed, arm=arm)
    with open(Path(directory) / f"ckpt_epoch{epochs:03d}.pt", "wb") as f:
        pickle.dump(dict(epoch=epochs, contract=contract, arm=arm), f)
    return t, dict(parameters=295474, updates=1)
'''
STUB_TC = '''
import numpy as np
class _M:
    def __init__(s, arm): s.arm = arm
class _P:
    def numel(s): return 295474
class Trainer:
    def __init__(s, data, indices, batch=64, epochs=8, device="cpu", seed=42, order_seed=None, arm="B0"):
        s.arm = arm; s.model = type("M", (), {"parameters": lambda self: [_P()]})()
        s.ema = type("E", (), {})(); s.ema.module = _M(arm)
    def step(s): return dict(loss=1.0)
    def load(s, state, contract):
        assert state["contract"] == contract
def infer(model, data, batch=64, guard=None):
    r = np.random.default_rng(3)
    noise = {"B0": .45, "T1": .40}[model.arm]
    fut = data["future"].astype(np.float64)
    return fut[:, None] + r.normal(0, noise, (len(fut), 20, 12, 2)), r.normal(size=(len(fut), 20))
'''
STUB_RF = '''
import hashlib, numpy as np
from fivefold_data import load_split, scene_tail_split
def index_sha(index): return hashlib.sha256(np.asarray(index, np.int64).tobytes()).hexdigest()
'''


def test_verdict():
    b = dict(minADE=.44, minFDE=.66)
    assert A.verdict(b, dict(minADE=.42, minFDE=.64))[0] == "helps"
    assert A.verdict(b, dict(minADE=.439, minFDE=.66))[0] == "no_help"
    assert A.verdict(b, dict(minADE=.45, minFDE=.67))[0] == "no_help"
    assert A.verdict(b, dict(minADE=.435, minFDE=.66))[0] == "unclear"
    assert A.verdict(b, dict(minADE=.42, minFDE=.66))[0] == "unclear"


def test_pipeline(ff):
    sys.path.insert(0, str(ff)); from test_fivefold_numpy import make_data  # noqa
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp); ex = tmp / "EXPLORE_FAST"
        (ex / "code").mkdir(parents=True); (ex / "exec").mkdir(); (ex / "parallel").mkdir()
        (ex / "code/explore_common.py").write_text(STUB_COMMON.replace("FIVE_CODE", repr(str(ff))))
        (ex / "code/run_explore.py").write_text(STUB_RUN)
        (ex / "code/fake_tc.py").write_text(STUB_TC); (ex / "code/fake_rf.py").write_text(STUB_RF)
        for n in ("train_core.py", "baseline_factory.py", "training_ext.py"):
            (ex / "exec" / n).write_text("#\n")
        (tmp / "st/torch").mkdir(parents=True); (tmp / "st/torch/__init__.py").write_text(STUB_TORCH)
        data = tmp / "data"; data.mkdir(); make_data(data)
        five_out = tmp / "FIVE"
        (ex / "parallel/PARALLEL_CONFIG.json").write_text(json.dumps(dict(data_dir=str(data), output=str(five_out))))
        # fake existing B0 budget run with the right train index sha
        sys.path.insert(0, str(ex / "code"))
        import importlib, pickle
        rf = importlib.import_module("fake_rf")
        tr = rf.load_split(data, "eth", "train"); idx, _ = rf.scene_tail_split(tr["scene"])
        brun = five_out / "runs/budget_eth_E64_s42"; brun.mkdir(parents=True)
        (brun / "DONE.json").write_text("{}")
        with open(brun / "ckpt_epoch064.pt", "wb") as f:
            pickle.dump(dict(epoch=64, contract=dict(fold="eth", seed=42, epochs=64, train_index_sha=rf.index_sha(idx))), f)
        env = dict(os.environ, PYTHONPATH=str(tmp / "st"))
        out = tmp / "ETH"
        run = lambda *x: subprocess.run([sys.executable, str(HERE / "run_eth_t1.py"), *x, "--explore-dir", str(ex), "--out", str(out)],
                                        capture_output=True, text=True, env=env)
        r = run("train"); assert r.returncode != 0 and "preflight" in r.stderr
        r = run("preflight"); assert r.returncode == 0, r.stdout + r.stderr
        r = run("train"); assert r.returncode == 0, r.stdout + r.stderr
        r = run("train"); assert "already trained" in r.stdout
        r = run("evaluate"); assert r.returncode == 0, r.stdout + r.stderr
        r = subprocess.run([sys.executable, "-I", str(HERE / "analyze_eth_t1.py"), "--out", str(out),
                            "--fivefold-code", str(ff), "--data-dir", str(data)], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        res = json.loads((out / "ETH_T1_RESULT.json").read_text())
        assert res["verdict"] == "helps" and res["speed"]["by_test_speed_tertile"], res["verdict"]
        # wrong B0 index -> preflight must fail
        with open(brun / "ckpt_epoch064.pt", "wb") as f:
            pickle.dump(dict(epoch=64, contract=dict(fold="eth", seed=42, epochs=64, train_index_sha="x")), f)
        (out / "PREFLIGHT.json").unlink()
        r = run("preflight"); assert r.returncode != 0 and "different index" in r.stderr


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fivefold-code", required=True); a = ap.parse_args()
    test_verdict(); print("PASS verdict rules")
    test_pipeline(Path(a.fivefold_code).resolve()); print("PASS preflight/train/evaluate/analyze chain (stubs)")
