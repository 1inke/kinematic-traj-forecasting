"""依次：单元测试 → smoke → short 两臂对照。"""
from __future__ import annotations
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(cmd, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("RUN", cmd, flush=True)
    with log_path.open("w", encoding="utf-8") as f:
        p = subprocess.run(cmd, cwd=str(HERE), stdout=f, stderr=subprocess.STDOUT, text=True)
    print(log_path.read_text(encoding="utf-8")[-4000:], flush=True)
    if p.returncode != 0:
        raise SystemExit(f"FAILED {cmd} -> {p.returncode}")


def main():
    t0 = time.time()
    # 1) unit
    run([PY, "-B", str(HERE / "tests/test_integrator.py")], HERE / "results/unit/test_integrator.log")
    (HERE / "results/unit/STATUS.json").write_text(
        json.dumps(dict(status="PASS", at=time.time()), indent=2), encoding="utf-8"
    )

    # 2) smoke + short via same driver
    run([PY, "-B", str(HERE / "kdec/run_train_driver.py"), "--phase", "smoke"], HERE / "results/smoke/run.log")
    run([PY, "-B", str(HERE / "kdec/run_train_driver.py"), "--phase", "short"], HERE / "results/short/run.log")

    summary = dict(status="DONE", elapsed_sec=time.time() - t0)
    (HERE / "results/SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("ALL_DONE", summary)


if __name__ == "__main__":
    main()
