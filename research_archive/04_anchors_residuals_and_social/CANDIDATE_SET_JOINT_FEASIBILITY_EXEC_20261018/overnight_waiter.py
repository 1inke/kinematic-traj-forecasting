"""Wait for run_k0.py to exit, then start run_k1.py once. Quiet overnight."""
from __future__ import annotations
import os, sys, time, subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(r"<WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018")
LOG = ROOT / "logs" / "overnight_chain.log"
RF = ROOT / "results" / "seed44" / "FIT_cal" / "result_fields.json"
MARKER = ROOT / "logs" / "k1_overnight.started"
K1_DONE = ROOT / "k1_exec" / "runtime_state.json"

def L(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().strftime('%Y-%m-%dT%H:%M:%S+09:00')}] {msg}\n"
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)

def python_cmds():
    try:
        import psutil  # type: ignore
    except Exception:
        psutil = None
    out = []
    if psutil is None:
        # fallback: powershell one-liner
        return None
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = p.info.get("cmdline") or []
            s = " ".join(cmd)
            if "run_k0.py" in s or "run_k1.py" in s:
                out.append((p.info["pid"], s))
        except Exception:
            pass
    return out

def k0_running_wmic() -> bool:
    # Use tasklist + wmic cmdline
    try:
        r = subprocess.run(
            ["wmic", "process", "where", "name='python.exe'", "get", "ProcessId,CommandLine", "/FORMAT:LIST"],
            capture_output=True, text=True, timeout=60,
        )
        return "run_k0.py" in (r.stdout or "")
    except Exception as e:
        L(f"wmic_err {e}")
        return True  # fail-safe: keep waiting

def main():
    for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(k, "2")
    L("python waiter start")
    while True:
        if not k0_running_wmic():
            break
        L("waiting K0")
        time.sleep(180)
    if not RF.exists():
        L("K0 done but result_fields missing; abort K1")
        return 3
    if K1_DONE.exists():
        L("K1 already complete")
        return 0
    if MARKER.exists():
        L("K1 marker exists; skip")
        return 0
    MARKER.write_text(datetime.now().isoformat(), encoding="utf-8")
    env = os.environ.copy()
    env["JOINT_FEAS_ROOT"] = r"<WORKSPACE_ROOT>"
    env["JOINT_FEAS_EXEC"] = str(ROOT)
    env["JOINT_FEAS_K1"] = str(ROOT / "k1_exec")
    (ROOT / "k1_exec").mkdir(parents=True, exist_ok=True)
    py = ROOT / ".venv-k0" / "Scripts" / "python.exe"
    k1log = ROOT / "logs" / "k1_overnight.log"
    k1err = ROOT / "logs" / "k1_overnight.err"
    L("starting K1")
    with k1log.open("ab") as out, k1err.open("ab") as err:
        p = subprocess.Popen(
            [str(py), "-u", str(ROOT / "run_k1.py")],
            cwd=str(ROOT), env=env, stdout=out, stderr=err,
        )
    L(f"K1 pid={p.pid}")
    # try BelowNormal via powershell
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f"try{{(Get-Process -Id {p.pid}).PriorityClass='BelowNormal'}}catch{{}}"],
            timeout=30,
        )
    except Exception:
        pass
    rc = p.wait()
    L(f"K1 exit={rc}")
    return rc

if __name__ == "__main__":
    sys.exit(main() or 0)
