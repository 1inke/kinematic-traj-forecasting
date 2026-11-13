import os, subprocess, sys
from pathlib import Path
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
root = Path(r"<WORKSPACE_ROOT>\DEPLOY_S_RANK_MOTION_ONLY_20261019")
log = root / "logs" / "run_deploy.log"
log.parent.mkdir(parents=True, exist_ok=True)
py = r"<WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018\.venv-k0\Scripts\python.exe"
cmd = [py, "-u", str(root / "run_deploy.py")]
CREATE_IDLE = 0x00000040
# truncate log for this run
open(log, "w", encoding="utf-8").write("LAUNCHING\n")
p = subprocess.Popen(
    cmd,
    stdout=open(log, "a", encoding="utf-8"),
    stderr=subprocess.STDOUT,
    creationflags=CREATE_IDLE,
    cwd=str(root),
)
try:
    import ctypes
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(0x0200 | 0x0800, False, p.pid)
    if handle:
        kernel32.SetProcessAffinityMask(handle, 0x3)  # cores 0,1
        kernel32.CloseHandle(handle)
except Exception as e:
    open(log, "a", encoding="utf-8").write(f"affinity_note={e}\n")
open(log, "a", encoding="utf-8").write(f"LAUNCH pid={p.pid} Idle affinity=0x3\n")
print(p.pid)
