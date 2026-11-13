import os, subprocess, sys, time
from pathlib import Path
root = Path(r"<WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018")
os.chdir(root)
for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ[k] = "2"
os.environ["K0_QUIET"] = "1"
py = sys.executable
log = root / "logs" / "overnight_quiet.log"
log.parent.mkdir(exist_ok=True)
def run(cmd):
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n==== {time.strftime('%Y-%m-%d %H:%M:%S')} START {cmd} ====\n")
        f.flush()
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        f.write(f"\n==== END rc={p.returncode} ====\n")
        return p.returncode
# finish K0 if anything left
if (root / "run_k0.py").exists():
    run([py, "-u", str(root / "run_k0.py")])
# then K1 if present
if (root / "run_k1.py").exists():
    run([py, "-u", str(root / "run_k1.py")])
