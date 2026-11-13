import os, sys, time
from pathlib import Path
# set idle + affinity from inside
try:
    import ctypes
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.GetCurrentProcess()
    # IDLE_PRIORITY_CLASS = 0x00000040
    kernel32.SetPriorityClass(handle, 0x00000040)
    kernel32.SetProcessAffinityMask(handle, 0x3)
except Exception as e:
    print("prio warn", e, flush=True)
os.environ["OMP_NUM_THREADS"]="2"
os.environ["MKL_NUM_THREADS"]="2"
os.environ["OPENBLAS_NUM_THREADS"]="2"
os.environ["NUMEXPR_NUM_THREADS"]="2"
os.environ["PYTHONUNBUFFERED"]="1"
script = Path(r"<WORKSPACE_ROOT>\ORIGINAL_CAL_TRANSFER_EVAL_20261019\run_cal_eval.py")
sys.argv = [str(script)]
# execute as __main__
code = compile(script.read_text(encoding="utf-8"), str(script), "exec")
g = {"__name__": "__main__", "__file__": str(script)}
exec(code, g)
