import os, subprocess, sys
from pathlib import Path
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['NUMEXPR_NUM_THREADS']='2'
log = Path(r'<WORKSPACE_ROOT>\STRATEGY_ONE_FACTOR_GTFREE_RANK_20261019\logs\run_gtf_rank.log')
log.parent.mkdir(parents=True, exist_ok=True)
cmd = [r'<WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018\.venv-k0\Scripts\python.exe', '-u', r'<WORKSPACE_ROOT>\STRATEGY_ONE_FACTOR_GTFREE_RANK_20261019\run_gtf_rank.py']
# CREATE_IDLE_PRIORITY_CLASS = 0x00000040
CREATE_IDLE = 0x00000040
p = subprocess.Popen(cmd, stdout=open(log,'a',encoding='utf-8'), stderr=subprocess.STDOUT, creationflags=CREATE_IDLE, cwd=r'<WORKSPACE_ROOT>\STRATEGY_ONE_FACTOR_GTFREE_RANK_20261019')
# set affinity to 2 cores
try:
    import ctypes
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(0x0200|0x0800, False, p.pid)  # PROCESS_SET_INFORMATION|QUERY
    if handle:
        kernel32.SetProcessAffinityMask(handle, 0x3)
        kernel32.CloseHandle(handle)
except Exception as e:
    open(log,'a',encoding='utf-8').write(f'affinity_note={e}\n')
open(log,'a',encoding='utf-8').write(f'LAUNCH pid={p.pid} Idle affinity=0x3\n')
print(p.pid)
