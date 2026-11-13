import os, subprocess
from pathlib import Path
os.environ['OMP_NUM_THREADS']='2'; os.environ['MKL_NUM_THREADS']='2'
log=Path(r'<WORKSPACE_ROOT>\STRATEGY_ONE_FACTOR_GTFREE_RANK_20261019\logs\run_gtf_rel.log'); log.parent.mkdir(exist_ok=True)
p=subprocess.Popen([r'<WORKSPACE_ROOT>\CANDIDATE_SET_JOINT_FEASIBILITY_EXEC_20261018\.venv-k0\Scripts\python.exe','-u',r'<WORKSPACE_ROOT>\STRATEGY_ONE_FACTOR_GTFREE_RANK_20261019\run_gtf_rel.py'],
  stdout=open(log,'a',encoding='utf-8'), stderr=subprocess.STDOUT,
  creationflags=0x00000040, cwd=r'<WORKSPACE_ROOT>\STRATEGY_ONE_FACTOR_GTFREE_RANK_20261019')
try:
  import ctypes
  h=ctypes.windll.kernel32.OpenProcess(0x0200|0x0800, False, p.pid)
  if h:
    ctypes.windll.kernel32.SetProcessAffinityMask(h, 0x3)
    ctypes.windll.kernel32.CloseHandle(h)
except Exception as e:
  open(log,'a',encoding='utf-8').write(f'aff={e}\n')
open(log,'a',encoding='utf-8').write(f'LAUNCH_REL pid={p.pid}\n')
print(p.pid)
