import subprocess,sys
from pathlib import Path
sys.dont_write_bytecode=True
root=Path(__file__).resolve().parent
with (root/'worker_stdout.log').open('xb') as out,(root/'worker_stderr.log').open('xb') as err:
    p=subprocess.Popen([sys.executable,'-B','-u',str(root/'worker.py')],cwd=root,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
    raise SystemExit(p.wait())
