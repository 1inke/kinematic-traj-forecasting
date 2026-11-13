import subprocess,sys
from pathlib import Path
p=Path(__file__).resolve().parent
with (p/'worker.stdout.log').open('xb') as out,(p/'worker.stderr.log').open('xb') as err:
    child=subprocess.Popen([sys.executable,'-B','-u',str(p/'worker.py')],cwd=p,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
    code=child.wait()
raise SystemExit(code)
