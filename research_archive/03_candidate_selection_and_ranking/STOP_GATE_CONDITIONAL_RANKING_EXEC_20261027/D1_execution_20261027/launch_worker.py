import subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parent
with (root/'worker.stdout.log').open('xb') as out,(root/'worker.stderr.log').open('xb') as err:
    p=subprocess.Popen([sys.executable,'-B','-u',str(root/'worker.py')],cwd=root,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW);code=p.wait()
raise SystemExit(code)
