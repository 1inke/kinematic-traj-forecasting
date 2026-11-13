import subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parent
if __name__=='__main__':
    with (root/'worker_stdout.log').open('ab') as out,(root/'worker_stderr.log').open('ab') as err:
        child=subprocess.Popen([sys.executable,'-B','-u',str(root/'worker.py')],cwd=root,stdout=out,stderr=err)
        raise SystemExit(child.wait())
