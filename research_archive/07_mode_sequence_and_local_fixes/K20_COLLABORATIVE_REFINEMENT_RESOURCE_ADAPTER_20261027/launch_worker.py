from core import *
import subprocess
if __name__=='__main__':
    with (HERE/'worker.stdout.log').open('xb') as out,(HERE/'worker.stderr.log').open('xb') as err:
        p=subprocess.Popen([sys.executable,'-B','-u',str(HERE/'worker.py')],cwd=HERE,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW);p.wait();sys.exit(p.returncode)
