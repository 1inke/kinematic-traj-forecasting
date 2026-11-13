"""Lightweight pre-Popen resource admission; current attempt evidence retained."""
from pathlib import Path
import sys,subprocess,time,traceback
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
from joint_common import *

def main():
    authorize('preflight',True);p=verify_package();verify(p['dependencies'])
    root=HERE/'receipts'/'preflight_attempt1_20261106'
    root.mkdir(parents=True,exist_ok=False)
    E=4520878080+21719*12*2*4+512*1024**2
    try:
        admission=resource_module().Guard(E).check(mode='startup',force=True)
        write(root/'PRE_POPEN_RESOURCE.json',dict(at=now(),E=E,admission=admission,
            scope='lightweight parent before CUDA child allocation',launcher_sha=sha(__file__)))
        command=[sys.executable,'-B','-u',str(HERE/'preflight_real.py'),'--output',str(root/'scientific'),'--execute']
        with (root/'stdout.log').open('x',encoding='utf-8') as out,(root/'stderr.log').open('x',encoding='utf-8') as err:
            result=subprocess.run(command,stdout=out,stderr=err,cwd=ROOT)
        write(root/'LAUNCH_EXIT.json',dict(at=now(),returncode=result.returncode,command=command))
        if result.returncode:
            print((root/'stderr.log').read_text(encoding='utf-8'),flush=True)
            raise RuntimeError('CURRENT_PREFLIGHT_FAILED_RETURN_'+str(result.returncode))
        print((root/'stdout.log').read_text(encoding='utf-8'),flush=True)
    except BaseException as e:
        write(root/'LAUNCH_FAILURE.json',dict(at=now(),error=repr(e),traceback=traceback.format_exc(),retain_all=True))
        raise

if __name__=='__main__':main()
