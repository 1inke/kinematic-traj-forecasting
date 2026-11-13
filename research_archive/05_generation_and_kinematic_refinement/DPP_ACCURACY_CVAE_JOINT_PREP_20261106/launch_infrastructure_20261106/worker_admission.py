"""Lightweight sole launcher, resource check immediately before worker Popen."""
from pathlib import Path
import sys,subprocess,time,traceback
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
from joint_common import *

def main(root):
    root=Path(root).resolve();authorize('train',True)
    ext=read(root/'INFRASTRUCTURE_LOCK.json');verify(ext['files'])
    if not read(HERE/'control.json').get('allow_OS_task_registration'):raise PermissionError('OS_START_NOT_AUTHORIZED')
    import msvcrt
    with (root/'launcher.lock').open('a+b') as handle:
        if handle.tell()==0:handle.write(b'1');handle.flush()
        handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        try:
            if (root/'WORKER_START.json').exists():raise RuntimeError('DO_NOT_REPEAT_EXISTING_WORKER_START')
            pre=read(ext['preflight']);admission=resource_module().Guard(pre['GPU_E']).check(mode='startup',force=True)
            write(root/'PRE_POPEN_RESOURCE.json',dict(at=now(),admission=admission,E=pre['GPU_E'],
                source='current real four-arm preflight measured inventory',launcher_pid=os.getpid()))
            cmd=[sys.executable,'-B','-u',str(Path(__file__).with_name('worker_authorized.py')),'--root',str(root)]
            with (root/'worker_stdout.log').open('x',encoding='utf-8') as out,(root/'worker_stderr.log').open('x',encoding='utf-8') as err:
                child=subprocess.Popen(cmd,cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
                write(root/'LAUNCHER_CHILD.json',dict(at=now(),launcher_pid=os.getpid(),child_pid=child.pid,
                    note='venv child may redirect; actual worker PID comes from WORKER_START.json',command=cmd))
                code=child.wait()
            write(root/'LAUNCHER_EXIT.json',dict(at=now(),returncode=code))
            return code
        except BaseException as e:
            write(root/f'LAUNCHER_FAILURE_{time.time_ns()}.json',dict(at=now(),error=repr(e),traceback=traceback.format_exc(),retain_all=True))
            raise
        finally:
            handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();sys.exit(main(a.root))
