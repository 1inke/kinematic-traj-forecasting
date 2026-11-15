"""Test suspending a waiting parent leaves its child alive and exact exit available."""
import subprocess
import sys
import time
import psutil
from pathlib import Path
import pause_controller as pc


def main():
    here=Path(__file__).resolve().parent
    child_script=here/'fixture_parent.py'
    child_script.write_text('import subprocess,sys,time\nfrom pathlib import Path\np=subprocess.Popen([sys.executable,"-c","import time; time.sleep(2)"])\nPath(sys.argv[1]).write_text(str(p.pid))\np.wait()\ntime.sleep(20)\n',encoding='utf-8')
    pidfile=here/'fixture_child.pid'
    parent=subprocess.Popen([sys.executable,str(child_script),str(pidfile)],creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        for _ in range(100):
            if pidfile.exists():break
            time.sleep(.05)
        childpid=int(pidfile.read_text());api,handle=pc.wait_exit_code(childpid)
        proc=psutil.Process(parent.pid);pc.suspend_verified(proc,str(child_script))
        if api.WaitForSingleObject(handle,10000)!=0:raise AssertionError('CHILD_DID_NOT_COMPLETE_INDEPENDENTLY')
        import ctypes
        code=ctypes.c_uint();api.GetExitCodeProcess(handle,ctypes.byref(code))
        assert code.value==0 and proc.status()==psutil.STATUS_STOPPED
        proc.resume();parent.terminate();parent.wait(timeout=10)
        api.CloseHandle.argtypes=[ctypes.c_void_p];api.CloseHandle(handle)
        pc.write_json(here/'GATE_TEST.json',dict(status='PASS',at=pc.now(),waiting_parent_suspended=True,
            child_finished_while_parent_suspended=True,real_child_exit_code=0,parent_resumed_then_fixture_only_terminated=True))
        print('PAUSE_GATE_TEST_PASS')
    finally:
        if parent.poll() is None:
            psutil.Process(parent.pid).resume();parent.terminate();parent.wait(timeout=10)


if __name__=='__main__':main()
