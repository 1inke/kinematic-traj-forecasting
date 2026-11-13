"""Single Windows on-demand task entry. This owns only the new P0 execution."""
import importlib.util
import os
from pathlib import Path
import runpy
import sys
from datetime import datetime
import json
ROOT=Path(__file__).resolve().parent
with (ROOT/'p0_worker_stdout.log').open('x',encoding='utf-8',buffering=1) as out, (ROOT/'p0_worker_stderr.log').open('x',encoding='utf-8',buffering=1) as err:
    sys.stdout=out;sys.stderr=err
    status='FAILURE'
    try:
        spec=importlib.util.spec_from_file_location('pre_popen_guard',ROOT.parents[1]/'qoder_work/K20_FINE_MODE_ERROR_DIAG_20261027/resource_guard_v3.py')
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        admission=m.Guard(2*1024**3).check('startup',force=True)
        with (ROOT/'p0_os_start.json').open('x',encoding='utf-8') as f:
            json.dump(dict(pid=os.getpid(),started_at=datetime.now().astimezone().isoformat(),admission=admission,scope='P0_CPU_RESUME_AND_INDEPENDENT_REVIEW_ONLY'),f,indent=2)
        runpy.run_path(str(ROOT/'p0_resume.py'),run_name='__main__')
        status='SUCCESS'
    except BaseException:
        import traceback
        traceback.print_exc()
        raise
    finally:
        with (ROOT/'p0_os_exit.json').open('x',encoding='utf-8') as f:
            json.dump(dict(status=status,at=datetime.now().astimezone().isoformat(),pid=os.getpid()),f,indent=2)
