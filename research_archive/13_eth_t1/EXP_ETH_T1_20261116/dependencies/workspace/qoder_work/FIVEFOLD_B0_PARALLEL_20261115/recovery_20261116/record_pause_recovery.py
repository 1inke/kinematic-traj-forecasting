"""Reuse original checkpoint-verifying pause receipt with resilient status I/O."""
from pathlib import Path
import runpy
import sys
import safe_status

HERE = Path(__file__).resolve().parent
OLD = HERE.parent.parent / 'FIVEFOLD_B0_EXEC_20261115'
sys.path.insert(0, str(OLD / 'code'))
import runtime as rt
from fivefold_common import read_json

if __name__ == '__main__':
    rt.verify_files(read_json(HERE / 'SOURCE_MANIFEST.json')['files'])
    safe_status.install(rt)
    runpy.run_path(str(HERE.parent / 'code/record_pause.py'), run_name='__main__')
