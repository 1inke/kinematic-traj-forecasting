"""Load the unchanged scientific child after installing execution-only I/O fixes."""
from pathlib import Path
import runpy
import sys
import safe_status

HERE = Path(__file__).resolve().parent
PARALLEL = HERE.parent
OLD = PARALLEL.parent / 'FIVEFOLD_B0_EXEC_20261115'
sys.path.insert(0, str(OLD / 'code'))
import runtime as rt
from fivefold_common import read_json

if __name__ == '__main__':
    rt.verify_files(read_json(HERE / 'SOURCE_MANIFEST.json')['files'])
    safe_status.install(rt)
    original = PARALLEL / 'code/parallel_child.py'
    sys.argv[0] = str(original)
    runpy.run_path(str(original), run_name='__main__')

