"""Only output namespace/resource contract changes; original science files exact."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'execution_staged_20261026'
sys.path.insert(0,str(PARENT))
import base
base.HERE=HERE
def run_original(name):
    path=PARENT/name
    exec(compile(path.read_text(encoding='utf-8'),str(path),'exec'),dict(__name__='__main__',__file__=str(path)))
