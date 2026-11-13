"""Only C2; same-shape 98,552-parameter pair, not C1+aWTA."""
from pathlib import Path
import sys,runpy
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
if __name__=='__main__':
    sys.argv.insert(1,'C2')
    runpy.run_path(str(Path(__file__).resolve().parents[1]/'training.py'),run_name='__main__')
