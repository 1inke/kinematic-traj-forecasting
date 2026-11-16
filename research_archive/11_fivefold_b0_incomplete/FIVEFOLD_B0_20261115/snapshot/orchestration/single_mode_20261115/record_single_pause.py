import runpy
from pathlib import Path
import worker_single
worker = worker_single.load_worker()
worker_single.install_single_mode(worker, worker.sha256(Path(__file__).resolve().parent / 'SOURCE_MANIFEST.json'))
runpy.run_path(str(worker_single.PARALLEL / 'code/record_pause.py'), run_name='__main__')
