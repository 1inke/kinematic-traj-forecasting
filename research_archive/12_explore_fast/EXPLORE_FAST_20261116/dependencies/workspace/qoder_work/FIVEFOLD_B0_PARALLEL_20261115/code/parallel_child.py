"""One disjoint fold/seed worker using the original sealed science and contracts."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
OLD = ROOT.parent / 'FIVEFOLD_B0_EXEC_20261115'
sys.path.insert(0, str(OLD / 'code'))
import runtime as rt
from fivefold_common import read_json, now


def main():
    label, arguments = sys.argv[1], sys.argv[2:]
    if not label.replace('_', '').isalnum():
        raise ValueError('INVALID_WORKER_LABEL')
    seal = ROOT / 'SOURCE_MANIFEST.json'
    if seal.exists():
        rt.verify_files(read_json(seal)['files'])
    elif not label.startswith('smoke_'):
        raise RuntimeError('PARALLEL_SOURCE_NOT_SEALED')
    # Every child owns its own status path; no shared temporary JSON file races.
    def own_status(**values):
        rt.atomic(ROOT / 'standalone/slots' / label / 'CURRENT.json', {'at': now(), **values})
    rt.status = own_status
    own_status(stage='STARTING', label=label)
    sys.argv = [str(OLD / 'code/runtime.py'), *arguments]
    rt.main()
    own_status(stage='SUCCESS', label=label)


if __name__ == '__main__':
    main()
