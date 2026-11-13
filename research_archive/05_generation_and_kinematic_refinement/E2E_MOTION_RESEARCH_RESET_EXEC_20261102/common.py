"""New execution namespace; no imports of historical training launchers."""
from pathlib import Path
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone, timedelta

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ASTRA = Path('<LOCAL_DESKTOP>/DRSE/ASTRA_ACCURACY_Q20')
DT = .4


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # New evidence must never silently replace a previous scientific receipt.
    with path.open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)


def write_csv(path, rows):
    rows = list(rows)
    with Path(path).open('x', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def seal(paths):
    return {str(Path(p).resolve()): sha(p) for p in paths}


def verify(files):
    for p, digest in files.items():
        if sha(p) != digest:
            raise ValueError('SOURCE_OR_OUTPUT_SHA_MISMATCH: ' + p)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))
