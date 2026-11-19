"""Execution-only JSON publication resilience; scientific values remain unchanged."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

HERE = Path(__file__).resolve().parent
PARALLEL = HERE.parent
OLD = PARALLEL.parent / 'FIVEFOLD_B0_EXEC_20261115'
MONITOR_NAMES = {'CURRENT.json', 'CHILD_STATE.json'}
RETRY_DELAYS = (0.0, .05, .1, .2, .4, .8, 1.0)


def transient(exc):
    return isinstance(exc, OSError) and getattr(exc, 'winerror', None) in (5, 32, 33)


def monitor_path(path):
    path = Path(path).resolve()
    return path.name in MONITOR_NAMES and any(
        path.is_relative_to(root.resolve())
        for root in (PARALLEL / 'standalone', OLD / 'standalone'))


def replace_json(path, value):
    """Unique temp, fsync and bounded replacement retries; never fake success."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    temp = None
    try:
        fd, name = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
        temp = Path(name)
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        for index, delay in enumerate(RETRY_DELAYS):
            if delay:
                time.sleep(delay)
            try:
                os.replace(temp, path)
                return
            except OSError as exc:
                if not transient(exc) or index == len(RETRY_DELAYS) - 1:
                    raise
    finally:
        if temp is not None and temp.exists():
            try:
                temp.unlink()
            except OSError:
                pass


def warning(path, exc, mirror=None):
    record = dict(at=time.time(), pid=os.getpid(), path=str(path),
                  winerror=getattr(exc, 'winerror', None), error=str(exc),
                  mirror=str(mirror) if mirror else None,
                  action='MONITOR_ONLY_FAILURE_NONFATAL')
    try:
        folder = HERE / 'warnings'
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / f'{os.getpid()}.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            stream.flush()
    except OSError:
        pass  # A diagnostic failure must not kill otherwise healthy training.


def atomic(path, value):
    path = Path(path)
    try:
        replace_json(path, value)
    except OSError as exc:
        if not monitor_path(path) or not transient(exc):
            raise  # Checkpoint receipts, decisions, DONE, etc. stay fail-closed.
        digest = hashlib.sha256(str(path.resolve()).encode('utf-8')).hexdigest()[:20]
        mirror = HERE / 'status_mirror' / (digest + '_' + path.name)
        try:
            replace_json(mirror, dict(original_path=str(path.resolve()), value=value))
        except OSError as mirror_exc:
            warning(mirror, mirror_exc)
        warning(path, exc, mirror)


def install(runtime):
    runtime.atomic = atomic

