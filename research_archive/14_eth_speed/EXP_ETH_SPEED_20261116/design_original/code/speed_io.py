"""File contracts and local dependencies; importing this module starts no work."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

PACKAGE = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, obj):
    """Atomic unique-temp replace with bounded retries for Windows sharing locks."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temp.open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    for attempt in range(7):
        try:
            os.replace(temp, path)
            return
        except OSError as exc:
            if getattr(exc, 'winerror', None) not in (5, 32, 33) or attempt == 6:
                raise
            time.sleep(.05 * 2**attempt)


def monitor(root, **obj):
    try:
        write(Path(root)/'PROGRESS.json', dict(at=now(), **obj))
    except OSError as exc:
        # A locked monitoring file must not destroy a healthy training session.
        try:
            write(Path(root)/'status_mirror'/f'{time.time_ns()}.json', dict(at=now(), warning=str(exc), **obj))
        except OSError:
            pass


def config(path=None):
    return read(path or PACKAGE/'CONFIG.local.json')


def imports(c, training=False):
    for key in ('exec_dir', 'fivefold_code'):
        sys.path.insert(0, c[key])
    import fivefold_data as fd
    import fivefold_metrics as fm
    if not training:
        return fd, fm
    import torch
    torch.set_num_threads(1)
    import train_core as tc
    import baseline_factory as bf
    return fd, fm, tc, bf


def verify(files):
    for file, digest in files.items():
        if sha(file) != digest:
            raise ValueError('INPUT_CHANGED: ' + file)


def package_files():
    return sorted([p for folder in ('code', 'tests', 'standalone') for p in (PACKAGE/folder).glob('*')
                   if p.is_file()] + [PACKAGE/'CONFIG.local.json'] + list(PACKAGE.glob('*.md')))


def frozen(c, root):
    """Before any model use, pin package, upstream scientific source, data and weights."""
    root = Path(root); target = root/'INPUT_CONTRACT.json'
    files = {str(p): sha(p) for p in package_files()}
    if sha(c['upstream_manifest']) != c['upstream_manifest_sha256']:
        raise ValueError('UPSTREAM_MANIFEST_CHANGED')
    # This is the existing sealed 456-file execution lineage, not a new copy.
    upstream = read(c['upstream_manifest'])['files']
    verify(upstream)
    files[c['upstream_manifest']] = c['upstream_manifest_sha256']
    files[c['recording_audit']] = c['recording_audit_sha256']
    audit = read(c['recording_audit'])
    if not audit['passed']:
        raise ValueError('RECORDING_AUDIT_FAILED')
    for row in audit['folds']:
        for key, digest in row['sha256'].items():
            split = key.split('_')[-1]; kind = key.split('_')[0]
            file = str(Path(c['data_dir']) / f"{row['fold']}_{kind}_{split}.npy")
            files[file] = digest
    files.update(audit['raw_label_sha256'])
    for fold in ('eth', 'hotel'):
        b = c['baselines'][fold]
        files[b['checkpoint']] = b['sha256']
        files[b['receipt']] = b['receipt_sha256']
    files.update(c['optional_saved_files'])
    for p in Path(c['astra_dir']).joinpath('data').glob('*.py'):
        if p.name in ('eth.py', 'eth_preprocessor.py', 'process_eth.py'):
            files[str(p)] = sha(p)
    files[str(Path(c['astra_dir'])/'utils/video2images.py')] = sha(Path(c['astra_dir'])/'utils/video2images.py')
    for video in Path(c['video_dir']).glob('*.avi'):
        files[str(video)] = sha(video)
    verify(files)
    value = dict(schema=1, study='EXP_ETH_SPEED_20261116', config=c, files=files)
    if target.exists():
        if read(target) != value:
            raise ValueError('RESUME_CONTRACT_MISMATCH: use a new output directory')
    else:
        write(target, value)
    return sha(target)
