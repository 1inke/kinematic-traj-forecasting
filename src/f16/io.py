from __future__ import annotations
import hashlib,json,os,time
from pathlib import Path
import numpy as np

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):h.update(block)
    return h.hexdigest()

def clean(value):
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,Path):return str(value)
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value):return None
    return value

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.{time.time_ns()}.tmp')
    with tmp.open('x',encoding='utf-8') as stream:
        json.dump(clean(value),stream,ensure_ascii=False,allow_nan=False,indent=2);stream.flush();os.fsync(stream.fileno())
    for i in range(40):
        try:os.replace(tmp,path);return
        except PermissionError:
            if i==39:raise
            time.sleep(.05)

def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def verify_manifest(manifest,root):
    for relative,expected in manifest.items():
        path=Path(root)/relative
        if sha256(path)!=expected:raise ValueError(f'SHA mismatch: {relative}')
