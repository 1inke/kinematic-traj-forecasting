from pathlib import Path
from datetime import datetime,timezone,timedelta
import hashlib,json,os,uuid
ROOT=Path(__file__).resolve().parents[1]
W=ROOT.parents[1]
OLD=W/'qoder_work/NEXT_PHASE_EXEC_20261114'
def now():return datetime.now(timezone(timedelta(hours=8))).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp')
    with tmp.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def npz(p,**v):
    import numpy as np
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp')
    with tmp.open('xb') as f:np.savez_compressed(f,**v);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def arrays(p):
    import numpy as np
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def save_torch(p,v):
    import torch
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp')
    with tmp.open('xb') as f:torch.save(v,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def verify(files):
    for p,h in files.items():
        if sha(p)!=h:raise ValueError('HASH_MISMATCH '+str(p))
def seal(paths):return {str(Path(p).resolve()):sha(p) for p in paths}
