import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
STAGE_A=ROOT.parent/'DPP_DUAL_FEASIBILITY_OFFLINE_EXEC_20261104'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024**2),b''):h.update(chunk)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)

def check_files(files):
    if not files:raise ValueError('EMPTY_SOURCE_EVIDENCE')
    for path,digest in files.items():
        if sha(path)!=digest:raise ValueError('SOURCE_SHA_MISMATCH '+path)

def seal(paths):return {str(Path(p).resolve()):sha(p) for p in paths}
