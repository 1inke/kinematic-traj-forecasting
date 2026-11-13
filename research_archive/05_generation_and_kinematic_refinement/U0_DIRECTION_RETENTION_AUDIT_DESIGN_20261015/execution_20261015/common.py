"""CPU saved-output audit helpers. No neural-network imports or legacy execution."""
import os,sys
sys.dont_write_bytecode=True
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import ast,json,hashlib,time,traceback,importlib.util,zipfile
from datetime import datetime
import numpy as np
H=Path(__file__).resolve().parent;ROOT=H.parents[1]
PARENT=ROOT/'GENERATION_UNPROJECTED_TIME_COUPLED_REFINEMENT_20261015/U0_execution_20261015'
PREV=PARENT/'audit_comparator_recovery_20261015'
POLICY=PARENT/'audit_memory_policy_20261015/memory_policy.py'
GROUPS=('overall','stable','transition','change');ARMS=('free_position','integrated_acceleration');SEEDS=(42,43,44)
SIZES=dict(FIT=18909,outer=1035,inner=1856,original_CAL=2879)

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):
    path=Path(path);path=path if path.is_absolute() else H/path;path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=lambda a:a.item() if isinstance(a,np.generic) else a.tolist()),encoding='utf-8');os.replace(tmp,path)
def save(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,**arrays);os.replace(tmp,path)
def sha(path):
    x=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):x.update(b)
    return x.hexdigest()
def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
spec=importlib.util.spec_from_file_location('approved_task_memory_policy',POLICY);policy=importlib.util.module_from_spec(spec);spec.loader.exec_module(policy);policy.H=H
RESOURCE=policy.RESOURCE
def guard(mode='runtime'):return policy.guard(mode)
def verify(files):
    for i,(p,h) in enumerate(files.items()):
        if i%32==0:guard()
        assert sha(p)==h,('BAD_OUTPUT_OR_SOURCE_SHA',p)
def stage(name,**kw):guard();write('progress.json',dict(stage=name,at=datetime.now().astimezone().isoformat(),**kw));print(name,kw,flush=True)
def masks(groups):return [('overall',np.ones(len(groups),bool))]+[(g,groups==i) for i,g in enumerate(GROUPS[1:])]
def close(a,b,tol=2e-12):
    a=np.asarray(a);b=np.asarray(b);assert a.shape==b.shape,(a.shape,b.shape)
    assert np.array_equal(np.isnan(a),np.isnan(b)) and np.array_equal(np.isposinf(a),np.isposinf(b)) and np.array_equal(np.isneginf(a),np.isneginf(b))
    ok=np.isfinite(a)&np.isfinite(b);e=float(np.max(abs(a[ok].astype(float)-b[ok].astype(float)))) if ok.any() else 0.;assert e<=tol,('ARITHMETIC_MISMATCH',e,tol);return e
def compare(a,b):
    if isinstance(a,dict):assert isinstance(b,dict) and a.keys()==b.keys();return max([compare(a[k],b[k]) for k in a] or [0.])
    if isinstance(a,(list,tuple)):assert isinstance(b,(list,tuple,np.ndarray)) and len(a)==len(b);return max([compare(x,y) for x,y in zip(a,b)] or [0.])
    if a is None:assert b is None;return 0.
    if isinstance(a,(str,bool)):assert a==b;return 0.
    return close(a,b)
def scalar_mean(x):return float(np.mean(x)) if len(x) else None
def failure(kind):
    d=H/'failures'/str(time.time_ns());d.mkdir(parents=True,exist_ok=True);(d/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
    for p in H.glob('*.py'):(d/p.name).write_bytes(p.read_bytes())
    for name in ('source_manifest.json','RESOURCE_ESTIMATE.json','MEMORY_ESTIMATE.json','PREFLIGHT_RESOURCE_START.json','FIXTURES.json','protocol.json'):
        p=H/name
        if p.exists():(d/name).write_bytes(p.read_bytes())
    write('FAILURE.json',dict(status=kind,source=str(d),resource=RESOURCE))

def payload(path):
    total=0
    with zipfile.ZipFile(path) as f:
        for name in f.namelist():
            with f.open(name) as b:
                version=np.lib.format.read_magic(b);shape,_,dtype=np.lib.format._read_array_header(b,version)
            total+=int(np.prod(shape))*dtype.itemsize
    return total

def estimate():
    jobs=read(PARENT/'JOBS.json');largest=max(payload(r['output'])+payload(r['aux']) for r in jobs.values())
    snap=policy.snapshot();base=max(snap['private_bytes'],snap['RSS_bytes'])
    candidates=18909*20;candidate_bytes=candidates*(32*8+12*8+12);window_bytes=18909*128*8
    # NPZ read/decompress copies, one native reference and diagnostics/review
    # coexistence; bootstrap stores at most20k draws for956+191 tracks.
    trackdraws=20000*(956+191+3)*8
    E=int(base+4*largest+3*candidate_bytes+window_bytes+2*trackdraws)
    write('RESOURCE_ESTIMATE.json',dict(status='TASK_SPECIFIC_CONSERVATIVE_ESTIMATE',estimated_total_peak_bytes=E,largest_saved_job_payload_bytes=largest,measured_import_baseline_bytes=base,candidate_diagnostic_bytes=candidate_bytes,window_diagnostic_bytes=window_bytes,bootstrap_draw_bytes_upper_bound=trackdraws,formula='baseline+4*largest_saved_job_payload+3*candidate_diagnostics+window_diagnostics+2*bootstrap_draws',not_chosen_from_available_memory=True))
    write('MEMORY_ESTIMATE.json',dict(estimated_total_peak_bytes=E));return E

def locks():
    old=read(PREV/'source_manifest.json');files=dict(old['files']);assert len(files)==2092
    paths=[H.parent/'EXPERIMENT_GUIDE.md',ROOT/'RESOURCE_POLICY.json',POLICY,*H.glob('*.py'),*H.glob('*.ps1'),H/'README.md',H/'protocol.json',H/'resource_policy.json',H/'MEMORY_ESTIMATE.json',H/'RESOURCE_ESTIMATE.json']
    excluded={'runtime_state.json','progress.json','worker.lock'}
    paths += [p for p in PREV.rglob('*') if p.is_file() and p.name not in excluded and 'failures' not in p.parts]
    paths += list((H/'failures').rglob('*'))
    for p in paths:
        if p.is_file():
            h=sha(p)
            if str(p) in files:assert files[str(p)]==h,('INHERITED_DIGEST_CHANGED',str(p))
            files[str(p)]=h
    verify(files);return dict(files=files,source_digest=digest(files),inherited_sources=2092,source_count=len(files))

def valid_done(row,source):
    assert row['status']=='PASS' and row['source_digest']==source,'BAD_SOURCE_CONTRACT';verify(row['files']);return True

def native(split):
    row=read(PARENT/'JOBS.json')['native/'+split];valid_done(row,digest(read(PARENT/'source_manifest.json')['files']))
    z=npz(row['output']);assert np.array_equal(z['paths'],z['native']);return z

def check_input(z,reference,row):
    required=('paths','native','gt','sample_id','groups','track','scene','sequence','raw','controls','correction')
    assert all(k in z for k in required),'MISSING_SAVED_FIELD'
    n=row['n'];assert z['paths'].shape==(n,20,12,2) and z['paths'].dtype==np.float64
    assert len(np.unique(z['sample_id']))==n and set(z['groups']).issubset({0,1,2})
    for k in ('sample_id','gt','groups','track','scene','sequence','native'):assert np.array_equal(z[k],reference[k]),('BAD_ID_OR_SLOT',k)
    for k in ('paths','native','gt','raw','controls','correction'):assert np.isfinite(z[k]).all(),k
    close(z['paths']-z['native'],z['correction'])
    # Read-only AST extraction of original NumPy recurrence, not model import.
    path=PARENT/'model.py';tree=ast.parse(path.read_text(encoding='utf-8-sig'));node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='reconstruct')
    ns=dict(np=np,ARMS=ARMS);exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
    q,_,r=ns['reconstruct'](z['native'],z['raw'],row['arm'],z['controls']);close(q,z['paths']);close(r,z['correction'])

def input_audit():
    jobs=read(PARENT/'JOBS.json');assert len(jobs)==28;source=digest(read(PARENT/'source_manifest.json')['files']);records=[];ids=[]
    for split,n in SIZES.items():
        z=native(split);assert len(z['sample_id'])==n;ids.append(z['sample_id']);records.append(dict(split=split,n=n,tracks=len(set(z['track'])),groups={g:int(m.sum()) for g,m in masks(z['groups'])},scenes={str(s):int((z['scene']==s).sum()) for s in np.unique(z['scene'])}));del z
    full=np.concatenate(ids);assert len(full)==24679 and len(np.unique(full))==24679
    scientific={k:r for k,r in jobs.items() if r['arm']!='native20'};assert len(scientific)==24
    for k,r in scientific.items():valid_done(r,source)
    value=dict(status='PASS_ID_CONTRACTS',native_splits=records,scientific_jobs=24,native_jobs=4,unique_IDs=24679,source_digest=source,scientific_repeated_window_rows=sum(r['n'] for r in scientific.values()),not_independent_repeats=True,full_scientific_arrays_checked_per_job=True)
    write('INPUT_ID_AUDIT.json',value);return scientific
