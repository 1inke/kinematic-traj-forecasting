from pathlib import Path
from datetime import datetime,timezone,timedelta
import hashlib,json,os,sys,time,traceback,uuid
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
sys.dont_write_bytecode=True
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def now():return datetime.now(timezone(timedelta(hours=8))).isoformat()
def write(p,v):
    p=Path(p);tmp=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(tmp,p)
def verify():
    for p,h in read(ROOT/'SOURCE_MANIFEST.json')['files'].items():assert sha(p)==h,p
def main():
    verify();config=read(ROOT/'CONFIG.json');out=Path(config['output_dir'])
    if (ROOT/'DONE.json').exists():
        for p,h in read(ROOT/'DONE.json')['files'].items():assert sha(p)==h,p
        print('ALREADY_COMPLETE_VERIFIED');return
    assert not out.exists(),'Output already exists without a completion seal; inspect before retry.'
    sys.path.insert(0,str(ROOT/'design_original/code'))
    import run_kine_oracle as run
    import kine_repr as kr
    original=kr.fit_templates;counter=0;receipts=[]
    def progress(family,kp,ve,seed,lam=kr.LAMBDA):
        nonlocal counter
        counter+=1;fold=config['folds'][(counter-1)//30];start=time.monotonic()
        write(ROOT/'CURRENT.json',dict(at=now(),phase='CLUSTERING',fold=fold,fit_started=counter,total_fits=150,family=family,seed=seed,lam=lam))
        print(f'FIT_START {counter}/150 fold={fold} family={family} seed={seed} lam={lam}',flush=True)
        value=original(family,kp,ve,seed,lam)
        receipts.append(dict(index=counter,fold=fold,family=family,seed=seed,lam=lam,seconds=time.monotonic()-start,fit_feature_rows=len(kp),at=now()))
        write(ROOT/'FIT_PROGRESS.json',dict(fits_completed=counter,total_fits=150,receipts=receipts))
        print(f'FIT_DONE {counter}/150 seconds={receipts[-1]["seconds"]:.3f}',flush=True)
        return value
    kr.fit_templates=progress
    args=['--data-dir',config['data_dir'],'--out',str(out),'--folds',*config['folds'],'--seeds',*[str(s) for s in config['seeds']],'--reference',*config['references']]
    result=run.main(args);assert counter==150
    assert 'torch' not in sys.modules
    verify()
    files={str(p):sha(p) for p in out.rglob('*') if p.is_file()}
    write(ROOT/'DONE.json',dict(at=now(),status='SUCCESS',verdict=result['decision']['verdict'],source_manifest_sha256=sha(ROOT/'SOURCE_MANIFEST.json'),files=files,training=False,inference=False,GPU_used=False))
    write(ROOT/'CURRENT.json',dict(at=now(),phase='COMPLETE',verdict=result['decision']['verdict'],fits_completed=counter))
    print('COMPLETE '+result['decision']['verdict'],flush=True)
if __name__=='__main__':
    try:main()
    except BaseException as e:
        write(ROOT/'FAILURE.json',dict(at=now(),error=str(e),traceback=traceback.format_exc()));write(ROOT/'CURRENT.json',dict(at=now(),phase='FAILED',error=str(e)));raise
