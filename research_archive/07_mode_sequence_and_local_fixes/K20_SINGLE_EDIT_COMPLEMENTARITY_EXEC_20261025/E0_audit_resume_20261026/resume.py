"""Fresh output namespace, exact prior bounded-audit implementation; no file deletion."""
from pathlib import Path
import sys,importlib.util,json,traceback
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'E0_audit_recovery_20261025'
spec=importlib.util.spec_from_file_location('prior_e0_bounded_audit',OLD/'recovery.py')
adapter=importlib.util.module_from_spec(spec);sys.modules[spec.name]=adapter;spec.loader.exec_module(adapter)
adapter.HERE=HERE
c=adapter.c
def prepare():
    if (HERE/'source_manifest.json').exists():raise FileExistsError('Existing lock must be reused without overwrite')
    m=c.read_json(OLD/'source_manifest.json');files=m['files'].copy()
    for path,ss in files.items():
        if c.sha256(c.ROOT/path)!=ss:raise ValueError('Inherited source/output '+path)
    additions=[OLD/'source_manifest.json',OLD/'PREFLIGHT_FAILURE.json',OLD/'RESULT_ANALYSIS.md',OLD/'RESULT_ANALYSIS.json']
    additions+=list(p for p in HERE.iterdir() if p.suffix in ('.py','.ps1','.md') or p.name=='protocol.json')
    for p in additions:
        key=p.relative_to(c.ROOT).as_posix();ss=c.sha256(p)
        if key in files and files[key]!=ss:raise ValueError('Inherited SHA overwritten')
        files.setdefault(key,ss)
    c.write_json(HERE/'source_manifest.json',dict(digest=c.digest(files),files=files,inherited_count=len(m['files']),previous_digest=m['digest'],parent_source_digest=m['parent_source_digest'],at=c.now(),no_files_deleted=True))
    print(json.dumps(dict(count=len(files),digest=c.digest(files))),flush=True)
def main():
    command=sys.argv[1]
    if command=='prepare':prepare()
    elif command=='preflight':
        if (HERE/'PREFLIGHT.json').exists():raise FileExistsError('Do not overwrite completed preflight')
        adapter.preflight()
    elif command=='worker':adapter.worker()
    else:raise ValueError('Unknown command')
if __name__=='__main__':
    try:main()
    except BaseException as exc:
        target=HERE/(sys.argv[1].upper()+'_FAILURE_'+str(__import__('time').time_ns())+'.json')
        c.write_json(target,dict(type=type(exc).__name__,message=str(exc),at=c.now(),traceback=traceback.format_exc(),all_old_files_preserved=True));raise
