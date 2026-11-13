"""Owned-process atomic replacement retry only; not science, solver, or training retry."""
import os,time
_original=os.replace
def replace(src,dst,*args,**kwargs):
    for i in range(40):
        try:return _original(src,dst,*args,**kwargs)
        except PermissionError:
            if i==39:raise
            time.sleep(.05)
def install():
    if os.replace is not replace:os.replace=replace
def fixture(folder):
    from pathlib import Path
    import json
    global _original
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);a=folder/'retry.tmp';b=folder/'retry.json';a.write_text('{"ok":true}',encoding='utf-8')
    original=_original;calls=[0]
    def transient(src,dst,*aa,**kw):
        calls[0]+=1
        if calls[0]<3:raise PermissionError('SYNTHETIC_OWNED_TRANSIENT')
        return original(src,dst,*aa,**kw)
    try:
        _original=transient;replace(a,b);assert calls[0]==3 and json.loads(b.read_text())['ok']
    finally:_original=original
    return dict(status='PASS',actual_atomic_write_after_two_injected_transients=True,bounded_40_retries_only_PermissionError=True,no_scientific_tolerance_or_optimizer_change=True)


