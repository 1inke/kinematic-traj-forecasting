"""Seal SCRIPT PREPARATION only; no model/data evaluation/optimizer/launch."""
import ast
from joint_common import *

def main():
    if (HERE/'SCRIPT_MANIFEST.json').exists():raise FileExistsError('Script package already sealed; use a new version for later repairs')
    c=read(HERE/'control.json')
    if c['allow_training'] or c['allow_real_preflight']:raise ValueError('PREPARATION_HOLD_REQUIRED')
    candidates=list(HERE.glob('CODE_VALIDATION*.json'))
    valid=[]
    for p in candidates:
        d=read(p)
        if d.get('source_files'):
            try:verify(d['source_files'])
            except ValueError:continue
            if set(d['source_files'])==set(str(p.resolve()) for p in HERE.rglob('*.py')):valid.append(p)
    if not valid:raise ValueError('CURRENT_ALL_PYTHON_CPU_VALIDATION_REQUIRED')
    receipt=max(valid,key=lambda p:p.stat().st_mtime_ns)
    files=[p for p in HERE.rglob('*') if p.is_file() and p.suffix in ('.py','.md')]
    for p in files:
        if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8'))
    files.extend([HERE/'protocol.json',receipt]);dependencies=inputs();verify(dependencies)
    write(HERE/'SCRIPT_MANIFEST.json',dict(status='SEALED_SCRIPT_PREPARATION_NOT_PRODUCTION_ADMISSION',at=now(),
        files=seal(files),dependencies=dependencies,validation_receipt=str(receipt),
        control_intentionally_mutable_only_after_new_human_launch_instruction=True,
        actual_GPU_preflight=False,production_started=False,optimizer_steps=0,new_OS_task=False))
    print('SCRIPT_PREPARATION_SHA_SEAL_ONLY',len(files),'files',len(dependencies),'read-only dependencies')

if __name__=='__main__':main()
