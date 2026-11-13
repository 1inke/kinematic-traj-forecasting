"""Narrow old audit API invocation repair; no objective/model/optimizer changes."""
import ast
import time
import traceback
import common as c

def main():
    path=c.ROOT/'preflight.py';src=path.read_text(encoding='utf-8')
    repairs={"check_saved(d['x'],saved,'END',g.check)":"check_saved(d['x'],saved,'END',check=g.check)",
             "out=c.ROOT/'preflight'":"out=c.ROOT/'preflight_v2'"}
    for old,new in repairs.items():assert src.count(old)==1;src=src.replace(old,new)
    ast.parse(src)
    c.write(c.ROOT/'PREFLIGHT_REPAIR_SOURCE_EQUIVALENCE.json',dict(status='NARROW_CALL_AND_OUTPUT_NAMESPACE_REPAIR',
        original_preflight_sha=c.sha(path),adapter_sha=c.sha(__file__),replacements=repairs,
        unchanged_files={str(c.ROOT/n):c.sha(c.ROOT/n) for n in ('common.py','objectives.py','trainer.py','run_tiny.py','review_tiny.py')},
        failure='Original old audit_saved.check_saved fourth positional argument is arithmetic_atol; passing g.check there produced TypeError before any scientific tiny.',
        actual_math_GPU_and_resume_v1_already_passed_but_full_current_preflight_rerun=True,
        old_preflight_outputs_and_failure_preserved=True,no_tolerance_change=True))
    def reuse_input_lock():
        manifest=c.json_read(c.ROOT/'INPUT_SOURCE_LOCK.json');c.verify(manifest['files']);return manifest['files']
    c.input_lock=reuse_input_lock
    ns={'__name__':'new_preflight_v2_namespace','__file__':str(path)}
    exec(compile(src,str(path),'exec'),ns);ns['main']()

if __name__=='__main__':
    try:main()
    except Exception:
        c.write(c.ROOT/('FAILURE_PREFLIGHT_V2_'+time.strftime('%H%M%S')+'.json'),dict(traceback=traceback.format_exc(),code_sha=c.code_sha()));raise
