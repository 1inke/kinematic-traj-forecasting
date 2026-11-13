"""Complete versioned preflight: old audit keyword and all colliding receipt paths explicit."""
import ast
import time
import traceback
import common as c

def main():
    path=c.ROOT/'preflight.py';src=path.read_text(encoding='utf-8')
    repairs={"check_saved(d['x'],saved,'END',g.check)":"check_saved(d['x'],saved,'END',check=g.check)",
             "out=c.ROOT/'preflight'":"out=c.ROOT/'preflight_v3'",
             "c.ROOT/'MEMORY_ESTIMATE.json'":"c.ROOT/'MEMORY_ESTIMATE_v3.json'"}
    for old,new in repairs.items():assert src.count(old)==1;src=src.replace(old,new)
    ast.parse(src)
    c.write(c.ROOT/'PREFLIGHT_V3_SOURCE_EQUIVALENCE.json',dict(status='VERSIONED_PREFLIGHT_CALL_AND_IO_REPAIR',
        files={str(path):c.sha(path),str(c.ROOT/'preflight_v3_adapter.py'):c.sha(__file__)},replacements=repairs,
        first_failure='audit API positional callback mistaken for arithmetic_atol, before scientific tiny',
        second_failure='exclusive resource receipt already exists; no overwrite attempted or permitted',
        all_math_model_train_unchanged=True,all_original_failures_sources_receipts_preserved=True,
        no_old_PASS_relabelled_as_current=True,no_tolerance_change=True))
    def reuse_input_lock():
        manifest=c.json_read(c.ROOT/'INPUT_SOURCE_LOCK.json');c.verify(manifest['files']);return manifest['files']
    c.input_lock=reuse_input_lock
    ns={'__name__':'new_preflight_v3_namespace','__file__':str(path)}
    exec(compile(src,str(path),'exec'),ns);ns['main']()

if __name__=='__main__':
    try:main()
    except Exception:
        c.write(c.ROOT/('FAILURE_PREFLIGHT_V3_'+time.strftime('%H%M%S')+'.json'),dict(traceback=traceback.format_exc(),code_sha=c.code_sha()));raise
