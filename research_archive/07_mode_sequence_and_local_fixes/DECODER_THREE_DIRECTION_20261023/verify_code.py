"""Non-production code verification: syntax, JSON, import-safe CLI, fixtures."""
import ast,json,subprocess,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
from common import write_json,sha256
from test_kernels import run_tests

def main():
    start=time.perf_counter();sources={}
    for path in sorted(HERE.rglob('*.py')):
        ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))
        sources[path.relative_to(HERE).as_posix()]=sha256(path)
    for path in sorted(HERE.rglob('protocol.json')):
        json.loads(path.read_text(encoding='utf-8-sig'))
    kernels=run_tests();cli={}
    for rel in ('direction_A_time_basis/screen.py','direction_A_time_basis/train.py',
                'direction_B_scene_prior/prepare_records.py','direction_B_scene_prior/probe.py',
                'direction_C_set_mechanisms/run_c1.py','direction_C_set_mechanisms/run_c2.py','profile_decoder.py'):
        p=subprocess.run([sys.executable,'-B',str(HERE/rel),'--help'],capture_output=True,text=True)
        if p.returncode:raise RuntimeError(p.stderr)
        cli[rel]='PASS_HELP_ONLY'
    report=dict(status='PASS_CODE_AND_SYNTHETIC_ONLY',python=sys.executable,python_version=sys.version,
        syntax_python_files=len(sources),source_SHA256=sources,kernels=kernels,CLI=cli,
        elapsed_seconds=time.perf_counter()-start,actual_GPU_FIT_preflight=False,
        old_source_manifest_full_verification=False,real_screen_or_probe=False,
        experiment_training=False,OS_tasks_created=False,automatic_next=False)
    write_json(HERE/'CODE_VERIFICATION.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_SHA256','CLI')},ensure_ascii=False))

if __name__=='__main__':main()
