"""Package the completed design; no experiment task registration or execution."""
from pathlib import Path
import json
import shutil
import subprocess
import sys
import zipfile

package = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(package/'code'))
from speed_io import now, read, sha, write

scanner = Path('C:/Users/linke/.agents/skills/avoid-zh-ai-coinage/scripts/scan_zh_coinage.py')
reports = {}
for name in ('README.md', 'EXP_ETH_DIAG.md', 'EXP_SPEED_AUG.md', 'DESIGN_REVIEW.md'):
    proc = subprocess.run([sys.executable, '-X', 'utf8', '-B', str(scanner), str(package/name), '--json'],
                          text=True, encoding='utf-8', capture_output=True, check=True)
    reports[name] = json.loads(proc.stdout)
write(package/'qa/LANGUAGE_REVIEW.json', dict(at=now(), scans=reports,
      contextual_review='锚定 refers to the explicitly defined trajectory scaling origin; retained as a technical operation. Other prose reviewed in context; scanner is advisory.'))
cpu = read(package/'qa/CPU_RUNTIME_PREFLIGHT.json')
for name, digest in cpu['code_sha256'].items():
    assert sha(package/'code'/name) == digest, name
raw_log = (package/'qa/CPU_TESTS.log').read_bytes()
log = raw_log.decode('utf-16' if raw_log.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig')
assert 'Ran 13 tests' in log and 'OK' in log
(package/'qa/CPU_TESTS.log').write_text(log, encoding='utf-8')
assert not Path(read(package/'CONFIG.local.json')['output_dir']).exists()
write(package/'qa/VALIDATION_SUMMARY.json', dict(at=now(), status='DESIGN_AND_CODE_VALIDATED_CPU',
      unit_and_synthetic_pipeline_tests=13, actual_B0_engine_CPU_preflight=cpu['status'],
      exact_full_state_resume=True, old_source_lineage_verified=456,
      powershell_scripts_parse_pass=4, new_task_registered=False, formal_output_directory_created=False,
      actual_preprocessing_audit_run=False, ETH_HOTEL_new_model_inference_run=False,
      real_data_training_started=False, GPU_preflight_run=False,
      Windows_scheduled_runtime_end_to_end_tested=False))
files = sorted(p for p in package.rglob('*') if p.is_file() and p.name != 'DELIVERY_MANIFEST.json')
manifest = dict(at=now(), status='DESIGN_CODE_ONLY_NOT_EXECUTED',
                files={p.relative_to(package).as_posix(): dict(sha256=sha(p), bytes=p.stat().st_size) for p in files})
write(package/'DELIVERY_MANIFEST.json', manifest)
destination = Path('C:/Users/linke/Downloads/1/EXP_ETH_SPEED_20261116')
if destination.exists():
    raise FileExistsError('Do not overwrite an existing delivery: '+str(destination))
shutil.copytree(package, destination)
for rel, record in manifest['files'].items():
    assert sha(destination/rel) == record['sha256'], rel
assert sha(destination/'DELIVERY_MANIFEST.json') == sha(package/'DELIVERY_MANIFEST.json')
archive = destination.with_suffix('.zip')
with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
    for file in sorted(destination.rglob('*')):
        if file.is_file(): z.write(file, destination.name+'/'+file.relative_to(destination).as_posix())
print(json.dumps(dict(delivery=str(destination), files=len(manifest['files'])+1,
      bytes=sum(p.stat().st_size for p in destination.rglob('*') if p.is_file()),
      zip=str(archive), zip_bytes=archive.stat().st_size, copy_SHA256='ALL_MATCH'), ensure_ascii=False))
