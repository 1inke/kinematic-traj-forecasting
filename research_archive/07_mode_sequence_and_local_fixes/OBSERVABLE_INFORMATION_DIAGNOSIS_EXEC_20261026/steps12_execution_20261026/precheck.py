"""Admission only: no Torch import, arrays/model loading, training or worker launch."""
from pathlib import Path
import sys,json,shutil,hashlib,datetime
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];sys.path.insert(0,str(ROOT))
from src.f16.resource import snapshot,decide
from src.f16.io import read_json,write_json,sha256
def main():
    now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
    estimate=read_json(HERE/'RESOURCE_ESTIMATE.json');disk=shutil.disk_usage(HERE);resource=decide(snapshot(),estimate['estimated_peak_bytes'],'startup')
    docs=ROOT/'docs/OBSERVABLE_INFORMATION_DIAGNOSIS_DESIGN_20261026'
    docfiles=[docs/n for n in ('README.md','00_COMMON_CONTRACT.md','01_SMALL_SAMPLE_FIT.md','02_SAME_INPUT_PROBES.md')]
    old=ROOT/'qoder_work/K20_SINGLE_EDIT_COMPLEMENTARITY_EXEC_20261025/E1_execution_20261026'
    required=[old/n for n in ('source_manifest.json','source_extension_prelaunch.json','PRESERVATION.json','RESULT_ANALYSIS.json','COMPLETION_EXTENSION.json','protocol.json','editor.py','feature_builder.py','prepared_v1/ready.json')]
    binding={str(p.relative_to(ROOT)).replace('\\','/'):dict(exists=p.is_file(),SHA=sha256(p) if p.is_file() else None) for p in docfiles+required}
    write_json(HERE/'CONTROL_INPUT_BINDING.json',dict(status='METADATA_BINDING_ONLY_NOT_FULL_ARRAY_SHA_AUDIT',at=now,files=binding,no_old_science_change=True))
    matrix=[dict(source='original E0 audited all4915200 opt actions + E1 audited holdinner6817200',object='native/KIN/GT/eligible/B/T/motion/ID',reuse='prospective same-ID/source-SHA compatible saved arrays only; no new enumeration',cannot_replace='full current read-back SHA admission, new objectives/MLP resume, step2 IPW/permutation'),dict(source='original E1 prepared_v1 features/stat/identity',object='324float32 features and full4096 opt-only stats',reuse='same pool seed; tiny uses pool42 fulloptstats, no tiny-stat fit',cannot_replace='new fitted models or sampled weighted objective'),dict(source='old K1/R1-R5 and tiny experiments',object='historical observations',reuse='background only: actions/IDs/objectives differ',cannot_replace='new18 max diagnostic fits'),dict(source='old E1 checkpoint',object='historical I/S behavior',reuse='reference only, no warmstart',cannot_replace='matched M_full sampled training baseline')]
    write_json(HERE/'REUSE_MATRIX.json',dict(status='PLANNED_PENDING_FULL_ARRAY_SHA_ADMISSION',rows=matrix))
    failures=[]
    if disk.free<estimate['disk_required_free_bytes']:failures.append('DISK_FREE_LT_CONSERVATIVE_OUTPUT_TEMP_X1p3')
    if resource['status']!='PASS':failures.append('AVAILABLE_COMMIT_OR_PHYSICAL_LT_TASK_ESTIMATE_X1p3')
    if not all(v['exists'] for v in binding.values()):failures.append('REQUIRED_INPUT_METADATA_MISSING')
    status='STOP_RESOURCE_NO_WORKER' if failures else 'ADMISSION_ONLY_PASS_NOT_SCIENTIFIC_PREFLIGHT'
    report=dict(status=status,at=now,failures=failures,disk=dict(total=disk.total,used=disk.used,free=disk.free,required=estimate['disk_required_free_bytes']),resource=resource,step1_training=False,step2_training=False,scientific_preflight_PASS=False,active_worker=False,PID=None,ETA=None,no_delete=True,no_periodic_resource_retry=True)
    write_json(HERE/'PRECHECK.json',report)
    write_json(HERE/'runtime_control.json',dict(**report,authorization='human2026-10-26 execute steps1and2 only',science_models_started=0,science_models_completed=0,step3='NOT_AUTHORIZED',step4='NOT_AUTHORIZED',next_action='Resource admission failed: preserve all files, report disk/commit reading. Need sufficient output storage/commit or explicit alternate destination; no delete, OS/pagefile/userkill, threshold lowering, background worker or repeated polling.'))
    print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
