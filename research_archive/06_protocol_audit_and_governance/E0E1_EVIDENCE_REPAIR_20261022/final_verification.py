"""Final contract closure and self-test of saved-result refusal paths."""
from common import *
import subprocess,platform,importlib.metadata,copy

def validate_record(record,expected_contract):
    if record['input_contract']!=expected_contract:raise ValueError('BAD_SOURCE_CONTRACT')
    if not record['outputs']:raise ValueError('EMPTY_OUTPUT_CONTRACT')
    for name,digest in record['outputs'].items():
        if sha256(HERE/name)!=digest:raise ValueError('BAD_OUTPUT_SHA')
    return True

def main():
    from preflight_v3 import verify_v3
    verify_sources();verify_v3();r=read_json(HERE/'frozen_eval/checkpoint_replay_manifest.json');review=read_json(HERE/'completion_review/REVIEW.json');expected={f'{a}_seed{s}_{split}' for s,a in RUNS for split in ['FIT','inner']};actual={x['job'] for x in r['records']}
    if expected!=actual or len(r['records'])!=22 or review['jobs']!=22 or review['status']!='PASS_CURRENT_SAVED_ARITHMETIC':raise RuntimeError('UNIQUE22_CONTRACT_NOT_CLOSED')
    digest=sha256(HERE/'source_manifest_before.json');outputbytes=0
    for job in sorted(expected):
        item=read_json(HERE/'frozen_eval/done'/f'{job}.json');validate_record(item,digest)
        if not item['record']['strict_weights_buffers_exact']:raise RuntimeError('NO_FROZEN_EVIDENCE')
        audit=read_json(HERE/'frozen_eval/audit'/f'{job}.json')
        if audit['status']!='PASS' or not audit['frozen_labels_verified'] or audit['script_sha']!=sha256(ROOT/'audit/verify.py'):raise RuntimeError('BAD_INDEPENDENT_AUDIT_CONTRACT')
        outputbytes+=sum((HERE/p).stat().st_size for p in item['outputs'])
    # Mutate in-memory manifest only, never scientific files.
    first=read_json(HERE/'frozen_eval/done'/f'{sorted(expected)[0]}.json');bad=copy.deepcopy(first);bad['input_contract']='deliberately-wrong-source';source_rejected=False;output_rejected=False
    try:validate_record(bad,digest)
    except ValueError:source_rejected=True
    bad=copy.deepcopy(first);key=next(iter(bad['outputs']));bad['outputs'][key]='0'*64
    try:validate_record(bad,digest)
    except ValueError:output_rejected=True
    if not source_rejected or not output_rejected:raise RuntimeError('BAD_SKIP_CONTRACT_WAS_ACCEPTED')
    code={p.relative_to(ROOT).as_posix():sha256(p) for p in HERE.rglob('*.py') if 'failures' not in p.parts};git=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip()
    write_json(HERE/'ENVIRONMENT_AND_CODE.json',dict(python=platform.python_version(),platform=platform.platform(),packages={k:importlib.metadata.version(k) for k in ['numpy','scipy','torch','psutil']},git_HEAD=git,dirty_worktree=True,new_repair_code_SHA256=code,note='Posthoc repair code record, not old training-time source lock. No pip install or environment alteration in this run.'))
    value=dict(status='COMPLETE_P0_P1_P2_P3_EVIDENCE_REPAIR',unique_models=11,unique_model_split_jobs=22,full_independent_geometry_motion_cluster_audits=22,extra_independent_diagnostic_jobs=review['jobs'],extra_review_maxdiff=review['max_abs_diff'],source_manifest_before_sha256=digest,all_original_locked_sources_unchanged=True,strict_weights_and_buffers_exact_all22=True,bad_source_skip_rejected=source_rejected,bad_output_skip_rejected=output_rejected,new_training_runs=0,new_optimizer_steps=0,P4_started=False,old_checkpoints_upgraded_to_resume=False,output_payload_bytes=outputbytes,source_extension_note='v2 constant-compute and v3 legacy-schema fixes explicitly locked; final review/report tools are later read-only stages, not old training sources',science_gates_relaxed=False)
    write_json(HERE/'FINAL_VERIFICATION.json',value);write_json(HERE/'source_manifest_final.json',dict(files=capture(current_sources()),original_input_preservation_verified=True,posthoc_new_source_extensions_explicit=True));progress('ANALYZED',done_jobs=22,P4_started=False,new_training_runs=0);print(json.dumps(value,ensure_ascii=False))

if __name__=='__main__':main()
