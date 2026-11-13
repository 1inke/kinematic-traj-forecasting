"""New namespace and explicit budget override; original sealed evidence is retained."""
from common import *
import ast,shutil
PRIOR=PROJECT/'qoder_work/GROUP_QUERY_MODE_SEQUENCE_EXEC_20261101'
def main():
    g=Guard(read(HERE/'MEMORY_ESTIMATE.json')['CPU_E']);g.check(force=True);files=dict(read(PRIOR/'INPUT_SOURCE_LOCK.json')['files'])
    for p,s in read(PRIOR/'FINAL_PRESERVATION.json')['files'].items():
        k=str((PRIOR/p).resolve())
        if k in files and files[k]!=s:raise ValueError('INHERITED_SHA_CONFLICT')
        files[k]=s
    for name in ('FINAL_PRESERVATION.json','FINAL_SEAL_CHECK.json'):
        p=PRIOR/name;files[str(p.resolve())]=sha(p)
    stage('AUTHORIZED_Q_PREFLIGHT_REUSE_SOURCE_SHA',count=len(files));verify(files,g.check)
    # Every mathematical evaluator, model, objective and statistics file exact.
    identical=[]
    for name in ('common.py','models.py','objectives.py','evaluate_saved.py','statistics_fixed.py','completion_review.py'):
        assert sha(PRIOR/name)==sha(HERE/name);identical.append(name)
    equivalent={}
    for name in ('train_stage.py','worker.py'):
        a=ast.parse((PRIOR/name).read_text(encoding='utf-8'));b=ast.parse((HERE/name).read_text(encoding='utf-8'))
        aa={n.name:n for n in a.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))};bb={n.name:n for n in b.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))};assert aa.keys()==bb.keys()
        for key in aa:
            if key!='main':assert ast.dump(aa[key],include_attributes=False)==ast.dump(bb[key],include_attributes=False),(name,key)
        def normalize(n):
            n=copy.deepcopy(n);out=[]
            for x in n.body:
                if isinstance(x,ast.Assert) and any(q in ast.unparse(x) for q in ('PREFLIGHT.json','BENCHMARK.json')):continue
                if isinstance(x,ast.ImportFrom) and x.module=='budget_admission':continue
                if isinstance(x,ast.Expr) and ast.unparse(x)=='validate(HERE)':continue
                out.append(x)
            n.body=out;return ast.dump(n,include_attributes=False)
        assert normalize(aa['main'])==normalize(bb['main']),name
        equivalent[name]='all functions EXACT; main only explicit budget admission changed'
    for folder in ('initial','orders'):
        for p in (PRIOR/folder).iterdir():assert sha(p)==sha(HERE/folder/p.name)
    for name in ('statistics.npz','split_ids.npz','diagnostic_ids.npz','BENCHMARK.json','MEMORY_ESTIMATE.json','IDENTITY_STATISTICS_AUDIT.json'):assert sha(PRIOR/name)==sha(HERE/name)
    pf=read(PRIOR/'PREFLIGHT_GPU.json');assert pf['status']=='PASS' and len(pf['independent_double'])==len(pf['resume'])==2
    assert all(r['status']=='PASS_EXACT' and r['updates']==6 for r in pf['resume'])
    assert all(r['all_parameter_gradient_error']<=1e-10 and all(z['winner_hinge_clip_faces_exact'] for z in r['FD']) for r in pf['independent_double'])
    for arm in ('Q1','Q3'):
        p=PRIOR/f'preflight/{arm}/saved128';audit=read(p/'audit.json');assert audit['status'].startswith('PASS_') and audit['N']==128
    write(HERE/'SOURCE_EQUIVALENCE.json',dict(status='PASS_BUDGET_AND_NAMESPACE_ONLY',byte_exact=identical,functions=equivalent,initial_stats_split_diagnostic_orders_byte_exact=True,old_numerical_preflight_SHA=sha(PRIOR/'PREFLIGHT_GPU.json'),no_old_result_overwritten=True))
    write(HERE/'PREFLIGHT.json',dict(status='PASS_NUMERICAL_PREFLIGHT_REUSED_WITH_EXPLICIT_BUDGET_AUTHORIZATION',numerical_GPU='PASS',independent_saved='PASS',exact_resume='PASS',
        reused_from=str(PRIOR),receipt_SHA=sha(PRIOR/'PREFLIGHT_GPU.json'),source_equivalence=sha(HERE/'SOURCE_EQUIVALENCE.json'),new_fixture_updates=0,old_fixture_updates=112,scientific_updates=0,not_claimed_fresh_GPU_reexecution=True))
    from budget_admission import validate,validate_values
    assert validate(HERE)
    rejected=[]
    for key,bad in [('allow_original_budget_override',False),('finite_stage','M'),('scientific_changes',True),('remove_or_reduce_audits',True)]:
        au=read(HERE/'authorization.json');au[key]=bad
        try:validate_values(read(HERE/'PREFLIGHT.json'),read(HERE/'BENCHMARK.json'),au)
        except ValueError:rejected.append(key)
        else:raise AssertionError('INVALID_ADMISSION_ACCEPTED '+key)
    write(HERE/'ADMISSION_REGRESSION.json',dict(status='PASS',actual_current_receipts_admitted=True,invalid_authorities_rejected=rejected,old_30min_STOP_not_changed=True))
    planned=5*2**30;free=shutil.disk_usage(HERE).free;assert free>=1.3*planned
    write(HERE/'PRELAUNCH_REVIEW.json',dict(status='PASS_AUTHORIZED_BUDGET_REUSE',actual_EST_seconds=read(HERE/'BENCHMARK.json')['conservative_full_seconds'],
        scope='finiteQ6fresh8epoch complete original audits',budget_override_explicit=True,scientific_sources_equivalent=True,original_preflight_all_numerical_gates_reused_by_SHA=True,
        no_science_retraining=True,source_count_before_extension=len(files),disk_free=free,disk_required=1.3*planned,no_delete=True))
    for p in HERE.rglob('*'):
        if p.is_file() and p.name not in ('progress.json','runtime_control.json','source_manifest.json') and '__pycache__' not in p.parts:files[str(p.resolve())]=sha(p)
    dg=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    write(HERE/'source_manifest.json',dict(status='PRODUCTION_SOURCE_LOCK_Q_HUMAN_BUDGET_OVERRIDE',count=len(files),digest=dg,files=files))
    write(HERE/'runtime_control.json',dict(status='READY_TO_START_AUTHORIZED_Q',active_worker=False,planning_seconds=read(HERE/'BENCHMARK.json')['conservative_full_seconds'],
        half_count=0,startup_count=0,no_app_automation=True,exit_chat_after_startup=True,no_delete=True))
    print('AUTHORIZED LOCK PASS',len(files),dg,flush=True)
if __name__=='__main__':main()
