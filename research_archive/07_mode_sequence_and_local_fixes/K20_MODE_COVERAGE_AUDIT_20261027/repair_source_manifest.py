from common import *
def main():
    old=read(HERE/'failures/v1_runtime_in_source_manifest/ARCHIVE_RECEIPT.json');exact=[]
    for name,h in old['files'].items():
        if name.endswith('.py') and name!='prepare_inputs.py':assert sha(HERE/name)==h;exact.append(name)
    assert sha(HERE/'config/analysis_config.json')==old['files']['config/analysis_config.json'];parent=read(PARENT/'source_manifest.json');files=parent['files'].copy();old_source=read(HERE/'config/sources_manifest.json')
    mutable={'qoder_work/K20_MODE_COVERAGE_AUDIT_20261027/progress.json','qoder_work/K20_MODE_COVERAGE_AUDIT_20261027/worker_state.json'}
    for p,h in old_source['files'].items():
        if p in mutable:continue
        actual=Path(p) if Path(p).is_absolute() else ROOT/p
        if actual.parent==HERE and actual.suffix=='.py':continue
        assert sha(actual)==h,p;files[p]=h
    def add(p):
        try:key=p.relative_to(ROOT).as_posix()
        except ValueError:key=p.as_posix()
        h=sha(p)
        if key in files:assert files[key]==h,key
        files[key]=h
    for p in HERE.glob('*.py'):add(p)
    for p in (HERE/'failures').rglob('*'):
        if p.is_file():add(p)
    digest=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
    write(HERE/'config/sources_manifest.json',dict(files=files,digest=digest,inherited_count=len(parent['files']),inherited_digest=parent['digest'],runtime_control_not_scientific_source=True))
    write(HERE/'config/CONFIG_LOCK.json',dict(config_SHA=sha(HERE/'config/analysis_config.json'),source=digest,at=now(),no_E2_GT_mode_matching_started=True))
    write(HERE/'audit/SOURCE_REPAIR.json',dict(status='PASS',cause='progress and worker_state erroneously admitted as immutable sources',exact_scientific_modules=exact,only_prepare_change='root source suffix whitelist excludes mutable JSON',config_and_all_input_SHA_EXACT=True,old_failed_manifest_and_code_preserved=True,all_parent_locked_sources_unchanged=True,current_source=verify_sources(),old_failure_not_relabelled=True))
    print('PASS REPAIRED SOURCE',flush=True)
if __name__=='__main__':main()
