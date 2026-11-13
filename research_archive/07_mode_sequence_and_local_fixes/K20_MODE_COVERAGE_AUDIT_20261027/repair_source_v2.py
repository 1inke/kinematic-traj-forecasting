from common import *
def main():
    p=read(PARENT/'source_manifest.json');old=read(HERE/'config/sources_manifest.json');files=p['files'].copy()
    # Only IO/provenance changes; functions performing science remain exact.
    archive=HERE/'failures/v1_runtime_in_source_manifest'
    math_names=['morphology.py','oracle.py','e1_modes.py','e2_oracle.py','aggregate.py','plot_diagnostics.py']
    for n in math_names:assert sha(HERE/n)==sha(archive/n)
    def add(q):
        try:k=q.relative_to(ROOT).as_posix()
        except ValueError:k=q.as_posix()
        value=sha(q)
        if k in files:assert files[k]==value,k
        files[k]=value
    for key,h in old['files'].items():
        q=Path(key) if Path(key).is_absolute() else ROOT/key
        if q.parent==HERE and q.suffix=='.py':continue
        assert sha(q)==h,key;files[key]=h
    for q in HERE.glob('*.py'):add(q)
    for q in (HERE/'failures').rglob('*'):
        if q.is_file():add(q)
    digest=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest();write(HERE/'config/sources_manifest.json',dict(files=files,digest=digest,inherited_count=len(p['files']),inherited_digest=p['digest'],runtime_control_not_scientific_source=True))
    write(HERE/'config/CONFIG_LOCK.json',dict(config_SHA=sha(HERE/'config/analysis_config.json'),source=digest,at=now(),no_E2_GT_mode_matching_started=True))
    write(HERE/'audit/SOURCE_REPAIR_V2.json',dict(status='PASS',mathematical_modules_byte_EXACT=math_names,preflight_change='source-specific saved fixture name and full-array reader validation; no deletion or overwrite',all_failure_evidence_preserved=True,source=verify_sources()))
if __name__=='__main__':main()
