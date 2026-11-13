"""Report-only aggregation and immutable SHA verification; no model or clustering replay."""
import csv, hashlib, json
from pathlib import Path
from datetime import datetime, timezone, timedelta

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
CHILD = PARENT / 'audit_recovery_20261024'
OUT = PARENT / 'q0_attribution'

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(4*2**20), b''):
            h.update(block)
    return h.hexdigest()

def verify(p):
    m = read(p)
    expected = {k: v['sha256'] for k, v in m['files'].items()}
    d = hashlib.sha256(json.dumps(expected,sort_keys=True,separators=(',', ':')).encode()).hexdigest()
    assert d == m['digest']
    for v in m['files'].values():
        assert sha(Path(v['path'])) == v['sha256'], v['path']
    return {'count':len(expected), 'digest':d}

def table(name):
    with (OUT/name).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))

def main():
    before = [verify(PARENT/'source_manifest.json'),verify(CHILD/'source_manifest.json')]
    audit = read(CHILD/'REVIEW.json')
    assert audit['status'] == 'PASS_INDEPENDENT_SAVED_Q0'
    assert read(CHILD/'RESULT.json')['status'] == 'SUCCESS_AUDIT_RECOVERY'
    assert read(CHILD/'worker_state.json')['status'] == read(CHILD/'exit_status.json')['status'] == 'SUCCESS'
    tables = {n:table(n+'.csv') for n in ('COVERAGE_TRANSITIONS','CLUSTER_CONTRIBUTIONS','MOTION_ATTRIBUTION','NATIVE_SUCCESS_SUPPORT','ALL_POPULATION_MOTION')}
    counts = {k:len(v) for k,v in tables.items()}
    assert counts == dict(COVERAGE_TRANSITIONS=72,CLUSTER_CONTRIBUTIONS=96,MOTION_ATTRIBUTION=288,NATIVE_SUCCESS_SUPPORT=288,ALL_POPULATION_MOTION=96)
    cells = tables['CLUSTER_CONTRIBUTIONS']
    summaries = []
    for split in ('FIT','inner'):
        for seed in ('42','43','44'):
            for group in ('overall','stable','transition','change'):
                c={r['cell']:r for r in cells if r['split']==split and r['seed']==seed and r['group']==group}
                assert len(c)==4
                assert abs(sum(float(r['signed_whole_population_contribution']) for r in c.values())-float(c['both_good']['total_delta']))<2e-12
                t=next(r for r in tables['COVERAGE_TRANSITIONS'] if r['split']==split and r['seed']==seed and r['group']==group and r['reference']=='iso')
                summaries.append(dict(seed=int(seed),split=split,group=group,N=int(t['N']),iso_good=int(t['reference_good']),mix_good=int(t['treatment_good']),rescue=int(t['treatment_only']),lost=int(t['reference_only']),both_good=int(t['both_good']),both_bad=int(t['both_bad']),coverage_delta_pp=100*int(t['net_good'])/int(t['N']),effectivegood_delta=float(c['both_good']['total_delta']),signed_contributions={k:float(v['signed_whole_population_contribution']) for k,v in c.items()},positive_shares={k:float(v['positive_contribution_share']) for k,v in c.items()},already_covered_more_clusters=int(c['both_good']['same_already_covered_more_clusters'])))
    motions = tables['ALL_POPULATION_MOTION']
    motion_counts = {arm:sum(r['all_five_pass']=='True' for r in motions if r['arm']==arm) for arm in ('iso','mix','oldA','native')}
    inv=read(PARENT/'q1_map_inventory/MAP_INVENTORY.json')
    maps=[{k:r[k] for k in ('recording_id','n_FIT','n_inner','complete_image_and_H_candidates','admission','missing')} for r in inv['records']]
    assert sum(r['n_FIT'] for r in maps)==18909 and sum(r['n_inner'] for r in maps)==1856
    after=[verify(PARENT/'source_manifest.json'),verify(CHILD/'source_manifest.json')]
    assert before==after
    result=dict(status='PASS_REPORT_TABLES_AND_FINAL_SHA',at=datetime.now(timezone(timedelta(hours=8))).isoformat(),source_before=before,source_after=after,csv_counts=counts,audit_status=audit['status'],paired_rows=audit['rows'],cluster_windows=audit['cluster_windows'],audit_maxerror=audit['maximum_reconstruction_error'],population_summaries=summaries,all_five_motion_pass_of_24=motion_counts,all_motion_rows=motions,all_event_motion_rows=tables['MOTION_ATTRIBUTION'],all_support_rows=tables['NATIVE_SUCCESS_SUPPORT'],all_coverage_rows=tables['COVERAGE_TRANSITIONS'],map_status=inv['status'],map_records=maps,formal_new_training=0,no_model_forward=True,no_gradient_replay=True,no_clustering_replay_in_this_completion_script=True,original_failure_preserved=True,recovery_peak_private_bytes=max(o['snapshot']['peak_private'] for o in audit['resource_observations']),recovery_peak_rss_bytes=max(o['snapshot']['peak_rss'] for o in audit['resource_observations']))
    (HERE/'RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','csv_counts','all_five_motion_pass_of_24','recovery_peak_private_bytes','recovery_peak_rss_bytes')}))
    for r in summaries:
        if r['group']=='change': print(json.dumps(r))
    for r in motions:
        if r['split']=='inner' and r['group']=='change': print(json.dumps(r))
    for r in tables['COVERAGE_TRANSITIONS']:
        if r['split']=='inner' and r['group']=='change' and r['reference']=='native':print(json.dumps(r))

if __name__ == '__main__': main()
