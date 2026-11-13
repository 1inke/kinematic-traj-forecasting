"""Independent scalar/partition checks of the new saved summaries; no model or NN replay."""
import json,hashlib,runpy,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent

def read(name):return json.loads((HERE/name).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    names=['RESULT.json','FULL_THREE_RECORDS.json','FULL_THREE_EXPLORATORY_CIS.json','SUPPORT.json','FEATURE_SHIFT.json','PAIRED_SUPPORT_GAPS.json','FIXTURES.json','SOURCE_LOCK.json','protocol.json','run_diagnosis.py','DESIGN.md','CLAIM_INTENT.json']
    before={str(HERE/n):sha(HERE/n) for n in names}
    result=read('RESULT.json');records=read('FULL_THREE_RECORDS.json');cis=read('FULL_THREE_EXPLORATORY_CIS.json')
    support=read('SUPPORT.json');shifts=read('FEATURE_SHIFT.json');gaps=read('PAIRED_SUPPORT_GAPS.json')
    assert result['status']=='PASS_FROZEN_DISTRIBUTION_DIAGNOSIS'
    assert len(records)==648 and len(cis)==720 and len(support)==56 and len(shifts)==308 and len(gaps)==240
    for r in records:
        assert 0<=r['good']<=r['n'] and r['good']-r['native_good']==r['rescued']-r['lost']
        assert r['effective_good_total']>=r['good'] and r['effective_good_total']<=r['raw_good_total']
        expected=all(r[k]<=r['native_'+k]*1.01 for k in ['speed_p95','accel_mean','accel_p95','jerk_mean','jerk_p95'])
        assert expected==r['motion_allfive_1pct']
    for c in cis:
        assert c['ci_low']<=c['ci_high'] and c['exploratory'] and c['models']==3 and c['repetitions']==2000
    for s in support:
        assert s['reference_n']+s['calibration_n']==s['source_n']
        assert 0<=s['target_out_rate']<=1 and abs(s['source_calibration_out_rate']-.05)<.005
        assert s['same_track_neighbors_excluded']
    keys={(r['study'],r['split'],r['family'],r['group']) for r in gaps}
    for key in keys:
        rows={r['condition']:r for r in gaps if (r['study'],r['split'],r['family'],r['group'])==key}
        assert rows['both_supported']['n']+rows['grouped_unsupported']['n']<=rows['all']['n']
        assert rows['random_supported_grouped_not']['n']+rows['both_unsupported']['n']==rows['grouped_unsupported']['n']
        for r in rows.values():
            assert r['thin']==(r['n']<100 or r['tracks']<20)
            if r['n']:
                assert abs(r['grouped_minus_random_pp']-100*(r['grouped_coverage']-r['random_coverage']))<1e-12
                assert abs(r['grouped_coverage']-r['native_coverage']-(sum(r['grouped_rescued'])-sum(r['grouped_lost']))/(3*r['n']))<1e-12
    # Exercise the exact production digest-rejection branch against a deliberate wrong digest.
    module=runpy.run_path(str(HERE/'run_diagnosis.py'),run_name='saved_audit_not_main')
    try:module['verify']({str(HERE/'protocol.json'):'0'*64})
    except AssertionError:bad_sha_rejected=True
    else:raise AssertionError('BAD_SHA_WAS_ACCEPTED')
    for p,h in before.items():assert sha(Path(p))==h,p
    out=dict(status='PASS_SAVED_SCALAR_PARTITION_AUDIT',records=648,original_CIs=720,support_rows=56,shift_rows=308,stratum_rows=240,all_quality_rescue_lost_and_motion_boolean_consistent=True,stratum_partitions_consistent=True,actual_bad_SHA_rejected=bad_sha_rejected,new_summary_SHA=before,no_neural_model_forward=True,no_optimizer=True,no_neighbor_refit_or_replay=True,no_new_bootstrap=True)
    (HERE/'SUMMARY_AUDIT.json').write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if k!='new_summary_SHA'}),flush=True)

if __name__=='__main__':main()
