"""Full saved128 per admissible arm: canonical and independent geometry/pdist/5motion."""
from core import *
import core,sys,time
sys.modules['base']=core
ev=module('_step3_preflight_saved',PREVIOUS.parent/'execution_staged_20261026/evaluate_saved.py')
def main():
    g=guard('CPU');source0=verify();gpu=read_json(HERE/'PREFLIGHT_GPU.json');records=[]
    for r in gpu['records']:
        path=HERE/r['saved128'];assert sha256(path)==r['saved_SHA']
        with np.load(path) as q:z={k:q[k] for k in q.files}
        start=time.perf_counter();ee,cc=ev.independent_check(z,z['scores'],z['chosen'],z['paths'])
        rows,_=ev.evaluate(z['paths'],z['gt'],z['group'],z['native'],z['sample_id'],z['track_key'],include_clusters=False)
        for i,name in enumerate(('overall','stable','transition','change')):
            mask=np.ones(len(z['sample_id']),bool) if not i else z['group']==i-1
            if mask.any():
                for paths,key in ((z['paths'],'motion'),(z['native'],'native_motion')):
                    other=ev.independent_motion(paths[mask]);assert all(abs(other[k]-rows[name][key][k])<=2e-12 for k in other)
        records.append(dict(arm=r['arm'],n=128,seconds=time.perf_counter()-start,groups=rows,all_fixed19_STOP_samecandidate_cluster_pdist_5motion=True))
        g.check()
    # Actual completed/bad-source/output readers, new tiny fixtures only.
    f=HERE/'reader_fixtures';f.mkdir();write_json(f/'payload.json',{'fixture':1});c=dict(source=source0,split='opt_only',stats=sha256(HERE/'prepared/stats_seed42.npz'))
    finish(f,c);completed(f,c)
    rejected=[]
    for key in ('source','split','stats'):
        bad=dict(c);bad[key]='wrong'
        try:completed(f,bad)
        except AssertionError:rejected.append(key)
    bad=HERE/'reader_bad_output_fixture';bad.mkdir();write_json(bad/'payload.json',{'fixture':2})
    write_json(bad/'done.json',dict(contract=c,outputs={'payload.json':sha256(f/'payload.json')}))
    try:completed(bad,c)
    except AssertionError:rejected.append('output')
    assert len(rejected)==4 and verify()==source0
    write_json(HERE/'PREFLIGHT_CPU_SAVED.json',dict(status='PASS',source=source0,records=records,completed_skip_actual=True,bad_readers_rejected=rejected,no_model_forward=True))
    resource_report(HERE/'PREFLIGHT_CPU_RESOURCE.json',g);print('PASS full saved128/arm geometry, clustering and all group motion; actual reader rejection',flush=True)
if __name__=='__main__':main()
