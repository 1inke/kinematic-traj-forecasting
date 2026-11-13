import common as c
import safety as s
import metrics as m
import numpy as np,time,gc

def validate(z,data,ix,reported,certificate):
    for k in ('sample_id','groups','track','sequence','scene','native','gt','slots'):assert np.array_equal(z[k],data[k][ix]),k
    ind=s.construct(data['native'][ix],data['gt'][ix],data['slots'][ix],True);maximum=0.
    for k,v in ind.items():
        if v.dtype==bool:assert np.array_equal(v,z[k]),('INDEPENDENT_CLASSIFICATION',k)
        else:maximum=max(maximum,c.close(v,z[k]))
    assert np.array_equal(ind['fallback'],z['fallback']),'FALLBACK_CLASSIFICATION_STOP'
    for p in (z['native'],z['safe4'],z['safe20']):assert np.isfinite(p).all() and not s.strict_violations(z['native'],p,True).any()
    assert np.array_equal(z['safe4'][~z['mask']],z['native'][~z['mask']])
    dist=lambda p:np.sqrt(((p-z['gt'][:,None])**2).sum(-1))
    assert np.all(dist(z['safe20'])<=dist(z['native'])+2e-12),'PERPOINT_QUALITY_ERROR'
    assert np.all(np.sqrt(((z['safe20']-z['native'])**2).sum(-1))<=s.RADIUS+2e-12)
    good=[c.errors(p,z['gt'],True)[3].any(1) for p in (z['native'],z['safe4'],z['safe20'],z['native']+z['ball_residual'])]
    lb=np.maximum(dist(z['native'])-s.RADIUS,0.)
    ub20=((lb.mean(-1)<=.2)&(lb[:,:,-1]<=.4)).any(1)
    ub4=np.where(z['mask'],(lb.mean(-1)<=.2)&(lb[:,:,-1]<=.4),c.errors(z['native'],z['gt'],True)[3]).any(1)
    assert all(np.all(~a|b) for a,b in zip(good,good[1:])) and np.all(~good[1]|ub4) and np.all(~good[2]|ub20)
    actual=m.metrics(z,z['groups'],z['track'],True);maximum=max(maximum,c.compare(actual,reported),c.compare(m.certificate_summary(ind|{k:z[k] for k in ('groups',)},z['groups']),certificate))
    for port in c.PORTS:
        for row in actual[port].values():
            if row['n']:assert row['lost']==0 and row['motion_literal_1pct'],'LITERAL_GROUP_SAFETY_OR_LOST_STOP'
    return maximum

def main():
    t=time.perf_counter();files=c.read(c.H/'source_manifest.json')['files'];c.verify(files);data,splits=c.load_data();records=c.read(c.H/'JOBS.json');rows=[]
    assert set(records)==set(c.SIZES)
    for name,row in records.items():
        c.stage('S0_INDEPENDENT_SAVED_AUDIT',split=name,done=len(rows));c.valid_done(c.read(c.H/'jobs'/f'{name}.done.json'),c.digest(files))
        z=c.npz(row['output']);error=validate(z,data,splits[name],row['metrics'],row['certificate']);assert error<=2e-12
        rows.append(dict(split=name,n=len(z['native']),maxerror=error,full_K20_quality_original_complete_link_five_motion=True,independent_ray_fallback_classification_EXACT=True,fixed16_EXACT=True,zero_lost=True));del z;gc.collect()
    c.verify(files);c.verify({p:h for r in records.values() for p,h in r['files'].items()})
    c.write('ARITHMETIC_AUDIT.json',dict(status='PASS',jobs=4,ports=12,groups=48,rows=rows,maxerror=max(x['maxerror'] for x in rows),elapsed_seconds=time.perf_counter()-t,no_model_or_optimizer=True,saved_arithmetic_not_model_replay=True))

if __name__=='__main__':main()
