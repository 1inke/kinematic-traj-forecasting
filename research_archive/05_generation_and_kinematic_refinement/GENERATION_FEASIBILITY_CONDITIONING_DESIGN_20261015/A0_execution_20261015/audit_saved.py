"""Independent saved-coordinate bound/quality/cluster/physical audit, no model imports."""
import common as c
import analytic_envelope as e
import numpy as np,time
def validate(z,data,ix,reported):
    assert z['native'].shape==(len(ix),20,12,2) and np.isfinite(z['ideal_ball_paths']).all() and np.isfinite(z['bottom4_paths']).all()
    for saved,key in [('sample_id','sample_id'),('groups','groups'),('track','track'),('sequence','sequence'),('scene','scene'),('slots','slots'),('gt','gt'),('native','native')]:assert np.array_equal(z[saved],data[key][ix]),saved
    independent=e.construct(data['native'][ix],data['gt'][ix],data['slots'][ix],True);maximum=0.
    for key,v in independent.items():maximum=max(maximum,c.close(z[key],v))
    # Geometry independently attains the analytic per-slot ADE/FDE lower errors.
    distance=np.hypot(z['ideal_ball_paths'][...,0]-z['gt'][:,None,:,0],z['ideal_ball_paths'][...,1]-z['gt'][:,None,:,1])
    maximum=max(maximum,c.close(distance,z['lower_distance']),c.close(distance.mean(-1),z['ball_slot_ADE']),c.close(distance[:,:,-1],z['ball_slot_FDE']))
    radius=np.hypot(*(z['ideal_ball_paths']-z['native']).transpose(3,0,1,2))
    assert np.all(radius<=e.RADIUS+2e-12),'NUMERIC_GEOMETRY_BOUND'
    mask=np.zeros((len(ix),20),bool);mask[np.arange(len(ix))[:,None],z['slots']]=True
    assert np.array_equal(z['bottom4_paths'][~mask],z['native'][~mask])
    nativeE=np.hypot(z['native'][...,0]-z['gt'][:,None,:,0],z['native'][...,1]-z['gt'][:,None,:,1])
    assert np.all(distance<=nativeE+2e-12),'QUALITY_MONOTONIC_NUMERIC_CHECK'
    ind=e.metrics(z,z['groups'],z['track'],True);maximum=max(maximum,c.compare(ind,reported))
    for port in c.PORTS:
        for group,row in ind[port].items():
            if not row['n']:continue
            assert row['lost']==0
    # Literal coverage must match closed form: tolerance never changes the gate.
    allgood=(z['ball_slot_J'].min(1)<=1);bottomgood=(z['bottom_slot_J'].min(1)<=1)
    actualall=c.errors(z['ideal_ball_paths'],z['gt'],True)[3].any(1);actualbottom=c.errors(z['bottom4_paths'],z['gt'],True)[3].any(1)
    assert np.array_equal(allgood,actualall) and np.array_equal(bottomgood,actualbottom),'CLASSIFICATION_ROUNDOFF_STOP'
    return maximum
def main():
    t=time.perf_counter();files=c.read(c.H/'source_manifest.json')['files'];c.verify(files);data,splits=c.load_data();records=c.read(c.H/'JOBS.json');rows=[]
    assert set(records)==set(c.SIZES)
    for name,r in records.items():
        c.stage('A0_INDEPENDENT_SAVED_AUDIT',split=name,done=len(rows),total=4);c.valid_done(c.read(c.H/'jobs'/f'{name}.done.json'),c.digest(files))
        z=c.npz(r['output']);error=validate(z,data,splits[name],r['metrics']);assert error<=2e-12
        rows.append(dict(split=name,n=len(z['sample_id']),maxerror=error,full_slots_bounds_and_all3K20_quality_cluster_motion=True,fixed16_EXACT=True,literal_analytic_constructed_coverage_EXACT=True))
        del z;c.gc.collect()
    c.verify(files);c.verify({p:h for r in records.values() for p,h in r['files'].items()})
    c.write('ARITHMETIC_AUDIT.json',dict(status='PASS',jobs=4,populations=12,groups=48,records=rows,maxerror=max(x['maxerror'] for x in rows),NoModelForward=True,NoOptimizer=True,no_newCI=True,source_output_SHA_initial_final=True,elapsed_seconds=time.perf_counter()-t));return rows
if __name__=='__main__':main()
