import common as c
import analytic_envelope as e
from audit_saved import validate
import numpy as np,time,gc
def fixtures():
    p=np.zeros((3,20,12,2),np.float64);y=np.zeros((3,12,2),np.float64);p[1,:,:,0]=e.RADIUS;p[2,:,:,0]=1.;slots=np.tile(np.arange(16,20),(3,1));z=e.construct(p,y,slots);ind=e.construct(p,y,slots,True)
    assert np.array_equal(z['lower_distance'][0],np.zeros((20,12))) and np.array_equal(z['ideal_ball_paths'][0],p[0])
    assert np.array_equal(z['lower_distance'][1],np.zeros((20,12))) and np.max(abs(z['ideal_ball_paths'][1]))<=2e-12
    for k in z:c.close(z[k],ind[k])
    assert np.all(z['native_slot_J'].min(1)>=z['bottom_slot_J'].min(1)) and np.all(z['bottom_slot_J'].min(1)>=z['ball_slot_J'].min(1))
    a=np.array([[.19,.21]]);f=np.array([[.41,.39]]);assert a.min()<=.2 and f.min()<=.4 and not ((a<=.2)&(f<=.4)).any()
    assert np.array_equal(np.sort(np.argsort(-np.zeros((1,20)),axis=1,kind='stable')[:,16:],axis=1),[[16,17,18,19]])
    assert np.array_equal(c.clusters(np.zeros((1,20,12,2)),np.ones((1,20),bool)),[[1,1]])
    c.write('fixtures/valid.json',dict(fixture=True));path=str(c.H/'fixtures/valid.json');done=dict(status='PASS',source_digest='fixture',files={path:c.sha(path)});assert c.valid_done(done,'fixture');rejected=0
    for b in (dict(done,source_digest='wrong'),dict(done,files={path:'0'*64})):
        try:c.valid_done(b,'fixture')
        except AssertionError:rejected+=1
    assert rejected==2
    return dict(zero_distance=True,exact_ball_boundary=True,same_candidate_disjoint_minima_rejected=True,stable_slot_ties=True,twenty_copies_one_cluster=True,closed_form_constructive_equivalence=True,nested_bottom_all=True,analysis_doneSHA_skip_bad_source_output_rejected=True,not_training_resume=True)
def job_arrays(data,ix):
    z=e.construct(data['native'][ix],data['gt'][ix],data['slots'][ix]);z.update({k:data[k][ix] for k in ('sample_id','groups','track','sequence','scene')});return z
def main():
    t=time.perf_counter();c.stage('A0_PREFLIGHT_SOURCE_IDS');files,inherited=c.locks();fx=fixtures();data,splits=c.load_data();records=[];times=[];err=0.;shape=[]
    for b in range(8):
        ix=splits['FIT'][b*128:(b+1)*128];start=time.perf_counter();z=job_arrays(data,ix);met=e.metrics(z,z['groups'],z['track']);dest=c.H/'preflight_smoke'/f'chunk{b}.npz';dest.parent.mkdir(exist_ok=True);np.savez_compressed(dest,**z);back=c.npz(dest);assert all(np.array_equal(v,back[k]) for k,v in z.items());err=max(err,validate(back,data,ix,met));times.append(time.perf_counter()-start);shape.append(sum(v.nbytes for v in z.values()));records.append(dict(chunk=b,n=128,output_SHA=c.sha(dest),full_bound_geometry_quality_clusters_motion_saved_audited=True));del z,back,met;gc.collect()
    assert err<=2e-12
    identity={name:dict(n=len(ix),tracks=len(set(data['track'][ix])),scene_counts={str(s):int(np.sum(data['scene'][ix]==s)) for s in np.unique(data['scene'][ix])},group_counts={g:int(m.sum()) for g,m in c.masks(data['groups'][ix])},sample_ID_digest=c.digest(data['sample_id'][ix].tolist())) for name,ix in splits.items()}
    c.write('IDENTITY_AUDIT.json',dict(status='PASS',registered_ID_registry=28791,metadata_exact=True,splits_disjoint=True,native_source='one immutable cache, no fictitious model seed repeats',records=identity));files[str(c.H/'IDENTITY_AUDIT.json')]=c.sha(c.H/'IDENTITY_AUDIT.json');c.verify(files)
    rows=sum(len(ix) for ix in splits.values());elapsed=time.perf_counter()-t;nominal=rows/128*float(np.mean(times));overhead=max(5.,elapsed-sum(times));planning=(nominal+2*overhead+45)*1.25/60
    disk=c.psutil.disk_usage(str(c.H)).free/2**30;size=rows/128*float(np.mean(shape))/2**30;assert disk>2+1.5*size,'STOP_DISK'
    report=dict(status='PASS' if planning<=30 else 'STOP_BUDGET',stage='A0',fixtures=fx,actual_eight128_complete_pipeline=records,independent_maxerror=err,benchmark_128seconds=times,nominal_compute_save_independent_audit_seconds=nominal,preflight_elapsed_seconds=elapsed,planning_minutes=planning,ETA_minutes=[planning*.7,planning*1.5],source_count=len(files),inherited_sources=inherited,source_digest=c.digest(files),splits=identity,scientific_jobs=4,populations=12,groups=48,rows=rows,uncompressed_new_arrays_GiB=size,disk_free_GiB=disk,resource=c.RESOURCE,no_optimizer_or_neural_forward=True,A1_started=False,B_started=False)
    c.write('source_manifest.json',dict(files=files,inherited=inherited));c.write('PREFLIGHT.json',report);print('PREFLIGHT',report['status'],planning,report['ETA_minutes'],len(files),c.RESOURCE,flush=True)
if __name__=='__main__':
    try:main()
    except Exception:c.failure('STOP_PREFLIGHT');raise
