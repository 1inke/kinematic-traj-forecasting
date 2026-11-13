import common as c
import safety as s
import metrics as m
from audit_saved import validate
import numpy as np,time,gc

def job_arrays(data,ix):
    z=s.construct(data['native'][ix],data['gt'][ix],data['slots'][ix]);z.update({k:data[k][ix] for k in ('sample_id','groups','track','sequence','scene')})
    for port in c.PORTS:
        a,f,j,g,temporal=c.errors(z['native' if port=='native20' else port],z['gt']);z[port+'_slot_ADE']=a;z[port+'_slot_FDE']=f;z[port+'_slot_J']=j;z[port+'_slot_good']=g
    return z

def main():
    t=time.perf_counter();c.stage('S0_PREFLIGHT_SOURCE_IDS');files,inherited=c.locks();fx=s.fixtures();data,splits=c.load_data();records=[];times=[];sizes=[];maximum=0.
    # A0 identity and coordinate contract are inherited; additionally check the
    # original label future agrees with frozen encoder train-relative GT.
    encoder=c.ROOT/'GENERATION_NATIVE_DECODER_SERIES/train_encoder.npz'
    with np.load(encoder,allow_pickle=False) as z:
        keys=z.files
        if 'sample_id' in keys:assert np.array_equal(z['sample_id'],data['sample_id'][:len(z['sample_id'])])
        if 'gt' in keys:assert np.array_equal(z['gt'],data['gt'][:len(z['gt'])])
    files[str(encoder)]=c.sha(encoder)
    # Analysis contract resume is SHA/source skip, never training RNG resume.
    c.write('fixtures/valid.json',dict(fixture=True));vp=str(c.H/'fixtures/valid.json');done=dict(status='PASS',source_digest='fixture',files={vp:c.sha(vp)});assert c.valid_done(done,'fixture');rejected=0
    for bad in (dict(done,source_digest='wrong'),dict(done,files={vp:'0'*64})):
        try:c.valid_done(bad,'fixture')
        except AssertionError:rejected+=1
    assert rejected==2
    assert np.array_equal(np.sort(np.argsort(-np.zeros((1,20)),axis=1,kind='stable')[:,16:],axis=1),[[16,17,18,19]])
    assert np.array_equal(c.clusters(np.zeros((1,20,12,2)),np.ones((1,20),bool)),[[1,1]])
    assert not ((np.array([.19,.21])<=.2)&(np.array([.41,.39])<=.4)).any()
    for b in range(8):
        ix=splits['FIT'][b*128:(b+1)*128];start=time.perf_counter();z=job_arrays(data,ix);met=m.metrics(z,z['groups'],z['track']);cert=m.certificate_summary(z,z['groups']);dest=c.H/'preflight_smoke'/f'chunk{b}.npz';dest.parent.mkdir(exist_ok=True);np.savez_compressed(dest,**z);back=c.npz(dest);assert all(np.array_equal(v,back[k]) for k,v in z.items());maximum=max(maximum,validate(back,data,ix,met,cert));times.append(time.perf_counter()-start);sizes.append(sum(v.nbytes for v in z.values()));records.append(dict(chunk=b,n=128,SHA=c.sha(dest),full_saved_pipeline_independently_audited=True));del z,back,met,cert;gc.collect()
    identity={name:dict(n=len(ix),tracks=len(set(data['track'][ix])),scene_counts={str(k):int((data['scene'][ix]==k).sum()) for k in np.unique(data['scene'][ix])},groups={k:int(v.sum()) for k,v in c.masks(data['groups'][ix])},sample_ID_digest=c.digest(data['sample_id'][ix].tolist())) for name,ix in splits.items()}
    c.write('IDENTITY_AUDIT.json',dict(status='PASS',records=identity,all24679_unique_disjoint=True,GT_native_groups_metadata_slots_EXACT=True,coordinates='Frozen native and future are relative to last observed origin; zero-prepend12v11a10j dt0.4. Prior frozen alignment sources inherited; encoder metadata checked without model.',encoder_keys=keys));files[str(c.H/'IDENTITY_AUDIT.json')]=c.sha(c.H/'IDENTITY_AUDIT.json');c.verify(files)
    elapsed=time.perf_counter()-t;rows=sum(map(len,splits.values()));nominal=rows/128*np.mean(times);overhead=max(5.,elapsed-sum(times));planning=(nominal+2*overhead+45)*1.25/60
    size=rows/128*np.mean(sizes)/2**30;disk=c.psutil.disk_usage(str(c.H)).free/2**30;assert disk>2+1.5*size,'STOP_DISK'
    report=dict(status='PASS' if planning<=30 else 'STOP_BUDGET',stage='S0',planning_minutes=planning,ETA_minutes=[.7*planning,1.5*planning],source_count=len(files),inherited_sources=inherited,source_digest=c.digest(files),fixtures=fx,analysis_SHA_skip_bad_source_output_rejected=True,actual_eight128_pipeline=records,independent_maxerror=maximum,benchmark_128seconds=times,preflight_elapsed_seconds=elapsed,nominal_full_construct_save_independent_audit_seconds=nominal,uncompressed_arrays_GiB=size,disk_free_GiB=disk,resource=c.RESOURCE,scientific_jobs=4,no_optimizer_model_forward_or_solver=True,S1_started=False)
    c.write('source_manifest.json',dict(files=files,inherited_sources=inherited));c.write('PREFLIGHT.json',report);print('PREFLIGHT',report['status'],planning,len(files),c.RESOURCE,flush=True)

if __name__=='__main__':
    try:main()
    except Exception:c.failure('STOP_S0_PREFLIGHT');raise
