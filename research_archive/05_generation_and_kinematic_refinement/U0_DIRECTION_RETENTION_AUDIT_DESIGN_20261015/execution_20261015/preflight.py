import sys,time,gc,shutil,json
import numpy as np
import common as c,fixtures as f,independent_review as rev,cluster_bootstrap as bs

def main():
    start=time.perf_counter();E=c.estimate();startup=c.guard('startup');c.write('PREFLIGHT_RESOURCE_START.json',startup)
    assert 'torch' not in sys.modules
    syntax={}
    for path in c.H.glob('*.py'):compile(path.read_text(encoding='utf-8-sig'),str(path),'exec');syntax[path.name]='PASS'
    geom=f.geometry();boot=f.bootstrap();c.write('FIXTURES.json',dict(status='PASS',geometry=geom,bootstrap=boot,syntax=syntax))
    t=time.perf_counter();lock=c.locks();hashseconds=time.perf_counter()-t;c.write('source_manifest.json',lock)
    inputs=c.input_audit();timings=[];records=[];io=[];payload=[]
    for arm in c.ARMS:
        key=f'{arm}/seed42/FIT';row=inputs[key];t=time.perf_counter();full=c.npz(row['output']);reference=c.native('FIT');io.append(time.perf_counter()-t)
        totalcompressed=__import__('pathlib').Path(row['output']).stat().st_size+__import__('pathlib').Path(c.read(c.PARENT/'JOBS.json')['native/FIT']['output']).stat().st_size
        rowtimes=[]
        for block in range(8):
            c.guard();ix=slice(block*128,(block+1)*128);z={k:v[ix].copy() for k,v in full.items()};ref={k:v[ix].copy() for k,v in reference.items()};small=dict(row,n=128)
            c.check_input(z,ref,small);t=time.perf_counter();d,w,s=rev.process(z)
            if block==0:
                cp=c.H/f'preflight_smoke/{arm}_candidate.npz';wp=c.H/f'preflight_smoke/{arm}_window.npz';jp=c.H/f'preflight_smoke/{arm}_summary.json';c.save(cp,**d);c.save(wp,**w);c.write(jp,s)
                review=rev.review_job(z,c.npz(cp),c.npz(wp),c.read(jp));contract=f.contracts(z,ref,small,lock['source_digest']);payload.append(sum(x.nbytes for x in d.values())+sum(x.nbytes for x in w.values()))
                records.append(dict(arm=arm,n=128,review=review,contract=contract,outputs_SHA={str(p):c.sha(p) for p in (cp,wp,jp)}))
            else:rev.review_job(z,d,w,s)
            rowtimes.append((time.perf_counter()-t)/128);del z,ref,d,w,s;gc.collect()
        timings.append(dict(arm=arm,blocks=8,n=1024,production_plus_independent_review_seconds_per_window=rowtimes,mean=float(np.mean(rowtimes)),max=float(max(rowtimes)),input_decompression_seconds=io[-1],input_compressed_bytes=totalcompressed))
        del full,reference;gc.collect()
    n=sum(r['n'] for r in inputs.values());rate=max(t['max'] for t in timings);compute=rate*n
    old=c.read(c.PARENT/'JOBS.json');readbytes=2*sum(__import__('pathlib').Path(r['output']).stat().st_size for r in inputs.values())+12*sum(__import__('pathlib').Path(r['output']).stat().st_size for r in old.values() if r['arm']=='native20')
    iorate=max(t['input_decompression_seconds']/t['input_compressed_bytes'] for t in timings);ioplan=readbytes*iorate
    # Additional primary/sensitivity/review CI, compressed diagnostic I/O,
    # final source/output SHA and report headroom, no droppedjob/seed/iterations.
    planning=(1.5*compute+1.5*ioplan+3*hashseconds+90)/60
    uncompressed=max(payload)/128*n;disk=shutil.disk_usage(c.H).free;required_disk=int(2*uncompressed+2*20000*(956+191+3)*8)
    assert disk>=required_disk,('STOP_DISK',disk,required_disk);assert planning<=30,('STOP_PLANNING_GT30',planning)
    c.verify(lock['files']);assert 'torch' not in sys.modules
    result=dict(status='PASS',scope='CPU_SAVED_OUTPUT_DIAGNOSIS_NO_MODEL',source_count=len(lock['files']),inherited_sources=2092,source_digest=lock['source_digest'],estimate_peak_bytes=E,startup_resource=startup,fixtures='FIXTURES.json',actual_saved_records=records,benchmark=timings,scientific_window_rows=n,compute_seconds=compute,input_IO_seconds=ioplan,source_hash_measure_seconds=hashseconds,planning_minutes=planning,ETA_minutes=[.7*planning,1.5*planning],disk_free_bytes=disk,conservative_disk_need_bytes=required_disk,new_training=False,no_torch_import=True,original_source_and_outputs_SHA_initial_final=True,elapsed_seconds=time.perf_counter()-start,resource=dict(c.RESOURCE))
    c.write('PREFLIGHT.json',result);print(json.dumps(result,ensure_ascii=False,default=lambda a:a.tolist()),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:c.failure('STOP_PREFLIGHT');raise
