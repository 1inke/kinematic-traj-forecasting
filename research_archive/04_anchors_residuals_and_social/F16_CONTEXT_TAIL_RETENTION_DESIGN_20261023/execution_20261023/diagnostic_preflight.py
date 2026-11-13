"""Actual saved128 fixtures/benchmark before the independent diagnostic worker."""
from source_contract import *
from saved_diagnostics import compute,summary,path_measures,temporal_transport,quality
from independent_diagnostics import review_job
import shutil

def fixtures():
    z=np.zeros((4,12,2));obs=np.zeros((4,2));r=path_measures(z,z,obs)
    assert np.isnan(r['endpoint_angle_radians']).all() and np.isnan(r['velocity_join_angle_radians']).all() and np.array_equal(r['W_time'],np.zeros(4))
    a=np.zeros((3,11));g=a.copy();a[1,0]=1;g[2,0]=1;assert np.isnan(temporal_transport(a,g)[1:]).all()
    t=np.arange(1,13)[None,:,None]/12;gt=np.repeat(np.concatenate((t,np.zeros_like(t)),2),4,0)
    endpoint=np.tile([.3,0.],(4,1));p=gt+t*endpoint[:,None];m=path_measures(p,gt,obs)
    assert np.max(m['shape_error'])<1e-14 and np.max(abs(m['along_error']-.3))<1e-14
    # Constant speed, one-turn and reverse paths are finite vector fixtures; wrong-slot success must not be assembled.
    turn=gt.copy();turn[:,6:,1]=np.arange(1,7)/12;reverse=-gt
    for pp in [gt,turn,reverse]:assert np.isfinite(path_measures(pp,gt,obs)['E_a']).all()
    wrong=np.zeros((1,20,12,2));truth=np.zeros((1,12,2));wrong[:]=1.;wrong[0,0,:11]=0;wrong[0,0,11]=[.5,0];wrong[0,1,11]=0
    q=quality(wrong,truth);assert q['ADE'].min()<=.2 and q['FDE'].min()<=.4 and not q['good'].any()
    return dict(status='PASS',zero_angles_NA=True,transport_both_zero_0_one_zero_NA=True,constant_speed_turn_reverse=True,endpoint_only_shape_zero=True,wrong_slot_oracle_rejected=True)

def main():
    guard=Guard(E_DIAG);guard.check('startup');fx=fixtures();data=job(42,'FIT');chosen=sorted(range(len(data['sample_id'])),key=lambda i:hashlib.sha256(f'tail-diagnostic-preflight::{int(data["sample_id"][i])}'.encode()).digest())[:128]
    sub={k:v[chosen] for k,v in data.items()};t=time.perf_counter();out=compute(sub);summ=summary(out);dest=HERE/'preflight_saved';dest.mkdir(exist_ok=True);np.savez_compressed(dest/'diagnostics128.npz',**out);write_json(dest/'SUMMARY.json',summ)
    with np.load(dest/'diagnostics128.npz') as z:back={k:z[k] for k in z.files}
    audit=review_job(sub,back,summ);assert audit['status']=='PASS';sec=time.perf_counter()-t;write_json(dest/'AUDIT.json',audit)
    # All formal production jobs use the same algorithms; no quality-based selection.
    nominal=sec*3*(18909+1856)/128;planning=nominal*1.35+60
    guard.check();memory=dict(E_bytes=E_DIAG,factor=1.3,inventory_GiB=dict(Python_numpy_scipy_baseline=.5,one_aligned_saved_population=.65,diagnostic_arrays_and_compressed_IO=.9,independent_block_review_and_summary=.7,hash_margin=.25),runtime_rule='dynamic remaining physical/commit and retained Windows process peaks',obsolete_fixed_limits=False)
    write_json(HERE/'MEMORY_ESTIMATE.json',memory)
    draft=read_json(HERE.parent/'protocol.draft.json');draft.update(status='FROZEN_DIAGNOSTIC_EXECUTION',human_execution_request='2026-10-23 执行新实验',current_stage='D0_D1_saved_only',T1_started=False)
    draft['runtime_fields_required_before_launch'].update(preflight_status='PASS',memory_estimate_E_bytes=E_DIAG,actual_benchmark_seconds=sec,actual_eta_seconds=planning,source_manifest_digest='bound_by_external_source_manifest_to_avoid_circular_hash',source_extension_digest='inherited_prelaunch_extension_exact')
    write_json(HERE/'protocol.json',draft);write_json(HERE/'FIXTURES.json',fx)
    free=shutil.disk_usage(HERE).free;disk_est=600*2**20;assert free>1.3*disk_est
    write_json(HERE/'PREFLIGHT.json',dict(status='PASS',stage='D0_D1_ONLY_NO_TORCH',actual_saved128_seconds=sec,nominal_compute_seconds=nominal,planning_seconds=planning,ETA_seconds=[.7*planning,1.5*planning],independent_audit=audit,fixtures=fx,disk_free_bytes=free,conservative_output_bytes=disk_est,resource=guard.observations))
    digest=inherit();write_json(HERE/'PRELAUNCH_REVIEW.json',dict(status='PASS',source_digest=digest,source_count=len(read_json(HERE/'source_manifest.json')['files']),no_training_authorized_before_D0_entry=True,inline_DA1='same-assistant not external; known prior contribution is not independent discovery; motion diagnosis cannot relax registered gates'))
    print('D0/D1 PREFLIGHT PASS',sec,'planning seconds',planning,flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        import traceback;write_json(HERE/'PREFLIGHT_FAILURE.json',dict(traceback=traceback.format_exc()));raise
