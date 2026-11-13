"""Strict frozen replay preflight, plus algebra-only tensor fixture (no training)."""
import time,gc,shutil
from common import *

def algebra_fixture():
    import torch
    rows=[]
    for nR,nN in [(3,4),(3,0),(0,4),(0,0)]:
        torch.manual_seed(20261022);p=torch.randn(2,20,12,2,dtype=torch.float64,requires_grad=True);R=torch.randn(2,nR,12,2,dtype=p.dtype);N=torch.randn(2,nN,12,2,dtype=p.dtype)
        def terms(t):
            if t.shape[1]==0:return p.sum()*0,0
            e=torch.linalg.vector_norm(t[:,:,None]-p[:,None],dim=-1);return (e.mean(-1)+.5*e[:,:,:,-1]).min(-1).values.sum(),2*t.shape[1]
        sr,nr=terms(R);sn,nn=terms(N);den=nr+nn
        union=(sr+sn)/max(den,1);decomposed=(nr/max(den,1))*(sr/max(nr,1))+(nn/max(den,1))*(sn/max(nn,1));gu=torch.autograd.grad(union,p,retain_graph=True)[0];gd=torch.autograd.grad(decomposed,p)[0]
        row=dict(nR=nr,nN=nn,scalar_abs_diff=abs(float(union.detach()-decomposed.detach())),all_output_gradient_maxdiff=float((gu-gd).abs().max()),fixture='synthetic output tensor only, no model optimizer')
        if max(row['scalar_abs_diff'],row['all_output_gradient_maxdiff'])>2e-12:raise RuntimeError('ALGEBRA_IDENTITY_FAILED')
        rows.append(row)
    return rows

def main():
    from f16.eval import evaluate,motion
    p=read_json(HERE/'protocol.json');guard=Guard(p['resource']['estimate_bytes']);guard.check('startup');verify_sources();data=data_load();started=time.monotonic()
    import torch
    torch.set_num_threads(2);device='cuda:0' if torch.cuda.is_available() else 'cpu';torch.backends.cuda.matmul.allow_tf32=False
    sample=np.array(sorted(data['split']['FIT'],key=lambda i:hashlib.sha256(f'e0e1-frozen-preflight-v1|{data["ids"][i]}'.encode()).hexdigest())[:128]);checks=[]
    for seed,arm in RUNS:
        guard.check();net=load_student(seed,arm,device);state=frozen_state(net);a=predict(net,data['past'][sample],device,128);b=predict(net,data['past'][sample],device,32);delta=float(np.max(np.abs(a-b)));assert_frozen(net,state)
        if delta>p['batch_invariance_max_abs']:raise RuntimeError('BATCH_INVARIANCE')
        checks.append(dict(seed=seed,arm=arm,strict_load=True,param_buffer_exact=True,batch128_vs32_maxdiff=delta,finite_K20=True));del net;gc.collect()
    # Full inner is a replay test, not a new S3/CAL selection.
    net=load_student(42,'A',device);state=frozen_state(net);ix=data['split']['inner'];paths=predict(net,data['past'][ix],device);metrics,_=evaluate(paths,data['gt'][ix],data['groups'][ix],data['native'][ix],data['ids'][ix],data['track'][ix],include_clusters=False);comparison=reconcile(metrics,old_result(42)['results']['A']['eval']);assert_frozen(net,state)
    if comparison['status']!='RECONCILED':
        write_json(HERE/'PREFLIGHT_REPLAY_DIFFERENCES.json',comparison);raise RuntimeError('REPLAY_NOT_RECONCILED')
    del net;gc.collect();producer=[];m=model_module(ROOT/'UNION_SELECTOR_K40_TO_K20_20261018/kdec',alias='_repair_actual_cache_producer_kdec');reference=motion(data['native'][data['split']['FIT']]);vmax=reference['speed_p95']*.995;amax=reference['accel_p95']*.90
    # Numeric row mapping validation, whole arrays, not equal-length guess.
    # No caches are regenerated or replaced; actual old arrays remain teachers.
    for seed in SEEDS:
        ck=ROOT/f'KINEMATIC_DECODER_SPEED_CLAMP_20261018/results/short/KIN_seed{seed}.pt';blob=torch.load(ck,map_location='cpu',weights_only=True)
        if abs(blob['v_max']-vmax)>2e-12 or abs(blob['a_max']-amax)>2e-12:raise ValueError('KIN_SCALE_RECIPE_MISMATCH')
        net=m.build('KIN',v_max=vmax,a_max=amax).to(device).eval();net.load_state_dict(blob['state_dict'],strict=True);net.requires_grad_(False);state=frozen_state(net)
        for split,ix,path in [('FIT',data['split']['FIT'],kin_path('FIT',seed)),('inner',data['split']['inner'],kin_path('inner',seed)),('E0_original_CAL_cache_rows',np.arange(25912,28791),Path(read_json(E0/'RESULT.json')['results'][str(seed)]['kin_src']))]:
            guard.check();current=predict(net,data['past'][ix],device);saved=np.load(path);delta=float(np.max(np.abs(current.astype(float)-saved.astype(float))))
            if delta>p['teacher_cache_replay_tolerance']:raise RuntimeError('KIN_ID_ORDER_NOT_RECONCILED '+str((seed,split,delta)))
            producer.append(dict(seed=seed,split=split,n=len(ix),path=str(path),file_sha256=sha256(path),registered_IDs_sha256=array_sha(data['ids'][ix]),numeric_full_row_replay_maxdiff=delta,producer_state_sha=sha256(ck),historical_lock_still_unknown=True));assert_frozen(net,state)
            del current,saved
        del net;gc.collect()
    # Actual complete saved-path pipeline smoke, independent implementation.
    from audit.verify import audit_file
    smoke=HERE/'preflight_smoke';smoke.mkdir(exist_ok=True);ix=sample;net=load_student(42,'A',device);pred=predict(net,data['past'][ix],device);del net
    np.savez_compressed(smoke/'saved.npz',paths=pred,native=data['native'][ix],gt=data['gt'][ix],groups=data['groups'][ix],sample_id=data['ids'][ix],track=data['track'][ix]);metrics,_=evaluate(pred,data['gt'][ix],data['groups'][ix],data['native'][ix],data['ids'][ix],data['track'][ix]);write_json(smoke/'metrics.json',metrics)
    audit=audit_file(smoke/'saved.npz',smoke/'metrics.json',frozen_cache=PATHS['cache'],frozen_labels=PATHS['labels'],frozen_metadata=PATHS['metadata'],frozen_manifest=ROOT/'config/data_manifest.v1.json',root=ROOT)
    if audit['status']!='PASS':raise RuntimeError('INDEPENDENT_PIPELINE_SMOKE_FAILED')
    write_json(smoke/'AUDIT.json',audit);fixtures=algebra_fixture();write_json(HERE/'ALGEBRA_IDENTITY_FIXTURE.json',dict(status='PASS',cases=fixtures,interpretation='equivalent weighted decomposition is an identity, not a new treatment'))
    guard.check();verify_sources();elapsed=time.monotonic()-started
    # Raw uncompressed paths/native/teacher + diagnostics, disk bound not promise.
    bytes_estimate=(18909+1856)*11*(20*12*2*(4+8+4)+12*2*8+20*20*4)*2
    if shutil.disk_usage(HERE).free<bytes_estimate:raise OSError('DISK_ESTIMATE_INSUFFICIENT')
    write_json(HERE/'PREFLIGHT.json',dict(status='PASS_FROZEN_EVALUATION_ONLY',seconds=elapsed,device=device,student_dtype='float32',students=checks,A42_complete_inner_reconciliation=comparison,KIN_complete_cache_row_replay=producer,actual_saved128_independent_audit=audit,algebra_fixture='ALGEBRA_IDENTITY_FIXTURE.json',resource=guard.observations,disk_conservative_bytes=bytes_estimate,disk_free_bytes=shutil.disk_usage(HERE).free,estimated_22_job_minutes_range=[3,12],estimate_kind='conservative engineering range, not production benchmark guarantee',no_optimizer_or_student_backward=True,no_new_training=True))
    progress('PREFLIGHT_PASS',done_jobs=0,device=device);print(f'PREFLIGHT PASS {elapsed:.2f}s; strict11 heads,9 whole teacher-cache row replays, fullA42inner and independent128 saved audit.',flush=True)

if __name__=='__main__':main()
