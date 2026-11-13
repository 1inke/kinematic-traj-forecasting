import core as c
import numpy as np,torch,time,copy,gc
from torch.utils.data import DataLoader,TensorDataset

def projection_fixtures():
    z=torch.tensor([[1.,0.],[-1.,0.],[0.,0.]],device=c.DEVICE,dtype=torch.float64)
    g,n=c.aggregate(z,'projected',0);assert torch.equal(g,torch.zeros(2,device=c.DEVICE,dtype=torch.float64)) and n==2
    for step in range(3):
        q=torch.tensor([[2.,1.],[-1.,3.],[-2.,-2.]],device=c.DEVICE,dtype=torch.float64)
        ordinary,_=c.aggregate(q,'ordinary',step);assert torch.equal(ordinary,q.sum(0))
        projected,_=c.aggregate(q,'projected',step);assert torch.isfinite(projected).all()
        # Independent NumPy loop, original q_j references and same fixed step order.
        a=q.cpu().numpy();h=a.copy()
        for i in range(3):
            for j in [step%3,(step+1)%3,(step+2)%3]:
                if i!=j and h[i]@a[j]<0 and a[j]@a[j]>0:h[i]-=(h[i]@a[j])/(a[j]@a[j])*a[j]
        assert np.max(abs(projected.cpu().numpy()-h.sum(0)))<=1e-14
    return dict(opposing_orthogonal_zero_empty_groups=True,original_reference_and_cyclic_order=True,no_epsilon_or_rescale=True)

def packstate(m,o,g,history,contract):
    return dict(contract=contract,state_dict={k:v.detach().cpu().clone() for k,v in m.state_dict().items()},optimizer=copy.deepcopy(o.state_dict()),generator=g.get_state(),python=c.random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),history=copy.deepcopy(history))

def resume_fixture(data,sp,arm):
    ix=sp['grouped_train_index'][:513];inner=sp['inner_cal_index'][:12]
    contract=dict(arm=arm,parent_SHA=c.sha(c.parent_checkpoint(42)),core_SHA=c.sha(c.H/'core.py'),protocol_SHA=c.sha(c.H/'protocol.json'),split_SHA=c.sha(c.SPLIT),FIT_ids=data['ids'][ix].tolist(),inner_ids=data['ids'][inner].tolist())
    def build():
        c.seed(42);m=c.build(42);o=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);g=torch.Generator().manual_seed(42)
        l=DataLoader(TensorDataset(torch.from_numpy(ix)),batch_size=256,shuffle=True,generator=g,num_workers=0);return m,o,g,l
    def epoch(m,o,l,ep,history):
        rows=[]
        for j,(ind,) in enumerate(l):rows.append(c.apply_step(m,o,data,ind.numpy(),arm,(ep-1)*3+j))
        p=c.predict(m,data,inner);history.append(dict(epoch=ep,steps=rows,inner12_Q=float(c.numpy_loss(p['refined4'],p['gt']).mean())))
    m,o,g,l=build();h=[];epoch(m,o,l,1,h);epoch(m,o,l,2,h);a=packstate(m,o,g,h,contract)
    d=c.H/'resume_smoke'/arm;d.mkdir(parents=True,exist_ok=True);torch.save(a,d/'continuous.pt');del m,o,l
    m,o,g,l=build();h=[];epoch(m,o,l,1,h);ck=packstate(m,o,g,h,contract);torch.save(ck,d/'interrupted.pt');del m,o,l,ck
    m,o,g,l=build();c.seed(1891);torch.rand(7,device=c.DEVICE);np.random.rand(11)
    z=torch.load(d/'interrupted.pt',map_location='cpu',weights_only=False);assert z['contract']==contract
    try:assert z['contract']==dict(contract,core_SHA='bad')
    except AssertionError:rejected=True
    else:raise AssertionError('BAD_RESUME_CONTRACT_ACCEPTED')
    m.load_state_dict(z['state_dict']);o.load_state_dict(z['optimizer']);g.set_state(z['generator']);c.random.setstate(z['python']);np.random.set_state(z['numpy']);torch.set_rng_state(z['torch']);torch.cuda.set_rng_state_all(z['cuda']);h=z['history'];epoch(m,o,l,2,h);b=packstate(m,o,g,h,contract);torch.save(b,d/'resumed.pt')
    aa=torch.load(d/'continuous.pt',map_location='cpu',weights_only=False);bb=torch.load(d/'resumed.pt',map_location='cpu',weights_only=False);assert c.equal(aa,bb),('RESUME_NOT_EXACT',arm)
    del m,o,l,a,b,aa,bb;gc.collect();torch.cuda.empty_cache()
    return dict(arm=arm,FIT_subset=513,inner_subset=12,epochs=2,all_parameters_buffers_optimizer_ALLRNG_history_EXACT=True,bad_resume_contract_rejected=rejected,not_full_FIT_training_resume=True)

def main():
    start=time.perf_counter();c.guard();assert torch.cuda.is_available();c.stage('D0_PREFLIGHT_SOURCE_IDENTITY')
    files,inherited=c.locks();data,sp=c.load_data();packs,probe,sel=c.selection(data,sp)
    np.savez_compressed(c.H/'selection.npz',source_packets=packs,probe_index=probe,source_ids=data['ids'][packs],probe_ids=data['ids'][probe],probe_tracks=data['track'][probe])
    files[str(c.H/'selection.npz')]=c.sha(c.H/'selection.npz');c.write('SELECTION.json',sel)
    projection=projection_fixtures();checks=[];benchmark=[]
    for s in c.SEEDS:
        c.stage('D0_PREFLIGHT_PARENT_ALIGN',seed=s);m=c.build(s);small=sp['outer_eval_index'][:128];z=c.predict(m,data,small)
        with np.load(c.P/f'runs/position3/seed_{s}/outer.npz',allow_pickle=False) as old:
            assert np.array_equal(z['sample_id'],old['sample_id'][:128]);err=float(np.max(abs(z['paths']-old['refined20'][:128])));assert err<=2e-12,err
        # CV not read in position3; changed CV cannot alter the forward.
        f,a,cv,y=c.batch(data,small)
        with torch.no_grad():assert torch.equal(m(f,a,cv,'position3')[0],m(f,a,cv+17,'position3')[0])
        copy_m=copy.deepcopy(m);assert c.equal(m.state_dict(),copy_m.state_dict());assert np.array_equal(z['paths'],c.predict(copy_m,data,small)['paths'])
        checks.append(dict(seed=s,original_saved128_path_maxerror=err,all24134_parameters_bool_scaler_basis_exact=True,NoUpdate_forward_EXACT=True,CV_unused_in_position3_verified=True));del m,copy_m,z
    ix=packs[0]
    for arm in c.ARMS:
        c.stage('D0_PREFLIGHT_GPU8',arm=arm);m=c.build(42);o=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);rows=[]
        initial={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
        for i in range(8):rows.append(c.apply_step(m,o,data,ix,arm,i,True))
        # All existing non-parameter buffers remain fixed through actual updates.
        for k,v in m.named_buffers():assert torch.equal(v.detach().cpu(),initial[k])
        assert all(torch.isfinite(v).all() for v in m.parameters())
        checks.append(dict(arm=arm,actual_GPU8_FIT256=True,scalar_maxerror=max(r['canonical_scalar_error'] for r in rows),gradient_L2_maxerror=max(r['canonical_gradient_L2_error'] for r in rows),finite_gradient_updates=True,all_buffers_fixed=True,group_counts=rows[0]['group_counts'],first_gradient_norms=rows[0]['group_norms']))
        del m,o;gc.collect();torch.cuda.empty_cache()
        m=c.build(42);o=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);times=[]
        for i in range(36):
            torch.cuda.synchronize();t=time.perf_counter();c.apply_step(m,o,data,ix,arm,i);torch.cuda.synchronize()
            if i>=4:times.append(time.perf_counter()-t)
        t=time.perf_counter();out=c.predict(m,data,probe);forward=time.perf_counter()-t;small={k:v[:128] for k,v in out.items()};t=time.perf_counter();c.metrics(small,small['paths']);metric=time.perf_counter()-t
        tmp=c.H/'preflight_saved128.npz';t=time.perf_counter();np.savez_compressed(tmp,**small);reload=c.npz(tmp);io=time.perf_counter()-t;assert all(np.array_equal(v,reload[k]) for k,v in small.items())
        from audit_saved import compare
        pipeline_error=compare(c.metrics(small,small['paths']),c.metrics(reload,reload['paths'],True));assert pipeline_error<=2e-12
        checks.append(dict(arm=arm,actual_saved128_fullquality_complete_link_pooled_motion_independent_maxerror=pipeline_error))
        benchmark.append(dict(arm=arm,FIT256_step_seconds=float(np.mean(times)),probe_n=len(probe),probe_forward_seconds=forward,full_quality_cluster_motion128_seconds=metric,save_read128_seconds=io,throwaway4warm32measure=True,no_scientific_model_saved=True));del m,o,out,reload;gc.collect();torch.cuda.empty_cache()
    resumes=[resume_fixture(data,sp,arm) for arm in c.ARMS]
    # Exercise exact locked-file rejection, without altering a science artifact.
    try:c.verify({str(c.H/'protocol.json'):'0'*64})
    except AssertionError:bad_source_rejected=True
    else:raise AssertionError('BAD_SHA_ACCEPTED')
    files[str(c.H/'SELECTION.json')]=c.sha(c.H/'SELECTION.json');c.verify(files)
    # 24 packet-seeds: common0 + A1,A4,B1,B4, each source+probe. Production one metric pass and independent auditor another.
    windows=3*8*5*(256+len(probe));compute=sum(3*8*4*b['FIT256_step_seconds'] for b in benchmark)
    compute+=3*8*5*sum(b['probe_forward_seconds'] for b in benchmark)/2
    meanmetric=np.mean([b['full_quality_cluster_motion128_seconds'] for b in benchmark]);meanio=np.mean([b['save_read128_seconds'] for b in benchmark])
    planning=(compute+windows/128*(2*meanmetric+meanio)+2*(time.perf_counter()-start)+60)/60
    disk=c.psutil.disk_usage(str(c.H)).free/2**30;estimated=windows*20*12*2*8/2**30
    assert disk>estimated*1.5+2,('STOP_DISK',disk,estimated)
    report=dict(status='PASS' if planning<=30 else 'STOP_BUDGET',stage='D0',selection=sel,checks=checks,projection_fixtures=projection,resume=resumes,bad_source_rejected=bad_source_rejected,benchmark=benchmark,planning_minutes=planning,ETA_minutes=[planning*.7,planning*1.5],saved_window_rows_estimate=windows,uncompressed_path_GiB_estimate=estimated,disk_free_GiB=disk,source_count=len(files),inherited_sources=inherited,source_digest=c.digest(files),elapsed_seconds=time.perf_counter()-start,resource=c.RESOURCE,U1_started=False)
    c.write('source_manifest.json',dict(files=files,inherited=inherited));c.write('PREFLIGHT.json',report);print('PREFLIGHT',c.read(c.H/'PREFLIGHT.json')['status'],planning,sel,c.RESOURCE,flush=True)

if __name__=='__main__':
    try:main()
    except Exception:c.reject_failure('STOP_PREFLIGHT');raise
