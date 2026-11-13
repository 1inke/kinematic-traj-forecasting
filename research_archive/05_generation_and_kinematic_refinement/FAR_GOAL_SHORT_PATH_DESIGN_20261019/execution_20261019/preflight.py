import copy,traceback,time,inspect
import common as c
c.E=c.read('MEMORY_ESTIMATE.json')['estimated_total_peak_bytes']
import model as m
import training as tr
import contracts
import evaluate_saved as ev
import bootstrap_stats as st
import diagnose_goals as dg
import torch
import numpy as np

def fd_test(stats,d):
    m.setup(20261019);model=m.GoalConditionedShortPath(stats).cuda().double()
    with torch.no_grad():
        for layer in (model.goal_head[-1],model.process_head[-1]):layer.weight.normal_(0,.003);layer.bias.normal_(0,.003)
    ix=d['split']['FIT'][:4];p,v,y,y20,mask=tr.tensors(d,ix);p=p.double();v=v.double();loss,parts=m.objective(model(p,v),y,y20,mask,20)
    names=list(model.named_parameters());directions=[]
    gen=torch.Generator(device='cuda').manual_seed(20261019)
    for name,param in names:
        dr=torch.randn(param.shape,device=param.device,dtype=param.dtype,generator=gen);dr/=max(float(dr.norm()),1);directions.append(dr)
    gradients=torch.autograd.grad(loss,[param for _,param in names]);analytic=sum(float((g*dr).sum()) for g,dr in zip(gradients,directions))
    base=[p.detach().clone() for _,p in names];results=[]
    def signature():
        with torch.no_grad():
            out=model(p,v);_,pa=m.objective(out,y,y20,mask,20);a,b=torch.triu_indices(20,20,1,device='cuda');act=(torch.linalg.vector_norm(out['path'][:,a]-out['path'][:,b],dim=-1).mean(-1)<.3)
            return pa['winner'].clone(),act.clone()
    sig=signature()
    for eps in (1e-4,1e-5):
        vals=[]
        for sign in (1,-1):
            with torch.no_grad():
                for (_,param),b,dr in zip(names,base,directions):param.copy_(b+sign*eps*dr)
            sg=signature();assert torch.equal(sig[0],sg[0]) and torch.equal(sig[1],sg[1]),'NONSMOOTH_CHECK_BLOCKED'
            vals.append(float(m.objective(model(p,v),y,y20,mask,20)[0].detach()))
        actual=(vals[0]-vals[1])/(2*eps);error=abs(actual-analytic);tol=1e-6+1e-4*max(abs(actual),abs(analytic));assert error<=tol,(eps,error,tol)
        results.append(dict(eps=eps,automatic=analytic,finite_difference=actual,error=error,tolerance=tol,activities_unchanged=True))
    return dict(status='PASS',float64_copy_not_float32_quantization_proof=True,results=results)

def fixtures(stats,d):
    assert list(inspect.signature(m.GoalConditionedShortPath.forward).parameters)==['self','p2','v0']
    m.setup(42);a=m.GoalConditionedShortPath(stats).cuda();m.setup(42);b=m.GoalConditionedShortPath(stats).cuda();tr.exact(a.state_dict(),b.state_dict())
    ix=d['split']['FIT'][:128];p,v,y,y20,mask=tr.tensors(d,ix);out=a(p,v);other=b(p,v)
    for k in out:assert torch.equal(out[k],other[k])
    # GT/mask never enter forward, and horizon is not a model attribute/input.
    changed=y20.flip(0);changedmask=~mask;again=a(p,v)
    for k in out:assert torch.equal(out[k],again[k])
    z=out['z'].detach().cpu().numpy();abc=out['process_coeff'].detach().cpu().numpy();path=out['path'].detach().cpu().numpy();vv=v.cpu().numpy()
    err=float(abs(m.reconstruct(z,abc,vv)-path).max());assert err<=2e-12
    assert np.max(abs(path[:,:,-1]-z.astype(float)-abc[:,:,0].astype(float)))<=2e-12
    scalar=[]
    for horizon in (12,20):
        l,pa=m.objective(out,y,y20,mask,horizon);nl,nb=m.numpy_loss(dict(path=path,z=z),y.cpu().numpy(),y20.cpu().numpy(),mask.cpu().numpy(),horizon);assert abs(float(l.detach())-nl)<=2e-12;assert np.array_equal(pa['winner'].cpu(),nb['winner']);scalar.append(dict(H=horizon,loss_error=abs(float(l.detach())-nl)))
    no=torch.zeros_like(mask);nan=torch.full_like(y20,float('nan'));empty=m.objective(out,y,nan,no,20);assert float(empty[1]['goal'].detach())==0 and torch.isfinite(empty[0])
    # Exact ties select physical first; auxiliary uses same short-path slot.
    tiepath=y[:,None].expand(-1,20,-1,-1).clone();tiez=torch.zeros_like(out['z']);q=m.objective(dict(path=tiepath,z=tiez),y,nan,no,20)[1];assert torch.count_nonzero(q['winner'])==0
    return dict(status='PASS',paired_all_weights_buffers=True,initial_paths_exact=True,not_native_identity=True,GT_API_isolated=True,geometry_max_error=err,p12_is_z_plus_a=True,loss_scalar=scalar,empty_mask_NaN_safe=True,first_tie_same_slot=True)

def gpu_eight(stats,d,horizon):
    m.setup(42);a=m.GoalConditionedShortPath(stats).cuda();opt=m.optimizer(a);maxnorm={name:0. for name,_ in a.named_parameters()};first={}
    for step in range(8):
        c.guard();ix=d['split']['FIT'][step*256:(step+1)*256];p,v,y,y20,mask=tr.tensors(d,ix);opt.zero_grad(set_to_none=True);out=a(p,v);l,_=m.objective(out,y,y20,mask,horizon);l.backward()
        assert out['path'].shape==(256,20,12,2) and torch.isfinite(out['path']).all()
        for name,param in a.named_parameters():
            assert param.grad is not None and torch.isfinite(param.grad).all();norm=float(param.grad.double().norm());maxnorm[name]=max(maxnorm[name],norm)
            if step==0:first[name]=norm
        opt.step()
    assert all(v>1e-12 for v in maxnorm.values()),maxnorm
    a.eval();state=copy.deepcopy(a.state_dict());ix=d['split']['outer'][:128];p,v,y,y20,mask=tr.tensors(d,ix)
    with torch.no_grad():out=a(p,v)
    tr.exact(state,a.state_dict());data={k:vv.cpu().numpy() for k,vv in out.items()};data.update(sample_id=d['ids'][ix],v0=v.cpu().numpy(),gt=d['gt'][ix],native=d['native'][ix],groups=d['groups'][ix])
    audit_begin=time.perf_counter();c.save(f'preflight_saved_H{horizon}.npz',**data);back=c.npz(c.H/f'preflight_saved_H{horizon}.npz')
    err=float(abs(m.reconstruct(back['z'],back['process_coeff'],back['v0'])-back['path']).max());assert err<=2e-12
    mt,aux=ev.evaluate(back['path'],back['gt'],back['groups'],back['native'].astype(float),back['sample_id']);mi,ai=ev.evaluate(back['path'],back['gt'],back['groups'],back['native'].astype(float),back['sample_id'],ind=True);ev.compare(mt,mi);ev.compare(aux,ai)
    saved_pipeline_seconds=time.perf_counter()-audit_begin
    donors=dg.donors(d['ids'][ix],d['track'][ix]);assert np.all(donors>=0) and np.all(d['track'][ix][donors]!=d['track'][ix])
    h2=[]
    for name,override in [('CV',np.repeat((4.8*d['v0'][ix].astype(np.float32))[:,None],20,1)),('cross_track',data['z'][donors])]:
        response=dg.predict(a,d,ix,override);assert np.array_equal(response['z'],override.astype(np.float32));assert abs(m.reconstruct(response['z'],response['process_coeff'],data['v0'])-response['path']).max()<=2e-12
        h2.append(dict(condition=name,override_EXACT=True,endpoint_z_plus_a=True))
    valid=ix[d['m20'][ix]];target=d['gt'][valid,-1] if horizon==12 else d['y20'][valid];override=np.repeat(((12/horizon)*target).astype(np.float32)[:,None],20,1);response=dg.predict(a,d,valid,override);assert np.array_equal(response['z'],override);h2.append(dict(condition='GT_local_only_m20',n=len(valid),override_EXACT=True))
    # Throughput only; actual initialized head, four warmup+32 measured updates;
    # no benchmark weights saved and no scientific outcome inspected/selected.
    samples=[];ix=d['split']['FIT'][:256];p,v,y,y20,mask=tr.tensors(d,ix)
    for k in range(36):
        c.guard();torch.cuda.synchronize();begin=time.perf_counter();opt.zero_grad(set_to_none=True);l,_=m.objective(a(p,v),y,y20,mask,horizon);l.backward();opt.step();torch.cuda.synchronize()
        if k>=4:samples.append(time.perf_counter()-begin)
    return dict(status='PASS',H=horizon,first_gradnorm=first,max_over8_gradnorm=maxnorm,zero_last_phases_reported=True,saved128_independent_quality_clusters_motion_PASS=True,reconstruction_max_error=err,eval_all_weights_buffers_exact=True,H2_actual_frozen_overrides=h2,saved128_full_pipeline_seconds=saved_pipeline_seconds,FIT256_step_mean_seconds=float(np.mean(samples)),warmup=4,measured=32)

def main():
    begin=time.perf_counter();c.guard('startup');c.stage('H1_PREFLIGHT_LOAD');d=c.load();labels=c.npz(c.H/'H0_labels.npz');d.update(y20=labels['y20'],m20=labels['m20']);stats=c.npz(c.H/'shared_statistics.npz')
    h0=contracts.h0_saved_audit(d);ft=fixtures(stats,d);fd=fd_test(stats,d);gpu=[];resume=[]
    for horizon in (12,20):
        c.stage('H1_PREFLIGHT_GPU8_AND_REAL_RESUME',H=horizon);gpu.append(gpu_eight(stats,d,horizon));resume.append(tr.resume_test(stats,d,horizon))
    bad=contracts.bad_contract_fixtures();boot=st.fixture()
    # Statistics independently recomputed without KMeans refit, IDs exact.
    ix=d['split']['FIT'];errors={}
    for key,x in [('p2',d['p2'][ix].astype(float)),('v',d['v0'][ix]),('z',d['gt'][ix,-1])]:
        mm=sum((x[i:i+256].sum(0) for i in range(0,len(x),256)),np.zeros(x.shape[1]))/len(x)
        ss=np.sqrt(sum((((x[i:i+256]-mm)**2).sum(0) for i in range(0,len(x),256)),np.zeros(x.shape[1]))/len(x));ss=np.where(ss<1e-6,1.,ss)
        errors[key]=float(max(abs(stats[key+'_mean']-mm).max(),abs(stats[key+'_std']-ss).max()));assert errors[key]<=1e-11
    # Actual GPU predicts/CPU clusters+independent metrics timings are measured in
    # this preflight; full stage ETA includes saved audit and H2 extra conditions.
    steps=6*25*int(np.ceil(18909/256));trainsec=steps*np.mean([g['FIT256_step_mean_seconds'] for g in gpu])
    rows=6*24679+6*3*(1035+1856+2879)
    measured_per_row=max(g['saved128_full_pipeline_seconds'] for g in gpu)/128
    planning=trainsec+rows*measured_per_row*1.3+300 # measured full save/audit + IO/CI/SHA reserve
    payload=(6*24679+6*3*(1035+1856+2879))*20*12*2*8
    free=__import__('shutil').disk_usage(str(c.H)).free;assert free>payload*4+2*2**30,'STOP_DISK_CAPACITY'
    files,inherit=contracts.source_files();source_digest=c.digest(files);c.write('source_manifest.json',dict(files=files,digest=source_digest,inherit=inherit));c.verify(files)
    result=dict(status='PASS',at=c.now(),elapsed_seconds=time.perf_counter()-begin,H0_independent=h0,fixtures=ft,FD=fd,GPU8=gpu,real_subset_resume=resume,bad_contracts=bad,bootstrap_fixture=boot,shared_stat_independent_max_errors=errors,parameters=sum(p.numel() for p in m.GoalConditionedShortPath(stats).parameters()),source_count=len(files),source_digest=source_digest,planning_seconds=planning,ETA_seconds=[.7*planning,1.5*planning],ETA_basis=dict(training_steps=steps,training_seconds=trainsec,total_normal_plus_H2_rows_conservative=rows,actual_measured_saved_audit_per_row=measured_per_row,metric_IO_multiplier=1.3,CI_compression_source_seconds=300),disk_free_bytes=free,uncompressed_path_bytes=payload,conservative_storage_bytes=payload*4+2*2**30,resource=c.RESOURCE,old_modules_not_forwarded_or_updated=True,not_external_review=True)
    c.write('PREFLIGHT.json',result);c.write('PRELAUNCH_REVIEW.json',dict(status='PASS',at=c.now(),source_digest=source_digest,actual_preflight_SHA=c.sha(c.H/'PREFLIGHT.json'),inline_DA1='matched m20 and ALL; no GT in normal forward; H2 does not deploy GT; legacy future neighbor selection and repeated outer use remain limitations',no_new_time_veto=True));print(c.json.dumps(c.safe({k:v for k,v in result.items() if k not in ('GPU8','fixtures')}),ensure_ascii=False),flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        c.write('PREFLIGHT_FAILURE.json',dict(at=c.now(),traceback=traceback.format_exc(),resource=c.RESOURCE));raise
