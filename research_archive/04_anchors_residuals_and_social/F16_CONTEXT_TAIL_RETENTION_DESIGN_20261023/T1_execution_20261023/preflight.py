from training import *
from evaluate_saved import save_evaluate

def parameter_grads(m,value):return [v.detach().cpu().clone() for v in torch.autograd.grad(value,tuple(m.parameters()))]

def fixed_nonzero(m):
    g=torch.Generator().manual_seed(20261023)
    with torch.no_grad():
        m.head.weight.copy_(torch.randn(m.head.weight.shape,generator=g).to(m.head.weight)*.03)
        m.head.bias.copy_(torch.randn(m.head.bias.shape,generator=g).to(m.head.bias)*.03)
    return m

def active_signature(p,y,nj):
    qi=quality(p,y);pd=torch.linalg.vector_norm(p[:,PAIR[0],-1]-p[:,PAIR[1],-1],dim=-1)
    return [qi['Qslot'],qi['Jslot'],qi['Jcomponent'],pd<.3,qi['Jmin']>nj]

def numpy_objective(p,y,nj,arm):
    dist=np.sqrt(((p-y[:,None])**2).sum(-1));a=dist.mean(-1);f=dist[:,:,-1];q=a+.5*f;j=np.maximum(a/.2,f/.4).min(1)
    dis=np.sqrt(((p[:,PAIR[0],-1]-p[:,PAIR[1],-1])**2).sum(-1));base=q.min(1).mean()+.05*np.maximum(.3-dis,0).mean()
    return base+(.2*np.maximum(j-nj,0).mean() if arm=='P_relative' else 0)

def fixtures(data):
    result={};d=data['FIT'];x=torch.from_numpy(d['x'][:8]).cuda();y=torch.from_numpy(d['gt'][:8].astype(np.float32)).cuda();nj=torch.from_numpy(d['native_j'][:8]).cuda()
    m=fixed_nonzero(new_model(42));p=m(x);a=objective(p,y,nj,'A_replay')[0]
    dummy=torch.zeros_like(p);mask=torch.zeros((8,20),device='cuda',dtype=torch.bool)
    b=old.loss(p,y,dummy,mask,dummy,mask,'A')[0];assert torch.equal(a,b)
    ga=parameter_grads(m,a);gb=parameter_grads(m,old.loss(m(x),y,dummy,mask,dummy,mask,'A')[0]);assert all(torch.equal(u,v) for u,v in zip(ga,gb))
    result['control_scalar_all_parameter_gradient_EXACT']=True
    # Fixed double model and direction: never search epsilon, seed or active face.
    m=m.double();xd=x.double();yd=y.double();nd=nj.double();p=m(xd)
    scalar=objective(p,yd,nd,'P_relative')[0];npdiff=abs(float(scalar.detach())-numpy_objective(p.detach().cpu().numpy(),yd.cpu().numpy(),nd.cpu().numpy(),'P_relative'));assert npdiff<=1e-10
    gg=parameter_grads(m,scalar);gen=torch.Generator().manual_seed(20261023);direction=[torch.randn(v.shape,generator=gen,dtype=torch.float64).to(v) for v in m.parameters()]
    norm=sum((v*v).sum() for v in direction).sqrt();direction=[v/norm for v in direction];analytic=sum((g.to(v)*v).sum() for g,v in zip(gg,direction));state=frozen_state(m);sig=active_signature(m(xd),yd,nd);fds=[]
    for eps in (1e-4,1e-5):
        values=[]
        for sign in (-1,1):
            m.load_state_dict(state)
            with torch.no_grad():
                for v,dr in zip(m.parameters(),direction):v.add_(dr,alpha=sign*eps)
            pred=m(xd);ss=active_signature(pred,yd,nd)
            assert all(torch.equal(a,b) for a,b in zip(sig,ss)), 'FD_NOT_APPLICABLE_FIXED_ACTIVE_FACE_CHANGED'
            values.append(float(objective(pred,yd,nd,'P_relative')[0].detach()))
        numeric=(values[1]-values[0])/(2*eps);error=abs(numeric-float(analytic));tol=1e-6+1e-4*max(abs(numeric),abs(float(analytic)));assert error<=tol
        fds.append(dict(eps=eps,analytic=float(analytic),numeric=numeric,error=error,tolerance=tol,all_active_faces_exact=True))
    result.update(P_double_numpy_error=npdiff,fixed_parameter_direction_FD=fds)
    z=torch.zeros((2,20,12,2),device='cuda',dtype=torch.float64,requires_grad=True);yy=torch.zeros((2,12,2),device='cuda',dtype=torch.float64)
    q=quality(z,yy);assert (q['Jslot']==0).all() and (q['Jcomponent']==0).all()
    excess=torch.relu(q['Jmin']);gz=torch.autograd.grad(excess.sum(),z)[0];assert torch.count_nonzero(gz)==0
    tie=torch.tensor([[1.,1.]],device='cuda',requires_grad=True);torch.max(tie,1).values.sum().backward();assert torch.equal(tie.grad,torch.tensor([[1.,0.]],device='cuda'))
    result['analytic_zero_norm_hinge0_slot_component_first_ties']=True
    connectivity={};initial=frozen_state(new_model(42));baseline=None
    for arm in ARMS:
        m=new_model(42);assert equal(initial,frozen_state(m));assert torch.count_nonzero(m(x))==0
        o=optimizer(m);norms=[]
        for step in range(8):
            ix=np.arange(step*256,(step+1)*256);xx=torch.from_numpy(d['x'][ix]).cuda();yy=torch.from_numpy(d['gt'][ix].astype(np.float32)).cuda();nn=torch.from_numpy(d['native_j'][ix]).cuda()
            o.zero_grad(set_to_none=True);value,_=objective(m(xx),yy,nn,arm);value.backward();gg={k:float(v.grad.norm()) for k,v in m.named_parameters()}
            assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in m.parameters())
            if step==0:assert gg['head.weight']>0 and all(v==0 for k,v in gg.items() if not k.startswith('head.'))
            if step==7:assert all(v>0 for v in gg.values())
            norms.append(gg);torch.nn.utils.clip_grad_norm_(m.parameters(),5.);o.step()
        fixed=predict(m,d['x'][:257],32);assert np.array_equal(fixed,predict(m,d['x'][:257],128));assert np.array_equal(fixed,predict(m,d['x'][:257],256))
        connectivity[arm]=norms
    result.update(actual_GPU8_FIT256_phased_gradients=connectivity,paired_all_weights_buffers_stats_exact=True,initial_zero_not_native_identity=True,storage_chunks_EXACT=True,forward_signature=['features'],old_F16_no_new_forward=True)
    # Synthetic active-reference fixture isolates penalty gradients without pretending natural activation.
    m=fixed_nonzero(new_model(42));pred=m(x);zero=torch.zeros(8,device='cuda');v,info=objective(pred,y,zero,'P_relative');pg=parameter_grads(m,.2*info['excess'].mean());assert all(torch.isfinite(v).all() for v in pg) and sum(float(v.norm()) for v in pg)>0
    result['synthetic_active_parameter_penalty_gradient_norm']=sum(float(v.norm()) for v in pg)
    return result

def resume_fixture(data,arm):
    d={k:v[:513].copy() for k,v in data['FIT'].items()};orders=np.stack([np.random.default_rng(20261023+ep).permutation(513) for ep in range(2)])
    con=dict(arm=arm,source='current_subset',IDs=old.arraysha(d['ids']),statistics=sha256(HERE/'statistics.npz'),initial=sha256(HERE/'initial_states/seed42.pt'),orders=old.arraysha(orders))
    def start():m=new_model(42);return m,optimizer(m),torch.Generator().manual_seed(42)
    def go(m,o,g,hist,slots,lo,hi):
        for ep in range(lo,hi):
            rec,slots=epoch(m,o,g,d,orders[ep],arm,slots);pred=predict(m,data['inner']['x'][:12]);rec['inner12_base']=numpy_objective(pred.astype(float),data['inner']['gt'][:12],data['inner']['native_j'][:12],arm);hist.append(rec)
        return hist,slots
    m,o,g=start();hist,slots=go(m,o,g,[],None,0,2);full=checkpoint(m,o,g,2,6,hist,slots,con)
    m,o,g=start();hist,slots=go(m,o,g,[],None,0,1);part=checkpoint(m,o,g,1,3,hist,slots,con)
    dest=HERE/'preflight_resume';dest.mkdir(exist_ok=True);torch.save(part,dest/f'{arm}.pt')
    m,o,g=start();blob=torch.load(dest/f'{arm}.pt',map_location='cpu',weights_only=False);restore(blob,m,o,g,con);hist,slots=go(m,o,g,blob['history'],blob['assignment'],1,2);res=checkpoint(m,o,g,2,6,hist,slots,con);assert equal(full,res)
    for key in con:
        bad=copy.deepcopy(con);bad[key]='BAD'
        try:restore(blob,m,o,g,bad)
        except ValueError:pass
        else:raise AssertionError('BAD_RESUME_ACCEPTED')
    return dict(status='PASS_EXACT',FIT=513,inner=12,epochs=2,all_model_buffers_optimizer_TorchCPU_CUDA_loader_numpy_python_RNG_history_assignment_exact=True,full_FIT_two_epochs=False)

def stats_fixture(data):
    x=data['FIT']['x'].astype(float);mean=x.mean(0);std=x.std(0);std[std<1e-6]=1
    st=np.load(HERE/'statistics.npz');assert np.array_equal(st['fit_ids'],data['FIT']['ids'])
    error=max(float(abs(mean-st['mean_float64']).max()),float(abs(std-st['std_float64']).max()));assert error<=1e-11
    assert np.array_equal(mean.astype(np.float32),st['mean']) and np.array_equal(std.astype(np.float32),st['std'])
    return dict(maxerror=error,FIT_only=True,mean_std_float32_EXACT=True)

def benchmark(data,guard):
    timings={};smokes={};idx=sorted(range(18909),key=lambda i:hashlib.sha256(f'tail-20261023::{int(data["FIT"]["ids"][i])}'.encode()).digest())[:128]
    for arm in ARMS:
        m=new_model(42);o=optimizer(m);d=data['FIT'];x=torch.from_numpy(d['x'][:256]).cuda();y=torch.from_numpy(d['gt'][:256].astype(np.float32)).cuda();nj=torch.from_numpy(d['native_j'][:256]).cuda();tt=[]
        for i in range(36):
            guard.check();torch.cuda.synchronize();start=time.perf_counter();o.zero_grad(set_to_none=True);value,_=objective(m(x),y,nj,arm);value.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5.);o.step();torch.cuda.synchronize()
            if i>=4:tt.append(time.perf_counter()-start)
        timings[arm]=dict(mean_step_seconds=float(np.mean(tt)),warm=4,measured=32)
        sub={k:v[idx] for k,v in d.items()};pred=predict(m,sub['x']);path=HERE/f'preflight_saved/{arm}';path.mkdir(parents=True,exist_ok=True)
        start=time.perf_counter();save_evaluate(path,'FIT',pred,sub,guard);smokes[arm]=time.perf_counter()-start
    train=sum(v['mean_step_seconds'] for v in timings.values())*3*8*74
    evaluate_est=max(smokes.values())*6*(18909+1856)/128
    planning=(train+evaluate_est+120)*1.35
    return dict(timings=timings,actual128_full_save_quality_cluster_motion_decomposition_independent_seconds=smokes,planning_seconds=planning,ETA_seconds=[.7*planning,1.5*planning],train_compute_seconds=train,includes_final_source_CI_IO_allowance_seconds=120,not_production_guarantee=True,benchmark_models_saved=False)

def completion_contract_fixture():
    con=contract('A_replay',42);p=HERE/'preflight_saved/probe_done.json';rel='preflight_saved/A_replay/predictions_FIT.npz'
    write_json(p,dict(contract=con,outputs={rel:sha256(HERE/rel)}));valid_done(p,con)
    for key in con:
        bad=copy.deepcopy(con);bad[key]='BAD'
        try:valid_done(p,bad)
        except ValueError:pass
        else:raise AssertionError('BAD_CONTRACT_ACCEPTED '+key)
    bad=read_json(p);bad['outputs'][rel]='0'*64;badpath=HERE/'preflight_saved/bad_done.json';write_json(badpath,bad)
    try:valid_done(badpath,con)
    except ValueError:pass
    else:raise AssertionError('BAD_OUTPUT_ACCEPTED')
    return dict(status='PASS',production_completed_branch_actual_file_SHA_skip=True,bad_source_stats_ID_initial_order_arm_output_rejected=True)

def main():
    t=time.perf_counter();guard=Guard(E);guard.check('startup')
    parent=read_json(DIAG/'source_manifest.json')
    for k,v in parent['files'].items():assert sha256(v['path'])==v['sha256'],k
    data=load_data();prepare(data);sf=stats_fixture(data);fx=fixtures(data);res={arm:resume_fixture(data,arm) for arm in ARMS};bench=benchmark(data,guard)
    # Actual production bootstrap and independent expanded-track implementation on fixed observations.
    stats=module('_tail_statistics_fixture',HERE/'statistical_analysis.py');n=96;tracks=np.array([f'track{i//3}' for i in range(n)]);groups=np.full(n,2);arr={}
    rng=np.random.default_rng(20261023)
    for arm in (*ARMS,'native'):
        arr[arm]=[dict(Jmin=rng.uniform(.1,3,n),coverage=rng.random(n)>.4,minADE=rng.uniform(.01,.5,n),minFDE=rng.uniform(.01,1,n)) for _ in SEEDS]
    boot=stats.bootstrap(arr,tracks,groups,save=HERE/'preflight_saved/bootstrap_fixture.npz');assert len(boot['results'])==17
    write_json(HERE/'BOOTSTRAP_PREFLIGHT.json',boot)
    free=shutil.disk_usage(HERE).free;disk=5*2**30;assert free>=1.3*disk,('STOP_DISK',free,disk)
    write_json(HERE/'MEMORY_ESTIMATE.json',dict(E_bytes=E,factor=1.3,inventory_GiB=dict(observed_previous_production_private_peak=3.294415,incremental_relative_logging_and_diagnostics=.8,independent_saved_npz_and_20k_CI_scratch=1.1,IO_margin=.805585),startup='physical and commit >=1.3E',runtime='dynamic remaining and retained peak <=1.3E',disk_free_bytes=free,conservative_disk_bytes=disk,no_fixed_6GiB_gate=True))
    protocol=read_json(HERE.parent/'protocol.draft.json');protocol.update(status='EXECUTION_ACTUAL_PREFLIGHT_COMPLETE',execution_authorized_by_user='2026-10-23 按照原计划执行',implementation=HERE.name,actual_stages=['T1_A_replay_P_relative','R'],D0_D1_entry='prior independently audited ALLOW_T1_ACTUAL_PREFLIGHT')
    protocol['secondary_lost_rate_interpretation']='For P-A, P lost among A successes / A successes (A self-reference0); P-native analogous. Engineering native-success lost P vs A separately evaluated.'
    write_json(HERE/'protocol.json',protocol)
    write_json(HERE/'FIXTURES.json',fx);write_json(HERE/'RESUME_CHECK.json',res);write_json(HERE/'BENCHMARK.json',bench);write_json(HERE/'STATISTICS_CHECK.json',sf)
    digest=lock_sources();cc=completion_contract_fixture();write_json(HERE/'COMPLETION_CONTRACT_PREFLIGHT.json',cc)
    guard.check();assert verify_sources()==digest
    write_json(HERE/'PREFLIGHT.json',dict(status='PASS',source_digest=digest,sources=len(read_json(HERE/'source_manifest.json')['files']),elapsed_seconds=time.perf_counter()-t,actual_GPU8_BOTH=True,subset_resume=res,control_scalar_gradient_EXACT=True,FD=fx['fixed_parameter_direction_FD'],full_saved_cluster_motion_decomposition_independent=True,production_manifest_completed_skip_contract_rejection=cc,resource_observations=guard.observations,estimated_peak_bytes=E,benchmark=bench))
    write_json(HERE/'PRELAUNCH_REVIEW.json',dict(status='PASS',source_digest=digest,all_actual_gates=True,old_F16_not_instantiated_no_new_frozen_GPU_claim=True,inline_same_assistant_not_external=True,limitations=['F16 upstream saw inner','repeated inner reuse','3 optimization seeds not independent scenes','relative quality does not guarantee physical safety'],no_automatic_BE_scorer_or_variants=True))
    print('PREFLIGHT_PASS',digest,'planning_seconds',bench['planning_seconds'],flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback
        write_json(HERE/'PREFLIGHT_FAILURE.json',dict(status='FAILURE',at=now(),traceback=traceback.format_exc()));raise
