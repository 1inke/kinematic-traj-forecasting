"""Actual GPU, exact subset resume, source and saved-arithmetic acceptance."""
from training import *
import ast,json,shutil

def old_formula():
    path=ROOT/'HYBRID_NATIVE_KIN_UNION_20261018/kdec/train.py';tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['winner_loss','diversity_hinge']]
    scope={'torch':torch};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope);return scope

def gradients(m,value):return [t.detach().cpu().clone() for t in torch.autograd.grad(value,tuple(m.parameters()),allow_unused=False)]

def fixtures(data,teachers):
    results={};old=old_formula();x,y,R,rm,N,nm=tensors(data['FIT'],teachers[42],np.arange(8),'cuda')
    m=new_model(42);seed(20822)
    with torch.no_grad():m.head.weight.normal_(0,.03);m.head.bias.normal_(0,.03)
    p=m(x);a=base_loss(p,y)[0];b=old['winner_loss'](p,y)+.05*old['diversity_hinge'](p)
    ga=gradients(m,a);p=m(x);gb=gradients(m,old['winner_loss'](p,y)+.05*old['diversity_hinge'](p))
    diff=max(float((u-v).abs().max()) for u,v in zip(ga,gb));assert abs(float(a-b))<=2e-6 and diff<=2e-6
    results['old_base_scalar_diff']=float(abs(a-b));results['old_base_all_parameter_grad_max']=diff
    pp=p.detach().cpu().numpy();npdiff=abs(numpy_base(pp,y.cpu().numpy())-float(a.detach()));assert npdiff<=2e-6;results['independent_numpy_base_diff']=npdiff
    # Four empty/nonempty group cases use a double model to test algebra without float32 reduction-order noise.
    md=m.double();xd=x.double();rd=R.double();nd=N.double();yd=y.double();cases={}
    for case in ['empty','R_only','N_only','both']:
        mm=torch.zeros_like(rm);mn=torch.zeros_like(nm)
        if case in ['R_only','both']:mm[0,:3]=True
        if case in ['N_only','both']:mn[1,:1]=True
        pred=md(xd);sr,nr,_,_=teacher_sums(pred,rd,mm);sn,nt,_,_=teacher_sums(pred,nd,mn)
        union=(sr+sn)/max(nr+nt,1);gu=gradients(md,union)
        pred=md(xd);sr,nr,_,_=teacher_sums(pred,rd,mm);sn,nt,_,_=teacher_sums(pred,nd,mn)
        den=max(nr+nt,1);weighted=nr/den*(sr/max(nr,1))+nt/den*(sn/max(nt,1));gw=gradients(md,weighted)
        gd=max(float((u-v).abs().max()) for u,v in zip(gu,gw));assert gd<=2e-12
        cases[case]=dict(scalar_diff=float(abs(union-weighted)),all_parameter_grad_diff=gd,n_R=nr,n_N=nt)
        if case=='both':
            pred=md(xd);sr,nr,_,_=teacher_sums(pred,rd,mm);sn,nt,_,_=teacher_sums(pred,nd,mn);bal=.5*sr/nr+.5*sn/nt
            assert abs(float(bal-union))>1e-6;cases[case]['Ebal_not_identity_difference']=float(bal-union)
    results['algebra_four_cases']=cases
    # Same observations are the only forward argument; label/ID/logit/teacher permutations cannot enter it.
    m=new_model(42);initial=frozen_state(m);zero=m(x);assert torch.count_nonzero(zero)==0
    gl=base_loss(zero,y)[0];gl.backward()
    assert m.head.weight.grad.norm()>0 and all(p.grad is None or p.grad.norm()==0 for n,p in m.named_parameters() if not n.startswith('head.'))
    connectivity={}
    for arm in ['A','B','E_bal']:
        model=new_model(42);assert equal(initial,frozen_state(model));opt=optimizer(model)
        for i in range(8):
            idx=np.arange(i*256,(i+1)*256);xx,yy,rr,mr,nnn,mn=tensors(data['FIT'],teachers[42],idx,'cuda')
            opt.zero_grad(set_to_none=True);value,_=loss(model(xx),yy,rr,mr,nnn,mn,arm);value.backward()
            assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters())
            if i==7:
                norms={n:float(p.grad.norm()) for n,p in model.named_parameters()};assert all(v>0 for v in norms.values());connectivity[arm]=norms
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
        fixed=predict(model,data['FIT']['x'][:257],32);assert np.array_equal(fixed,predict(model,data['FIT']['x'][:257],128))
        assert np.array_equal(fixed,predict(model,data['FIT']['x'][:257]))
    results.update(actual8steps_each_arm_all_parameter_connectivity=connectivity,initial_zero_not_native=True,paired_initial_all_buffers_exact=True,compute256_storage32_128_256_exact=True,forward_arguments=['features'])
    return results

def resume_fixture(data,teacher,arm):
    dd={k:v[:513].copy() for k,v in data['FIT'].items()};tt={k:v[:513].copy() for k,v in teacher.items() if isinstance(v,np.ndarray)}
    orders=np.stack([np.random.default_rng(42+ep).permutation(513) for ep in range(2)]);c=dict(fixture=True,arm=arm,IDs=arraysha(dd['ids']),source='subset_same_current_source')
    def start():
        m=new_model(42);opt=optimizer(m);g=torch.Generator().manual_seed(42);return m,opt,g
    def go(m,opt,g,hist,lo,hi):
        for ep in range(lo,hi):
            rec=epoch(m,opt,g,dd,tt,orders[ep],arm);pred=predict(m,data['inner']['x'][:12]);rec['inner12_base']=numpy_base(pred,data['inner']['gt'][:12]);hist.append(rec)
        return hist
    m,o,g=start();h=go(m,o,g,[],0,2);full=checkpoint(m,o,g,2,6,h,h,c)
    m,o,g=start();h=go(m,o,g,[],0,1);part=checkpoint(m,o,g,1,3,h,h,c);path=HERE/f'preflight_resume/{arm}.pt';path.parent.mkdir(exist_ok=True);torch.save(part,path);del m,o
    m,o,g=start();blob=torch.load(path,map_location='cpu',weights_only=False);restore(blob,m,o,g,c);h=go(m,o,g,blob['history'],1,2);res=checkpoint(m,o,g,2,6,h,h,c)
    assert equal(full,res)
    for key in ['source','IDs','arm']:
        bad=dict(c);bad[key]='deliberately_wrong'
        try:restore(blob,m,o,g,bad)
        except ValueError:pass
        else:raise AssertionError('Bad resume accepted')
    return dict(status='PASS_EXACT',FIT=513,inner=12,epochs=2,all_params_buffers_optimizer_CPU_CUDA_loader_numpy_python_RNG_history_assignment_exact=True,full_FIT_two_epochs_claimed=False)

def saved_smoke(data,m):
    d={k:v[:128] for k,v in data['FIT'].items()};p=predict(m,d['x']);base=HERE/'preflight_saved';base.mkdir(exist_ok=True)
    np.savez_compressed(base/'prediction.npz',paths=p,native=d['native'],gt=d['gt'],groups=d['groups'],sample_id=d['ids'],track=d['track'])
    start=time.perf_counter();metric,ee=evaluate(p,d['gt'],d['groups'],d['native'],d['ids'],d['track']);write_json(base/'metrics.json',metric)
    review=independent.audit_file(base/'prediction.npz',base/'metrics.json',frozen_cache=prior.PATHS['cache'],frozen_labels=prior.PATHS['labels'],frozen_metadata=prior.PATHS['metadata'],frozen_manifest=ROOT/'config/data_manifest.v1.json',root=ROOT)
    assert review['status']=='PASS';write_json(base/'REVIEW.json',review)
    return dict(seconds=time.perf_counter()-start,review=review)

def setup(data):
    for name in ['initial_states','batch_orders','teachers']: (HERE/name).mkdir(exist_ok=True)
    fit=data['FIT'];diag=sorted(range(len(fit['ids'])),key=lambda i:hashlib.sha256(f'context-diag-20261022::{int(fit["ids"][i])}'.encode()).digest())[:1024]
    np.savez_compressed(HERE/'diagnostic_ids.npz',FIT_index=np.array(diag),sample_id=fit['ids'][diag])
    out={}
    for s in SEEDS:
        model=new_model(s,'cpu');torch.save(model.state_dict(),HERE/f'initial_states/seed{s}.pt')
        rng=np.random.RandomState(s);orders=np.stack([rng.permutation(len(fit['ids'])) for _ in range(8)])
        np.savez_compressed(HERE/f'batch_orders/seed{s}.npz',orders=orders)
        out[s]=make_teacher(fit,s);np.savez_compressed(HERE/f'teachers/seed{s}.npz',**out[s],sample_id=fit['ids'])
        # Independently recompute teacher membership by per-slot distances, without errors().
        for arr,key in [(fit['native'],'R_mask'),(out[s]['N'],'N_mask')]:
            # KIN original float64 source is mandatory for masks, not the cast training array.
            aa=fit['native'] if key=='R_mask' else np.load(prior.kin_path('FIT',s)).astype(float)
            ds=np.sqrt(np.sum((aa-fit['gt'][:,None])**2,-1));good=(ds.sum(-1)/12<=.2)&(ds[:,:,-1]<=.4)
            if key=='N_mask':good &= ~out[s]['R_mask'].any(1)[:,None]
            assert np.array_equal(good,out[s][key])
    return out

def benchmark(data,teachers):
    timings={};guard=Guard(E);guard.check('startup')
    for arm in ['A','B','E_bal']:
        m=new_model(42);o=optimizer(m);x,y,R,rm,N,nm=tensors(data['FIT'],teachers[42],np.arange(256),'cuda');measure=[]
        for i in range(36):
            guard.check();torch.cuda.synchronize();t=time.perf_counter();o.zero_grad(set_to_none=True);v,_=loss(m(x),y,R,rm,N,nm,arm);v.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5.);o.step();torch.cuda.synchronize()
            if i>=4:measure.append(time.perf_counter()-t)
        timings[arm]=dict(mean_step_seconds=float(np.mean(measure)),measured_steps=32,warm=4)
    t=time.perf_counter();predict(m,data['inner']['x']);torch.cuda.synchronize();inner=time.perf_counter()-t
    smoke=saved_smoke(data,m);total_rows=4*(18909+1856) # Six model outputs plus two native references.
    eval_est=smoke['seconds']*total_rows/128
    train_est=timings['A']['mean_step_seconds']*3*8*74
    planning=(train_est+eval_est+60)*1.35
    return dict(arms=timings,inner_full_forward_seconds=inner,save_quality_full_cluster_independent128=smoke['seconds'],S1_train_compute_seconds=train_est,S1_audit_scaled_seconds=eval_est,S1_planning_seconds=planning,S1_ETA_seconds=[.7*planning,1.5*planning],includes_source_IO_allowance_seconds=60,not_production_guarantee=True,benchmark_models_saved=False,resource=guard.observations)

def lock_sources():
    inherited=read_json(ROOT/'qoder_work/E0E1_EVIDENCE_REPAIR_20261022/source_manifest_final.json')['files'];files={}
    for key,v in inherited.items():
        expected=v['sha256'] if isinstance(v,dict) else v;p=ROOT/key
        assert sha256(p)==expected,('INHERITED_SOURCE_CHANGED',key)
        files[key]=dict(path=str(p),sha256=expected,inherited=True)
    sa=read_json(HERE/'source_audit/RESULT.json')
    paths=[Path(p) for p in sa['source_paths']]
    paths+=list(HERE.parent.glob('*.md'))+[HERE.parent/'protocol.draft.json']+list(HERE.glob('*.py'))+list(HERE.glob('*.ps1'))
    paths += [HERE/n for n in ['protocol.json','input_contract.json','statistics.npz','split_ids.npz','diagnostic_ids.npz','MEMORY_ESTIMATE.json']]
    paths+=list((HERE/'initial_states').glob('*.pt'))+list((HERE/'batch_orders').glob('*.npz'))+list((HERE/'teachers').glob('*.npz'))
    old=ROOT/'qoder_work/E0E1_EVIDENCE_REPAIR_20261022'
    paths += [p for p in old.rglob('*') if p.is_file() and '__pycache__' not in str(p)]
    for p in sorted(set(p.resolve() for p in paths)):
        key=p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p);actual=sha256(p)
        if key in files:assert files[key]['sha256']==actual
        else:files[key]=dict(path=str(p),sha256=actual,inherited=False)
    digest=hashlib.sha256(json.dumps({k:v['sha256'] for k,v in sorted(files.items())},sort_keys=True,separators=(',',':')).encode()).hexdigest()
    write_json(HERE/'source_manifest.json',dict(digest=digest,files=files,previous139_preserved=True,scientific_source_locked=True))
    verify_sources();return len(files)

def main():
    timer=time.perf_counter();guard=Guard(E);guard.check('startup');data=load_data();teachers=setup(data);fx=fixtures(data,teachers)
    resumes={a:resume_fixture(data,teachers[42],a) for a in ['A','B','E_bal']}
    bench=benchmark(data,teachers);write_json(HERE/'BENCHMARK.json',bench)
    write_json(HERE/'FIXTURES.json',fx)
    write_json(HERE/'RESUME_CHECK.json',resumes)
    # Production completion contract is tested against actual files; altered source/output reject.
    expected=dict(actual=True,source='current');probe=HERE/'preflight_saved/probe_done.json';write_json(probe,dict(contract=expected,outputs={'preflight_saved/prediction.npz':sha256(HERE/'preflight_saved/prediction.npz')}));valid_done(probe,expected)
    for kind in ['source','output']:
        b=read_json(probe)
        if kind=='source':b['contract']['source']='wrong'
        else:b['outputs']['preflight_saved/prediction.npz']='0'*64
        bad=HERE/f'preflight_saved/bad_{kind}.json';write_json(bad,b)
        try:valid_done(bad,expected)
        except ValueError:pass
        else:raise AssertionError('bad completion accepted')
    draft=read_json(HERE.parent/'protocol.draft.json');ic=read_json(HERE/'input_contract.json')
    draft.update(status='EXECUTION_FROZEN_AFTER_ACTUAL_PREFLIGHT',execution_authorized_by_user='2026-10-22 按照路径下的跑实验',execution_authorized_by_this_file=False,implementation='execution_20261022',S1_only_initial_launch=True)
    draft['input'].update(producer_checkpoint=ic['checkpoint'],producer_sha256=ic['checkpoint_sha256'],cache_path=ic['cache'],cache_sha256=ic['cache_sha256'])
    draft['resource'].update(estimate_bytes=E,benchmark_eta_seconds=bench['S1_planning_seconds'])
    write_json(HERE/'protocol.json',draft)
    free=shutil.disk_usage(HERE).free;estimate=3*2*(18909+1856)*20*12*2*8*1.5+400*2**20
    assert free>estimate*1.3
    guard.check();write_json(HERE/'MEMORY_ESTIMATE.json',dict(E_bytes=E,factor=1.3,inventory_GiB=dict(runtime_torch_scipy_baseline=1.8,aligned_inputs_and_copies=.9,one_model_optimizer_activations=1.1,teachers_and_distances=.65,saved_quality_cluster_audit_scratch=.9,hash_IO_margin=.65),total_GiB=6.,startup_rule='both available physical and commit >=1.3E',runtime_rule='physical>=1.3max(E-RSS,0);commit>=1.3max(E-private,0);current/Windows-retained peaks<=1.3E',disk_free_bytes=free,S1_conservative_disk_bytes=estimate,resource_observations=guard.observations))
    write_json(HERE/'PREFLIGHT.json',dict(status='PASS',elapsed_seconds=time.perf_counter()-timer,actual_GPU_8steps_each_arm=True,actual_subset_resume=resumes,source_lineage='source_audit/RESULT.json',saved_arithmetic='preflight_saved/REVIEW.json',completed_source_output_rejection='PASS',all_old_weights_read_only=True,old_F16_replay_not_training=True,benchmark='BENCHMARK.json'))
    count=lock_sources();write_json(HERE/'PRELAUNCH_REVIEW.json',dict(status='PASS',sources=count,source_digest=verify_sources(),inline_same_assistant_not_external=True,DA1=dict(verdict='PASS_WITH_EXPLORATORY_LIMITS',major_limitations=['upstream inner exposure','zero-output initialization may fail; never extend8epochs','recent+mean not full token information','S1 gate is investment decision not statistical noninferiority']),stage_gates_unchanged=True))
    print('PREFLIGHT PASS',count,'planning_seconds',bench['S1_planning_seconds'],flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        import traceback;write_json(HERE/'PREFLIGHT_FAILURE.json',dict(traceback=traceback.format_exc()));raise
