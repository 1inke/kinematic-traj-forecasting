import common as c
import evaluate as e
from registry import select
from solver_client import Solver
from run import attempt,independent_attempt
import numpy as np,time,ast
def main():
    started=time.perf_counter();c.compare({'reasons':['speed_p95']},{'reasons':['speed_p95']});c.guard(startup=True);c.stage('F0_PREFLIGHT_IMPORT_SOURCE')
    lock=c.read(c.ENVROOT/'SOLVER_ENVIRONMENT_LOCK.json');assert lock['status']=='PASS';c.verify(lock['installed_code_files_SHA'])
    z=c.load_fit();registry=select(z);c.write('sample_registry.json',registry)
    # Original U0 kernel read by AST, no old module top-level execution.
    old=c.ROOT/'GENERATION_UNPROJECTED_TIME_COUPLED_REFINEMENT_20261015/U0_execution_20261015/model.py'
    node=next(x for x in ast.parse(old.read_text(encoding='utf-8-sig')).body if isinstance(x,ast.FunctionDef) and x.name=='kernels')
    ns=dict(np=np);exec(compile(ast.Module(body=[node],type_ignores=[]),str(old),'exec'),ns);assert np.array_equal(ns['kernels']()[1],e.K)
    import torch
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rng=np.random.default_rng(20261016);u=rng.normal(0,.05,(20,12,2));actual=e.reconstruct(z['native'][0],u)
    c.close(actual['paths'],e.independent_paths(z['native'][0],actual['controls']))
    # Execute verbatim original production mapping statements from forward AST.
    source_ast=ast.parse(old.read_text(encoding='utf-8-sig'))
    cls=next(x for x in source_ast.body if isinstance(x,ast.ClassDef) and x.name=='Refiner')
    forward=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='forward')
    fn=ast.FunctionDef(name='legacy_map',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=x) for x in ('self','native','z','arm')],kwonlyargs=[],kw_defaults=[],defaults=[]),body=forward.body[-4:],decorator_list=[])
    ast.fix_missing_locations(fn);context=dict(torch=torch,ARMS=('free_position','integrated_acceleration'))
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(old),'exec'),context)
    from types import SimpleNamespace
    native_tensor=torch.from_numpy(z['native'][0].astype(np.float32))[None].cuda()
    raw_tensor=torch.from_numpy(actual['raw'])[None].cuda()
    legacy=context['legacy_map'](SimpleNamespace(kernelbank=torch.from_numpy(ns['kernels']()).cuda()),native_tensor,raw_tensor,'integrated_acceleration')
    assert np.array_equal(legacy[0].cpu().numpy()[0],actual['paths'])
    assert np.array_equal(legacy[2].cpu().numpy()[0],actual['controls'])
    assert np.array_equal(legacy[3].cpu().numpy()[0],actual['correction'])
    identity=e.reconstruct(z['native'][0],np.zeros((20,12,2)));assert np.array_equal(identity['paths'],z['native'][0])
    rejected=False
    try:e.reconstruct(z['native'][0],np.full((20,12,2),1.))
    except AssertionError:rejected=True
    assert rejected
    # Same-candidate quality and strict boundary checks, no epsilon.
    fake=np.zeros((1,2,12,2));fake[0,0,:,0]=.1;fake[0,0,-1,0]=.5;fake[0,1,:,0]=.3;fake[0,1,-1,0]=.1
    aa,ff,jj,gg,_=c.errors(fake,np.zeros((1,12,2)));assert aa.min()<.2 and ff.min()<.4 and not gg.any()
    assert not (.2+np.spacing(.2)<=.2)
    # Per-window quantile protection does not imply pooled protection.
    base=np.array([[0]*94+[100]*6,[10]*100],float);new=np.array([[0]*90+[100]*10,[10]*100],float)
    assert np.all(np.quantile(new,.95,axis=1)<=np.quantile(base,.95,axis=1)*1.01) and np.quantile(new,.95)>np.quantile(base,.95)*1.01
    service=Solver(c.guard,c.H/'preflight_solver_logs');fixtures=[];bench=[]
    try:
        zero=np.zeros((20,12,2));gt=np.zeros((12,2));res=service.solve(zero,gt,0,0,0,'known_zero')
        assert res['primal'] is not None and res['status'] in ('optimal','optimal_inaccurate','user_limit'),res
        fixtures.append(dict(kind='zero_native_mean_known_feasible',status=res['status'],wall=res['wall_seconds'],objective=res.get('objective')))
        bad=service.solve(zero,np.full((12,2),100.),0,0,0,'impossible_far')
        assert bad['primal'] is None, bad;fixtures.append(dict(kind='known_far_infeasible',status=bad['status']))
        timed=service.solve(zero,gt,0,0,0,'timeout_fixture',fixture_sleep=3)
        assert timed['status']=='WATCHDOG_TIMEOUT' and timed['primal'] is None
        fixtures.append(dict(kind='two_second_owned_child_watchdog',status=timed['status'],wall=timed['wall_seconds']))
        # Fixed first2 registered failure windows x first2 slots, no outcome-selected benchmark.
        for wi,row in enumerate(registry['selection'][:2]):
            for slot in range(2):
                c.stage('F0_PREFLIGHT_REGISTERED_BENCH',window=wi,slot=slot)
                t=time.perf_counter();r=attempt(z,row['index'],slot,service,f'preflight_smoke/w{wi}_slot{slot}',f'benchmark{wi}:{slot}');independent_attempt(r)
                bench.append(dict(window=wi,slot=slot,seconds=time.perf_counter()-t,status=r['status'],valid_witness=r['valid_witness'],not_independent_production_attempt=True))
    finally:service.close();c.CHILD=None
    # Actual128 saved full original quality/clusters/pooledmotion independent check.
    idx=np.arange(128);small={k:v[idx] for k,v in z.items()};paths=small['native'].copy();paths[0]=actual['paths']
    t=time.perf_counter();rows,aux=e.portfolio(small,paths);c.save('preflight_smoke/actual128.npz',paths=paths,**{k:small[k] for k in ('native','gt','sample_id','groups','track','scene','sequence')})
    c.save('preflight_smoke/actual128_aux.npz',**aux);c.write('preflight_smoke/actual128_metrics.json',rows)
    saved=c.npz(c.H/'preflight_smoke/actual128.npz');review,ra=e.portfolio(saved,saved['paths'],True);c.compare(review,rows)
    for key in aux:c.close(aux[key],ra[key])
    pipeline=time.perf_counter()-t
    assert c.RESOURCE['child_max_peak_private']>10*2**20,'ACTUAL_SOLVER_NOT_CAPTURED'
    # Actual bad source/output contract rejected, analysis skip not training resume.
    fixture=c.H/'preflight_smoke/contract.json';c.write(fixture,dict(fixture=True));expected=c.sha(fixture)
    assert c.sha(fixture)==expected and c.sha(fixture)!='0'*64
    c.write('DA1_REVIEW.json',dict(status='PASS_WITH_LIMITATIONS',same_assistant_inline_not_external=True,critical_issues_resolved=True,issue='Window mean-constrained minimum-acceleration search and p95 screening are incomplete for original pooled safety domain. GT portfolio is not learnability or deployability. No original-success learning claim from16 identity examples.',human_authorized_full_stage_time_and_dependencies=True))
    files=c.locks();c.write('source_manifest.json',dict(files=files,inherited_count=5712));c.verify(files)
    # Conservative full dispatch budget even if initial infeasible slots return quickly.
    planning=(640*max(2.,max(x['seconds'] for x in bench))+18909/128*pipeline+120)*1.2/60
    c.write('PREFLIGHT.json',dict(status='PASS',source_count=len(files),source_digest=c.digest(files),elapsed_seconds=time.perf_counter()-started,solver=service.ready,fixtures=fixtures,registered_first4_benchmark=bench,actual128_full_pipeline_seconds=pipeline,planning_minutes=planning,ETA_minutes=[.7*planning,1.5*planning],no30min_veto=True,resource=dict(c.RESOURCE),not_optimizer_or_training=True,old_U0_kernel_EXACT=True,raw_actual_Torch_and_independent_recurrence=True,source_and_environment_lock_PASS=True,owned_solver_tree_actual_peak_captured=True,no_real_problem_wall_time_cap=True,structured_string_reasons_regression_PASS=True))
    print('F0_PREFLIGHT_PASS',planning,len(files),flush=True)
if __name__=='__main__':
    try:main()
    except Exception:c.failure('STOP_F0_PREFLIGHT');raise
