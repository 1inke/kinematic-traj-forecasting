"""Isolated CVXPY/Clarabel SOCP service. No data discovery, GT only oracle payload."""
import os,sys
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,time,traceback
import numpy as np,scipy.sparse as sp,cvxpy as cp,clarabel
def operators():
    kernel=np.asarray([[.20/144*(2*(t-q)+1) if q<=t else 0. for q in range(12)] for t in range(12)])
    vel=np.eye(12);vel[np.arange(1,12),np.arange(11)]=-1;vel/=.4
    diff=np.diff(np.eye(12),axis=0)/.4
    acc=diff@vel;jerk=diff[:-1,:-1]@acc
    return kernel,vel,acc,jerk
def templates():
    k,v,a,j=operators();K=sp.block_diag([k]*20,format='csc');A=sp.block_diag([a]*20,format='csc');J=sp.block_diag([j]*20,format='csc')
    u=cp.Variable((240,2));base=cp.Parameter((240,2));gt=cp.Parameter((12,2));ab=cp.Parameter(nonneg=True);jb=cp.Parameter(nonneg=True)
    correction=K@u;p=base+correction;acc=cp.norm(A@p,axis=1);jerk=cp.norm(J@p,axis=1)
    common=[cp.norm(u,axis=1)<=1,cp.norm(correction,axis=1)<=np.tile(.20*(np.arange(1,13)/12)**2,20),
      cp.sum(acc)<=220*ab,cp.sum(jerk)<=200*jb]
    problems=[]
    base.value=np.zeros((240,2));gt.value=np.zeros((12,2));ab.value=jb.value=0.
    for slot in range(20):
        target=p[slot*12:(slot+1)*12]-gt
        problem=cp.Problem(cp.Minimize(cp.sum(acc)),common+[cp.sum(cp.norm(target,axis=1))<=12*.2,cp.norm(target[-1])<=.4])
        assert problem.is_dcp() and problem.is_dpp()
        problem.get_problem_data(cp.CLARABEL);problems.append(problem)
    return u,base,gt,ab,jb,problems
def main():
    u,base,gt,ab,jb,problems=templates()
    print(json.dumps(dict(status='READY',actual_service_pid=os.getpid(),cvxpy=cp.__version__,clarabel=clarabel.__version__,templates=20)),flush=True)
    for line in sys.stdin:
        request=json.loads(line);start=time.perf_counter();slot=int(request['slot']);assert 0<=slot<20
        base.value=np.asarray(request['native'],float).reshape(240,2);gt.value=np.asarray(request['gt'],float);ab.value=request['accel_budget'];jb.value=request['jerk_budget'];u.value=None
        result=dict(status='SOLVER_EXCEPTION',primal=None)
        try:
            if request.get('fixture_sleep'):time.sleep(request['fixture_sleep'])
            problems[slot].solve(solver=cp.CLARABEL,warm_start=False,verbose=False,max_iter=200,max_threads=1,
              tol_gap_abs=1e-9,tol_gap_rel=1e-9,tol_feas=1e-9)
            stats=problems[slot].solver_stats
            result=dict(status=problems[slot].status,primal=None if u.value is None else u.value.reshape(20,12,2).tolist(),
              iterations=stats.num_iters,solver_seconds=stats.solve_time,objective=problems[slot].value if problems[slot].value is not None and np.isfinite(problems[slot].value) else None)
        except Exception:result['traceback']=traceback.format_exc()
        result.update(request_id=request['request_id'],elapsed_seconds=time.perf_counter()-start)
        print(json.dumps(result,allow_nan=False),flush=True)
if __name__=='__main__':main()
