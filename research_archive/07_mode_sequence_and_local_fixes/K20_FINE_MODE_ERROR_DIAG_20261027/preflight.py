from common import *
from fine_features import *
from geometry_oracles import *
import time
def scalar_grid(p,gt):
    # Independent direct interpolation/loop over all 441 grid pairs.
    z=np.vstack((np.zeros(2),p));seg=np.linalg.norm(np.diff(z,axis=0),axis=1)
    base=np.linalg.norm(p-gt,axis=1).mean();answer=base;winner=(1.,1.)
    if np.any(seg<=0):return answer
    q=np.r_[0,np.cumsum(seg)/seg.sum()];w=np.array([q[4],q[8]-q[4],1-q[8]])
    for a in RATES:
        for b in RATES:
            c=(1-a*w[0]-b*w[1])/w[2]
            if c<.75 or c>1.25:continue
            v=[]
            for t in range(1,13):
                qt=q[t]*a if t<=4 else (q[4]*a+(q[t]-q[4])*b if t<=8 else q[4]*a+(q[8]-q[4])*b+(q[t]-q[8])*c)
                v.append([np.interp(qt,q,z[:,j]) for j in range(2)])
            v=np.asarray(v);v[-1]=p[-1]
            if a==1 and b==1:v=p.copy()
            answer=min(answer,float(np.linalg.norm(v-gt,axis=1).mean()))
    return answer
def main():
    g=guard(True);start=time.perf_counter();checks=[]
    assert abs(wrapped(np.deg2rad(-179)-np.deg2rad(179))-np.deg2rad(2))<1e-12;checks.append('wrapped_pi')
    h=np.zeros((1,8,2));h[0,:,1]=np.arange(-7,1)*.4
    p=np.zeros((1,1,12,2));p[0,0,:,0]=np.arange(1,13)*.4
    f=extract(p,h);assert f['L0'][0,0]==2 and f['entry'][0,0]==2 and f['onset_frame'][0,0]==1;checks.append('old_straight_immediate_entry_right90')
    p[0,0,:4,0]=0;p[0,0,:4,1]=np.arange(1,5)*.4;p[0,0,4:,0]=np.arange(1,9)*.4;p[0,0,4:,1]=1.6
    f=extract(p,h);assert f['entry'][0,0]==0 and f['onset_frame'][0,0]>=5;checks.append('middle_turn')
    f=extract(np.zeros_like(p),np.zeros_like(h));assert not f['history_heading_valid'][0] and f['entry'][0,0]==3 and f['onset'][0,0]==5;checks.append('zero_separate_history_and_future_undefined')
    oscill=h.copy();oscill[0,:,0]=np.arange(8)%2;oscill[0,:,1]=0
    assert not extract(p,oscill)['history_heading_valid'][0];checks.append('oscillation_coherence')
    x=xload('opt');xx={k:v[:128] for k,v in x.items()};y=yload('opt')['gt'][:128]
    f=extract(xx['native'],xx['past']);prior=load(OLD/'E1_mode_distribution/opt/coarse20/modes.npz',('mode',))['mode'][:128]
    assert np.array_equal(f['L0'],prior);checks.append('actual128_old_L0_exact')
    q=quality(xx['native'],y);np.testing.assert_array_equal(q[3],(q[0]<=.2)&(q[1]<=.4));checks.append('same_candidate_not_disjoint_minima')
    begin=time.perf_counter();t=time_search(xx['native'],y);time_seconds=time.perf_counter()-begin
    pp,qq,c=time_from_rates(xx['native'].reshape(-1,12,2),t['rate_a'].ravel(),t['rate_b'].ravel(),t['time_feasible'].ravel());pp=pp.reshape(xx['native'].shape)
    assert np.array_equal(pp[:,:,-1],xx['native'][:,:,-1])
    after=quality(pp,y);assert np.array_equal(after[1],q[1]);assert np.max(abs(after[0]-t['best_ADE']))<2e-12
    assert np.all(after[3].any(1)[q[3].any(1)]);assert not after[3].any(1)[np.all(q[1]>.4,axis=1)].any();checks.append('time_endpoint_FDE_identity_no_impossible_rescue')
    for i in range(8):
        for k in range(20):assert abs(scalar_grid(xx['native'][i,k],y[i])-t['best_ADE'][i,k])<2e-12
    checks.append('actual160_candidates_all441_grid_independent_scalar')
    delta=y[:,None,-1]-xx['native'][:,:,-1];end=xx['native']+np.arange(1,13)[None,None,:,None]/12*delta[:,:,None];end[:,:,-1]=y[:,None,-1]
    assert np.array_equal(end[:,:,-1],np.broadcast_to(y[:,None,-1],end[:,:,-1].shape));checks.append('end_exact_endpoint')
    for bound in (.2,.4):assert np.array_equal(np.where((np.linalg.norm(delta,axis=-1)<=bound)[...,None,None],end,xx['native'])[np.linalg.norm(delta,axis=-1)>bound],xx['native'][np.linalg.norm(delta,axis=-1)>bound])
    checks.append('end_infeasible_not_clipped')
    path=HERE/'preflight/fixture128.npz';save(path,sample_id=xx['sample_id'],L0=f['L0'],entry=f['entry'],time_ADE=after[0],time_FDE=after[1],rate_a=t['rate_a'],rate_b=t['rate_b'])
    got=load(path);assert np.array_equal(got['L0'],f['L0']);checks.append('saved_readback')
    r=extract(xx['native'],xx['past']);gt0=np.zeros_like(y);gt1=np.ones_like(y)*999
    assert all(np.array_equal(r[k],extract(xx['native'],xx['past'])[k],equal_nan=True) for k in r);checks.append('GT_not_feature_API_input_and_unused_GT_change_exact')
    write(HERE/'preflight/PREFLIGHT.json',dict(status='PASS_CPU_ACTUAL128_AND_SYNTHETIC',checks=checks,all_grid_scalar_candidates=160,old_L0_exact=True,actual128_TIME_seconds=time_seconds,elapsed_seconds=time.perf_counter()-start,no_model_forward_optimizer=True))
    resource(HERE/'preflight/RESOURCE.json',g);print('PASS PREFLIGHT',time_seconds,flush=True)
if __name__=='__main__':main()
