"""No labels, groups, identities, logits or future enter block assembly."""
from core import *
MASK_INDEX=np.arange(1092+2,1284,3)
CONTINUOUS=np.setdiff1d(np.arange(1284),np.r_[323,MASK_INDEX])
def moments(x):
    # All extra features shared by actions => original equal-window/eligible
    # action weighting reduces EXACTLY to equal-window weighting for this block.
    x=np.asarray(x,np.float64);mean=x.mean(0);std=np.sqrt(np.maximum(((x-mean)**2).mean(0),0));std[std<1e-6]=1
    # Independent sequential Welford + second-pass/chunk aggregation.
    m=np.zeros(x.shape[1]);v=np.zeros_like(m)
    for i,row in enumerate(x,1):d=row-m;m+=d/i;v+=d*(row-m)
    st=np.sqrt(np.maximum(v/len(x),0));st[st<1e-6]=1
    assert np.max(abs(mean-m))<=1e-11 and np.max(abs(std-st))<=1e-11
    for size in (31,256):
        mm=sum((x[i:i+size].sum(0) for i in range(0,len(x),size)),np.zeros(x.shape[1]))/len(x)
        vv=sum((((x[i:i+size]-mm)**2).sum(0) for i in range(0,len(x),size)),np.zeros(x.shape[1]))/len(x)
        ss=np.sqrt(np.maximum(vv,0));ss[ss<1e-6]=1
        assert np.max(abs(mm-mean))<=1e-11 and np.max(abs(ss-std))<=1e-11
    return mean,std
def prepare_statistics():
    g=guard('CPU');require12();ad=read_json(HERE/'FEATURE_PROVENANCE.json')
    with np.load(HERE/'prepared/observed_blocks_opt.npz') as z:
        ids=z['sample_id'];token=z['token'].reshape(len(ids),768);neighbor=z['neighbor'].reshape(len(ids),192)
    # Blocked neighbor arms do not get fabricated admitted inputs/statistics.
    tm,ts=moments(token);nm,ns=moments(neighbor)
    nm[2::3]=0;ns[2::3]=1
    for seed in SEEDS:
        bm,bs=base_stats(seed);mean=np.r_[bm,0,tm,nm];std=np.r_[bs,1,ts,ns]
        assert len(mean)==1284 and np.array_equal(mean[:323].astype(np.float32),bm)
        assert np.array_equal(std[:323].astype(np.float32),bs)
        save_npz(HERE/f'prepared/stats_seed{seed}.npz',sample_id=ids,mean64=mean,std64=std,mean32=mean.astype(np.float32),std32=std.astype(np.float32))
    write_json(HERE/'STATISTICS_AUDIT.json',dict(status='PASS',fit_population='opt4096_only',window_equal=True,
        shared_action_extra_block_equivalence=True,base_original_stats_exact=True,two_pass_welford_chunks31_256_tolerance=1e-11,
        STOP_mask_unstandardized=True,blocked_neighbor_stats_not_permission=True,admissible_arms=ad['admissible_arms']))
    resource_report(HERE/'STATISTICS_RESOURCE.json',g)
def assemble(base,token,neighbor):
    base=np.asarray(base,np.float32);token=np.asarray(token,np.float32);neighbor=np.asarray(neighbor,np.float32)
    assert base.shape[-1]==324 and token.shape[-2:]==(8,96) and neighbor.shape[-3:]==(8,8,3)
    assert base.shape[0]==token.shape[0]==neighbor.shape[0]
    # Build only current batch, never4096x401x1284 materialization.
    a=base.shape[1];t=np.broadcast_to(token.reshape(-1,1,768),(len(base),a,768))
    n=np.broadcast_to(neighbor.reshape(-1,1,192),(len(base),a,192))
    return np.concatenate((base,t,n),-1)
def read_stats(seed):
    with np.load(HERE/f'prepared/stats_seed{seed}.npz') as z:return z['mean32'],z['std32']
if __name__=='__main__':prepare_statistics()
