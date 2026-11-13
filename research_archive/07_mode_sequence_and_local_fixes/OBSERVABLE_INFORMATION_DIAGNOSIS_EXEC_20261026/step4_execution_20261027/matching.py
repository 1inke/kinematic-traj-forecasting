"""GT-free time/past/candidate matching only. No B, quality or group arguments."""
from core import *
from scipy.optimize import linear_sum_assignment
import scipy
def checked_pools(path,identity):
    p=load_npz(path);assert set(p)=={'sample_id','track_key','native','kin','eligible'}
    assert np.array_equal(p['sample_id'],identity['sample_id']) and np.array_equal(p['track_key'],identity['track_key'])
    n=len(identity['sample_id']);assert p['native'].shape==p['kin'].shape==(n,20,12,2) and p['eligible'].shape==(n,401)
    assert p['eligible'].dtype==bool and p['eligible'][:,0].all() and np.isfinite(p['native']).all() and np.isfinite(p['kin']).all()
    return p
def legal_rows(q,qi,lib):
    if not q['known'][qi]:return np.zeros(len(lib['sample_id']),bool)
    start=int(q['start'][qi]);ls=lib['start']
    return lib['known']&(lib['sequence']==q['sequence'][qi])&(lib['track_key']!=q['track_key'][qi])&((ls+19<start)|(start+19<ls))
def shortlist(q,qi,lib):
    legal=legal_rows(q,qi,lib);ix=np.flatnonzero(legal)
    if not len(ix):return np.empty(0,np.int32),np.empty(0)
    rms=np.sqrt(((lib['past'][ix]-q['past'][qi])**2).sum(-1).mean(-1))
    best={}
    for i,d in zip(ix,rms):
        t=str(lib['track_key'][i]);key=(float(d),int(lib['sample_id'][i]))
        if t not in best or key<best[t][0]:best[t]=(key,int(i))
    selected=sorted(best.values(),key=lambda row:(row[0][0],str(lib['track_key'][row[1]]),row[0][1]))[:32]
    indices=np.asarray([r[1] for r in selected],np.int32);distance=np.asarray([r[0][0]/.1 for r in selected])
    return indices,distance
def matrix(a,b):
    delta=a[:,None]-b[None,:];point=np.sqrt((delta*delta).sum(-1))
    return np.maximum(point.mean(-1)/.1,point[:,:,-1]/.2)
def assignment(cost,reverse=False):
    n=len(cost)
    if reverse:
        rows,cols=linear_sum_assignment(cost[::-1,::-1]);mapping=np.empty(n,np.int16);mapping[n-1-rows]=n-1-cols
    else:
        rows,cols=linear_sum_assignment(cost);mapping=np.empty(n,np.int16);mapping[rows]=cols
    assert sorted(mapping.tolist())==list(range(n))
    return mapping,float(cost[np.arange(n),mapping].sum())
def certificate(cost,mapping,total):
    best=np.inf;n=len(cost)
    for i in range(n):
        c=cost.copy();c[i,int(mapping[i])]=np.inf
        try:r,k=linear_sum_assignment(c);alternative=float(c[r,k].sum())
        except ValueError:continue
        assert alternative>=total-2e-12*max(1,abs(total))
        best=min(best,alternative-total)
    return best,bool(best<=2e-12*max(1,abs(total))),n
def pair(qnative,qkin,pnative,pkin,dH):
    primary=[];reverse=[];costs=[];rcosts=[];gaps=[];amb=[];nextra=0
    for a,b in ((qnative,pnative),(qkin,pkin)):
        c=matrix(a,b);m,total=assignment(c);r,revtotal=assignment(c,True)
        assert abs(total-revtotal)<=2e-12*max(1,abs(total))
        gap,ambiguous,n=certificate(c,m,total);primary.append(m);reverse.append(r);costs.append(total/20);rcosts.append(revtotal/20);gaps.append(gap);amb.append(ambiguous);nextra+=n
    dP=float(np.mean(costs));rdP=float(np.mean(rcosts))
    return dict(native_map=primary[0],kin_map=primary[1],reverse_native_map=reverse[0],reverse_kin_map=reverse[1],
        native_mean_cost=costs[0],kin_mean_cost=costs[1],reverse_native_mean_cost=rcosts[0],reverse_kin_mean_cost=rcosts[1],
        dH=float(dH),dP=dP,d=max(float(dH),dP),reverse_dP=rdP,reverse_d=max(float(dH),rdP),
        alternative_gap=np.array(gaps),numerically_ambiguous=np.array(amb),audit_forbidden_edge_problems=nextra)
def action_map(native,kin):
    out=np.r_[0,1+(20*np.asarray(native)[:,None]+np.asarray(kin)[None,:]).reshape(-1)].astype(np.int16)
    assert len(out)==401 and np.array_equal(np.sort(out),np.arange(401)) and out[0]==0
    return out
def selection(pack,identity):
    """Lock all geometry before posthoc labels; pure saved pack input."""
    summaries={};selected={}
    kinds=pack['kind'];positions=pack['reference_position'];qrows=pack['query_row'];prows=pack['library_row']
    for mode,key in (('primary','d'),('reverse','reverse_d')):
        nearest=[]
        for position in range(len(identity['opt']['reference_rows'])):
            ix=np.flatnonzero((kinds==0)&(positions==position))
            if len(ix):nearest.append(float(pack[key][ix].min()))
        tau=float(np.quantile(nearest,.25,method='linear')) if nearest else None
        chosen=[];perquery=[]
        for pos,qi in enumerate(identity['hold']['reference_rows']):
            ix=np.flatnonzero((kinds==1)&(positions==pos))
            eligible=[] if tau is None else [int(i) for i in ix if pack[key][i]<=tau]
            eligible.sort(key=lambda i:(float(pack[key][i]),str(identity['opt']['track_key'][prows[i]]),int(identity['opt']['sample_id'][prows[i]])))
            keep=eligible[:5];chosen+=keep
            reason='SUPPORTED' if keep else ('UNKNOWN_QUERY_TIME' if not identity['hold']['known'][qi] else 'NO_LEGAL_CROSS_TRACK_TIME_PAIR' if not len(ix) else 'NO_PAIR_WITHIN_FIXED_TAU')
            perquery.append(dict(reference_position=pos,query_row=int(qi),sample_id=int(identity['hold']['sample_id'][qi]),track_key=str(identity['hold']['track_key'][qi]),n=len(keep),pair_indices=keep,status=reason))
        selected[mode]=np.asarray(chosen,np.int32)
        summaries[mode]=dict(calibration_valid=len(nearest),calibration_reference_status='PASS' if len(nearest)>=50 else 'UNDERPOWERED',tau=tau,calibration_nearest=nearest,queries=perquery,supported_tracks=sum(r['n']>0 for r in perquery))
    return selected,summaries
def compute(seed,g,limit_references=None):
    ids={s:load_npz(HERE/f'prepared/identity_{s}.npz') for s in ('opt','hold')}
    pools={s:checked_pools(HERE/f'prepared/pools_seed{seed}_{s}.npz',ids[s]) for s in ('opt','hold')}
    rows=[];t=time.perf_counter()
    for kind,split in ((0,'opt'),(1,'hold')):
        refs=ids[split]['reference_rows']
        if limit_references is not None:refs=refs[:limit_references]
        for position,qi in enumerate(refs):
            pi,dh=shortlist(ids[split],int(qi),ids['opt'])
            for librow,d in zip(pi,dh):
                r=pair(pools[split]['native'][qi],pools[split]['kin'][qi],pools['opt']['native'][librow],pools['opt']['kin'][librow],float(d))
                r.update(kind=kind,reference_position=position,query_row=int(qi),library_row=int(librow));rows.append(r)
            if position%8==0:g.check();mark('STEP4_GT_FREE_MATCHING',seed=seed,population=split,reference_position=position+1,pairs=len(rows))
    # Empty legal population is retained, with typed arrays for every field.
    templates=dict(native_map=np.zeros(20,np.int16),kin_map=np.zeros(20,np.int16),reverse_native_map=np.zeros(20,np.int16),reverse_kin_map=np.zeros(20,np.int16),alternative_gap=np.zeros(2),numerically_ambiguous=np.zeros(2,bool))
    numeric=('native_mean_cost','kin_mean_cost','reverse_native_mean_cost','reverse_kin_mean_cost','dH','dP','d','reverse_dP','reverse_d')
    integer=('audit_forbidden_edge_problems','kind','reference_position','query_row','library_row')
    pack={k:np.asarray([r[k] for r in rows]) if rows else np.empty((0,)+v.shape,v.dtype) for k,v in templates.items()}
    pack.update({k:np.asarray([r[k] for r in rows],dtype=np.float64) for k in numeric})
    pack.update({k:np.asarray([r[k] for r in rows],dtype=np.int32) for k in integer})
    chosen,summary=selection(pack,ids)
    summary.update(seed=seed,scientific_assignment_problems=2*len(rows),reverse_audit_problems=2*len(rows),forbidden_edge_audit_problems=int(pack['audit_forbidden_edge_problems'].sum()),seconds=time.perf_counter()-t,scipy_version=scipy.__version__)
    return pack,chosen,summary
