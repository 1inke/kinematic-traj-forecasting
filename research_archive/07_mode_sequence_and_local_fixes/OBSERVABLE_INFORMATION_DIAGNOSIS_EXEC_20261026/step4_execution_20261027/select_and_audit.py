"""All matching inputs saved first; second algorithm audits every actual pair."""
from core import *
from matching import compute,selection,matrix,action_map
from independent import hungarian
def independent_shortlist(q,qi,lib):
    if not q['known'][qi]:return [],[]
    by={};start=int(q['start'][qi]);seq=q['sequence'][qi];track=q['track_key'][qi]
    for i in range(len(lib['sample_id'])):
        if not lib['known'][i] or lib['sequence'][i]!=seq or lib['track_key'][i]==track:continue
        pstart=int(lib['start'][i])
        if not (pstart+19<start or start+19<pstart):continue
        delta=lib['past'][i]-q['past'][qi];dh=float(np.sqrt(sum(float(np.dot(r,r)) for r in delta)/8)/.1)
        key=(dh,int(lib['sample_id'][i]));t=str(lib['track_key'][i])
        if t not in by or key<by[t][0]:by[t]=(key,i)
    rows=sorted(by.values(),key=lambda x:(x[0][0],str(lib['track_key'][x[1]]),x[0][1]))[:32]
    return [int(x[1]) for x in rows],[float(x[0][0]) for x in rows]
def independent_matrix(a,b):
    rows=[]
    for p in a:
        diff=p-b;point=np.sqrt(np.sum(diff*diff,axis=-1))
        rows.append(np.maximum(np.sum(point,axis=-1)/12/.1,np.sqrt(np.sum(diff[:,-1]*diff[:,-1],axis=-1))/.2))
    return np.asarray(rows)
def audit(pack,chosen,summary,seed,g,limit_references=None):
    identities={s:load_npz(HERE/f'prepared/identity_{s}.npz') for s in ('opt','hold')}
    pools={s:load_npz(HERE/f'prepared/pools_seed{seed}_{s}.npz') for s in ('opt','hold')};count=0;maxerr=0.
    for kind,split in ((0,'opt'),(1,'hold')):
        refs=identities[split]['reference_rows'];refs=refs if limit_references is None else refs[:limit_references]
        for pos,qi in enumerate(refs):
            rows,dh=independent_shortlist(identities[split],int(qi),identities['opt'])
            ix=np.flatnonzero((pack['kind']==kind)&(pack['reference_position']==pos))
            assert pack['library_row'][ix].tolist()==rows
            for i,pr,h in zip(ix,rows,dh):
                assert int(pack['query_row'][i])==int(qi) and abs(pack['dH'][i]-h)<=2e-12
                costs=[];rcosts=[]
                for family in ('native','kin'):
                    c=independent_matrix(pools[split][family][qi],pools['opt'][family][pr]);m,total=hungarian(c)
                    original=pack[family+'_map'][i];reverse=pack['reverse_'+family+'_map'][i]
                    assert np.array_equal(np.sort(original),np.arange(20)) and np.array_equal(np.sort(reverse),np.arange(20))
                    actual=float(c[np.arange(20),original].sum());reversed_cost=float(c[np.arange(20),reverse].sum())
                    assert abs(actual-total)<=2e-12*max(1,abs(total)) and abs(reversed_cost-total)<=2e-12*max(1,abs(total))
                    mean=actual/20;revmean=reversed_cost/20;maxerr=max(maxerr,abs(mean-pack[family+'_mean_cost'][i]))
                    assert abs(mean-pack[family+'_mean_cost'][i])<=2e-12
                    costs.append(pack[family+'_mean_cost'][i]);rcosts.append(pack['reverse_'+family+'_mean_cost'][i])
                dp=sum(costs)/2;rdp=sum(rcosts)/2
                assert abs(dp-pack['dP'][i])<=2e-12 and abs(max(float(pack['dH'][i]),dp)-pack['d'][i])<=2e-12
                assert abs(rdp-pack['reverse_dP'][i])<=2e-12 and abs(max(float(pack['dH'][i]),rdp)-pack['reverse_d'][i])<=2e-12
                action_map(pack['native_map'][i],pack['kin_map'][i]);action_map(pack['reverse_native_map'][i],pack['reverse_kin_map'][i]);count+=1
            if pos%8==0:g.check();mark('STEP4_INDEPENDENT_MATCH_TIME_AUDIT',seed=seed,population=split,position=pos+1,pairs_checked=count)
    assert count==len(pack['d'])
    # Literal independent threshold/supported-row audit on verified saved costs.
    for mode,dkey in (('primary','d'),('reverse','reverse_d')):
        nearest=[]
        for pos in range(len(identities['opt']['reference_rows'])):
            ix=np.flatnonzero((pack['kind']==0)&(pack['reference_position']==pos))
            if len(ix):nearest.append(min(float(pack[dkey][i]) for i in ix))
        if nearest:
            s=sorted(nearest);at=(len(s)-1)*.25;low=int(np.floor(at));high=int(np.ceil(at));tau=s[low]+(s[high]-s[low])*(at-low)
            assert abs(tau-summary[mode]['tau'])<=2e-12
        else:assert summary[mode]['tau'] is None
        tau=summary[mode]['tau'];actual=[]
        for pos in range(len(identities['hold']['reference_rows'])):
            ix=[int(i) for i in np.flatnonzero((pack['kind']==1)&(pack['reference_position']==pos)) if tau is not None and pack[dkey][i]<=tau]
            ix.sort(key=lambda i:(float(pack[dkey][i]),str(identities['opt']['track_key'][pack['library_row'][i]]),int(identities['opt']['sample_id'][pack['library_row'][i]])))
            actual+=ix[:5]
        assert actual==chosen[mode].tolist()
    return dict(status='PASS',actual_pairs=count,science_assignment_problems=2*count,all_time_source_shortlist_independent=True,
        all_Hungarian_costs_second_primal_dual_algorithm=True,max_cost_error=maxerr,all_action_maps_bijections=True,all_threshold_selection_arithmetic=True,
        uniqueness_certificate='production forbidden-edge alternatives; conservative numerical ambiguity, no exact-math uniqueness claim')
def main(seed):
    g=guard();c=dict(source=source(),seed=seed,prepared_SHA=sha256(HERE/'prepared/done.json'))
    out=HERE/f'selections/seed{seed}';out.mkdir(parents=True,exist_ok=True)
    if (out/'done.json').exists():completed(out,c);return
    if list(out.iterdir()):raise FileExistsError('Preserve partial matching evidence')
    pack,chosen,summary=compute(seed,g);save_npz(out/'MATCHES.npz',**pack);save_npz(out/'SELECTED.npz',**chosen);write_json(out/'SELECTION.json',summary)
    reread=load_npz(out/'MATCHES.npz');rrsel=load_npz(out/'SELECTED.npz')
    review=audit(reread,rrsel,summary,seed,g);write_json(out/'REVIEW.json',review);resource(out/'RESOURCE.json',g);finish(out,c)
    print('PASS geometry and independent matching',seed,len(pack['d']),flush=True)
if __name__=='__main__':main(int(sys.argv[1]))
