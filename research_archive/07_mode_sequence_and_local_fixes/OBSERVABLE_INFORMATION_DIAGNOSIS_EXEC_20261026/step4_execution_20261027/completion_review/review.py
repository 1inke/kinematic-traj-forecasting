"""New, saved-data-only final supplement. No matching solver, Torch or models."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import core as c
import numpy as np
from collections import Counter

OUT=Path(__file__).resolve().parent
TOL=2e-12

def close(a,b):
    x=np.asarray(a);y=np.asarray(b)
    assert x.shape==y.shape
    assert np.array_equal(np.isnan(x),np.isnan(y))
    assert np.max(abs(x[np.isfinite(x)]-y[np.isfinite(y)]),initial=0)<=TOL

def quality(native,kin,gt):
    # Explicitly construct every final K20, rather than reuse producer minima.
    paths=np.broadcast_to(native,(400,20,12,2)).copy()
    r=np.repeat(np.arange(20),20);l=np.tile(np.arange(20),20)
    paths[np.arange(400),r]=kin[l]
    paths=np.concatenate((native[None],paths))
    d=np.sqrt(((paths-gt)**2).sum(-1));a=d.mean(-1);f=d[:,:,-1]
    j=np.maximum(a/.2,f/.4)
    return np.stack((a.min(1),f.min(1),j.min(1)),-1)

def positives(q,e):
    out=np.zeros(401,bool)
    for a in range(1,401):
        out[a]=bool(e[a] and all(q[a]<=q[0]) and any(q[a]<q[0]) and (q[0,2]>1 or q[a,2]<=1))
    return out

def files_check():
    assert c.verify()==c.source()
    p=c.read_json(c.HERE/'PRESERVATION.json');assert p['status']=='PASS'
    for name,h in p['outputs'].items():assert c.sha256(c.HERE/name)==h,name
    for folder in [*(c.HERE/f'selections/seed{s}' for s in c.SEEDS),*(c.HERE/f'posthoc/seed{s}' for s in c.SEEDS),c.HERE/'statistics']:
        d=c.read_json(folder/'done.json');assert d['contract']['source']==c.source()
        for name,h in d['outputs'].items():assert c.sha256(folder/name)==h,name
    return len(p['outputs'])

def refs(ident,tag,windowtag):
    # Independent fixed-hash population recreation without future/group inputs.
    import hashlib
    def key(prefix,val):return hashlib.sha256((prefix+'|'+val).encode()).digest()
    tracks=sorted(set(ident['track_key']),key=lambda x:(key(tag,str(x)),str(x)))[:128]
    answer=[]
    for tr in tracks:
        ii=np.flatnonzero(ident['track_key']==tr)
        answer.append(min(ii,key=lambda i:(key(windowtag,str(tr)+'|'+str(int(ident['sample_id'][i]))),int(ident['sample_id'][i]))))
    assert np.array_equal(answer,ident['reference_rows'])

def main():
    g=c.guard();count0=files_check();roots=['RESULT.json','RESOURCE.json','ARITHMETIC_AUDIT.json','PRESERVATION.json','ALL_SELECTIONS_LOCK.json','PRODUCTION_SOURCE_LOCK.json','source_manifest.json','worker_state.json','exit_status.json','statistics/SUMMARY.json','statistics/BOOTSTRAP_DRAWS.npz']
    initial={p:c.sha256(c.HERE/p) for p in roots}
    assert c.read_json(c.HERE/'worker_state.json')['status']==c.read_json(c.HERE/'exit_status.json')['status']=='SUCCESS'
    lock=c.read_json(c.HERE/'ALL_SELECTIONS_LOCK.json');assert lock['source']==c.source() and lock['science_assignment_problems']<=49152
    ids={s:c.load_npz(c.HERE/f'prepared/identity_{s}.npz') for s in ('opt','hold')}
    refs(ids['opt'],'ambiguity-calibration-v1','ambiguity-calibration-window-v1');refs(ids['hold'],'ambiguity-query-v1','ambiguity-query-window-v1')
    assert len(set(ids['opt']['track_key'])&set(ids['hold']['track_key']))==0
    summary=c.read_json(c.HERE/'statistics/SUMMARY.json');records={};groups=[];examples={};matching=[];calibration=[];allpairs=0;maxquality=0.;science=0
    rootgroups=c.old_columns(42,'hold',('sample_id','group'))
    assert np.array_equal(rootgroups['sample_id'],ids['hold']['sample_id'])
    for seed in c.SEEDS:
        fold=c.HERE/f'selections/seed{seed}';assert c.sha256(fold/'done.json')==lock['done_SHA'][str(seed)]
        pack=c.load_npz(fold/'MATCHES.npz');selected=c.load_npz(fold/'SELECTED.npz');sel=c.read_json(fold/'SELECTION.json')
        audit=c.read_json(fold/'REVIEW.json');assert audit['status']=='PASS' and audit['max_cost_error']<=TOL
        n=len(pack['d']);science+=2*n
        for k in ('native_map','kin_map','reverse_native_map','reverse_kin_map'):
            assert np.array_equal(np.sort(pack[k],axis=1),np.broadcast_to(np.arange(20),pack[k].shape))
        assert (pack['audit_forbidden_edge_problems']==40).all()
        for i in range(n):
            split='opt' if pack['kind'][i]==0 else 'hold';q=ids[split];p=ids['opt'];qi=int(pack['query_row'][i]);pi=int(pack['library_row'][i])
            assert q['known'][qi] and p['known'][pi] and q['sequence'][qi]==p['sequence'][pi] and q['track_key'][qi]!=p['track_key'][pi]
            assert int(q['start'][qi])+19<int(p['start'][pi]) or int(p['start'][pi])+19<int(q['start'][qi])
        close(pack['dP'],(pack['native_mean_cost']+pack['kin_mean_cost'])/2)
        close(pack['d'],np.maximum(pack['dH'],pack['dP']))
        close(pack['reverse_d'],np.maximum(pack['dH'],pack['reverse_dP']))
        p=c.HERE/f'posthoc/seed{seed}';raw=c.read_json(p/'PAIRS.json');arr=c.load_npz(p/'PAIR_SCALARS.npz');sensitivity=c.read_json(p/'ALIGNMENT_SENSITIVITY.json')
        assert c.read_json(p/'done.json')['contract']['selection_lock_SHA']==c.sha256(c.HERE/'ALL_SELECTIONS_LOCK.json')
        assert c.read_json(p/'REVIEW.json')['status']=='PASS'
        assert len(raw)==len(arr['mode']) and len(sensitivity)==len(selected['primary'])
        pools={s:c.load_npz(c.HERE/f'prepared/pools_seed{seed}_{s}.npz') for s in ('opt','hold')}
        labs={s:c.old_columns(seed,s,('sample_id','quality','B','gt','group')) for s in ('opt','hold')};seen={}
        for split in labs:assert np.array_equal(labs[split]['sample_id'],pools[split]['sample_id'])
        for j,r in enumerate(raw):
            mode=r['mode'];i=int(r['pair_index']);qr=int(r['query_row']);pr=int(r['library_row']);prefix='' if mode=='primary' else 'reverse_'
            assert arr['mode'][j]==mode and int(arr['pair_index'][j])==i
            assert i in selected[mode] and pack['kind'][i]==1 and pack['query_row'][i]==qr and pack['library_row'][i]==pr
            assert r['query_ID']==int(ids['hold']['sample_id'][qr]) and r['library_ID']==int(ids['opt']['sample_id'][pr])
            for split,row in (('hold',qr),('opt',pr)):
                key=(split,row)
                if key not in seen:
                    q=quality(pools[split]['native'][row],pools[split]['kin'][row],labs[split]['gt'][row]);saved=labs[split]['quality'][row]
                    maxquality=max(maxquality,float(np.max(abs(q-saved))));close(q,saved)
                    assert np.array_equal(positives(q,pools[split]['eligible'][row]),labs[split]['B'][row]);seen[key]=True
            nm=pack[prefix+'native_map'][i];km=pack[prefix+'kin_map'][i]
            amap=np.array([0]+[1+20*int(nm[(a-1)//20])+int(km[(a-1)%20]) for a in range(1,401)])
            assert sorted(amap.tolist())==list(range(401))
            e=pools['hold']['eligible'][qr]&pools['opt']['eligible'][pr,amap];assert e[0]
            qb=labs['hold']['B'][qr];pb=labs['opt']['B'][pr];bq=qb&e;bp=pb[amap]&e
            jq=labs['hold']['quality'][qr,:,2];jp=labs['opt']['quality'][pr,amap,2]
            for key,value in dict(domain=e,q_positive=bq,p_positive=bp,qJ=jq,pJ=jp).items():
                if value.dtype==bool:assert np.array_equal(value,arr[key][j])
                else:close(value,arr[key][j])
            inter=int((bq&bp).sum());union=int((bq|bp).sum());both=bool(bq.any() and bp.any())
            event=('COMMON_POSITIVE' if inter else 'CONFLICT') if both else ('BOTH_ORIGINAL_NO_OPPORTUNITY' if not qb.any() and not pb.any() else 'COMMON_DOMAIN_OPPORTUNITY_REMOVAL' if qb.any() and pb.any() else 'ONE_ORIGINAL_NO_OPPORTUNITY')
            nums=dict(domain_count=int(e.sum()),q_original_B=int(qb.sum()),p_original_B=int(pb.sum()),q_common_B=int(bq.sum()),p_common_B=int(bp.sum()),intersection=inter,union=union)
            assert all(r[k]==v for k,v in nums.items());assert r['event']==event and r['both_positive']==both
            assert r['conflict']==(not bool(inter) if both else None)
            if both:close(r['overlap'],inter/union)
            else:assert r['overlap'] is None
            expected=min(max(float(jq[a]-jq[e].min()),float(jp[a]-jp[e].min())) for a in np.flatnonzero(e));close(r['regret'],expected)
            delta=labs['hold']['gt'][qr]-labs['opt']['gt'][pr];close(r['future_mean_distance'],np.linalg.norm(delta,axis=-1).mean());close(r['future_endpoint_distance'],np.linalg.norm(delta[-1]))
            allpairs+=1
        records[seed]=raw
        for mode in ('primary','reverse'):
            nearest=[]
            for qi in range(128):
                xx=np.flatnonzero((pack['kind']==0)&(pack['reference_position']==qi))
                if len(xx):nearest.append(float(pack['d' if mode=='primary' else 'reverse_d'][xx].min()))
            close(nearest,sel[mode]['calibration_nearest']);tau=float(np.quantile(nearest,.25,method='linear'));close(tau,sel[mode]['tau'])
            expected=[]
            for qi in range(128):
                xx=[int(i) for i in np.flatnonzero((pack['kind']==1)&(pack['reference_position']==qi)) if pack['d' if mode=='primary' else 'reverse_d'][i]<=tau]
                xx.sort(key=lambda i:(float(pack['d' if mode=='primary' else 'reverse_d'][i]),str(ids['opt']['track_key'][pack['library_row'][i]]),int(ids['opt']['sample_id'][pack['library_row'][i]])))
                expected+=xx[:5]
            assert np.array_equal(expected,selected[mode])
            reasons=Counter(x['status'] for x in sel[mode]['queries'])
            calibration.append(dict(seed=seed,mode=mode,tau=tau,valid=len(nearest),query_reasons=dict(reasons),selected_pairs=len(expected)))
            for group in (-1,0,1,2):
                rr=[x for x in raw if x['mode']==mode and (group==-1 or x['group']==group)]
                gr=rootgroups['group'][ids['hold']['reference_rows']];total=int((gr==group).sum()) if group!=-1 else len(gr)
                supported=len(set(x['query_ID'] for x in rr));comparable=len(set(x['query_ID'] for x in rr if x['both_positive']))
                groups.append(dict(seed=seed,mode=mode,group='overall' if group==-1 else ('stable','transition','change')[group],query_tracks=total,supported_tracks=supported,comparable_tracks=comparable,pairs=len(rr),both_positive_pairs=sum(x['both_positive'] for x in rr),events=dict(Counter(x['event'] for x in rr))))
            for event in sorted(set(x['event'] for x in raw)):
                candidates=[x for x in raw if x['mode']==mode and x['event']==event]
                if candidates:
                    # Fixed hash illustration, never chosen to strengthen a claim.
                    ex=min(candidates,key=lambda x:c.rank('ambiguity-example-v1',f"{seed}|{mode}|{x['query_ID']}|{x['library_ID']}"));examples[f'{seed}_{mode}_{event}']=ex
        matching.append(dict(seed=seed,pairs=n,scientific_assignments=2*n,reverse_audit_problems=2*n,forbidden_edge_audit_problems=int(pack['audit_forbidden_edge_problems'].sum()),second_algorithm_max_cost_error=audit['max_cost_error'],all_quality_windows=len(seen),all_selected_primary_reverse_pairs=len(raw),all_shortlist_optimality_certification_inherited_worker_REVIEW_SHA=c.sha256(fold/'REVIEW.json')))
        g.check()
    assert science==48702==lock['science_assignment_problems']
    z=c.load_npz(c.HERE/'statistics/BOOTSTRAP_DRAWS.npz');draws=z['track_draws'];rr=ids['hold']['reference_rows'];qids=ids['hold']['sample_id'][rr]
    assert np.array_equal(draws,np.random.default_rng(20261026).integers(128,size=(20000,128),dtype=np.int32))
    assert np.array_equal(z['query_ID'],qids) and np.array_equal(z['group'],rootgroups['group'][rr])
    counts=np.stack([np.bincount(ix,minlength=128) for ix in draws]);boot={}
    for key,r in summary['records'].items():
        mode,subgroup,metric=key.split('_',2);v=np.full((3,128),np.nan)
        for si,seed in enumerate(c.SEEDS):
            for qi,sid in enumerate(qids):
                pp=[x[metric] for x in records[seed] if x['mode']==mode and x['query_ID']==int(sid) and x[metric] is not None]
                if pp and (subgroup=='overall' or z['group'][qi]==2):v[si,qi]=float(np.mean(pp))
        close(v,z['perquery_'+key]);answers=np.full((20000,3),np.nan)
        for si in range(3):
            finite=np.isfinite(v[si]);den=counts@finite.astype(np.int64);num=counts@np.where(finite,v[si],0)
            np.divide(num,den,out=answers[:,si],where=den>0)
            point=float(v[si,finite].mean()) if finite.any() else None
            if point is None:assert r['point_per_seed'][si] is None
            else:close(point,r['point_per_seed'][si])
            ok=np.isfinite(answers[:,si]);assert int(ok.sum())==r['seed_records'][si]['valid_draws'];assert int((~ok).sum())==r['seed_records'][si]['invalid_draws']
            if ok.any():close(np.quantile(answers[ok,si],[.025,.975]),r['seed_records'][si]['CI95'])
        close(answers,z['draws_'+key]);ok=np.isfinite(answers).all(1);combined=np.full(20000,np.nan);combined[ok]=answers[ok].mean(1)
        close(combined,z['combined_'+key]);assert int(ok.sum())==r['combined_valid'] and int((~ok).sum())==r['combined_invalid']
        if ok.any():close(np.quantile(combined[ok],[.025,.975]),r['combined_CI95'])
        boot[key]=dict(valid=int(ok.sum()),invalid=int((~ok).sum()),max_saved_draw_error=float(np.nanmax(abs(answers-z['draws_'+key]))))
        g.check()
    production=c.read_json(c.HERE/'RESOURCE.json');peaks=[];physical=[];commit=[]
    for item in production['records']:
        for obs in item['report']['observations']+[item['report']['last']]:
            assert obs['status']=='PASS';s=obs['snapshot'];assert max(s['peak_rss'],s['peak_private'],s['rss'],s['private'])<=obs['retained_peak_limit'];peaks.append((s['peak_private'],s['peak_rss']));physical.append(s['physical']);commit.append(s['commit'])
    count1=files_check();assert count0==count1 and initial=={p:c.sha256(c.HERE/p) for p in roots}
    assert 'torch' not in sys.modules
    c.write_json(OUT/'ALL_GROUPS.json',groups);c.write_json(OUT/'CALIBRATION_SUPPORT.json',calibration);c.write_json(OUT/'HASH_EXAMPLES.json',examples)
    c.write_json(OUT/'OUTPUT_SHA.json',dict(explicit_root_SHA=initial,original_worker_output_files_checked=count1,supplement_source_SHA=c.sha256(Path(__file__)),no_control_file_immutability_claim=True))
    c.write_json(OUT/'REVIEW.json',dict(status='PASS_SAVED_COMPLETION_SUPPLEMENT',source=c.source(),source_count=3536,all_source_and_worker_output_SHA_initial_final=True,all7_scientific_done_SHA=True,explicit_root_SHA_count=len(initial),all_fixed_query_calibration_hash_and_48702_maps_time_intervals=True,scientific_assignment_problems=science,matching=matching,
        no_new_matching_solver_or_model_replay=True,original_second_Hungarian_optimality_audit_inherited_SHA_verified=True,all_selected_pairs_primary_plus_reverse_checked=allpairs,all401_explicit_finalK20_quality_B_reconstructed=True,max_quality_error=maxquality,all_common_domains_B_sets_J_regret_actual_arrays_recomputed=True,all_actual20000_draws_each16_statistics_recomputed=boot,
        production=dict(retained_private_peak_bytes=max(x[0] for x in peaks),retained_RSS_peak_bytes=max(x[1] for x in peaks),min_available_physical_bytes=min(physical),min_available_commit_bytes=min(commit),all_original_1p3_PASS=True),
        support_conclusion='UNDERPOWERED_34_overall_10_change_per_seed_no_threshold_adjustment',same_assistant_not_external_review=True))
    c.resource(OUT/'RESOURCE.json',g);print('PASS SAVED COMPLETION',science,allpairs,maxquality,flush=True)

if __name__=='__main__':main()
