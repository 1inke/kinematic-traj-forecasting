"""Independent saved P0 arithmetic, literal filters and all-window oracle review.

CPU only. No held GT, Torch, training or model forward. Same assistant, not an
external review. This is an explicit source-locked completion extension.
"""
import csv
import json
import sys
import time
from pathlib import Path
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code'))
from io_contract import json_new, json_read, npz_new, load_npz, sha, checked_seal, verify_manifest, code_digest
from action_geometry import space, enumerate_actions
from evaluate import evaluate_population
from p0_driver import guard,resource,emit,independent_six


def independent_cluster(path,good):
    # SciPy's condensed physical-pair order, independently from production's
    # square distance matrix / squareform conversion.
    left,right=np.triu_indices(20,1)
    delta=np.sqrt(np.sum((path[left]-path[right])**2,axis=-1))
    d=np.maximum(delta.mean(1)/.10,delta[:,-1]/.20)
    cl=fcluster(linkage(d,method='complete'),t=1.,criterion='distance')
    return len(np.unique(cl)),len(np.unique(cl[good]))


def check_quality(x,gt,paths,saved,groups):
    # Whole arrays, all physical slots and each group's pooled derivatives.
    d=np.sqrt(np.sum((paths-gt[:,None])**2,axis=-1))
    ade,fde=d.mean(2),d[:,:,-1]
    js=np.maximum(ade/.2,fde/.4); good=(ade<=.2)&(fde<=.4)
    for name,value in [('ADE',ade),('FDE',fde),('J',js),('good',good),('covered',good.any(1))]:
        if not np.array_equal(value,saved[name]): raise AssertionError('quality '+name)
    for i,p in enumerate(paths):
        dist,eff=independent_cluster(p,good[i])
        if dist!=saved['distinct'][i] or eff!=saved['effective_good'][i]: raise AssertionError('full-K20 cluster')
    for row in groups:
        if row['N']==0: continue
        mask=np.ones(len(paths),bool) if row['group']=='overall' else x['group']==['stable','transition','change'].index(row['group'])
        p=paths[mask]
        zeros=np.zeros(p.shape[:-2]+(1,2)); velocity=(np.concatenate((zeros,p),axis=-2)[...,1:,:]-np.concatenate((zeros,p),axis=-2)[...,:-1,:])/.4
        acceleration=(velocity[...,1:,:]-velocity[...,:-1,:])/.4
        first=velocity[...,1:,:]-velocity[...,:-1,:]
        jerk=(first[...,1:,:]-first[...,:-1,:])/(.4**2)
        v,a,j=[np.sqrt(np.sum(q*q,axis=-1)) for q in (velocity,acceleration,jerk)]
        metrics=np.array([np.quantile(v,.95),a.mean(),np.quantile(a,.95),j.mean(),np.quantile(j,.95)])
        original=np.array(list(row['motion'].values()))
        if not np.array_equal(metrics,original): raise AssertionError('group pooled motion')


def main():
    out=ROOT/'p0_completion_review';out.mkdir(exist_ok=False)
    json_new(out/'source_extension.json',dict(status='EXPLICIT_COMPLETION_REVIEW_EXTENSION',
        original_source_lock_sha=sha(ROOT/'source_manifest.json'),code_digest=code_digest(),
        files={str(Path(__file__).resolve()):sha(__file__)}))
    verify_manifest(ROOT/'source_manifest.json');g=guard();records=[];counts=[];allfiles={};decisions={}
    for branch in ('END','TIME'):
        s=space(branch);decisions[branch]={}
        for pop in ('train','val'):
            begin=time.perf_counter();x=load_npz(ROOT/f'prepared/X_{pop}.npz');gt=load_npz(ROOT/f'prepared/GT_{pop}.npz')['gt']
            registry=json_read(ROOT/'P0_STAGE_REGISTRY.json')
            maskroot=ROOT/registry[branch][pop]['masks'];labelroot=ROOT/registry[branch][pop]['labels']
            checked_seal(maskroot);checked_seal(labelroot)
            n=len(gt);native=np.asarray(x['native'],float)
            paths={name:np.empty_like(native) for name in ('geometry','motion','safe')}
            choices={name:np.empty((n,20),np.int32) for name in paths}
            arrays={name:np.empty((n,20),float) for name in ('safe_count','geometry_count','motion_count','active','exact_ties','near_ties','second_gap','diameter')}
            failures=np.zeros(7,np.int64);actions=0;maxdiam=0.;window=0
            for mf in sorted(maskroot.glob('[0-9]*.npz')):
                m=load_npz(mf);y=load_npz(labelroot/mf.name)
                if not np.array_equal(m['sample_id'],y['sample_id']) or not np.array_equal(m['sample_id'],x['sample_id'][window:window+len(m['sample_id'])]): raise AssertionError('shard IDs')
                for row in range(len(m['sample_id'])):
                    i=window+row;g.check()
                    for k,p in enumerate(native[i]):
                        pp,legal=enumerate_actions(p,s)
                        metric=independent_six(x['past'][i],pp);base=independent_six(x['past'][i],p)
                        fail=np.c_[~(metric<=base*1.01),~np.isfinite(pp).all(axis=(1,2))]
                        fail[np.all(pp==p,axis=(1,2))]=False
                        safe=legal & ~fail.any(1)
                        if not np.array_equal(legal,m['legal'][row,k]) or not np.array_equal(fail,m['failed'][row,k]) or not np.array_equal(safe,m['safe'][row,k]): raise AssertionError('all-action literal mask')
                        dist=np.sqrt(np.sum((pp-gt[i])**2,axis=2));js=np.maximum(dist.mean(1)/.2,dist[:,-1]/.4)
                        if not np.array_equal(js,y['J_matrix'][row,k]): raise AssertionError('full J matrix')
                        subsets=dict(geometry=legal,motion=legal & ~fail[:,:5].any(1) & ~fail[:,6],safe=safe)
                        for name,allowed in subsets.items():
                            winner=min(np.flatnonzero(allowed),key=lambda j:(js[j],j!=s.identity,np.sum(s.normalized[j]**2),int(j)))
                            choices[name][i,k]=winner;paths[name][i,k]=pp[winner]
                        win=choices['safe'][i,k];vals=np.sort(js[safe]);best=js[win]
                        equivalent=s.normalized[safe & (js<=best+.01)]
                        diameter=0. if len(equivalent)<2 else float(np.linalg.norm(equivalent[:,None]-equivalent[None,:],axis=-1).max())
                        maxdiam=max(maxdiam,abs(diameter-y['diameter'][row,k]))
                        expected=dict(action_id=win,target=s.normalized[win],best_J=best,identity_J=js[s.identity],unfiltered_J=js[legal].min(),
                            exact_ties=np.count_nonzero(safe & (js==best)),near_ties=np.count_nonzero(safe & (js<=best+1e-6)),
                            second_gap=vals[1]-vals[0] if len(vals)>1 else np.nan,safe_count=safe.sum())
                        for name,val in expected.items():
                            if not np.array_equal(np.asarray(val),np.asarray(y[name][row,k]),equal_nan=True): raise AssertionError('teacher '+name)
                        if abs(diameter-y['diameter'][row,k])>2e-12: raise AssertionError('parameter diameter')
                        for name,val in dict(safe_count=safe.sum(),geometry_count=legal.sum(),motion_count=subsets['motion'].sum(),active=win!=s.identity,
                            exact_ties=expected['exact_ties'],near_ties=expected['near_ties'],second_gap=expected['second_gap'],diameter=diameter).items(): arrays[name][i,k]=val
                        failures+=(fail & legal[:,None]).sum(0);actions+=len(pp)
                window+=len(m['sample_id'])
                if window%256==0:emit('P0_ALL_ACTIONS_REVIEW',branch=branch,population=pop,windows=window)
            if window!=n:raise AssertionError('missing population')
            local=out/f'{branch}_{pop}';local.mkdir()
            npz_new(local/'ORACLES.npz',sample_id=x['sample_id'],**{'paths_'+k:v for k,v in paths.items()},**{'actions_'+k:v for k,v in choices.items()},**arrays)
            outcomes={}
            for name,p in paths.items():
                sample,rows=evaluate_population(x,gt,p,g.check)
                check_quality(x,gt,p,sample,rows)
                npz_new(local/(name+'_metrics.npz'),sample_id=x['sample_id'],**sample)
                json_new(local/(name+'_groups.json'),dict(rows=rows))
                for r in rows: records.append(dict(branch=branch,population=pop,port=name,**r))
                change=x['group']==2; rescue=change & ~sample['native_covered'] & sample['covered']
                outcomes[name]=dict(change_N=int(change.sum()),native_good=int(sample['native_covered'][change].sum()),good=int(sample['covered'][change].sum()),
                    rescue=int(rescue.sum()),rescue_tracks=int(len(np.unique(x['track_key'][rescue]))),lost=int((change & sample['native_covered'] & ~sample['covered']).sum()))
            original=json_read(labelroot/'opportunity.json');safe=outcomes['safe']
            if safe['good']!=original['change_safe_good'] or safe['rescue']!=original['change_rescue'] or safe['rescue_tracks']!=original['rescue_tracks'] or safe['lost']!=0:raise AssertionError('opportunity summary')
            gate=bool(safe['rescue']>=int(np.ceil(.02*safe['change_N'])) and safe['rescue_tracks']>=5)
            if gate!=original['opportunity_pass']:raise AssertionError('gate')
            decisions[branch][pop]=dict(stages=outcomes,opportunity_pass=gate,authorized_population_for_gate=pop=='train',
                nonidentity_candidates=int(arrays['active'].sum()),candidate_denominator=n*20,
                nonidentity_windows=int((arrays['active'].any(1)).sum()),exact_ties_over_one=int((arrays['exact_ties']>1).sum()),
                near_ties_over_one=int((arrays['near_ties']>1).sum()),safe_count_mean=float(arrays['safe_count'].mean()),
                failures_among_geometry_legal=failures.tolist(),motion_and_junction_order_is_descriptive_not_causal=True)
            counts.append(dict(branch=branch,population=pop,windows=n,actions=actions,max_diameter_error=maxdiam,seconds=time.perf_counter()-begin))
            emit('P0_POPULATION_REVIEW_PASS',branch=branch,population=pop,outcome=decisions[branch][pop])
    for branch in decisions:
        decisions[branch]['next_stage']='OWN_GPU_RESUME_TINY_PREFLIGHT_REQUIRED' if decisions[branch]['train']['opportunity_pass'] else 'STOP_LOW_SAFE_OPPORTUNITY_IN_THIS_ACTION_CLASS'
    json_new(out/'ALL_GROUPS.json',dict(rows=records))
    scalar_keys=['branch','population','port','group','N','native_good','good','rescue','lost','coverage_gain','minADE','minFDE','minJ','distinct','effective_good','motion_literal_pass','junction_literal_pass']
    with (out/'ALL_GROUPS.csv').open('x',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=scalar_keys,extrasaction='ignore');writer.writeheader();writer.writerows(records)
    json_new(out/'DECISION.json',dict(branches=decisions,held_GT_read=False,no_training=True))
    verify_manifest(ROOT/'source_manifest.json');verify_manifest(out/'source_extension.json')
    json_new(out/'RESOURCE.json',resource(g))
    for p in out.rglob('*'):
        if p.is_file():allfiles[str(p.relative_to(ROOT))]=sha(p)
    json_new(out/'REVIEW.json',dict(status='PASS_INDEPENDENT_SAVED_P0_ALL_ACTIONS_AND_ORACLES',counts=counts,
        all_groups=len(records),source_unchanged=True,source_lock_sha=sha(ROOT/'source_manifest.json'),files=allfiles,
        same_assistant_not_external=True,no_model_or_gradient_replay=True))
    emit('P0_FULL_REVIEW_PASS',decisions={k:v['next_stage'] for k,v in decisions.items()})


if __name__=='__main__':main()
