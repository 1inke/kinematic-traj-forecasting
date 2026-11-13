"""D0: source inheritance, train-only feature stats, all set-utility contracts."""
import time
import numpy as np
import contracts_discrete as c
from features_discrete import geometry8,fit8,transform,independent_geometry8
from targets_discrete import targets,label_data,independent_targets
from decode_discrete import select,apply,selection_metrics
from build_teacher import samepath_quality

def full_action_review(d,check=lambda:None):
    maxerr=0.;n=c.ACTIONS
    for i in range(len(d['gt'])):
        check();native=d['x']['native'][i];gt=d['gt'][i];oldq=float(d['nj'][i].min())
        for k in range(20):
            # Materialize only 48 temporary K20 sets (one window/slot), never N960K20.
            paths=np.repeat(native[None],48,0)
            paths[:,k]=native[k]+(np.arange(1,13)/12)[None,:,None]*n.params[c.NONIDENTITY,None,:]
            jj=samepath_quality(paths,gt[None,None])[2];q=jj.min(1)
            val=(q<=1).astype(float)-float(oldq<=1)+.5*(np.clip(oldq,0,2)-np.clip(q,0,2))
            err=float(np.max(np.abs(val-d['value'][i,k])));maxerr=max(maxerr,err)
            assert err<=2e-12
            assert np.array_equal(np.sign(val),np.sign(d['value'][i,k])),'utility sign mismatch'
            assert np.array_equal(q<=1,d['q_edit'][i,k]<=1)
    return dict(status='PASS_ALL_SINGLE_EDIT_K20_ACTIONS',windows=len(d['gt']),actions=len(d['gt'])*960,maxerror=maxerr)

def synthetic():
    nj=np.full((6,20),2.5);nj[0,0]=.8;nj[1,:2]=.8;nj[2]=.8;nj[3]=1.;nj[4]=0.;nj[5]=3.
    ja=np.broadcast_to(nj[:,:,None],(6,20,48)).copy()
    ja[0,0]=1.1;ja[1,0]=1.1;ja[3,0]=1.;ja[4,0]=0.;ja[5,0]=2.8
    v,q=targets(nj,ja);vv,qq=independent_targets(nj,ja)
    assert np.array_equal(v,vv) and np.array_equal(q,qq)
    assert (v[0,0]<-1).all() and (v[1,0]==0).all() and (v[2]==0).all() and (q[3]<=1).all()
    valid=np.zeros_like(v,bool);score=np.ones_like(v,dtype=np.float32);assert not select(score,valid).any()
    valid[:]=True;score[:]=0;assert not select(score,valid).any();score[0,1,5]=1;score[0,0,8]=1
    assert select(score,valid)[0]==9
    return dict(status='PASS_UNIQUE_MULTIPLE_DUPLICATE_BOUNDARY_ZERO_NOLEGAL_NOOP_FIRSTTIE',invalid_not_negative=True)

def main():
    start=time.perf_counter();c.inputs();g=c.resource();g.check(force=True)
    out=c.ROOT/'prepared';out.mkdir(exist_ok=False)
    train=c.base_data('train');val=c.base_data('val')
    assert len(train['gt'])==3103 and len(val['gt'])==993
    stats=fit8(train['x']['native'],31);ind=geometry8(train['x']['native']).reshape(-1,8)
    mu=ind.mean(0);sd=np.sqrt(((ind-mu)**2).mean(0));sd[sd<1e-6]=1
    alt=fit8(train['x']['native'],256)
    maxstat=max(float(np.max(np.abs(stats[k]-r[k]))) for k in ('mean','std') for r in (dict(mean=mu,std=sd),alt))
    assert maxstat<=1e-11
    c.npz_new(out/'statistics8.npz',**stats,train_sample_id=train['x']['sample_id'])
    ix=c.load_npz(c.OLD/'tiny_v2/indices.npz');tiny=c.subset(train,ix['indices'])
    assert np.array_equal(tiny['x']['sample_id'],ix['sample_id']) and len(np.unique(tiny['x']['track_key']))==97 and int((tiny['nj']<=1).any(1).sum())==69
    assert np.max(np.abs(geometry8(tiny['x']['native'])-independent_geometry8(tiny['x']['native'])))<=2e-12
    for name,d in [('train',train),('val',val),('tiny',tiny)]:
        label_data(d);zH=transform(d,stats,'H');zG=transform(d,stats,'G')
        assert np.array_equal(zH[...,16:],zG[...,16:]) and not zG[...,:16].any()
        assert np.array_equal(zH[...,:40],d['z'])
        c.npz_new(out/f'{name}.npz',zH=zH,zG=zG,valid=d['valid'],value=d['value'],q_edit=d['q_edit'],sample_id=d['x']['sample_id'])
    full=full_action_review(tiny,g.check);v,q=independent_targets(tiny['nj'],tiny['labels']['J_matrix'][:,:,c.NONIDENTITY])
    assert np.array_equal(v,tiny['value']) and np.array_equal(q,tiny['q_edit'])
    old=c.load_npz(c.FEAS/'O0_witness/witness.npz');assert np.array_equal(old['sample_id'],tiny['x']['sample_id'])
    newpath=tiny['x']['native'].copy();ng=(tiny['nj']<=1).any(1)
    for i in range(128):
        if ng[i]:continue
        candidates=[(float(tiny['labels']['J_matrix'][i,k,a]),k,a) for k in range(20) for a in range(49) if tiny['safe'][i,k,a]]
        j,k,a=min(candidates)
        if j<=1:newpath[i,k]+=(np.arange(1,13)/12)[:,None]*c.ACTIONS.params[a]
    assert np.array_equal(newpath,old['paths'])
    # New utility oracle is separate, no assumption its pooled gates match O0.
    bestchoice=select(tiny['value'],tiny['valid']);op,slot,act=apply(tiny['x']['native'],tiny['x']['past'],bestchoice,tiny['valid'])
    sample,rows=c.reference.full_metrics(op,tiny);native_rows=c.reference.groups(tiny['x']['native'],tiny)[1]
    oracle_gates=[]
    for row,native in zip(rows,native_rows):
        limit=1. if row['group']=='change' else 1.01
        oracle_gates.append(dict(group=row['group'],errors=all(row[k]<=native[k]*limit for k in ('minADE','minFDE')),motion=all(row[k]<=native[k]*1.01 for k in c.reference.NAMES)))
    assert int((sample['covered']&~ng).sum())==21
    c.npz_new(out/'new_utility_oracle.npz',paths=op,choice=bestchoice,slot=slot,action=act,**sample)
    c.write(out/'NEW_UTILITY_ORACLE.json',dict(status='DIAGNOSTIC_ONLY_SEPARATE_FROM_O0',groups=rows,gates=oracle_gates,selected=selection_metrics(tiny['value'],bestchoice,tiny)))
    # GT perturbation changes utility but cannot change observation-only construction.
    changed=c.subset(tiny,np.arange(8));changed['gt']=changed['gt']+100
    assert np.array_equal(transform(changed,stats,'H'),transform(c.subset(tiny,np.arange(8)),stats,'H'))
    old40=c.load_npz(c.P0/'prepared/statistics.npz');assert np.array_equal(old40['train_sample_id'],train['x']['sample_id'])
    track_train=set(train['x']['track_key']);track_val=set(val['x']['track_key'])
    intersections=sorted(track_train&track_val)
    # Do not impose an invented split condition: disclose actual original contract.
    c.write(c.ROOT/'SPLIT_AUDIT.json',dict(train_windows=3103,val_windows=993,train_change=int((train['x']['group']==2).sum()),val_change=int((val['x']['group']==2).sum()),
        exact_ID_intersection=sorted(set(map(int,train['x']['sample_id']))&set(map(int,val['x']['sample_id']))),sequence_target_track_intersection=intersections,
        train_tracks=len(track_train),val_tracks=len(track_val),original_split_unchanged=True,
        scene_counts={pop:{str(k):int((d['x']['scene']==k).sum()) for k in np.unique(d['x']['scene'])} for pop,d in [('train',train),('val',val)]},held_evaluated=False))
    c.write(out/'done.json',dict(status='PREPARED_D0_SOURCE_COMPATIBLE',files={p.name:c.sha(p) for p in out.iterdir() if p.is_file()}))
    c.write(c.ROOT/'D0_RESULT.json',dict(status='PASS_D0_LABELS_EXECUTION',full_actions=full,synthetic=synthetic(),
        old_O0_paths_bitwise=True,new_oracle_good=int(sample['covered'].sum()),new_oracle_groups_not_prefilled=oracle_gates,
        original40_stats_sha=c.sha(c.P0/'prepared/statistics.npz'),new8_stats_max_two_pass_chunk_error=maxstat,
        tiny_original_IDs97tracks_Native69=True,hold_inner_GT_not_parsed=True,seconds=time.perf_counter()-start))
    c.verify(c.json_read(c.ROOT/'INPUT_SOURCE_LOCK.json')['files'],g.check);g.check(force=True)
    c.write(c.ROOT/'RESOURCE_D0.json',dict(estimate=g.estimate,observations=g.observations))
    print('D0 COMPLETE full122880 single-edit labels / oldO0 exact / stats trainonly',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        import datetime,traceback
        c.write(c.ROOT/('FAILURE_D0_'+datetime.datetime.now().strftime('%H%M%S')+'.json'),dict(traceback=traceback.format_exc(),scientific_updates=0))
        raise
