"""D0/D1 joint saved-data scan; no model, Torch, new trajectories or solver."""
from source_contract import *

def quality(p,y):
    dist=np.linalg.norm(p.astype(float)-y[:,None].astype(float),axis=-1);a=dist.mean(-1);f=dist[:,:,-1];j=np.maximum(a/.2,f/.4);q=a+.5*f
    return dict(ADE=a,FDE=f,J=j,Q=q,J_slot=j.argmin(1),Q_slot=q.argmin(1),good=(a<=.2)&(f<=.4),Jmin=j.min(1),minADE=a.min(1),minFDE=f.min(1))

def physics(p):
    v=np.diff(p,axis=-2,prepend=np.zeros_like(p[...,:1,:]))/.4;a=np.diff(v,axis=-2)/.4;j=np.diff(a,axis=-2)/.4
    return v,a,j

def temporal_transport(m,truth):
    msum=m.sum(-1);tsum=truth.sum(-1);both=(msum==0)&(tsum==0);valid=(msum>0)&(tsum>0);out=np.full(msum.shape,np.nan);out[both]=0
    out[valid]=.4*np.abs(np.cumsum(m[valid]/msum[valid,None],axis=-1)[...,:-1]-np.cumsum(truth[valid]/tsum[valid,None],axis=-1)[...,:-1]).sum(-1)
    return out

def path_measures(p,y,vobs):
    p=p.astype(float);y=y.astype(float);pe=p[:,-1];ge=y[:,-1];gn=np.linalg.norm(ge,axis=-1);pn=np.linalg.norm(pe,axis=-1);delta=pe-ge;unit=np.divide(ge,gn[:,None],out=np.zeros_like(ge),where=gn[:,None]>0)
    defined=gn>=.2;angdefined=defined&(pn>=.2)
    along=np.full(len(p),np.nan);cross=along.copy();angle=along.copy()
    along[defined]=(delta[defined]*unit[defined]).sum(-1);cross[defined]=unit[defined,0]*delta[defined,1]-unit[defined,1]*delta[defined,0]
    angle[angdefined]=np.arccos(np.clip((pe[angdefined]*ge[angdefined]).sum(-1)/(pn[angdefined]*gn[angdefined]),-1,1))
    error=p-y;distance=np.linalg.norm(error,axis=-1);shape=np.linalg.norm(error-np.arange(1,13)[None,:,None]/12*delta[:,None],axis=-1).mean(-1)
    v,a,j=physics(p);gv,ga,gj=physics(y);an=np.linalg.norm(a,axis=-1);gan=np.linalg.norm(ga,axis=-1)
    firstnorm=np.linalg.norm(v[:,0],axis=-1);obsnorm=np.linalg.norm(vobs,axis=-1);vm=(firstnorm>=.05)&(obsnorm>=.05);va=np.full(len(p),np.nan)
    va[vm]=np.arccos(np.clip((v[vm,0]*vobs[vm]).sum(-1)/(firstnorm[vm]*obsnorm[vm]),-1,1))
    return dict(along_error=along,lateral_error=cross,abs_lateral_error=np.abs(cross),endpoint_angle_radians=angle,amplitude_error=pn-gn,shape_error=shape,early_ADE=distance[:,:4].mean(-1),middle_ADE=distance[:,4:8].mean(-1),late_ADE=distance[:,8:].mean(-1),E_v=np.linalg.norm(v-gv,axis=-1).mean(-1),E_a=np.linalg.norm(a-ga,axis=-1).mean(-1),E_j=np.linalg.norm(j-gj,axis=-1).mean(-1),accel_amplitude_error=np.abs(an-gan).mean(-1),accel_by_time=an,GT_accel_by_time=gan,accel_vector_error_by_time=np.linalg.norm(a-ga,axis=-1),W_time=temporal_transport(an,gan),velocity_join=np.linalg.norm(v[:,0]-vobs,axis=-1),velocity_join_angle_radians=va)

def compute(d):
    y=d['gt'];p=d['paths'];native=d['native'];a=quality(p,y);n=quality(native,y);ag=a['good'].any(1);ng=n['good'].any(1)
    #0 retained,1 rescue,2 lost,3 neither. Labels are posthoc, never training inputs.
    label=np.where(ng,np.where(ag,0,2),np.where(ag,1,3)).astype(np.int8)
    out=dict(sample_id=d['sample_id'],groups=d['groups'],track=d['track'],sequence=d['sequence'],scene=d['scene'],partition=label)
    for pref,q in [('student',a),('native',n)]:
        for k,v in q.items():out[pref+'_'+k]=v
    vobs=(d['past'][:,-1].astype(float)-d['past'][:,-2].astype(float))/.4;out['observed_speed']=np.linalg.norm(vobs,axis=-1)
    for pref,paths,q in [('student',p,a),('native',native,n)]:
        for choice in ['J','Q']:
            slot=q[choice+'_slot'];chosen=paths[np.arange(len(p)),slot]
            fields=path_measures(chosen,y,vobs)
            out.update({pref+'_'+choice+'_'+key:value for key,value in fields.items()})
        # All20 counterpart is descriptive; average per-candidate GT errors, not likelihood.
        parts={key:[] for key in ['E_v','E_a','E_j','accel_amplitude_error','accel_by_time','W_time']}
        for i in range(0,len(p),256):
            block=paths[i:i+256].reshape(-1,12,2);truth=np.repeat(y[i:i+256],20,axis=0);vo=np.repeat(vobs[i:i+256],20,axis=0);bm=path_measures(block,truth,vo)
            for key in parts:
                val=bm[key].reshape(len(paths[i:i+256]),20,-1)
                finite=np.isfinite(val);den=finite.sum(1);avg=np.divide(np.where(finite,val,0).sum(1),den,out=np.full(den.shape,np.nan),where=den>0)
                parts[key].append(avg[:,0] if avg.shape[1]==1 else avg)
        out.update({pref+'_ALL20_'+key:np.concatenate(v) for key,v in parts.items()})
    for k in ['minADE','minFDE','Jmin']:out['delta_'+k]=a[k]-n[k]
    return out

def summary(out):
    rows=[]
    numerical=[k for k,v in out.items() if k.startswith(('student_J_','native_J_','student_Q_','native_Q_','student_ALL20_','native_ALL20_'))]
    for group,mask in group_masks(out['groups']):
        n=int(mask.sum());row=dict(group=group,n=n,tracks=len(np.unique(out['track'][mask])),parts={},delta={},positive_tail={})
        for part,label in [('retained',0),('rescue',1),('lost',2),('neither',3)]:
            take=mask&(out['partition']==label);r=dict(n=int(take.sum()),tracks=len(np.unique(out['track'][take])),quality={},contribution={},diagnostics={})
            for key in ['minADE','minFDE','Jmin']:
                delta=out['delta_'+key];r['quality'][key]=dict(student=stats(out['student_'+key],take),native=stats(out['native_'+key],take),delta=stats(delta,take));r['contribution'][key]=float(delta[take].sum()/n)
            for key in numerical:r['diagnostics'][key]=stats(out[key],take)
            row['parts'][part]=r
        for key in ['minADE','minFDE','Jmin']:
            delta=out['delta_'+key][mask];positive=np.maximum(delta,0);negative=np.minimum(delta,0)
            row['delta'][key]=float(delta.mean());assert abs(sum(v['contribution'][key] for v in row['parts'].values())-row['delta'][key])<=1e-10
            total=float(positive.sum());ordered=np.sort(positive)[::-1];row['positive_tail'][key]=dict(positive_sum=total,negative_sum=float(negative.sum()),top_fraction={str(f):dict(n=int(np.ceil(f*n)),share=float(ordered[:int(np.ceil(f*n))].sum()/total) if total else None) for f in [.01,.05,.10]})
        row['motion_alignment']={k:stats(out[k],mask) for k in numerical};rows.append(row)
    strata=[]
    speed=out['observed_speed'];bins=[(0,.1),(.1,.5),(.5,1),(1,np.inf)]
    takes=[('speed',f'[{lo},{hi})',(speed>=lo)&(speed<hi)) for lo,hi in bins]+[('sequence',str(s),out['sequence']==s) for s in np.unique(out['sequence'])]
    for kind,label,mask in takes:
        strata.append(dict(kind=kind,label=label,n=int(mask.sum()),tracks=len(np.unique(out['track'][mask])),delta={k:stats(out['delta_'+k],mask) for k in ['minADE','minFDE','Jmin']}))
    return dict(groups=rows,strata=strata,undefined_policy='NA with explicit defined count; exact zero-acceleration branches; no epsilon or row removal')

def main():
    from independent_diagnostics import review_job
    t=time.perf_counter();guard=Guard(E_DIAG);guard.check('startup');digest=verify();records=[];audits=[]
    for s in SEEDS:
        for split in ['FIT','inner']:
            stage('D0_D1_SAVED_SCAN',status='RUNNING',seed=s,split=split,done=len(records));guard.check();d=job(s,split);out=compute(d);summ=summary(out)
            dest=HERE/f'D0_D1/seed{s}_{split}';dest.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest/'per_window.npz',**out);write_json(dest/'SUMMARY.json',summ)
            review=review_job(d,out,summ);assert review['status']=='PASS',review;write_json(dest/'AUDIT.json',review)
            # Reuse original full GT-free cluster/all-K20 motion audit only after its file binding is checked.
            prior_audit=read_json(OLD/f'runs/A_seed{s}/audit_{split}.json');assert prior_audit['status']=='PASS'
            for key,h in prior_audit['inputs_sha'].items():assert sha256(ROOT/key)==h
            write_json(dest/'REGISTERED_SET_METRICS.json',dict(status='SOURCE_BOUND_REUSE',metrics=read_json(OLD/f'runs/A_seed{s}/metrics_{split}.json'),prior_audit_sha=sha256(OLD/f'runs/A_seed{s}/audit_{split}.json'),clusters_not_recomputed=True))
            outputs={str(p.relative_to(HERE)):sha256(p) for p in dest.iterdir() if p.is_file()};write_json(dest/'done.json',dict(source_digest=digest,outputs=outputs))
            records.append(dict(seed=s,split=split,summary=summ,outputs=outputs));audits.append(review);del d,out;guard.check()
    inn=[next(g for g in r['summary']['groups'] if g['group']=='change') for r in records if r['split']=='inner'];positive=all(any(g['parts']['neither']['contribution'][k]>0 for k in ['minADE','minFDE']) for g in inn)
    largest=any(np.mean([g['parts']['neither']['contribution'][k] for g in inn])>=max(np.mean([g['parts'][part]['contribution'][k] for g in inn]) for part in ['retained','rescue','lost','neither']) and np.mean([g['parts']['neither']['contribution'][k] for g in inn])>0 for k in ['minADE','minFDE'])
    allow=bool(positive and largest);ea=[[g['motion_alignment'][pref+'_J_E_a']['mean'] for pref in ['student','native']] for g in inn]
    motion_language='GT_SELECTED_MOTION_ALIGNMENT_IMPROVED_REGISTERED_SET_GATE_STILL_FAILS' if all(a<b for a,b in ea) else 'GT_SELECTED_MOTION_ALIGNMENT_WORSE' if all(a>b for a,b in ea) else 'MIXED_GT_SELECTED_MOTION_ALIGNMENT'
    assert digest==verify();guard.check();write_json(HERE/'D0_D1/RESOURCE.json',dict(status='PASS',estimate_bytes=E_DIAG,observations=guard.observations))
    write_json(HERE/'D0_D1/RESULT.json',dict(status='PASS_DIAGNOSTIC_ONLY_TAIL_AND_MOTION',jobs=6,source_digest=digest,records=records,independent_audits=audits,elapsed_seconds=time.perf_counter()-t,no_torch_or_model=True))
    write_json(HERE/'D0_D1/DECISION.json',dict(status='ALLOW_T1_ACTUAL_PREFLIGHT' if allow else 'NO_SUPPORT_FOR_RELATIVE_REGRESSION_TEST',allow_T1_preflight=allow,all_seed_neither_positive=positive,mean_neither_largest_positive=largest,GT_selected_accel_vector_error=ea,D1_interpretation=motion_language,does_not_change_motion_gate=True,new_formal_trainings=0))
    stage('D0_D1_COMPLETE',status='COMPLETED',active_worker=False,done=6,allow_T1_preflight=allow,elapsed_seconds=time.perf_counter()-t)

if __name__=='__main__':main()
