"""Full second saved-array implementation for nine tiny models and frozen utility gates."""
import time
import traceback
import numpy as np
from scipy.cluster.hierarchy import linkage,fcluster
from scipy.spatial.distance import pdist
import common as c
from objectives import numpy_loss_and_grad
from train import checkpoint_load_trusted
from preflight_checks import trees_equal

def eq(a,b,tol=2e-12):
    a,b=np.asarray(a),np.asarray(b)
    if a.dtype.kind in 'biuUS' or b.dtype.kind in 'biuUS':assert np.array_equal(a,b)
    else:assert np.allclose(a,b,atol=tol,rtol=0,equal_nan=True),float(np.max(np.abs(a-b)))

def quality(p,gt):
    d=np.sqrt(((p-gt[:,None])**2).sum(-1));a=d.mean(-1);f=d[...,-1];j=np.maximum(a/.2,f/.4);return a,f,j,(a<=.2)&(f<=.4)

def physics(p,past):
    p=p.astype(float);past=past.astype(float)
    v=(p-np.concatenate((np.zeros(p.shape[:-2]+(1,2)),p[...,:-1,:]),axis=-2))/.4
    acc=np.diff(v,axis=-2)/.4;jerk=np.diff(v,n=2,axis=-2)/(.4**2)
    vn,an,jn=[np.sqrt((vv*vv).sum(-1)) for vv in (v,acc,jerk)]
    jun=np.sqrt(((p[:,:,0]/.4-((past[:,-1]-past[:,-2])/.4)[:,None])**2).sum(-1))/.4
    cand=np.stack((np.quantile(vn,.95,axis=-1),an.mean(-1),np.quantile(an,.95,axis=-1),jn.mean(-1),np.quantile(jn,.95,axis=-1),jun),-1)
    return cand,(vn,an,jn,jun)

def main():
    started=time.perf_counter();g=c.resource();g.check(force=True)
    manifest=c.json_read(c.ROOT/'source_manifest.json');c.verify(manifest['files'],g.check)
    full=c.data();ix=c.load_npz(c.OLD/'tiny_v2/indices.npz')['indices'];d=c.subset(full,ix)
    N=d['x']['native'];past=d['x']['past'];gt=d['gt'];na,nf,nj,ng=quality(N,gt);nc=ng.any(1);supports=ng.sum(1)
    st=np.where(ng,np.where(supports[:,None]==1,0,1),np.where(supports[:,None]>0,2,3));active=np.any(d['target']!=0,axis=-1)
    strnames=('unique_support','multiple_support','badslot_in_good_window','bad_window')
    nativephysics,nphys=physics(N,past);action=c.space('END');rows=[];strata=[];events=[];summaries=[];initials={};orders={}
    oldevents=c.load_npz(c.ATTR/'derived_ports.npz');oldcv=oldevents['good'].any(2);old15=set(oldevents['sample_id'][oldcv[0]!=oldcv[3]].tolist())
    max_scalar=0.;max_double=0.;models=[]
    for seed in (42,43,44):
        for arm in ('M','R','RS'):
            g.check();folder=c.ROOT/'tiny'/f'{arm}_{seed}';done=c.json_read(folder/'done.json')
            for rel,s in done['files'].items():assert c.sha(folder/rel)==s
            con=done['contract'];assert con['source_manifest_sha']==c.sha(c.ROOT/'source_manifest.json')
            assert con['source_code']==manifest['code_sha'] and all(con[k]==v for k,v in c.binding(d).items())
            state=checkpoint_load_trusted(folder/'checkpoint.pt');initial=checkpoint_load_trusted(folder/'initial_checkpoint.pt')
            assert state['updates']==256 and state['epoch']==64 and state['cursor']==0 and len(state['history'])==64 and len(state['batch_order'])==256
            assert sum(v.numel() for v in state['model'].values())==4770
            assert all(__import__('torch').isfinite(v).all() for v in state['model'].values())
            if seed not in initials:initials[seed]=state['initial_state'];orders[seed]=state['batch_order']
            else:assert trees_equal(initials[seed],state['initial_state']) and orders[seed]==state['batch_order']
            assert trees_equal(initial['model'],state['initial_state'])
            for v in state['optimizer']['state'].values():assert int(v['step'])==256
            assert all(k in state for k in ('python_rng','numpy_rng','torch_rng','cuda_rng','sampler','diagnostics','aux_binding'))
            trace=c.load_npz(folder/'training_trace.npz');assert trace['prediction'].shape==(256,32,20,2)
            eq(trace['batch_indices'],state['batch_order'])
            # Every2304 pre-update candidate prediction is stored. This is loss arithmetic,
            # not a second model or optimizer/gradient replay.
            for step,(u,ids) in enumerate(zip(trace['prediction'],trace['batch_indices'])):
                if step%64==0:g.check()
                _,_,parts=numpy_loss_and_grad(u,d['target'][ids],N[ids],gt[ids],d['best'][ids],arm)
                float32_mse=((u-d['target'][ids])**2).mean(axis=(1,2)).mean()
                delta=abs(float(float32_mse)-trace['component_mse'][step]);max_scalar=max(max_scalar,delta);assert delta<=2e-7
                for nm in ('retention','rescue'):
                    err=abs(parts[nm]-trace['component_'+nm][step]);max_double=max(max_double,err);assert err<=2e-12
            epochloss=trace['component_mse'].copy()
            if arm in ('R','RS'):epochloss+=trace['component_retention']
            if arm=='RS':epochloss+=trace['component_rescue']
            for epoch in range(64):eq(epochloss[epoch*4:epoch*4+4].mean(),state['history'][epoch]['loss'])
            z=c.load_npz(folder/'final.npz');eq(z['sample_id'],d['x']['sample_id']);u=z['continuous_param'].astype(float)
            eq(z['continuous_paths'],N+.4*(np.arange(1,13)/12)[None,None,:,None]*u[:,:,None])
            raw=np.empty_like(N)
            for i in range(128):
                for k in range(20):
                    dd=((action.normalized-u[i,k])**2).sum(1);best=dd.min();picked=next(int(a) for a in action.priority if dd[a]==best)
                    assert picked==z['raw_action_id'][i,k]
                    raw[i,k]=N[i,k] if picked==action.identity else N[i,k]+(np.arange(1,13)/12)[:,None]*action.params[picked]
            eq(raw,z['raw_paths']);cmetric,pp=physics(raw,past)
            failed=np.concatenate((~(cmetric<=nativephysics*1.01),np.zeros((128,20,1),bool)),-1)
            identity=np.all(raw==N,axis=(-2,-1));failed[identity]=False
            eq(failed,z['rejection_reason']);eq(np.where(failed.any(-1)[...,None,None],N,raw),z['guarded_paths'])
            eq(np.where(failed.any(-1),action.identity,z['raw_action_id']),z['executed_action_id'])
            metrics=c.json_read(folder/'METRICS.json');bygroup={}
            for port,p in [('C',z['continuous_paths']),('Q',raw),('S',z['guarded_paths'])]:
                aa,ff,jj,good=quality(p,gt);covered=good.any(1)
                for key,val in [('ADE',aa),('FDE',ff),('J',jj),('good',good),('covered',covered)]:eq(val,z[port+'_'+key])
                for i in range(128):
                    ds=np.stack([pdist(p[i,:,t]) for t in range(12)],1)
                    lab=fcluster(linkage(np.maximum(ds.mean(1)/.1,ds[:,11]/.2),method='complete'),1,criterion='distance')
                    assert len(np.unique(lab))==z[port+'_distinct'][i] and len(np.unique(lab[good[i]]))==z[port+'_effective_good'][i]
                    winner=int(jj[i].argmin());ee=np.sqrt(((p[i,winner]-gt[i])**2).sum(-1))
                    assert winner==z[port+'_winner'][i]
                    for key,sl in [('early_ADE',slice(0,4)),('middle_ADE',slice(4,8)),('late_ADE',slice(8,12))]:eq(ee[sl].mean(),z[port+'_'+key][i])
                    events.append(dict(arm=arm,seed=seed,port=port,sample_id=int(d['x']['sample_id'][i]),track=str(d['x']['track_key'][i]),group=int(d['x']['group'][i]),
                        native_good=bool(nc[i]),good=bool(covered[i]),rescue=bool(~nc[i]&covered[i]),lost=bool(nc[i]&~covered[i]),retained=bool(nc[i]&covered[i]),
                        old15_event_case=int(d['x']['sample_id'][i]) in old15,stage=''.join(map(str,[int(nc[i]),int(z['C_covered'][i]),int(z['Q_covered'][i]),int(z['S_covered'][i])])) ))
                pm,ph=physics(p,past)
                for group,mask in [('overall',np.ones(128,bool))]+[(nm,d['x']['group']==i) for i,nm in enumerate(('stable','transition','change'))]:
                    b,cc=nc[mask],covered[mask]
                    motion=lambda vals:np.array([np.quantile(vals[0][mask],.95),vals[1][mask].mean(),np.quantile(vals[1][mask],.95),vals[2][mask].mean(),np.quantile(vals[2][mask],.95)])
                    mv,nv=motion(ph),motion(nphys);junction=ph[3][mask];jn=nphys[3][mask]
                    row=dict(arm=arm,seed=seed,port=port,group=group,N=int(mask.sum()),native_good=int(b.sum()),good=int(cc.sum()),rescue=int((~b&cc).sum()),lost=int((b&~cc).sum()),retained=int((b&cc).sum()),
                        lost_fraction=(b&~cc).sum()/b.sum() if b.any() else None,minADE=aa[mask].min(1).mean(),minFDE=ff[mask].min(1).mean(),minJ=jj[mask].min(1).mean(),
                        native_minADE=na[mask].min(1).mean(),native_minFDE=nf[mask].min(1).mean(),native_minJ=nj[mask].min(1).mean(),
                        motion_literal_pass=bool((mv<=nv*1.01).all()),junction_literal_pass=bool(junction.mean()<=jn.mean()*1.01 and np.quantile(junction,.95)<=np.quantile(jn,.95)*1.01),
                        junction_mean=junction.mean(),junction_p95=np.quantile(junction,.95),native_junction_mean=jn.mean(),native_junction_p95=np.quantile(jn,.95))
                    row.update({nm:float(v) for nm,v in zip(('speed_p95','accel_mean','accel_p95','jerk_mean','jerk_p95'),mv)})
                    row.update({'native_'+nm:float(v) for nm,v in zip(('speed_p95','accel_mean','accel_p95','jerk_mean','jerk_p95'),nv)})
                    oldrow=next(r for r in metrics['groups'] if r['port']==port and r['group']==group)
                    for key in ('N','good','rescue','lost','retained','lost_fraction','minADE','minFDE','minJ','junction_mean','junction_p95'):
                        if row[key] is not None:eq(row[key],oldrow[key])
                    assert row['motion_literal_pass']==oldrow['motion_literal_pass'] and row['junction_literal_pass']==oldrow['junction_literal_pass']
                    rows.append(row)
                    if port=='S':bygroup[group]=row
            allerr=((u-d['target'].astype(float))**2).mean(-1);ej=nj-z['S_J']
            for group,gm in [('overall',np.ones(128,bool))]+[(nm,d['x']['group']==i) for i,nm in enumerate(('stable','transition','change'))]:
                for s,name in enumerate(strnames):
                    for label,lm in [('all',np.ones_like(active)),('active',active),('identity',~active)]:
                        mask=(st==s)&gm[:,None]&lm;ws=mask.any(1)
                        strata.append(dict(arm=arm,seed=seed,group=group,stratum=name,teacher_type=label,candidates=int(mask.sum()),windows=int(ws.sum()),tracks=len(np.unique(d['x']['track_key'][ws])),
                            parameter_MSE=allerr[mask].mean() if mask.any() else None,deltaJ=ej[mask].mean() if mask.any() else None,
                            native_support_lost=int((mask&ng&~z['S_good']).sum()),status='DESCRIPTIVE' if mask.any() else 'NA_EMPTY'))
            summaries.append(dict(arm=arm,seed=seed,**{k:metrics[k] for k in metrics if k!='groups'},S_groups=bygroup))
            models.append(dict(arm=arm,seed=seed,all_checkpoint_contract=True,all256_loss_components=True))
            print('SAVED REVIEW',arm,seed,'good/rescue/lost',bygroup['overall']['good'],bygroup['overall']['rescue'],bygroup['overall']['lost'],flush=True)
    c.csv_new(c.ROOT/'ALL_GROUPS.csv',rows);c.csv_new(c.ROOT/'SUPPORT_STRATA.csv',strata);c.csv_new(c.ROOT/'WINDOW_EVENTS.csv',events)
    lookup={(s['arm'],s['seed']):s for s in summaries};gates=[]
    for seed in (42,43,44):
        m=lookup['M',seed]['S_groups']['overall'];r=lookup['R',seed]['S_groups']['overall'];s=lookup['RS',seed]['S_groups']['overall'];rs=lookup['RS',seed]['S_groups']
        errors=True;physicsgate=True
        for group,row in rs.items():
            limit=1. if group=='change' else 1.01
            errors &= row['minADE']<=row['native_minADE']*limit and row['minFDE']<=row['native_minFDE']*limit
            physicsgate &= row['motion_literal_pass'] and row['junction_literal_pass']
        gates.append(dict(seed=seed,R_lost_le_M=r['lost']<=m['lost'],R_lost_strict=r['lost']<m['lost'],RS_rescue_ge_R=s['rescue']>=r['rescue'],RS_rescue_strict=s['rescue']>r['rescue'],
            RS_lost_le_M=s['lost']<=m['lost'],RS_rescue_ge_M=s['rescue']>=m['rescue'],RS_coverage_gt_native=s['good']>s['native_good'],RS_zero_lost=s['lost']==0,
            RS_protected_errors=bool(errors),RS_all_motion_junction=bool(physicsgate)))
    checks=dict(R_lost_eachseed=all(x['R_lost_le_M'] for x in gates),R_atleast1_strict=any(x['R_lost_strict'] for x in gates),
        RS_rescue_ge_R_eachseed=all(x['RS_rescue_ge_R'] for x in gates),RS_atleast1_strict_rescue=any(x['RS_rescue_strict'] for x in gates),
        RS_relative_M_and_nontrivial=all(x['RS_lost_le_M'] and x['RS_rescue_ge_M'] and x['RS_coverage_gt_native'] for x in gates),
        RS_zero_lost_eachseed=all(x['RS_zero_lost'] for x in gates),RS_all_protected_errors=all(x['RS_protected_errors'] for x in gates),
        RS_all_motion_junction=all(x['RS_all_motion_junction'] for x in gates),all_actual_preflight_source_audit=True)
    passed=all(checks.values())
    c.write(c.ROOT/'TINY_DECISION.json',dict(status='PASS_TINY_ALLOW_F_OWN_ADMISSION' if passed else 'STOP_TINY_UTILITY_NO_FORMAL',
        checks=checks,perseed=gates,models_completed=9,updates=2304,formal_models=0,
        primary_family6='NOT_RUN_IF_TINY_STOP',old90_gate_unchanged=True,no_select_R_if_RS_fails=True))
    out=c.ROOT/'tiny_review';out.mkdir(exist_ok=False)
    c.verify(manifest['files'],g.check);g.check(force=True)
    c.write(out/'REVIEW.json',dict(status='PASS_ALL_NINE_TINY_SAVED_ARITHMETIC',models=models,candidate_ports=9*3*128*20,
        window_ports=9*3*128,all2304_preupdate_losses=True,float32_MSE_max_backend_error=max_scalar,double_R_S_max_error=max_double,
        independent_allquality_clusters_motion=True,allcheckpoint_RNG_history_contract=True,source_unchanged=True,
        saved_arithmetic_not_all_model_gradient_replay=True,same_assistant_not_external=True,elapsed_seconds=time.perf_counter()-started))
    c.write(out/'RESOURCE.json',dict(policy='warning_first_remaining_v3',estimate_bytes=g.estimate,observations=g.observations))
    c.write(c.ROOT/'TINY_ANALYSIS_DATA.json',dict(summaries=summaries,decision=c.json_read(c.ROOT/'TINY_DECISION.json')))
    print('TINY UTILITY DECISION',passed,checks,flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        c.write(c.ROOT/('FAILURE_TINY_REVIEW_'+time.strftime('%H%M%S')+'.json'),dict(traceback=traceback.format_exc(),code_sha=c.code_sha()));raise
