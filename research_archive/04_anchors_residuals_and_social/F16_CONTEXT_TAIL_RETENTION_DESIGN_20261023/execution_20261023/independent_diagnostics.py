"""Second implementation of diagnostic arrays and summary arithmetic, no production imports."""
import numpy as np

def independent_path(p,y,obs):
    p=np.asarray(p,float);y=np.asarray(y,float);n=len(p);v=np.empty_like(p);gv=np.empty_like(y)
    for t in range(12):
        v[:,t]=(p[:,t]-(p[:,t-1] if t else 0))/.4
        gv[:,t]=(y[:,t]-(y[:,t-1] if t else 0))/.4
    a=(v[:,1:]-v[:,:-1])/.4;ga=(gv[:,1:]-gv[:,:-1])/.4;j=(a[:,1:]-a[:,:-1])/.4;gj=(ga[:,1:]-ga[:,:-1])/.4
    mag=lambda z:np.sqrt(np.sum(z*z,axis=-1))
    gtend=y[:,11];end=p[:,11];gn=np.linalg.norm(gtend,axis=1);pn=np.linalg.norm(end,axis=1);enderr=end-gtend
    along=np.full(n,np.nan);lateral=np.full(n,np.nan);angle=np.full(n,np.nan);take=gn>=.2
    unit=np.zeros_like(gtend);np.divide(gtend,gn[:,None],out=unit,where=gn[:,None]>0)
    along[take]=np.sum(enderr[take]*unit[take],axis=1);lateral[take]=unit[take,0]*enderr[take,1]-unit[take,1]*enderr[take,0]
    valid=take&(pn>=.2);angle[valid]=np.arccos(np.clip(np.sum(end[valid]*gtend[valid],axis=1)/(pn[valid]*gn[valid]),-1,1))
    ds=mag(p-y);shape=[]
    for t in range(12):shape.append(mag(p[:,t]-y[:,t]-((t+1)/12)*enderr))
    am=mag(a);gam=mag(ga);total=am.sum(1);g_total=gam.sum(1);transport=np.full(n,np.nan);zero=(total==0)&(g_total==0);transport[zero]=0
    ok=(total>0)&(g_total>0);cum1=np.cumsum(am[ok]/total[ok,None],1);cum2=np.cumsum(gam[ok]/g_total[ok,None],1);transport[ok]=np.sum(np.abs(cum1[:,:10]-cum2[:,:10]),axis=1)*.4
    obs_angle=np.full(n,np.nan);vn=np.linalg.norm(v[:,0],axis=1);on=np.linalg.norm(obs,axis=1);good=(vn>=.05)&(on>=.05)
    obs_angle[good]=np.arccos(np.clip(np.sum(v[good,0]*obs[good],axis=1)/(vn[good]*on[good]),-1,1))
    return dict(along_error=along,lateral_error=lateral,abs_lateral_error=abs(lateral),endpoint_angle_radians=angle,amplitude_error=pn-gn,shape_error=np.stack(shape,1).mean(1),early_ADE=ds[:,:4].sum(1)/4,middle_ADE=ds[:,4:8].sum(1)/4,late_ADE=ds[:,8:].sum(1)/4,E_v=mag(v-gv).sum(1)/12,E_a=mag(a-ga).sum(1)/11,E_j=mag(j-gj).sum(1)/10,accel_amplitude_error=abs(am-gam).sum(1)/11,accel_by_time=am,GT_accel_by_time=gam,accel_vector_error_by_time=mag(a-ga),W_time=transport,velocity_join=mag(v[:,0]-obs),velocity_join_angle_radians=obs_angle)

def review_job(data,out,summary):
    maximum=0.;fields=0
    def same(a,b,name):
        nonlocal maximum,fields
        a=np.asarray(a);b=np.asarray(b);assert a.shape==b.shape,name
        assert np.array_equal(np.isnan(a),np.isnan(b)),('UNDEFINED_MASK',name)
        valid=np.isfinite(a)&np.isfinite(b);assert not (np.isinf(a)|np.isinf(b)).any(),name
        error=float(np.max(np.abs(a[valid].astype(float)-b[valid].astype(float)))) if valid.any() else 0.;maximum=max(maximum,error);fields+=1
        assert error<=1e-10,(name,error)
    y=data['gt'].astype(float);n=len(y);obs=(data['past'][:,-1].astype(float)-data['past'][:,-2].astype(float))/.4;native_good=None;student_good=None
    for pref,paths in [('student',data['paths']),('native',data['native'])]:
        e=np.sqrt(np.sum((paths.astype(float)-y[:,None])**2,axis=-1));a=np.sum(e,2)/12;f=e[:,:,11];j=np.maximum(a/.2,f/.4);q=a+.5*f;g=(a<=.2)&(f<=.4)
        reference=dict(ADE=a,FDE=f,J=j,Q=q,good=g,J_slot=np.argmin(j,1),Q_slot=np.argmin(q,1),minADE=np.min(a,1),minFDE=np.min(f,1),Jmin=np.min(j,1))
        for key,value in reference.items():same(out[pref+'_'+key],value,pref+'_'+key)
        if pref=='native':native_good=g.any(1)
        else:student_good=g.any(1)
        for which in ['J','Q']:
            slots=reference[which+'_slot'];chosen=paths[np.arange(n),slots]
            # Bounded independent blocks, all windows, same selected path for all times.
            for i in range(0,n,256):
                m=independent_path(chosen[i:i+256],y[i:i+256],obs[i:i+256])
                for key,value in m.items():same(out[pref+'_'+which+'_'+key][i:i+256],value,pref+'_'+which+'_'+key)
        for i in range(0,n,256):
            count=len(paths[i:i+256]);block=paths[i:i+256].reshape(-1,12,2);truth=np.repeat(y[i:i+256],20,0);vo=np.repeat(obs[i:i+256],20,0);m=independent_path(block,truth,vo)
            for key in ['E_v','E_a','E_j','accel_amplitude_error','accel_by_time','W_time']:
                vals=m[key].reshape(count,20,-1);defined=np.isfinite(vals);den=defined.sum(1);mean=np.divide(np.where(defined,vals,0).sum(1),den,out=np.full(den.shape,np.nan),where=den>0)
                if mean.shape[1]==1:mean=mean[:,0]
                same(out[pref+'_ALL20_'+key][i:i+256],mean,pref+'_ALL20_'+key)
    partition=np.full(n,3,np.int8);partition[native_good&student_good]=0;partition[~native_good&student_good]=1;partition[native_good&~student_good]=2
    assert np.array_equal(partition,out['partition'])
    for row in summary['groups']:
        group=row['group'];mask=np.ones(n,bool) if group=='overall' else data['groups']==['stable','transition','change'].index(group)
        assert row['n']==int(mask.sum()) and sum(v['n'] for v in row['parts'].values())==int(mask.sum())
        for metric in ['minADE','minFDE','Jmin']:
            delta=out['student_'+metric]-out['native_'+metric];contributions=[]
            for label,key in enumerate(['retained','rescue','lost','neither']):
                take=mask&(partition==label);part=row['parts'][key];assert part['n']==int(take.sum());assert part['tracks']==len(np.unique(data['track'][take]))
                contributions.append(float(delta[take].sum()/mask.sum()));same(part['contribution'][metric],contributions[-1],group+key+metric)
                for source,arr in [('student',out['student_'+metric]),('native',out['native_'+metric]),('delta',delta)]:
                    if take.any():
                        for stat,value in [('mean',arr[take].mean()),('median',np.percentile(arr[take],50)),('p90',np.percentile(arr[take],90)),('p95',np.percentile(arr[take],95))]:same(part['quality'][metric][source][stat],value,group+key+metric+stat)
            same(sum(contributions),delta[mask].mean(),group+metric+'SUM')
            dd=delta[mask];pos=dd[dd>0];total=float(pos.sum());tail=row['positive_tail'][metric];same(tail['positive_sum'],total,group+metric+'POS');same(tail['negative_sum'],float(dd[dd<0].sum()),group+metric+'NEG')
            for frac in [.01,.05,.1]:
                count=int(np.ceil(frac*mask.sum()));expected=float(np.sort(np.maximum(dd,0))[-count:].sum()/total) if total else None;value=tail['top_fraction'][str(frac)]['share']
                if expected is None:assert value is None
                else:same(value,expected,group+metric+'TAIL'+str(frac))
    return dict(status='PASS',max_abs_difference=maximum,tolerance=1e-10,compared_fields_blocks=fields,all_window_slots_and_geometry_and_motion=True,summary_contributions_tails_partitions_recomputed=True,independent_code=True,no_model=True)
