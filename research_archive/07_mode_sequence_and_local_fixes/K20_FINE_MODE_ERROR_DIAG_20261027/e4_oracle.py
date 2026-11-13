from common import *
from geometry_oracles import *
def main():
    g=guard(True);rows=[];overlaps=[];motionrows=[]
    for split in SPLITS:
        x=xload(split);gt=yload(split)['gt'];p=x['native'];n=len(p)
        base=quality(p,gt);bm=motion(p);basegood=base[3].any(1);metrics={}
        delta=gt[:,None,-1]-p[:,:,-1];norm=np.linalg.norm(delta,axis=-1)
        ts=time_search(p,gt,progress=True)
        save(HERE/f'E4_oracle/{split}/time_parameters.npz',sample_id=x['sample_id'],**ts)
        ports={}
        for name in ('Native','END_unbounded','END_b02','END_b04','TIME_grid'):
            if name=='Native':out=p;feasible=np.ones((n,20),bool)
            elif name=='TIME_grid':
                tmp,qq,c=time_from_rates(p.reshape(-1,12,2),ts['rate_a'].ravel(),ts['rate_b'].ravel(),ts['time_feasible'].ravel());out=tmp.reshape(p.shape);feasible=ts['time_feasible']
                assert np.array_equal(out[:,:,-1],p[:,:,-1])
            else:
                bound={'END_unbounded':np.inf,'END_b02':.2,'END_b04':.4}[name];feasible=norm<=bound
                out=p+np.arange(1,13)[None,None,:,None]/12*delta[:,:,None];out[:,:,-1]=gt[:,None,-1];out=np.where(feasible[...,None,None],out,p)
            a,f,j,good=quality(out,gt);covered=good.any(1);m=motion(out)
            if name=='TIME_grid':
                assert np.array_equal(f,base[1]);assert np.max(abs(a-ts['best_ADE']))<=2e-12
                assert not covered[np.all(base[1]>.4,axis=1)].any();assert np.all(covered[basegood])
            if name.startswith('END'):assert np.all(f[feasible]==0)
            save(HERE/f'E4_oracle/{split}/{name}.npz',sample_id=x['sample_id'],ADE=a,FDE=f,J=j,good=good,covered=covered,feasible=feasible,window_motion=m['window'],motion_literal_pass=np.all(m['window']<=bm['window']*1.01,axis=-1),set_level_GT_fallback_covered=covered|basegood)
            ports[name]=covered
            for group,mask in masks(x):
                rr=dict(population=split,port=name,group=group,N=int(mask.sum()),native_good=int(basegood[mask].sum()),oracle_good=int(covered[mask].sum()),oracle_rescue=int((mask&~basegood&covered).sum()),oracle_lost=int((mask&basegood&~covered).sum()),all_FDE_unrescuable=int((mask&np.all(base[1]>.4,axis=1)).sum()),feasible_candidates=int(feasible[mask].sum()),candidate_den=int(mask.sum()*20),window_all20_motion_literal_pass=int(np.all(m['window']<=bm['window']*1.01,axis=(1,2))[mask].sum()))
                rows.append(rr);raw=pooled(m,mask);ref=pooled(bm,mask)
                motionrows.append(dict(population=split,port=name,group=group,N=int(mask.sum()),speed_p95=raw[0],accel_mean=raw[1],accel_p95=raw[2],jerk_mean=raw[3],jerk_p95=raw[4],native_values=ref,ratio=[v/b if b>0 else (1 if v==0 else None) for v,b in zip(raw,ref)],all5_literal_PASS=bool(np.all(np.array(raw)<=np.array(ref)*1.01))))
            g.check()
        for group,mask in masks(x):
            fail=mask&~basegood
            for endname in ('END_unbounded','END_b02','END_b04'):
                e=ports[endname];t=ports['TIME_grid'];overlaps.append(dict(population=split,group=group,endpoint_port=endname,N_failure=int(fail.sum()),only_END=int((fail&e&~t).sum()),only_TIME=int((fail&~e&t).sum()),both=int((fail&e&t).sum()),neither=int((fail&~e&~t).sum())))
        mark('E4_ORACLE_COMPLETE',population=split,no_new_prediction_paths_saved=True)
    table(HERE/'E4_oracle/oracle_coverage.csv',rows);table(HERE/'E4_oracle/oracle_overlap.csv',overlaps);table(HERE/'E4_oracle/motion_audit.csv',motionrows)
    resource(HERE/'E4_oracle/RESOURCE.json',g);finish(HERE/'E4_oracle',fixed_K20=True,GT_oracle_only=True,no_paths_written=True,TIME_finite_predeclared_grid_not_continuous_global_optimum=True)
if __name__=='__main__':main()
