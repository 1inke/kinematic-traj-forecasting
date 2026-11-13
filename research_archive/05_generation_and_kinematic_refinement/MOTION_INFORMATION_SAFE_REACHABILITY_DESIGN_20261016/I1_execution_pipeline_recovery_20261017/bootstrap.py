import common as c
import numpy as np
N=20000;SEEDS=(42,43,44)
def reduce_stats(values,tracks,ind=False):
    names=np.unique(tracks);out=np.zeros((3,len(names),3));count=np.zeros(len(names))
    if ind:
        for t,name in enumerate(names):
            mask=tracks==name;out[:,t]=values[:,mask].sum(1);count[t]=mask.sum()
    else:
        index=np.searchsorted(names,tracks);np.add.at(count,index,1)
        for s in range(3):np.add.at(out[s],index,values[s])
    return names,out,count
def boot(values,tracks,ind=False,save=None,family=2):
    names,st,count=reduce_stats(values,tracks,ind);gen=np.random.Generator(np.random.PCG64(20261016));sd=[];td=[];draw=[]
    for start in range(0,N,128):
        c.guard();b=min(128,N-start);ss=gen.integers(0,3,(b,3));tt=gen.integers(0,len(names),(b,len(names)));den=count[tt].sum(1)
        x=st[ss[:,:,None],tt[:,None,:]].sum(2)/den[:,None,None];draw.append(np.stack(((x[:,:,0]-x[:,:,1]).mean(1),(x[:,:,0]-x[:,:,2]).mean(1)),1));sd.append(ss);td.append(tt)
    sd=np.concatenate(sd);td=np.concatenate(td);draw=np.concatenate(draw);point=np.stack(((values[:,:,0]-values[:,:,1]).mean(1),(values[:,:,0]-values[:,:,2]).mean(1)),1)
    if save is not None:
        if ind:
            z=c.npz(save);assert np.array_equal(sd,z['modelseed']) and np.array_equal(td,z['tracks']) and np.array_equal(names,z['track_order'])
        else:np.savez_compressed(save,modelseed=sd,tracks=td,track_order=names)
    rows=[]
    for k in range(2):
        good=np.isfinite(draw[:,k]);valid=int(good.sum());x=draw[good,k]
        rows.append(dict(estimate=float(point[:,k].mean()),per_seed=point[:,k],valid_draws=valid,invalid_draws=N-valid,usable=valid>=19000,CI95=np.quantile(x,[.025,.975]) if valid else None,CI_family=np.quantile(x,[.05/(2*family),1-.05/(2*family)]) if valid else None))
    return dict(family=family,contrasts=['H-S','H-CV'],n=len(tracks),tracks=len(names),draws=N,rows=rows,per_seed_MSE=values.mean(1),relative_gain=1-float(values[:,:,0].mean()/values[:,:,1].mean()) if values[:,:,1].mean()!=0 else None,relative_gain_per_seed=1-values[:,:,0].mean(1)/values[:,:,1].mean(1),target_track_order_digest=c.digest(names.tolist()))
def arrays(split,group):
    values=[];ids=None;tracks=None
    for seed in SEEDS:
        pair=[]
        for arm in ('S','H'):
            z=c.npz(c.H/f'runs/{arm}/seed{seed}/{split}_aux.npz');mask=np.ones(len(z['groups']),bool) if group=='overall' else z['groups']=={'stable':0,'transition':1,'change':2}[group]
            if ids is None:ids=z['sample_id'][mask];tracks=z['track'][mask]
            assert np.array_equal(ids,z['sample_id'][mask]) and np.array_equal(tracks,z['track'][mask]);pair.append(z)
        assert np.array_equal(pair[0]['CV_MSE'],pair[1]['CV_MSE'])
        values.append(np.column_stack((pair[1]['MSE'][mask],pair[0]['MSE'][mask],pair[0]['CV_MSE'][mask])))
    return np.stack(values),tracks
def compute(ind=False):
    records=[]
    for split in ('FIT','outer','inner','original_CAL'):
        for group in c.GROUPS:
            val,tr=arrays(split,group);row=boot(val,tr,ind,c.H/'bootstrap_draws.npz' if split=='outer' and group=='change' else None,family=2 if split=='outer' and group=='change' else 1);row.update(split=split,group=group,exploratory=not(split=='outer' and group=='change'));records.append(row)
    main=next(x for x in records if x['split']=='outer' and x['group']=='change');r=main['rows'];per=np.asarray(main['per_seed_MSE']);protected=[]
    for g in ('overall','stable','transition'):
        val,_=arrays('outer',g);means=val.mean(1);protected.append(dict(group=g,H=means[:,0],S=means[:,1],PASS=bool(np.all(means[:,0]<=1.01*means[:,1]))))
    signal=bool(np.all(per[:,0]<per[:,1]) and main['relative_gain'] is not None and main['relative_gain']>=.05 and all(x['usable'] and x['CI_family'][1]<0 for x in r) and all(x['PASS'] for x in protected))
    return dict(primary=main,exploratory=records,protected_outer=protected,INFORMATION_READOUT_SIGNAL=signal,CV_is_one_shared_deterministic_reference=True,no_baseline_upgrade=True,F0_dependency_unavailable_G1_not_triggered=True)
