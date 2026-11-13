import common as c
import numpy as np
SEEDS=(42,43,44);N=20000

def cell_arrays(cell,split,group='change'):
    out=[]
    for seed in SEEDS:
        z=c.npz(c.H/f'runs/{cell}/seed{seed}/{split}_aux.npz');p=c.npz(c.H/f'runs/{cell}/seed{seed}/{split}_paths.npz');sel=np.ones(len(p['groups']),bool) if group=='overall' else p['groups']=={'stable':0,'transition':1,'change':2}[group]
        good=z['native_good'][sel];new=z['good'][sel].any(1);v=np.stack((new.astype(float),(good&~new).astype(float),(~good&new).astype(float),z['fixed_deltaJ'][sel],np.ones(sel.sum()),good.astype(float),(~good).astype(float),np.ones(sel.sum())),1)
        absvalues=np.stack((z['ADE'][sel].min(1),z['FDE'][sel].min(1),z['J'][sel].min(1)),1)
        native=z['native_vectors4'][sel]
        out.append((p['sample_id'][sel],p['track'][sel],v,absvalues-native[:,1:],new.astype(float)-native[:,0]))
    for row in out[1:]:assert np.array_equal(row[0],out[0][0]) and np.array_equal(row[1],out[0][1])
    return out

def intervals(draw,point,family):
    rows=[]
    for k in range(draw.shape[1]):
        vals=draw[:,k];vals=vals[np.isfinite(vals)];valid=len(vals)
        rows.append(dict(estimate=float(point[k]),valid_draws=valid,invalid_draws=N-valid,usable=valid>=19000,CI95=np.quantile(vals,[.025,.975]) if valid else None,CI_family=np.quantile(vals,[.05/(2*family),1-.05/(2*family)]) if valid else None))
    return rows

def exploratory(ind=False):
    records=[]
    for split in ('FIT','outer','inner','original_CAL'):
        for group in c.GROUPS:
            arrays={cell:cell_arrays(cell,split,group) for cell in ('A','B')};tracks=np.unique(arrays['A'][0][1]);index=np.searchsorted(tracks,arrays['A'][0][1]);stats={}
            for cell in arrays:
                stats[cell]=np.zeros((3,len(tracks),8))
                for s,(_,tr,values,_,_) in enumerate(arrays[cell]):
                    if ind:
                        for t,name in enumerate(tracks):stats[cell][s,t]=values[tr==name].sum(0)
                    else:np.add.at(stats[cell][s],index,values)
            gen=np.random.Generator(np.random.PCG64(20261016));results=[]
            for start in range(0,N,128):
                c.guard();b=min(128,N-start);ss=gen.integers(0,3,(b,3));tt=gen.integers(0,len(tracks),(b,len(tracks)));row=[]
                for cell in ('A','B'):
                    sums=stats[cell][ss[:,:,None],tt[:,None,:]].sum(2);ratio=np.divide(sums[:,:,:4],sums[:,:,4:],out=np.full(sums[:,:,:4].shape,np.nan),where=sums[:,:,4:]!=0);row.append(ratio.mean(1))
                results.append(row[1]-row[0])
            point=[]
            for cell in ('A','B'):
                sums=stats[cell].sum(1);point.append(np.divide(sums[:,:4],sums[:,4:],out=np.full((3,4),np.nan),where=sums[:,4:]!=0).mean(0))
            records.append(dict(split=split,group=group,n=len(arrays['A'][0][0]),tracks=len(tracks),contrast='B-A',rows=intervals(np.concatenate(results),point[1]-point[0],1),exploratory_only=True,not_primary_family_replacement=True))
    return records

def compute(ind=False):
    cells={x:cell_arrays(x,'outer') for x in ('A','B')};tracks=np.unique(cells['A'][0][1]);trackindex=np.searchsorted(tracks,cells['A'][0][1]);seedrows=[];sufficient={};absolute={};coverage={}
    for cell,arr in cells.items():
        sufficient[cell]=np.zeros((3,len(tracks),8));absolute[cell]=np.zeros((3,len(tracks),4));coverage[cell]=np.zeros((3,len(tracks),2))
        for s,(_,tr,v,ab,co) in enumerate(arr):
            assert np.array_equal(tr,cells['A'][0][1])
            if ind:
                for t,name in enumerate(tracks):
                    mask=tr==name;sufficient[cell][s,t]=v[mask].sum(0);absolute[cell][s,t]=np.r_[ab[mask].sum(0),mask.sum()];coverage[cell][s,t]=[co[mask].sum(),mask.sum()]
            else:
                np.add.at(sufficient[cell][s],trackindex,v);np.add.at(absolute[cell][s],trackindex,np.column_stack((ab,np.ones(len(v)))));np.add.at(coverage[cell][s],trackindex,np.column_stack((co,np.ones(len(v)))))
    gen=np.random.Generator(np.random.PCG64(20261016));sd=np.empty((N,3),np.int16);td=np.empty((N,len(tracks)),np.int32)
    for first in range(0,N,128):
        b=min(128,N-first);sd[first:first+b]=gen.integers(0,3,(b,3));td[first:first+b]=gen.integers(0,len(tracks),(b,len(tracks)))
    drawpath=c.H/'bootstrap_draws.npz'
    if ind:
        saved=c.npz(drawpath);assert np.array_equal(saved['modelseed'],sd) and np.array_equal(saved['tracks'],td)
    else:np.savez_compressed(drawpath,modelseed=sd,tracks=td,track_order=tracks)
    estimates={};draws={};absdraw={};covdraw={}
    for cell in cells:
        points=sufficient[cell].sum(1);rates=np.divide(points[:,:4],points[:,4:],out=np.full((3,4),np.nan),where=points[:,4:]!=0);estimates[cell]=rates
        vals=np.empty((N,4));av=np.empty((N,3));cv=np.empty(N)
        for first in range(0,N,128):
            c.guard();ss=sd[first:first+128];tt=td[first:first+128];g=sufficient[cell][ss[:,:,None],tt[:,None,:]].sum(2);den=g[:,:,4:];rr=np.divide(g[:,:,:4],den,out=np.full(g[:,:,:4].shape,np.nan),where=den!=0);vals[first:first+len(ss)]=rr.mean(1)
            ag=absolute[cell][ss[:,:,None],tt[:,None,:]].sum(2);av[first:first+len(ss)]=(ag[:,:,:3]/ag[:,:,3:]).mean(1)
            cg=coverage[cell][ss[:,:,None],tt[:,None,:]].sum(2);cv[first:first+len(ss)]=(cg[:,:,0]/cg[:,:,1]).mean(1)
        draws[cell]=vals;absdraw[cell]=av;covdraw[cell]=cv
    diff=draws['B']-draws['A'];point=(estimates['B']-estimates['A']).mean(0);per=estimates['B']-estimates['A'];main=dict(status='PASS',family=24,comparisons_not_run=['C-A','D-A','D-B','D-C','D-C-B+A'],contrast='B-A',metrics=['coverage','lost_rate','rescue_rate','fixed_j0_actual_deltaJ'],per_seed=per,rows=intervals(diff,point,24),draws=N,valid_threshold=19000,ID_digest=c.digest(cells['A'][0][0].tolist()),track_order_digest=c.digest(tracks.tolist()),draw_SHA=c.sha(drawpath),not_confirmatory=True)
    abpoint=np.stack([row[3].mean(0) for row in cells['B']]).mean(0);copoint=np.mean([row[4].mean() for row in cells['B']]);absresult=dict(family9=intervals(absdraw['B'],abpoint,9),coverage_family3=intervals(covdraw['B'][:,None],np.array([copoint]),3),other_cells_not_run=['C','D'])
    r=main['rows'];retention=bool(point[1]<=-.03 and np.all(per[:,1]<0) and r[1]['usable'] and r[1]['CI_family'][1]<0 and point[2]>=0 and r[2]['usable'] and r[2]['CI_family'][0]>=-.02 and np.all(per[:,0]>=0))
    return dict(primary=main,absolute=absresult,exploratory95=exploratory(ind),decision=dict(RETENTION_SIGNAL=retention,T2_independently_allowed_after_review=True,T3_allowed=False,major_claimed=False,no_automatic_coefficients_sweep=True))
