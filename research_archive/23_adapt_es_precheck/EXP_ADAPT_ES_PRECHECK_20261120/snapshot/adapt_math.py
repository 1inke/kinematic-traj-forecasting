"""Energy score, archived HPD and paired track summaries."""
import numpy as np

from coupling_analysis import TrackBootstrap
from calib_metrics import hpd_mc, log_density

GRID = np.arange(4,15)/10
ANISO = np.arange(5,15)/10
ES_NAMES=("path_weighted","path_equal","endpoint_weighted","endpoint_equal")
METRICS=("minADE","minFDE","J","HPD90","spread")


def probabilities(z):
    if z is None:
        return None
    e=np.exp(z.astype(np.float64)-z.max(1,keepdims=True))
    return e/e.sum(1,keepdims=True)


def errors(paths,y):
    diff=paths-y[:,None]
    d=np.sqrt(np.square(diff).sum(-1))
    ade,fde=d.mean(-1),d[:,:,-1]
    return dict(minADE=ade.min(1),minFDE=fde.min(1),J=((ade<=.2)&(fde<=.4)).any(1),
                spread=np.sqrt(np.square(paths[:,:,-1]-paths[:,:,-1].mean(1,keepdims=True)).sum(-1).mean(1)))


def scatter(path,p):
    """Half of the double sum, accumulating each unordered pair once."""
    index=np.triu_indices(path.shape[1],1)
    chunks=[]
    for i in range(0,len(path),128):
        row=slice(i,i+128)
        dx=path[row,index[0]]-path[row,index[1]]
        squared=np.square(dx).sum(-1)
        a=np.sqrt(squared.sum(-1)); b=np.sqrt(squared[:,:,-1])
        w=p[row,index[0]]*p[row,index[1]]
        k=path.shape[1]
        chunks.append(np.column_stack(((a*w).sum(1),a.sum(1)/(k*k),
                                        (b*w).sum(1),b.sum(1)/(k*k))))
    return np.concatenate(chunks)


def energy(path,y,p,scattering=None):
    d2=np.square(path-y[:,None]).sum(-1)
    dt,de=np.sqrt(d2.sum(-1)),np.sqrt(d2[:,:,-1])
    a=np.column_stack(((dt*p).sum(1),dt.mean(1),(de*p).sum(1),de.mean(1)))
    s=scatter(path,p) if scattering is None else scattering
    return a-s,a,s


def check_energy():
    single=np.array([[[[0.,0.]]]])
    val,_,_=energy(single,np.array([[[3.,4.]]]),np.ones((1,1)))
    np.testing.assert_allclose(val,5.,atol=1e-14,rtol=0)
    pair=np.array([[[[-1.,0.]],[[1.,0.]]]])
    val,a,s=energy(pair,np.zeros((1,1,2)),np.full((1,2),.5))
    np.testing.assert_allclose([val,s,a],[[[.5]*4],[[.5]*4],[[1.]*4]],atol=1e-14,rtol=0)
    rng=np.random.default_rng(20261120)
    sample=rng.normal(size=(8192,20,1,2))
    truth=rng.normal(size=(8192,1,2))
    p=np.full((len(truth),20),.05)
    mean={str(a):float(energy(sample*a,truth,p)[0][:,0].mean()) for a in (.5,1.,2.)}
    assert mean["1.0"]<min(mean["0.5"],mean["2.0"]),mean
    return dict(single=5.,two_atoms=.5,energy_of_normal_forecasts=mean)


def hpd(paths,y,h,draws):
    e=paths[:,:,-1]
    p=np.full(e.shape[:2],1/e.shape[1])
    q=np.square(e-y[:,None,-1]).sum(-1)
    ld=log_density(q,np.log(p),h)
    return hpd_mc(e,p,h,ld,*draws)[0]<=.9


def median_draws(helper,x):
    order=np.argsort(x,kind="stable")
    idx=helper.index[order]
    v=x[order]
    parts=[]
    for lo in range(0,len(helper.draw),128):
        c=np.cumsum(helper.draw[lo:lo+128,idx],axis=1)
        n=c[:,-1].astype(int)
        a=(c<=((n-1)//2)[:,None]).sum(1)
        b=(c<=(n//2)[:,None]).sum(1)
        parts.append(.5*(v[a]+v[b]))
    return np.concatenate(parts)


class Summary:
    def __init__(self,masks,track,keep):
        self.groups={}
        for name,group in masks.items():
            for label,mask in (("all_dev",group),("without_zara02",group&keep)):
                if mask.any():
                    self.groups[label,name]=mask,TrackBootstrap(track[mask])

    def table(self,after,before=None):
        output={}
        for (scope,name),(m,helper) in self.groups.items():
            unit=np.ones(int(m.sum()))
            metrics={key:(v[m].astype(np.float64),unit) for key,v in after.items() if key!="spread"}
            if before is not None:
                metrics.update({f"delta_{k}":(v[m].astype(np.float64)-before[k][m].astype(np.float64),unit)
                                for k,v in after.items() if k!="spread"})
            avg=helper.summarize(metrics)
            if "spread" in after:
                a=median_draws(helper,after["spread"][m])
                central=float(np.median(after["spread"][m]))
                avg["spread"]=dict(estimate=central,ci95=np.quantile(a,[.025,.975]).tolist())
                if before is not None:
                    b=median_draws(helper,before["spread"][m])
                    avg["delta_spread"]=dict(estimate=central-float(np.median(before["spread"][m])),
                                             ci95=np.quantile(a-b,[.025,.975]).tolist())
            if before is not None:
                den=float(before["minFDE"][m].mean())
                avg["relative_FDE_gain"]=-avg["delta_minFDE"]["estimate"]/den
            output.setdefault(scope,{})[name]=dict(population=helper.population,**avg)
        return output
