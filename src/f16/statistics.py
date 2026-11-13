"""Track then seed paired resampling, no window-wise pseudoreplication."""
from statistics import NormalDist
import numpy as np

def paired_bootstrap(treatment,control,tracks,*,draws=20000,seed=20261020):
    t=np.asarray(treatment,np.float64);b=np.asarray(control,np.float64);track=np.asarray(tracks)
    if t.ndim!=2 or t.shape!=b.shape or t.shape[1]!=len(track) or not len(track) or not np.isfinite(t).all() or not np.isfinite(b).all():raise ValueError('Expected paired seed x window arrays and nonempty population')
    unique,codes=np.unique(track,return_inverse=True);ntrack=len(unique);nseed=len(t)
    if nseed<1 or draws<2:raise ValueError('Need seeds and >=2 draws')
    counts=np.bincount(codes,minlength=ntrack);sums=np.stack([np.bincount(codes,weights=x,minlength=ntrack) for x in t-b]);rng=np.random.Generator(np.random.PCG64(seed));boot=np.empty(draws)
    for i in range(draws):
        ti=rng.integers(0,ntrack,ntrack);si=rng.integers(0,nseed,nseed);den=counts[ti].sum();boot[i]=sums[si[:,None],ti[None,:]].sum(1).mean()/den
    point=float((t.mean(1)-b.mean(1)).mean());ci=np.quantile(boot,[.025,.975]);se=float(boot.std(ddof=1))
    return dict(point=point,CI95=ci.tolist(),standard_error=se,n_tracks=ntrack,n_windows=len(track),n_model_seeds=nseed,draws=draws,valid_draws=int(np.isfinite(boot).sum()),seed=seed,order='track_then_seed',per_seed=(t.mean(1)-b.mean(1)).tolist()),boot

def holm_adjust(pvalues):
    p=np.asarray(pvalues,float)
    if p.ndim!=1 or not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('Invalid p-values')
    order=np.argsort(p,kind='stable');adjusted=np.empty(len(p));running=0.
    for rank,i in enumerate(order):running=max(running,(len(p)-rank)*p[i]);adjusted[i]=min(1.,running)
    return adjusted

def percentile_tail_p(draws,null=0.):
    # Descriptive percentile-tail convention, NOT exact randomization p-value.
    x=np.asarray(draws,float)
    if not len(x) or not np.isfinite(x).all():raise ValueError('Invalid bootstrap draws')
    return min(1.,2*min((1+np.count_nonzero(x<=null))/(len(x)+1),(1+np.count_nonzero(x>=null))/(len(x)+1)))

def mde_normal_approx(standard_error,power=.8,alpha=.05):
    if not np.isfinite(standard_error) or standard_error<0 or not 0<power<1 or not 0<alpha<1:raise ValueError('Invalid power parameters')
    # Explicit two-sided normal approximation, not empirical power simulation.
    return (NormalDist().inv_cdf(1-alpha/2)+NormalDist().inv_cdf(power))*standard_error

def verdict(primary_interval,guards,*,multiplicity_pass=None,multiplicity_required=False):
    low,high=primary_interval
    if not np.isfinite([low,high]).all() or low>high:raise ValueError('Invalid interval')
    if any(x is not None and not isinstance(x,(bool,np.bool_)) for x in guards.values()):raise ValueError('Guard results must be boolean or unknown')
    if any(x is not None and not bool(x) for x in guards.values()):return 'FAIL'
    if high<0:return 'FAIL'
    if not all(k in guards for k in ['G1','G2','G3','G4','G5']) or any(x is None for x in guards.values()):return 'INCONCLUSIVE'
    if multiplicity_required and multiplicity_pass is None:return 'INCONCLUSIVE'
    if low>0 and (not multiplicity_required or multiplicity_pass):return 'PASS'
    if low>-.02:return 'NON-INFERIOR'
    return 'INCONCLUSIVE'
