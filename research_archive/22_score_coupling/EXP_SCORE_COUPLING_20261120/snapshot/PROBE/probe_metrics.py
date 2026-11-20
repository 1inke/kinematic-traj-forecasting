"""Held-out metrics, track intervals and the predeclared probe comparisons."""

import numpy as np

from coupling_analysis import TrackBootstrap, grouping
from calib_metrics import hpd_mc, log_density

SEEDS = ("0","1","2")
ARMS = ("R-couple","R-detach","R-nogeom")


def metrics(paths, truth, scores, bandwidth, draws):
    d = np.linalg.norm(paths - truth[:,None],axis=-1)
    ade,fde = d.mean(-1), d[..., -1]
    winners = (ade+.25*fde).argmin(1)
    row = np.arange(len(d))
    top = scores.argmax(1)
    score = scores-scores.max(1,keepdims=True)
    lp = score-np.log(np.exp(score).sum(1,keepdims=True))
    p = np.exp(lp)
    good = (ade<=.2)&(fde<=.4)
    selected_bin = np.minimum((p*10).astype(int),9)
    # Per-window signed calibration residual, additive before taking bin absolutes.
    bins = np.stack([np.where(selected_bin==i,p-good,0.).sum(1) for i in range(10)],axis=1)
    endpoint = paths[:,:, -1]
    spread = np.sqrt(np.square(endpoint-endpoint.mean(1,keepdims=True)).sum(-1).mean(1))
    log_w = np.full(p.shape,-np.log(20))
    density = log_density(np.square(endpoint-truth[:,None,-1]).sum(-1),log_w,bandwidth)
    u, _ = hpd_mc(endpoint,np.exp(log_w),bandwidth,density,*draws)
    return dict(minADE=ade.min(1),minFDE=fde.min(1),top1_ADE=ade[row,top],top1_FDE=fde[row,top],
                winner_NLL=-lp[row,winners],J_coverage=good.any(1),HPD90_coverage=u<=.90,
                spread=spread,ece_bins=bins,hpd_level=u)


def aggregate(w, data):
    masks = grouping(data["maneuver"],data["group"])
    obs = data["observation"]
    masks.update({f"obs_{name}":obs==name for name in ("stationary","steady","other")})
    simple = tuple(k for k in w if k not in ("spread","ece_bins","hpd_level"))
    result = {}
    for name, m in masks.items():
        helper = TrackBootstrap(data["track"][m])
        n = int(m.sum())
        vals = helper.summarize({k:(w[k][m],np.ones(n)) for k in simple})
        ordered = np.argsort(w["spread"][m],kind="stable")
        positions = helper.index[ordered]
        spread = w["spread"][m][ordered]
        medians = []
        for start in range(0,len(helper.draw),128):
            window_weights = helper.draw[start:start+128,positions]
            cumulative = np.cumsum(window_weights,axis=1)
            total = cumulative[:,-1]
            low = (cumulative <= ((total.astype(np.int64)-1)//2)[:,None]).sum(1)
            high = (cumulative <= (total.astype(np.int64)//2)[:,None]).sum(1)
            medians.extend(.5*(spread[low]+spread[high]))
        vals["median_spread"] = dict(estimate=float(np.median(spread)),ci95=np.quantile(medians,[.025,.975]).tolist())
        ec = w["ece_bins"][m]
        mat = np.column_stack([np.bincount(helper.index,weights=ec[:,i],minlength=helper.draw.shape[1])
                               for i in range(10)])
        counts = np.bincount(helper.index,minlength=helper.draw.shape[1])
        boot = np.abs(helper.draw@mat).sum(1)/(20*(helper.draw@counts))
        vals["ECE_J"] = dict(estimate=float(np.abs(ec.sum(0)).sum()/(20*n)),
                             ci95=np.quantile(boot,[.025,.975]).tolist())
        result[name]=dict(population=helper.population,**vals)
    return result


def calculate_decisions(runs, b0_minfde):
    def series(arm,group,key):
        return np.asarray([runs[arm][s]["groups"][group][key]["estimate"] for s in SEEDS])

    def diff(arm,ref,group,key):
        delta = series(arm,group,key)-series(ref,group,key)
        threshold = 2*series("R-couple",group,key).std(ddof=1)/np.sqrt(3)
        mean = float(delta.mean())
        return dict(deltas=delta.tolist(),mean=mean,threshold=float(threshold),
                    signal=bool(abs(mean)>threshold and (np.all(delta>0) or np.all(delta<0))))

    paired = {}
    for arm,ref in (("R-detach","R-couple"),("R-nogeom","R-detach")):
        paired[arm]={g:{k:diff(arm,ref,g,k) for k in v if k!="population"}
                     for g,v in runs[arm]["0"]["groups"].items()}
    h=paired["R-detach"]["hard"]
    overall=paired["R-detach"]["overall"]
    s0 = series("R-couple","overall","minFDE")/b0_minfde - 1
    s1 = (h["J_coverage"]["signal"] and h["J_coverage"]["mean"]>0 and h["minFDE"]["mean"]<=0 and
          all(overall[k]["mean"] <= .01*series("R-couple","overall",k).mean() for k in ("minADE","minFDE")))
    s2 = any(overall[k]["signal"] and overall[k]["mean"]>0 for k in ("top1_FDE","winner_NLL"))
    ng=paired["R-nogeom"]["overall"]
    s3 = any(ng[k]["signal"] and ng[k]["mean"]>0 for k in ("top1_FDE","winner_NLL"))
    spread=paired["R-detach"]["obs_stationary"]["median_spread"]
    s4 = (spread["signal"] and spread["mean"]<0 and
          np.all(series("R-detach","obs_stationary","median_spread") < series("R-detach","obs_steady","median_spread")))
    state=dict(S0=bool(np.all(abs(s0)<=.05)),S1=bool(s1),S2=bool(s2),S3=bool(s3),S4=bool(s4))
    return dict(flags=state,paired=paired,B0_comparison=s0.tolist())
