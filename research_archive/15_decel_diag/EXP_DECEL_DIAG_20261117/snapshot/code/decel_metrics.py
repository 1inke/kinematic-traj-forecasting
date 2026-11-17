"""World-coordinate float64 candidate-generation/selection diagnostics, NumPy only."""
from __future__ import annotations

import numpy as np

DT = .4
HISTORY_NAMES = ('decelerating', 'steady', 'accelerating')
MANEUVERS = ('stationary','start','stop','turn_and_speed','turn','acceleration','deceleration','steady')


def speed(path, origin):
    points = np.concatenate([origin[..., None, :], path], axis=-2)
    return np.linalg.norm(np.diff(points, axis=-2), axis=-1).mean(-1)/DT


def compute(past, future, paths, logits):
    p, y, x, z = [np.asarray(v, np.float64) for v in (past, future, paths, logits)]
    if p.ndim == 4: p = p[:, 0]
    n = len(p)
    if p.shape != (n,8,2) or y.shape != (n,12,2) or x.shape != (n,20,12,2) or z.shape != (n,20):
        raise ValueError('BAD_INPUT_SHAPE')
    if not all(np.isfinite(v).all() for v in (p,y,x,z)): raise ValueError('NONFINITE_INPUT')
    observed_steps = np.linalg.norm(np.diff(p, axis=1), axis=-1)/DT
    observed = observed_steps.mean(-1); denominator = np.maximum(observed, .1)
    mean_displacement = (p[:, -1]-p[:, -4])/3
    norm = np.linalg.norm(mean_displacement, axis=-1)
    u = mean_displacement/np.maximum(norm[:, None], 1e-12)
    normal = np.stack([-u[:,1],u[:,0]],axis=-1)
    rg = speed(y,p[:,-1])/denominator
    rk = speed(x,np.broadcast_to(p[:,-1,None],(n,20,2)))/denominator[:,None]
    dec = rk <= .8
    centered = z-z.max(-1,keepdims=True)
    prob = np.exp(centered); prob /= prob.sum(-1,keepdims=True)
    top = z.argmax(-1); row = np.arange(n)
    order = np.argsort(-z,axis=-1,kind='stable')
    ranks = np.empty_like(order); np.put_along_axis(ranks,order,np.broadcast_to(np.arange(1,21),(n,20)),axis=-1)
    history_ratio = observed_steps[:,-3:].mean(-1)/np.maximum(observed_steps[:,:4].mean(-1),.1)
    history = np.where(history_ratio <= .85,0,np.where(history_ratio >= 1.15,2,1)).astype(np.int8)
    out = dict(direction_valid=norm>.03, direction_valid_previous_total=(norm*3)>.03,
        mean_displacement_length=norm, observed_speed=observed, r_g=rg, candidate_r=rk,
        candidate_r_min=rk.min(-1),candidate_r_max=rk.max(-1),candidate_r_mean=rk.mean(-1),
        n_decel=dec.sum(-1), speed_ratio_reachable=(np.abs(rk-rg[:,None])<=.15).any(-1),
        P_dec=np.clip((prob*dec).sum(-1),0,1),true_deceleration=rg<=.8,top1_r=rk[row,top],
        top1_index=top,history_ratio=history_ratio,history_bin=history,
        candidate_decel=dec,candidate_probability=prob)
    for horizon in (6,12):
        candidate_delta = x[:,:,horizon-1]-p[:,-1,None]
        target_delta = y[:,horizon-1]-p[:,-1]
        candidate_prog = (candidate_delta*u[:,None]).sum(-1)
        target_prog = (target_delta*u).sum(-1)
        prog_error = candidate_prog-target_prog[:,None]
        lat_error = ((x[:,:,horizon-1]-y[:,horizon-1,None])*normal[:,None]).sum(-1)
        distance = np.sqrt(prog_error**2+lat_error**2)
        # For a zero direction the projections are not Euclidean, so compute
        # selection from raw Euclidean distance and mask invalid directions later.
        euclidean = np.linalg.norm(x[:,:,horizon-1]-y[:,horizon-1,None],axis=-1)
        best = euclidean.argmin(-1)
        suffix = f'_t{horizon}'
        values = dict(candidate_prog_error=prog_error,candidate_lat_error=lat_error,
            envelope_covered=(candidate_prog.min(-1)<=target_prog)&(target_prog<=candidate_prog.max(-1)),
            all_overshoot=target_prog<candidate_prog.min(-1),all_undershoot=target_prog>candidate_prog.max(-1),
            longitudinal_reachable=(np.abs(prog_error).min(-1)<=.4),
            best_prog_error=prog_error[row,best],best_lat_error=lat_error[row,best],
            best_displacement_error=euclidean[row,best],best_index=best,best_rank=ranks[row,best],
            best_r=rk[row,best],top1_prog_error=prog_error[row,top],
            top1_lat_error=lat_error[row,top],top1_displacement_error=euclidean[row,top],
            top1_overshoot=prog_error[row,top]>.4)
        out.update({k+suffix:v for k,v in values.items()})
        valid = out['direction_valid']
        np.testing.assert_allclose(distance[valid],euclidean[valid],rtol=1e-11,atol=1e-11)
    return out


def error_stats(x):
    return dict(signed_mean=float(x.mean()),absolute_mean=float(np.abs(x).mean()),
                absolute_p90=float(np.quantile(np.abs(x),.9)),signed_p90=float(np.quantile(x,.9)))


def calibration(w, mask):
    edges = np.linspace(0,1,6); result = []
    for i in range(5):
        select = mask & (w['P_dec']>=edges[i]) & ((w['P_dec']<edges[i+1]) if i<4 else (w['P_dec']<=1))
        n = int(select.sum()); p = float(w['P_dec'][select].mean()) if n else None
        actual = float(w['true_deceleration'][select].mean()) if n else None
        result.append(dict(bin=i,lower=float(edges[i]),upper=float(edges[i+1]),n=n,small_group=n<20,
            mean_P_dec=p,actual_deceleration_rate=actual,gap_actual_minus_predicted=actual-p if n else None))
    assert sum(x['n'] for x in result)==int(mask.sum())
    return result


def aggregate(w, mask):
    n = int(mask.sum())
    if not n: return dict(n=0,small_group=True,t6=None,t12=None,calibration=calibration(w,mask))
    out = dict(n=n,small_group=n<20,mean_P_dec=float(w['P_dec'][mask].mean()),
        actual_deceleration_rate=float(w['true_deceleration'][mask].mean()),
        mean_top1_r=float(w['top1_r'][mask].mean()),median_top1_r=float(np.median(w['top1_r'][mask])),
        mean_r_g=float(w['r_g'][mask].mean()),mean_n_decel=float(w['n_decel'][mask].mean()),
        n_decel_distribution={'zero':float((w['n_decel'][mask]==0).mean()),
                              'one_to_two':float(((w['n_decel'][mask]>=1)&(w['n_decel'][mask]<=2)).mean()),
                              'three_or_more':float((w['n_decel'][mask]>=3).mean())},
        speed_ratio_reachable_rate=float(w['speed_ratio_reachable'][mask].mean()),
        calibration=calibration(w,mask))
    for horizon in (6,12):
        get = lambda k:w[k+f'_t{horizon}'][mask]
        out[f't{horizon}'] = dict(envelope_coverage_rate=float(get('envelope_covered').mean()),
            all_overshoot_rate=float(get('all_overshoot').mean()),all_undershoot_rate=float(get('all_undershoot').mean()),
            longitudinal_reachable_rate=float(get('longitudinal_reachable').mean()),
            best_prog_error=error_stats(get('best_prog_error')),best_lat_error=error_stats(get('best_lat_error')),
            best_displacement_error_mean=float(get('best_displacement_error').mean()),
            best_rank_median=float(np.median(get('best_rank'))),best_rank_top3_rate=float((get('best_rank')<=3).mean()),
            mean_best_r=float(get('best_r').mean()),
            top1_prog_error=error_stats(get('top1_prog_error')),
            top1_overshoot_rate=float(get('top1_overshoot').mean()),
            top1_displacement_error_mean=float(get('top1_displacement_error').mean()))
    return out


def summarize(w, labels, validity='direction_valid'):
    valid = w[validity]
    cohorts = dict(all=np.ones(len(labels),bool),
                   state_change=np.isin(labels,['deceleration','stop','turn_and_speed']),
                   controls=np.isin(labels,['steady','acceleration']))
    cohorts.update({name:labels==name for name in MANEUVERS})
    result = {}
    for name, subset in cohorts.items():
        result[name] = dict(total_windows=int(subset.sum()),direction_invalid=int((subset&~valid).sum()),
                           all_history=aggregate(w,subset&valid),
                           by_history={h:aggregate(w,subset&valid&(w['history_bin']==i))
                                       for i,h in enumerate(HISTORY_NAMES)})
    return result


def decide(w, labels, validity='direction_valid'):
    valid = w[validity]; dec = valid&(labels=='deceleration'); n = int(dec.sum())
    if not n: return dict(label='NOT_ESTIMABLE',n=0,A=False,B=False)
    s = aggregate(w,dec)['t12']
    A = s['all_overshoot_rate']>=.30 or s['longitudinal_reachable_rate']<.60
    B = s['longitudinal_reachable_rate']>=.70 and s['top1_overshoot_rate']>=.50
    hd, hs = dec&(w['history_bin']==0), dec&(w['history_bin']==1)
    over_d = float(w['top1_overshoot_t12'][hd].mean()) if hd.any() else None
    over_s = float(w['top1_overshoot_t12'][hs].mean()) if hs.any() else None
    all_hd = valid&(w['history_bin']==0)
    gap = float(w['true_deceleration'][all_hd].mean()-w['P_dec'][all_hd].mean()) if all_hd.any() else None
    overshoot = dec&w['top1_overshoot_t12']
    steady_share = float((overshoot&(w['history_bin']==1)).sum()/overshoot.sum()) if overshoot.any() else None
    cal = calibration(w,valid); nonempty = [x for x in cal if x['n']]
    calibrated = bool(nonempty) and all(abs(x['gap_actual_minus_predicted'])<.1 for x in nonempty)
    rate_condition = over_d is not None and over_s is not None and over_d>=over_s
    underprediction = gap is not None and gap>=.2
    B1 = bool(B and (rate_condition or underprediction))
    B2 = bool(B and steady_share is not None and steady_share>.5 and calibrated)
    main = 'A+B' if A and B else 'A' if A else 'B' if B else 'NEITHER'
    subtype = 'B1' if B1 else 'B2' if B2 else 'B_UNRESOLVED' if B else None
    label = main if main!='B' else subtype
    return dict(label=label,main_class=main,n=n,small_group=n<20,A=bool(A),B=bool(B),B1=B1,B2=B2,
        B_subtype=subtype,all_overshoot_rate=s['all_overshoot_rate'],longitudinal_reachable_rate=s['longitudinal_reachable_rate'],
        top1_overshoot_rate=s['top1_overshoot_rate'],historical_deceleration_n=int(hd.sum()),historical_steady_n=int(hs.sum()),
        historical_deceleration_top1_overshoot=over_d,historical_steady_top1_overshoot=over_s,
        historical_deceleration_calibration_n=int(all_hd.sum()),historical_deceleration_underprediction_gap=gap,
        B1_rate_condition=bool(rate_condition),B1_underprediction_condition=bool(underprediction),
        overshoot_share_from_historical_steady=steady_share,all_nonempty_calibration_gaps_below_point1=calibrated,
        interpretation='Exploratory operational label, not causal or intrinsic-unpredictability proof.')
