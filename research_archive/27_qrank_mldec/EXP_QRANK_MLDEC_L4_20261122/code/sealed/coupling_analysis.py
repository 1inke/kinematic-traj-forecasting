"""Track-clustered summaries of candidate forces and missed coverage."""

import numpy as np

RNG = 20261121
BOOT = 2000
HARD = ("turn", "turn_and_speed")
SECONDARY = ("acceleration", "deceleration", "stop", "start")
LABELS = (*HARD, *SECONDARY, "steady", "stationary")
RANKS = {"r1": slice(None, 1), "r2_3": slice(1, 3), "r4_6": slice(3, 6),
         "r7_20": slice(6, None), "r2_6": slice(1, 6)}
CLASSES = ("shape", "range", "direction", "near", "sector_radius_compatible")


def grouping(label, c0):
    return {"overall": np.ones(len(label), dtype=bool),
            "hard": np.isin(label, HARD), "secondary": np.isin(label, SECONDARY),
            "change": c0 == 2, **{g: label == g for g in LABELS}}


def population(track):
    keys, ix, counts = np.unique(track, return_inverse=True, return_counts=True)
    details = dict(windows=len(track), tracks=len(keys),
                   top10_window_fraction=float(np.sort(counts)[-10:].sum()/len(track)) if len(track) else None)
    return details, ix


class TrackBootstrap:
    def __init__(self, track):
        self.population, self.index = population(track)
        n = self.population["tracks"]
        self.draw = (np.random.default_rng(RNG).multinomial(n, np.full(n, 1/n), size=BOOT)
                     if n else np.empty((BOOT,0))).astype(np.float64)

    def summarize(self, series):
        if not self.population["tracks"]:
            return {k: dict(estimate=None, ci95=None, valid_bootstrap=0) for k in series}
        values = list(series.values())
        cols = []
        for numer, denom in values:
            for v in (numer, denom):
                cols.append(np.bincount(self.index, weights=v, minlength=self.draw.shape[1]))
        track_totals = np.column_stack(cols)
        average = track_totals.sum(0).reshape((-1,2))
        boot = (self.draw @ track_totals).reshape((BOOT,-1,2))
        out = {}
        for i, name in enumerate(series):
            n, d = average[i]
            b = boot[:,i]
            valid = b[:,1] > 0
            draws = b[valid,0] / b[valid,1]
            out[name] = dict(estimate=float(n/d) if d else None,
                             ci95=np.quantile(draws,[.025,.975]).tolist() if len(draws) else None,
                             valid_bootstrap=len(draws))
        return out


def a_stat(raw, term_names):
    groups = grouping(raw["maneuver"], raw["c0_group"])
    output = {}
    ordered = {}
    order = np.argsort(raw["quality_rank"], axis=1, kind="stable")
    for k in ("P", "E", "norm"):
        ordered[k] = np.take_along_axis(raw[k], order[...,None], axis=1)
    for k in ("P_valid", "E_valid"):
        ordered[k] = np.take_along_axis(raw[k], order, axis=1)
    cls, geom = (term_names.index(name) for name in ("cls", "geom"))
    for name, m in groups.items():
        helper = TrackBootstrap(raw["track_id"][m])
        stat = {}
        pending = {}
        for rank, sl in RANKS.items():
            G, P, E = [ordered[k][m][:,sl] for k in ("norm","P","E")]
            pv, ev = [ordered[k][m][:,sl] for k in ("P_valid","E_valid")]
            pending[f"{rank}:D"] = (G[:,:,cls].sum(1), G[:,:,geom].sum(1))
            for i, term in enumerate(term_names):
                p, e = P[:,:,i], E[:,:,i]
                pending[f"{rank}:{term}_P"] = (np.where(pv,p,0).sum(1), pv.sum(1))
                pending[f"{rank}:{term}_P_negative"] = ((pv & (p<0)).sum(1), pv.sum(1))
                pending[f"{rank}:{term}_E"] = (np.where(ev,e,0).sum(1), ev.sum(1))
        for key, value in helper.summarize(pending).items():
            rank, stat_key = key.split(":")
            stat.setdefault(rank,{})[stat_key] = value
        output[name] = dict(population=helper.population, ranks=stat)
    hard = groups["hard"]
    nearest = raw["fde_winner"]
    rr = np.arange(len(nearest))
    p = raw["P"][rr, nearest, cls]
    geo = raw["P"][rr, nearest, geom]
    pv = raw["P_valid"][rr, nearest]
    blocked = (p < 0) & (-p >= .5*geo)
    a4 = {}
    for name,m in (("covered",hard & raw["covered"]),("uncovered",hard & ~raw["covered"])):
        helper = TrackBootstrap(raw["track_id"][m])
        valid = pv[m]
        values = {
            "P_cls": (p[m]*valid,valid),
            "P_geom": (geo[m]*valid,valid),
            "blocked": (blocked[m]&valid,valid),
            "geom_nonpositive": ((geo[m]<=0)&valid,valid),
        }
        a4[name] = dict(population=helper.population, **helper.summarize(values))
    # Shared draws preserve a track's co-occurrence across covered and uncovered windows.
    helper = TrackBootstrap(raw["track_id"][hard])
    cov = raw["covered"][hard]
    v = pv[hard]
    if helper.population["tracks"]:
        num = blocked[hard] & v
        mat = np.column_stack([np.bincount(helper.index, weights=x, minlength=helper.draw.shape[1])
                               for x in (num&~cov, v&~cov, num&cov, v&cov)])
        a,b,c,d = (helper.draw @ mat).T
        ok = (b > 0) & (d > 0)
        a4["gap_pp_ci95"] = (100*np.quantile(a[ok]/b[ok] - c[ok]/d[ok],[.025,.975])).tolist() if ok.any() else None
    return dict(A1_A3=output,A4=a4)


def misses(paths, target, logits):
    # Coordinates here are already relative to the last observed focal point.
    error = paths - target[:,None]
    dist = np.linalg.norm(error,axis=-1)
    ade, fde = dist.mean(-1), dist[:,:,-1]
    joint = ((ade<=.2)&(fde<=.4)).any(1)
    end, truth = paths[:,:,-1], target[:,-1]
    length, r = np.linalg.norm(end,axis=-1), np.linalg.norm(truth,axis=-1)
    valid = (r[:,None] > 0) & (length > 0)
    cos = np.divide(np.einsum("nki,ni->nk",end,truth),length*r[:,None],
                    out=np.full_like(length,-1.), where=valid)
    in_sector = (cos>=np.cos(np.deg2rad(15)))&(length>=.3*r[:,None])
    radial = np.min(np.where(in_sector,np.abs(length-r[:,None]),np.inf),axis=1)
    shape = ((fde<=.4)&(ade>.2)).any(1)
    cat = np.select(
        [joint, shape, (r>=1)&in_sector.any(1)&(radial>.4),
         (r>=1)&~in_sector.any(1), r<1],
        ["covered", "shape", "range", "direction", "near"], default="sector_radius_compatible")
    best = fde.argmin(1)
    row = np.arange(len(best))
    logp = np.exp(logits-logits.max(1,keepdims=True))
    prob = logp/logp.sum(1,keepdims=True)
    rank = np.argsort(np.argsort(-logits,axis=1,kind="stable"),axis=1)+1
    return dict(category=cat, covered=joint, nearest_fde=fde[row,best], nearest_ade=ade[row,best],
                abs_longitude=np.abs(error[row,best,-1,0]), abs_lateral=np.abs(error[row,best,-1,1]),
                signed_longitude=error[row,best,-1,0], signed_lateral=error[row,best,-1,1],
                nearest_probability=prob[row,best], probability_rank=rank[row,best],
                truth_angle_deg=np.rad2deg(np.arctan2(truth[:,1],truth[:,0])))


def b_stat(raw, label, c0, track):
    out = {}
    groups = grouping(label,c0)
    fields = ("nearest_fde","nearest_ade","abs_longitude","abs_lateral","signed_longitude","signed_lateral",
              "nearest_probability","probability_rank")
    edges = np.arange(-180,181,30)
    angles = np.minimum(np.searchsorted(edges,raw["truth_angle_deg"],side="right")-1,11)
    for name, mask in groups.items():
        helper = TrackBootstrap(track[mask])
        c = raw["covered"][mask]
        overall = helper.summarize({"joint_coverage":(c,np.ones(len(c)))})
        missed = mask & ~raw["covered"]
        helper_miss = TrackBootstrap(track[missed])
        category = raw["category"][missed]
        result = {}
        queries = {}
        for state in CLASSES:
            select = category == state
            result[state] = dict(population=population(track[missed][select])[0])
            queries[f"{state}:fraction"] = (select,np.ones(len(select)))
            for k in fields:
                queries[f"{state}:{k}"] = (raw[k][missed]*select,select)
            if state=="direction":
                for i in range(12):
                    queries[f"direction:angle_{edges[i]}_{edges[i+1]}"] = (select & (angles[missed]==i), select)
        for key,value in helper_miss.summarize(queries).items():
            state,k = key.split(":")
            result[state][k] = value
        assert sum(v["population"]["windows"] for v in result.values()) == int(missed.sum())
        out[name] = dict(population=helper.population, uncovered_population=helper_miss.population,
                         **overall, categories=result)
    return out


def decision(summaries):
    per = {}
    for seed, value in summaries.items():
        hard = value["A1_A3"]["hard"]["ranks"]["r2_6"]
        steady = value["A1_A3"]["steady"]["ranks"]["r2_6"]
        p = hard["cls_P"]
        g = hard["geom_P"]["estimate"]
        cov, uncov = (value["A4"][k]["blocked"]["estimate"] for k in ("covered","uncovered"))
        low, high = p["ci95"]
        t = hard["D"]["estimate"]
        gap = uncov-cov
        per[seed] = dict(K1=dict(pass_gate=t>=1,D=t),
                         K2=dict(pass_gate=p["estimate"]<0 and -p["estimate"]>=.5*g and high<0,
                                 cls_P=p["estimate"], geom_P=g, cls_P_ci95=[low,high]),
                         K3=dict(pass_gate=uncov>=.5 and gap>=.15, uncovered=uncov, covered=cov, gap_pp=100*gap),
                         K4=dict(pass_gate=steady["D"]["estimate"]<.5,D=steady["D"]["estimate"]))
    all_markers = {key:sum(bool(s[key]["pass_gate"]) for s in per.values()) for key in ("K1","K2","K3","K4")}
    flags = {key: dict(active=n>=4,seeds_passing=n) for key,n in all_markers.items()}
    return dict(per_seed=per, flags=flags, phase_two_triggered=flags["K1"]["active"])
