"""Track-clustered curves, paired corrections and recording sensitivities."""
import numpy as np

from bias_inputs import OBS, bins, cells
from coupling_analysis import TrackBootstrap, population

GEOMETRY = ("minADE","minFDE","J","path_ES","Top1_ADE","Top1_FDE")
PROBABILITY = ("minADE","minFDE","Top1_ADE","Top1_FDE","path_ES","endpoint_ES","winner_NLL","HPD90")


def hard(d):
    return np.isin(d["label"], ("turn","turn_and_speed"))


def bias_values(model):
    e, w = model["error"], model["w"]
    d = np.linalg.norm(e, axis=-1)
    idx = np.arange(len(w))
    values = dict(nearest_FDE=e[idx,d[:,:,-1].argmin(1)], nearest_ADE=e[idx,d.mean(2).argmin(1)],
                  weighted_center=np.einsum("nk,nktc->ntc",w,e), top1=e[idx,w.argmax(1)])
    series = {f"{kind}_{axis}_t{t+1}":x[:,t,j] for kind,x in values.items()
              for t in range(x.shape[1]) for j,axis in enumerate(("lon","lat"))}
    preceding = e[...,0] < 0
    series.update({f"PIT_equal_t{t+1}":preceding[:,:,t].mean(1) for t in range(e.shape[2])})
    series.update({f"PIT_weighted_t{t+1}":(w*preceding[:,:,t]).sum(1) for t in range(e.shape[2])})
    return series


def counts(tracks):
    p = population(tracks)[0]
    return dict(**p, small_group=p["windows"] < 150)


def bias_group(columns, d, masks, intervals=True):
    output = {}
    for key,mask in masks.items():
        if not mask.any():
            continue
        train = d["track"][mask]
        p = counts(train)
        values = {k:v[mask] for k,v in columns.items()}
        if intervals:
            support = TrackBootstrap(train)
            unit = np.ones(len(train))
            row = support.summarize({k:(v,unit) for k,v in values.items()})
        else:
            row = {k:dict(estimate=float(v.mean())) for k,v in values.items()}
        output[key] = dict(population=p, values=row)
    return output


def contrast(value, track, low, high):
    mask = low | high
    b = TrackBootstrap(track[mask])
    support = (low[mask], high[mask])
    v = value[mask]
    n = b.draw.shape[1]
    nlow, nhigh = (np.bincount(b.index, weights=x, minlength=n) for x in support)
    vlow, vhigh = (np.bincount(b.index, weights=v*x, minlength=n) for x in support)
    nums = b.draw @ np.column_stack((vlow,vhigh))
    denom = b.draw @ np.column_stack((nlow,nhigh))
    good = np.all(denom > 0, axis=1)
    rep = nums[good]/denom[good]
    return dict(estimate=float(value[low].mean()-value[high].mean()),
                ci95=np.quantile(rep[:,0]-rep[:,1],[.025,.975]).tolist(),
                low=counts(track[low]),high=counts(track[high]))


def pit_fraction(table):
    entries = [row for row in table.values() if row["population"]["windows"] >= 150]
    n = sum(row["population"]["windows"] for row in entries)
    return dict(eligible_cells=len(entries), eligible_windows=n,
                **{kind:(sum(row["population"]["windows"] for row in entries
                             if not .4 <= row["values"][f"PIT_{kind}_t12"]["estimate"] <= .6)/n if n else None)
                   for kind in ("equal","weighted")})


def bias_tables(model,d,cuts):
    columns = bias_values(model)
    groups = bins(d,cuts)
    valid = d["valid"]
    by_feature = {}
    for field,category in groups.items():
        by_feature[field] = bias_group(columns,d,{str(k):valid&(category==k) for k in np.unique(category)})
    joint = cells(groups, OBS)
    primary = bias_group(columns,d,{k:valid&(joint==k) for k in np.unique(joint[valid])})
    descriptions = dict(future=bias_group(columns,d,{k:valid&(d["label"]==k) for k in np.unique(d["label"])}),
                        direction_invalid=bias_group(columns,d,{"all":~valid,
                           **{str(k):~valid&(d["label"]==k) for k in np.unique(d["label"])}},False),
                        curvature_validity=bias_group(columns,d,{"valid":valid&d["curvature_valid"],
                                                                "invalid":valid&~d["curvature_valid"]}))
    speed = groups["v_obs"]
    t12 = ("weighted_center_lon_t12","top1_lon_t12","nearest_FDE_lon_t12","PIT_weighted_t12","PIT_equal_t12")
    keep = {"all":valid,"without_zara02":valid&(d["scene"]!="crowds_zara02"),
            **{f"without_{q}":valid&(d["scene"]!=q) for q in np.unique(d["scene"])}}
    contrasts = {name:{k:contrast(columns[k],d["track"],m&(speed==0),m&(speed==4)) for k in t12}
                 for name,m in keep.items() if (m&(speed==0)).any() and (m&(speed==4)).any()}
    return dict(cuts=cuts, feature_bins=by_feature,K_joint=primary,
                speed_low_minus_high=contrasts, pit=dict(K_joint=pit_fraction(primary),
                    **{g:pit_fraction(by_feature[g]) for g in OBS})), descriptions


class Paired:
    def __init__(self,d,cuts):
        track, scene = d["track"], d["scene"]
        group_masks = {"overall":np.ones(len(track),bool),"hard":hard(d),
                       **{f"speed_q{k}":bins(d,cuts)["v_obs"]==k for k in range(5)}}
        self.groups = {}
        self.leaving = {}
        for group, windows in group_masks.items():
            for scope, mask in (("all_dev",windows),("without_zara02",windows&(scene!="crowds_zara02"))):
                if mask.any():
                    self.groups[scope,group] = mask,TrackBootstrap(track[mask])
            if group in ("overall","hard") and windows.any():
                entry = {}
                for clip in np.unique(scene[windows]):
                    dropped = windows & (scene==clip)
                    keep = windows & ~dropped
                    entry[str(clip)] = dropped,keep,TrackBootstrap(track[keep])
                self.leaving[group] = entry

    def summary(self, before, after, kind):
        measures = GEOMETRY if kind == "geometry" else PROBABILITY
        focus = ("minFDE",) if kind=="geometry" else ("path_ES","Top1_FDE")
        groups = {}
        for (scope,group),(m,b) in self.groups.items():
            one = np.ones(m.sum())
            pairs = {}
            for k in measures:
                origin,update = before[k][m].astype(float),after[k][m].astype(float)
                pairs[f"{k}_before"] = origin, one
                pairs[f"{k}_after"] = update, one
                pairs[f"{k}_delta"] = update-origin,one
                if k not in ("J","HPD90"):
                    pairs[f"{k}_relative_gain"] = origin-update,origin
            groups.setdefault(scope,{})[group] = dict(population=b.population,metrics=b.summarize(pairs))
        recordings = {}
        for group, entries in self.leaving.items():
            parts = {}
            for clip, (drop,m,b) in entries.items():
                pairs, contribution = {},{}
                for k in focus:
                    change = before[k]-after[k]
                    contribution[k] = float(change[drop].sum())
                    pairs[k] = (change[m],before[k][m])
                parts[clip] = dict(population=b.population,relative_gain=b.summarize(pairs),contribution=contribution)
            share = {}
            for k in focus:
                contributions = {clip:v["contribution"][k] for clip,v in parts.items()}
                largest = max(contributions,key=contributions.get)
                total = sum(contributions.values())
                share[k] = dict(recording=largest,share=contributions[largest]/total if total>0 else None,
                                total_gain=float(total))
            recordings[group] = dict(leave_one=parts,maximum_contribution=share)
        return dict(groups=groups,recordings=recordings)

