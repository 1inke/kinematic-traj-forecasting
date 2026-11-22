"""Compare supplied preliminary scripts and archived projections before the formal run."""
import re

import numpy as np

from bias_inputs import (CAL, ROOT, RUNS, arrays, data_model, decel_reference, load_pair, observed,
                         read, ready, status, verify, weighted_widths, write)
from bias_models import SHIFT, TILTS, grid_geometry, grid_probability, scoring, tilt_weights


def error(a, b, tolerance):
    gap = float(np.max(abs(np.asarray(a, float)-np.asarray(b, float))))
    assert gap <= tolerance, (gap, tolerance)
    return gap


def legacy_global(d, rng):
    take = d["direction_valid"]
    longitudinal, lateral = (d[f"candidate_{k}_error_t12"][take] for k in ("prog", "lat"))
    track = np.array([k.rsplit("|", 1)[0] for k in d["ids"][take]])
    keys, inv = np.unique(track, return_inverse=True)
    for _ in range(4):
        rng.integers(0, len(keys), (2000,len(keys)))
    assignment = np.empty(len(keys), int)
    assignment[rng.permutation(len(keys))] = np.arange(len(keys)) % 5
    folds = assignment[inv]
    base = np.hypot(longitudinal,lateral).min(1)
    grid = np.round(np.arange(-.5, .501, .05), 3)
    result = {}
    for mode, beta in (("prog", [0.]), ("prog_lat", grid)):
        scores = np.column_stack([np.hypot(longitudinal-a,lateral-b).min(1) for a in grid for b in beta])
        corrected = np.empty(len(track))
        for k in range(5):
            objective = scores[folds!=k].mean(0)
            best = np.flatnonzero(objective <= objective.min()+1e-12)[0]
            corrected[folds==k] = scores[folds==k,best]
        result[f"GLOBAL_{mode}"] = float((base-corrected).sum()/base.sum())
    for _ in range(8):
        rng.integers(0, len(keys), (2000,len(keys)))
    return result


def legacy_tilt(d, rng):
    take = d["direction_valid"]
    spd, ranks, weights = (d[k][take] for k in ("observed_speed","candidate_r","candidate_probability"))
    columns = np.digitize(spd, np.quantile(spd, [.2,.4,.6,.8]))
    track = np.array([k.rsplit("|", 1)[0] for k in d["ids"][take]])
    keys, inv = np.unique(track, return_inverse=True)
    folds = (rng.permutation(len(keys)) % 5)[inv]
    e = np.stack([d[f"candidate_{k}_error_t12"][take] for k in ("prog","lat")], axis=-1)
    dist = np.linalg.norm(e, axis=2)
    pair = np.linalg.norm(e[:, :, None] - e[:, None], axis=-1)
    feature = ranks-np.median(ranks, axis=1, keepdims=True)
    beta = np.round(np.arange(-3.,3.01,.25), 2)
    logit = np.log(weights[:, None]+1e-12)+beta[None,:,None]*feature[:,None]
    value = np.exp(logit-logit.max(-1,keepdims=True))
    value /= value.sum(-1, keepdims=True)
    score = (value*dist[:,None]).sum(-1)-.5*((value@pair)*value).sum(-1)
    base = (weights*dist).sum(-1)-.5*((weights[:,None]@pair)[:,0]*weights).sum(-1)
    new = np.empty(len(w := weights))
    new_top = np.empty(len(w))
    for k in range(5):
        for b in range(5):
            q = columns == b
            te = (folds==k) & q
            j = score[(folds!=k) & q].mean(0).argmin()
            new[te] = score[te,j]
            new_top[te] = dist[te, value[te,j].argmax(1)]
    top = dist[np.arange(len(w)), w.argmax(1)]
    return dict(endpoint_ES_change=float(new.sum()/base.sum()-1),
                Top1_FDE_change=float(new_top.sum()/top.sum()-1))


def verify_grids(model, width, draws):
    e = model["error"]
    geom = grid_geometry(e, lambda *_: None)
    prob = grid_probability(e[:, :, -1], model["w"], model["tilt"], lambda *_: None)
    base = scoring(model,e,model["w"],width,draws)
    pf2 = {}
    pf3 = {}
    indices = (0, 9, 137, len(SHIFT)//2, len(SHIFT)-1)
    t = np.arange(13)/12
    x = np.concatenate([np.zeros_like(e[:, :, :1]), e], axis=2)
    for i in indices:
        translated = x + SHIFT[i]*t[None,None,:,None]
        pf2[str(i)] = dict(pair=error(translated[:,1:]-translated[:,:1], x[:,1:]-x[:,:1], 1e-9),
                           t0=error(translated[:,:,0], x[:,:,0],1e-9))
        direct = scoring(model,translated[:,:,1:],model["w"])
        for k in geom:
            error(geom[k][:,i],direct[k],1e-12)
        w = tilt_weights(model["w"], model["tilt"], np.broadcast_to(TILTS[i],(len(e),2)))
        comparison = scoring(model,e,w,width,draws)
        pf3[str(i)] = {k:error(comparison[k],base[k],1e-12) for k in ("minADE","minFDE")}
        for k in prob:
            error(prob[k][:,i],comparison[k],1e-12)
        if i == len(SHIFT)//2:
            for k in base:
                error(comparison[k],base[k],1e-12)
    return pf2, pf3


def preflight():
    data, archive, replay, files, draws = ready()
    ref = read(ROOT/"reference/bias_check.json")
    tilt_log = (ROOT/"reference/tilt_stdout.log").read_text("utf-8-sig")
    expected = {int(s):[float(a)/100,float(b)/100] for s,a,b in
                re.findall(r"^(\d+) endpoint wES change ([+\-\d.]+)%\s+top1 FDE change ([+\-\d.]+)%",tilt_log,re.M)}
    rng_shift, rng_tilt, random_future = (np.random.default_rng(seed) for seed in (20261120,7,20261121))
    p1,p2,p3,p4,p5 = {},{},{},{},{}
    for split,d in data.items():
        changed = dict(d, future=random_future.normal(size=d["future"].shape))
        fresh = observed(changed)
        assert all(np.array_equal(v,d[k]) for k,v in fresh.items())
        p4[split] = dict(windows=len(d["ids"]),features=list(fresh),exact=True)
    d = data["dev"]
    valid = d["valid"]
    past = d["past"][:,0].astype(float)
    late = (past[:,-1]-past[:,-4])/3
    u = late / np.maximum(np.linalg.norm(late,axis=1)[:,None], 1e-12)
    frame_legacy = np.stack((u,u[:,::-1]*[-1,1]),axis=-1)
    for arm,seed in RUNS:
        key = f"{arm}_s{seed}"
        status("PREFLIGHT", current=key)
        _, forecasts = load_pair(arm,seed,data,files)
        x,w = forecasts["dev"]
        model = data_model(d,x,w)
        h = weighted_widths(archive,seed,d)
        if arm == "B0":
            old = decel_reference(seed,files)
            np.testing.assert_array_equal(d["ids"],old["ids"])
            np.testing.assert_array_equal(valid,old["direction_valid"])
            legacy = np.einsum("nkti,nij->nktj",x-model["y"][:,None],frame_legacy)
            measured = {}
            for horizon in (6,12):
                for dim, name in enumerate(("prog","lat")):
                    a = old[f"candidate_{name}_error_t{horizon}"]
                    measured[f"{horizon}_{name}_all_archive_frame"] = error(legacy[:,:,horizon-1,dim],a,1e-9)
                    measured[f"{horizon}_{name}_valid_formal"] = error(model["error"][valid,:,horizon-1,dim],a[valid],1e-9)
            hp = CAL/f"windows/B0_s{seed}.npz"
            verify(hp,read(CAL/"DONE.json")["files"][hp.relative_to(CAL).as_posix()],files)
            hp_ref = arrays(hp)
            np.testing.assert_array_equal(d["ids"],hp_ref["ids"])
            hpd = scoring(model,model["error"],w,h,draws)["HPD90"]
            np.testing.assert_array_equal(hpd,hp_ref["t12__weighted__u__global"]<=.9)
            p1[key] = dict(windows=len(d["ids"]),valid_windows=int(valid.sum()),max_error_m=measured,
                           weighted_HPD90_matches_archive=True)
            old_shift = legacy_global(old,rng_shift)
            for name,gain in old_shift.items():
                error(gain,ref[str(seed)]["correction"][name]["rel_all"],1e-12)
            old_tilt = legacy_tilt(old,rng_tilt)
            for value, rounded in zip(old_tilt.values(),expected[seed]):
                assert round(value,4) == round(rounded,4), (seed, value, rounded)
            p5[key] = dict(shift=old_shift, tilt=old_tilt, comparison="supplied_original_scripts_on_archive15")
        idx = np.linspace(0,len(x)-1,24,dtype=int)
        p2[key],p3[key] = verify_grids({k:v[idx] for k,v in model.items()},h[idx],tuple(a[idx] for a in draws))
        print(f"Preflight passed for {key}",flush=True)
    report = dict(status="PASS",PF1=p1,PF2=p2,PF3=p3,PF4=p4,PF5=p5,
                  holdout_replay=replay["checks"],files=files)
    write(ROOT/"PREFLIGHT.json",report)
    status("PREFLIGHT_PASSED")


if __name__ == "__main__":
    preflight()

