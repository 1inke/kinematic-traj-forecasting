"""Run the sealed C0 conditional bias diagnostic in an independent CPU process."""
import argparse
import json
from pathlib import Path
import time
import traceback

import numpy as np

from bias_analysis import Paired, bias_tables
from bias_decisions import evaluate, render
from bias_inputs import (ROOT, RUNS, TAU, data_model, load_pair, quantiles, read, ready, sha, status,
                         verify, weighted_widths, write)
from bias_models import grid_geometry, grid_probability, learn, scoring, take_model, tilt_weights


def select_rows(d, idx):
    return {k:v[idx] for k,v in d.items()}


def interventions(pred, after, kind, width, draws, base):
    active = np.any(after != 0,axis=1)
    result = {k:v.copy() for k,v in base.items()}
    if not active.any():
        return result, 0
    subset = take_model(pred,active)
    e, w, parameters = subset["error"], subset["w"], after[active]
    if kind == "geometry":
        output = scoring(subset, e + parameters[:,None,None]*TAU[None,None,:,None], w)
    else:
        updated = tilt_weights(w,subset["tilt"],parameters)
        output = scoring(subset,e,updated,width[active],tuple(a[active] for a in draws))
        for metric in ("minADE", "minFDE"):
            np.testing.assert_array_equal(output[metric],base[metric][active])
    for k,v in output.items():
        result[k][active] = v
    return result, int(active.sum())


def smoke_rows(d,n):
    """Pick evenly spaced valid and invalid rows, preserving the normal masks."""
    good = np.flatnonzero(d["valid"])
    bad = np.flatnonzero(~d["valid"])
    return np.sort(np.r_[good[np.linspace(0,len(good)-1,n,dtype=int)],
                         bad[np.linspace(0,len(bad)-1,max(1,n//6),dtype=int)]])


def run_one(arm,seed,index,data,archive,draws,files,sample):
    clock = time.monotonic()

    def progress(phase, **values):
        status("RUNNING", arm=arm, seed=seed, arms_complete=index, arms_total=len(RUNS),
               component=phase, elapsed_arm_seconds=round(time.monotonic()-clock,1), **values)

    progress("LOAD")
    selection, forecasts = load_pair(arm,seed,data,files)
    d, train = data["dev"], data[selection]
    h = weighted_widths(archive,seed,d)
    if sample:
        used = {"dev":smoke_rows(d,sample), selection:smoke_rows(train,sample)}
        d,train = (select_rows(data[part],used[part]) for part in ("dev",selection))
        h, draws = h[used["dev"]], tuple(v[used["dev"]] for v in draws)
        forecasts = {part:tuple(v[used[part]] for v in pair) for part,pair in forecasts.items()}
    dev_all = data_model(d,*forecasts["dev"])
    cuts = quantiles(train,train["valid"])
    progress("DESCRIPTIVE_BIAS")
    tables, descriptive = bias_tables(dev_all,d,cuts)
    counters = dict(selection_population=int(train["valid"].sum()), valid_dev=int(d["valid"].sum()),
                    invalid_dev=int((~d["valid"]).sum()))
    valid = d["valid"]
    dev = select_rows(d,valid)
    model = take_model(dev_all,valid)
    del dev_all
    width, ref = h[valid], tuple(v[valid] for v in draws)
    base = scoring(model,model["error"],model["w"],width,ref)
    summary = Paired(dev,cuts)
    geometry,probability = {},{}
    sources = (selection,"dev_crossfit") if arm=="B0" else (selection,)
    for source in sources:
        if source == "dev_crossfit":
            a, source_model = dev, model
        else:
            mask = train["valid"]
            a = select_rows(train,mask)
            source_model = data_model(a,*(v[mask] for v in forecasts[selection]))
        e = source_model["error"]
        geo_grid = grid_geometry(e,lambda i,n:progress("GEOMETRY_GRID",source=source,step=i,component_total=n))
        prob_grid = grid_probability(e[:,:,-1],source_model["w"],source_model["tilt"],
                                      lambda i,n:progress("PROBABILITY_GRID",source=source,step=i,component_total=n))
        progress("FIT_CORRECTIONS",source=source)
        fitted = learn(a,dev,e,geo_grid,prob_grid,source)
        del geo_grid,prob_grid,source_model,e
        g,p = {},{}
        for ordinal, ((feature,objective),(values,details)) in enumerate(fitted.items(),1):
            key = f"{feature}/{objective}"
            kind = "probability" if objective in ("endpoint_ES","winner_NLL") else "geometry"
            progress("DEV_COMPARISONS",source=source,step=ordinal,component_total=len(fitted),case=key)
            update, active = interventions(model,values,kind,width,ref,base)
            out = p if kind=="probability" else g
            out[key] = dict(fits=details,changed_windows=active,
                            parameter_absolute_max=np.max(abs(values),axis=0).tolist(),
                            **summary.summary(base,update,kind))
        geometry[source],probability[source] = g,p
    return dict(tables=tables, descriptive=descriptive, geometry=geometry, probability=probability,
                populations=counters, seconds=round(time.monotonic()-clock,2))


def run(sample):
    started = time.monotonic()
    done = ROOT/"DONE.json"
    if done.exists() and sample is None:
        assert all(sha(ROOT/name)==digest for name,digest in read(done)["files"].items())
        return
    status("CHECK_PINNED_INPUTS")
    data, archive, _, files, draws = ready()
    pre = read(ROOT/"PREFLIGHT.json")
    assert pre["status"]=="PASS" and len(pre["PF1"])==5 and len(pre["PF5"])==5 and len(pre["PF3"])==7
    for name,digest in pre["files"].items():
        verify(Path(name),digest,files)
    if sample:
        examples = {}
        for i,(arm,seed) in enumerate((("B0",42),("E1B",42))):
            a = run_one(arm,seed,i,data,archive,draws,files,sample)
            assert len(a["geometry"]) == (2 if arm=="B0" else 1)
            assert all(len(v)==12 for v in a["geometry"].values())
            assert all(len(v)==8 for v in a["probability"].values())
            examples[arm] = a
            print(f"Smoke passed for {arm}_s{seed}",flush=True)
        tables, geometry, probability = ({f"{arm}_s{seed}":examples[arm][section] for arm,seed in RUNS}
                                        for section in ("tables","geometry","probability"))
        decisions = evaluate(tables,geometry,probability)
        assert set(decisions["summary"]) == {"G1","G2","P1","P2","P3"}
        json.dumps(decisions,allow_nan=False)
        assert all(key in render(decisions) for key in decisions["summary"])
        status("SMOKE_PASSED",valid_windows_per_split=sample)
        return
    sealed = dict(inputs=files.copy(),preflight=sha(ROOT/"PREFLIGHT.json"))
    checkpoints = ROOT/"checkpoints"
    checkpoints.mkdir(exist_ok=True)
    tables, description, geom, prob, pops, times = {},{},{},{},{},{}
    outputs = ["PREFLIGHT.json"]
    for i,(arm,seed) in enumerate(RUNS):
        name = f"{arm}_s{seed}"
        point = checkpoints/f"{name}.json"
        prior = read(point) if point.exists() else {}
        if prior.get("seal") == sealed:
            row = prior["result"]
        else:
            row = run_one(arm,seed,i,data,archive,draws,files,None)
            write(point,dict(seal=sealed,result=row))
        tables[name],description[name],geom[name],prob[name],pops[name],times[name] = (
            row[k] for k in ("tables","descriptive","geometry","probability","populations","seconds"))
        target = f"bias_tables_{name}.json"
        write(ROOT/target,row["tables"])
        outputs.append(target)
        status("ARM_COMPLETE", arms_complete=i+1,arms_total=len(RUNS), arm=arm,seed=seed)
        print(f"Completed {name} in {times[name]} seconds",flush=True)
    status("FINAL_SUMMARY",arms_complete=len(RUNS),arms_total=len(RUNS))
    result = evaluate(tables,geom,prob)
    result.update(populations=pops,seconds_by_arm=times)
    for target, contents in (("geom_correction",geom),("prob_correction",prob),
                             ("descriptive_outcome_groups",description),("RESULT",result)):
        name = f"{target}.json"
        write(ROOT/name,contents)
        outputs.append(name)
    (ROOT/"REPORT.md").write_text(render(result),encoding="utf-8")
    outputs.append("REPORT.md")
    write(done,dict(status="PASS",elapsed_seconds=round(time.monotonic()-started,2),
                    inputs=files,files={name:sha(ROOT/name) for name in outputs}))
    status("DONE",arms_complete=len(RUNS),arms_total=len(RUNS))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke",type=int)
    arguments = parser.parse_args()
    try:
        run(arguments.smoke)
    except Exception:
        status("FAILED",traceback=traceback.format_exc())
        raise

