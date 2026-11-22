"""Z2: run the five-seed pool diagnostic on sealed C0 forecasts, entirely on CPU."""
from datetime import datetime
from pathlib import Path
import sys
import time
import os
import traceback

import numpy as np

from pool_select import NMS_THRESHOLDS, medoids, nms

ROOT = Path(__file__).resolve().parent
OUT = ROOT/"Z2"
sys.path.insert(0, str(ROOT.parent/"EXP_ADAPT_ES_PRECHECK_EXEC_20261120"))
from adapt_data import SEEDS, old_paths, prepare_data, read, sha, write
from coupling_analysis import TrackBootstrap

BASE = "B0_five_seed_mean"
NMS = tuple(f"NMS_{v:.1f}" for v in NMS_THRESHOLDS)
METHODS = (BASE, "pool100", *NMS, "weighted_kmedoids20", "top20")
FIELDS = ("minADE", "minFDE", "J")


def status(phase, **payload):
    write(OUT/"CURRENT.json", dict(phase=phase, at=datetime.now().astimezone().isoformat(),
                                   pid=os.getpid(), **payload))


def quality(distance, selected=None):
    ade, fde = distance.mean(2), distance[:, :, -1]
    if selected is not None:
        ade, fde = (np.take_along_axis(v, selected, axis=1) for v in (ade, fde))
    return np.column_stack((ade.min(1), fde.min(1), ((ade<=.2)&(fde<=.4)).any(1)))


def preflight():
    rng = np.random.default_rng(20261121)
    points = rng.normal(size=(4, 100, 2))
    points[:, 5] = points[:, 0]
    dist = np.linalg.norm(points[:, :, None]-points[:, None], axis=-1)
    w = rng.random((4, 100))
    w /= w.sum(1, keepdims=True)
    for s in NMS_THRESHOLDS:
        a = nms(dist,w,s)
        expected = []
        for d,prob in zip(dist,w):
            order = np.argsort(-prob, kind="stable")
            active = []
            for j in order:
                if not active or (d[j,active]>=s).all():
                    active.append(j)
                if len(active)==20:
                    break
            active.extend(j for j in order if j not in active)
            expected.append(active[:20])
        np.testing.assert_array_equal(a,expected)
    result, initial, final, pending = medoids(dist,w)
    np.testing.assert_array_equal(result,medoids(dist,w)[0])
    np.testing.assert_array_equal(medoids(np.zeros_like(dist),w)[0],
                                  medoids(np.zeros_like(dist),w)[0])
    assert (final<=initial+1e-12).all() and pending == 0
    return dict(NMS_matches_reference=True, kmedoids_deterministic=True, duplicate_endpoints_checked=True)


def evaluate(split, d, files):
    arrays = [old_paths(f"E1C_E64_s{seed}",split,d,files) for seed in SEEDS]
    p = np.concatenate([a for a,w in arrays],axis=1)
    w = np.concatenate([w/len(SEEDS) for a,w in arrays],axis=1)
    np.testing.assert_allclose(w.sum(1),1.,atol=1e-14,rtol=0)
    n = len(p)
    output = {name:np.empty((n,len(FIELDS))) for name in METHODS}
    references = np.empty((n,len(SEEDS),len(FIELDS)))
    selections = {name:np.empty((n,20),np.uint8) for name in (*NMS,"weighted_kmedoids20","top20")}
    cap = 0
    objectives = np.empty((n,2))
    for start in range(0,n,128):
        loc = slice(start,start+128)
        batch = p[loc]
        errors = np.linalg.norm(batch-d["target_rel"][loc,None],axis=-1)
        base = np.stack([quality(errors[:,s:s+20]) for s in range(0,100,20)],axis=1)
        references[loc] = base
        output[BASE][loc] = base.mean(1)
        output["pool100"][loc] = quality(errors)
        end = batch[:, :, -1]
        distances = np.linalg.norm(end[:, :, None]-end[:, None],axis=-1)
        chosen = {name:nms(distances,w[loc],value) for name,value in zip(NMS,NMS_THRESHOLDS)}
        m, initial, final, pending = medoids(distances,w[loc])
        cap += pending
        objectives[loc] = np.column_stack((initial,final))
        chosen.update(weighted_kmedoids20=m,top20=np.argsort(-w[loc],axis=1,kind="stable")[:,:20])
        for key,index in chosen.items():
            selections[key][loc] = index
            output[key][loc] = quality(errors,index)
            assert np.all(output["pool100"][loc,:2] <= output[key][loc,:2]+1e-12)
        if start%1024==0:
            status("RUNNING",split=split,windows_done=start,windows_total=n)
    return output, selections, references, objectives, cap


def summarize(table, d):
    groups = {"overall":np.ones(len(d["ids"]),bool),
              "hard":np.isin(d["label"],("turn","turn_and_speed")),
              **{k:d["obs"]==k for k in ("stationary","steady","other")}}
    out = {}
    for scope,keep in (("all",groups["overall"]),("without_zara02",d["scene"]!="crowds_zara02")):
        rows = {}
        for g,condition in groups.items():
            mask = keep&condition
            if not mask.any():
                continue
            helper = TrackBootstrap(d["track"][mask])
            unit = np.ones(mask.sum())
            baseline = table[BASE][mask]
            pairs = {}
            for name,v in table.items():
                for i,k in enumerate(FIELDS):
                    before,after = baseline[:,i],v[mask,i]
                    pairs[name,k,"value"] = after,unit
                    pairs[name,k,"delta"] = after-before,unit
                    if k!="J":
                        pairs[name,k,"gain"] = before-after,before
            rows[g] = dict(population=helper.population,metrics={})
            for (name,k,stat), value in helper.summarize(pairs).items():
                rows[g]["metrics"].setdefault(name,{}).setdefault(k,{})[stat] = value
        out[scope] = rows
    return out


def main():
    OUT.mkdir(exist_ok=True)
    done = OUT/"DONE.json"
    if done.exists():
        assert all(sha(OUT/k)==v for k,v in read(done)["files"].items())
        return
    began = time.monotonic()
    status("PREFLIGHT")
    checks = preflight()
    files = {}
    data, _, _ = prepare_data(files)
    summaries, convergence, clips = {},{},{}
    for split, d in data.items():
        table, selections, individual, objectives, cap = evaluate(split,d,files)
        summaries[split] = summarize(table,d)
        convergence[split] = dict(initial_mean_objective_m=float(objectives[:,0].mean()),
                                  final_mean_objective_m=float(objectives[:,1].mean()),hit_iteration_cap=cap)
        clips[split] = [float(table[name][:,1].mean()) for name in NMS]
        target = OUT/f"windows_{split}.npz"
        temp = target.with_suffix(".tmp")
        with temp.open("wb") as stream:
            np.savez_compressed(stream, ids=d["ids"], metric_names=FIELDS, seed_metrics=individual,
                seeds=np.array(SEEDS), medoid_objective=objectives, tracks=d["track"], scene=d["scene"],
                observation=d["obs"], maneuver=d["label"], **table,
                **{f"{key}_indices":v for key,v in selections.items()})
        temp.replace(target)
    chosen = NMS[int(np.argmin(clips["fit"]))]
    dev = summaries["dev"]["all"]["overall"]["metrics"]
    names = (chosen,"weighted_kmedoids20")
    meets = [k for k in names if dev[k]["minFDE"]["gain"]["estimate"] >= .015]
    result = dict(status="COMPLETE", elapsed_seconds=round(time.monotonic()-began,2),
                  NMS_chosen_fit_only=chosen, NMS_fit_and_dev_curve_m=clips,
                  pool_seed_count=len(SEEDS), seed_list=list(SEEDS), metrics=summaries,
                  meets_dev_1_5_percent=meets, medoid_convergence=convergence, preflight=checks,
                  definitions=dict(baseline="Within-window arithmetic mean of five B0 seeds",
                                   inference="Selection only uses endpoints and probabilities",
                                   scope="Five-model ensemble diagnostic on C0, with fixed K20 comparisons"),
                  files=files)
    write(OUT/"RESULT.json",result)
    verdict = "至少一种压缩方法达到 1.5% 门槛。" if meets else "两种预设压缩方法均未达到 1.5% 门槛。"
    lines = ["# EXP-POOL-POTENTIAL", "", verdict, "",
             "| C0 dev 方案 | minADE (m) | minFDE (m) | minFDE 改善 |",
             "|---|---:|---:|---:|"]
    for key in (BASE,"pool100",*names,"top20"):
        metrics = dev[key]
        lines.append(f"| {key} | {metrics['minADE']['value']['estimate']:.5f} | "
                     f"{metrics['minFDE']['value']['estimate']:.5f} | {metrics['minFDE']['gain']['estimate']:+.2%} |")
    lines += ["", "NMS 距离阈值由 fit 的 minFDE 选择。dev 汇总包括轨迹配对区间、转向难组和去掉 zara02 的结果。",
              "预测池来自五个独立模型，结果用于衡量集成池的潜力。", "", "[逐组结果](RESULT.json)", ""]
    (OUT/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")
    sealed = ("RESULT.json","REPORT.md","windows_fit.npz","windows_dev.npz")
    write(done,dict(status="PASS",files={k:sha(OUT/k) for k in sealed},
                    source_sha256={k:sha(ROOT/k) for k in ("run_pool.py","pool_select.py")}))
    status("DONE")


if __name__=="__main__":
    try:
        main()
    except Exception:
        status("FAILED",traceback=traceback.format_exc())
        raise

