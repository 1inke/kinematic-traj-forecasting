"""Apply the locked C0 and five-fold gates to verified per-window metrics."""
import json
from pathlib import Path

import numpy as np

from prepare_inputs import digest
from qrank_model import FORMAL
from run_b0 import write_json


FOLDS = ("eth","zara01","hotel","zara02","univ")
SEEDS = (42,43,44)


def read(path):
    return json.loads(path.read_bytes())


def run(root,stage,arm,fold,seed):
    name = f"{arm}_s{seed}" if fold=="C0" else f"{arm}_{fold}_s{seed}"
    folder = root/stage/"runs"/name
    if stage=="stage2" and not (folder/"RESULT.json").is_file():
        return root/stage/"reused"/name
    return folder


def evidence(paths):
    arrays, results = [],[]
    for folder in paths:
        r = read(folder/"RESULT.json")
        p = folder/"dev_metrics.npz"
        assert r["status"]=="PASS" and digest(p)==r["artifacts"][p.name],folder
        with np.load(p,allow_pickle=False) as z:
            arrays.append({k:z[k] for k in z.files})
        results.append(r)
    return arrays,results


def vector(rows,metric):
    return np.asarray([a[metric].astype("float64") for a in rows])


def gain(a,b):
    denominator = float(np.mean(b))
    return float((np.mean(b)-np.mean(a))/max(denominator,1e-12))


def paired(a,b,units,level=.95):
    _,ix = np.unique(units,return_inverse=True)
    s = np.bincount(ix,b-a)
    d = np.bincount(ix,b)
    ind = np.random.default_rng(20261122).integers(0,len(s),size=(2000,len(s)))
    res = s[ind].sum(1)/np.maximum(d[ind].sum(1),1e-12)
    tail = (1-level)/2
    return dict(gain=gain(a,b),ci=np.quantile(res,[tail,1-tail]).tolist(),level=level,
                windows=len(a),units=len(s))


def short_compare(a,b,metric,group="overall"):
    mask = b[0][group]
    ids = b[0]["ids"]
    assert all(np.array_equal(ids,r["ids"]) and np.array_equal(mask,r[group]) for r in (*a,*b))
    av,bv = vector(a,metric)[:,mask],vector(b,metric)[:,mask]
    units = np.array([x.rsplit("|",1)[0] for x in ids]) if ids.dtype.kind in "US" else np.arange(len(ids))//15
    val = paired(av.mean(0),bv.mean(0),units[mask])
    val["by_seed"] = [gain(x,y) for x,y in zip(av,bv)]
    return val


def matches(results):
    totals = {}
    for r in results:
        for key, count in r["final_epoch"]["emta_counts"].items():
            totals.setdefault(key,np.zeros(2,dtype="int64"))
            totals[key] += count
    return {k:dict(matched=int(a),windows=int(b),rate=float(a/b) if b else None,
                   fallback=float(1-a/b) if b else None) for k,(a,b) in totals.items()}


def screen(root):
    data,runs = {},{}
    for arm in FORMAL:
        data[arm],runs[arm] = evidence(run(root,"stage1",arm,"C0",seed) for seed in SEEDS)
    ids = data["B0"][0]["ids"]
    tracks = np.array([s.rsplit("|",1)[0] for s in ids])
    recordings = np.array([s.split("|",1)[0] for s in ids])
    assert all(np.array_equal(ids,r["ids"]) for rows in data.values() for r in rows)
    mask = data["B0"][0]["without_zara02"]
    table, selected, mc = {},set(),{}
    random_overlap = np.mean([r["manipulation"]["top5_overlap"] for r in runs["O-rand"]])
    mc["O-rand"] = dict(top5=float(random_overlap),valid=bool(.20<=random_overlap<=.30))
    for arm in ("O-risk","O-qual","O-riskp"):
        stats = [r["manipulation"] for r in runs[arm]]
        overlap = float(np.mean([m["top5_overlap"] for m in stats]))
        first = float(np.mean([m["first_diff"] for m in stats]))
        spearman = {}
        for layer in ("0","3"):
            spearman[layer] = {k:float(np.mean([m["quality_spearman"][layer][k] for m in stats]))
                               for k in ("head","confidence")}
        trained = all(v["head"]>=max(.50,v["confidence"]-.05) for v in spearman.values())
        distinct = (overlap<=.80 and first>=.15) if arm!="O-qual" else True
        valid = mc["O-rand"]["valid"] and distinct and (trained or arm=="O-riskp")
        mc[arm] = dict(overlap=overlap,first_changed=first,spearman=spearman,head_valid=trained,
                       distinct=distinct,near_conf=arm=="O-qual" and overlap>.90)
        tests = {}
        primary = arm=="O-risk"
        for peer,threshold in (("O-conf",.015),("O-rand",.01),("D3-full",.01)):
            a,b = (vector(data[x],"minFDE") for x in (arm,peer))
            full = paired(a.mean(0),b.mean(0),tracks,1-.05/3 if primary else .95)
            full["by_seed"] = [gain(x,y) for x,y in zip(a,b)]
            full["without_zara02"] = paired(a.mean(0)[mask],b.mean(0)[mask],tracks[mask])
            full["leave_one_recording"] = {rec:gain(a[:,recordings!=rec],b[:,recordings!=rec])
                                          for rec in sorted(set(recordings))}
            full["same_sign_recordings"] = int(sum(np.sign(g)==np.sign(full["gain"])
                                                   for g in full["leave_one_recording"].values()))
            full["pass"] = full["gain"]>=threshold and (peer!="O-conf" or full["ci"][0]>0)
            tests[peer] = full
        brier = short_compare(data[arm],data["O-conf"],"brier_minFDE")
        top = short_compare(data[arm],data["B0"],"Top1_FDE")
        ok = valid and all(v["pass"] for v in tests.values()) and brier["gain"]>=0 and top["gain"]>=-.01
        table[arm] = dict(marker="M-ORDER" if primary else "M-ORDER-2",eligible=valid,passed=ok,
                          tests=tests,brier=brier,top1=top)
        if ok: selected.add(arm)
    for order in ("O-conf","O-risk"):
        for new,ref in ((order+"+E0",order),(order+"+QFL",order+"+E0")):
            match = matches(runs[new])
            valid = .30<=match["overall"]["rate"]<=.95
            checks = {m:short_compare(data[new],data[ref],m) for m in ("minFDE","Top1_FDE","brier_minFDE")}
            checks["stationary_minFDE"] = short_compare(data[new],data[ref],"minFDE","stationary")
            ok = (valid and (checks["Top1_FDE"]["gain"]>=.03 or checks["brier_minFDE"]["gain"]>=.02)
                  and checks["minFDE"]["gain"]>=-.01 and checks["stationary_minFDE"]["gain"]>=-.03)
            table[new] = dict(marker="M-QFL" if new.endswith("QFL") else "M-EMTA",
                              reference=ref,eligible=valid,passed=ok,matches=match,tests=checks)
            if ok: selected.add(new)
    score = {arm:short_compare(data[arm],data["B0"],"brier_minFDE")["gain"] for arm in FORMAL[1:]}
    ranked = sorted(selected,key=lambda a:(-score[a],a))[:2]
    if len(ranked)==1:
        ranked.append("D3-full")
    elif not ranked:
        ranked = ["D3-full",max((x for x in FORMAL[1:] if x!="D3-full"),key=lambda a:(score[a],a))]
    information = dict(depth=short_compare(data["D3-full"],data["B0"],"minFDE"),
                       synergy={m:short_compare(data["O-risk+E0"],data["O-conf+E0"],m)["gain"]-
                                  short_compare(data["O-risk"],data["O-conf"],m)["gain"]
                                for m in ("minFDE","Top1_FDE")},
                       posthoc={a:{str(k):paired(vector(d,f"riskp_minFDE{k}").mean(0),
                                                 vector(d,f"conf_minFDE{k}").mean(0),tracks)
                                  for k in (3,5)} for a,d in data.items()})
    for arm in FORMAL[1:]:
        per_seed = [r["manipulation"]["anchor_move_median_m"] for r in runs[arm]]
        mc.setdefault(arm,{})["MC6"] = dict(per_seed=per_seed,passed=bool((np.mean(per_seed,0)>=.02).all()))
    write_json(root/"STAGE1_RESULT.json",dict(status="COMPLETE",comparisons=table,manipulation=mc,
                                            information=information,arm_brier_vs_B0=score))
    result = dict(status="LOCKED",arms=ranked,mode="CONFIRMATORY" if selected else "EXPLORATORY",
                  qualified=sorted(selected),stage2_seeds={f:list(range(42,47) if f=="eth" else SEEDS) for f in FOLDS})
    write_json(root/"ADMISSION.json",result)
    return result


def confirm(root,admission):
    values = {}
    for fold in FOLDS:
        seeds = admission["stage2_seeds"][fold]
        values[fold] = {arm:evidence(run(root,"stage2",arm,fold,s) for s in seeds)[0]
                        for arm in ("B0",*admission["arms"])}
    output = {}
    for arm in admission["arms"]:
        folds = {fold:{metric:short_compare(values[fold][arm],values[fold]["B0"],metric)
                       for metric in ("minADE","minFDE","Top1_FDE","brier_minFDE")} for fold in FOLDS}
        mean = {k:float(np.mean([f[k]["gain"] for f in folds.values()])) for k in folds["eth"]}
        wins = sum(v["minFDE"]["gain"]>0 for v in folds.values())
        eth = folds["eth"]["minFDE"]
        gates = dict(avg_minFDE=mean["minFDE"]>=.02,fold_wins=wins>=4,eth_guard=eth["gain"]>=-.01,
                     top1_guard=mean["Top1_FDE"]>=-.01)
        result = {"mean":mean,"folds":folds,"fold_wins":wins,"gates":gates,
                  "F-PASS":all(gates.values()) if admission["mode"]=="CONFIRMATORY" else None,
                  "F-ETH":eth["gain"]>=.01 and eth["ci"][0]>0,"unit":"block15"}
        if arm.startswith("O-") and "D3-full" in admission["arms"]:
            g = [short_compare(values[f][arm],values[f]["D3-full"],"minFDE") for f in FOLDS]
            result["F-ORDER"] = np.mean([x["gain"] for x in g])>=.01 and sum(x["gain"]>0 for x in g)>=4
            result["F-ORDER"] = bool(result["F-ORDER"])
        output[arm] = result
    write_json(root/"STAGE2_RESULT.json",dict(status="COMPLETE",mode=admission["mode"],arms=output))
    return output
