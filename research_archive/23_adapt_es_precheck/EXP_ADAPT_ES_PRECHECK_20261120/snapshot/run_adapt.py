"""Run P1 and P2 from sealed predictions after the separate scale check."""
from datetime import datetime
import itertools
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from scipy.stats import spearmanr

from adapt_data import (ROOT, CAL, OLD, PROBE, SEEDS, LEGACY, PROBE_ARMS, OBS, BINS,
                        arrays, groups, old_paths, prepare_data, probe_paths, probe_target,
                        quantizer, read, sha, split_halves, verify, write)
from calib_metrics import SEED, DRAW_COUNT
from adapt_math import ANISO, GRID, ES_NAMES, Summary, check_energy, energy, errors, hpd, scatter

LABELS=("PREFLIGHT.json","P1_alpha_curves.json","P1_levels.json","P2_ranking.json","P2_es_alpha.json",
        "RESULT.json","REPORT.md")


def status(phase,**kw):
    write(ROOT/"CURRENT.json",dict(at=datetime.now().astimezone().isoformat(),pid=os.getpid(),phase=phase,**kw))


def save_npz(path,**data):
    temp=path.with_suffix(".tmp")
    with temp.open("wb") as f:
        np.savez_compressed(f,**data)
    temp.replace(path)


def scaled(paths,alpha):
    return paths*alpha[:,None,None,None]


def fields(es,a,s):
    return {f"{name}_{stat}":values[:,i] for i,name in enumerate(ES_NAMES)
            for stat,values in (("ES",es),("A",a),("S",s))}


def grid_measure(paths, y, w):
    base_s=scatter(paths,w)
    out={}
    for alpha in GRID:
        x=paths*alpha
        out[float(alpha)]=dict(**errors(x,y),**fields(*energy(x,y,w,base_s*alpha)))
    return out


def select(grid, labels, metric):
    out={}
    for label in sorted(set(labels)):
        m=labels==label
        out[label]=float(min(grid,key=lambda a:grid[a][metric][m].mean()))
    return out


def apply_scale(choices,labels):
    return np.asarray([choices[k] for k in labels],dtype=np.float64)


def fit_aniso(path,fit):
    out={}
    r=fit["rotation"]
    e=np.einsum("nij,nki->nkj",r,path[:,:,-1])
    y=np.einsum("nij,ni->nj",r,fit["target_rel"][:,-1])
    for name in OBS:
        group=fit["obs"]==name
        tx,ty=e[group].transpose(2,0,1)
        px,py=y[group].T
        dx=[np.square(scale*tx-px[:,None]) for scale in ANISO]
        dy=[np.square(scale*ty-py[:,None]) for scale in ANISO]
        i,j=min(itertools.product(range(len(ANISO)),repeat=2),
                key=lambda k:np.sqrt((dx[k[0]]+dy[k[1]]).min(1)).mean())
        out[name]=[float(ANISO[i]),float(ANISO[j])]
    return out


def eval_aniso(path,dev,choices):
    x=np.einsum("nij,nkti->nktj",dev["rotation"],path)
    m=np.asarray([choices[k] for k in dev["obs"]])
    return np.einsum("nij,nktj->nkti",dev["rotation"],x*m[:,None,None])


def curves(grid,data):
    table={}
    mask=groups(data)
    for a,v in grid.items():
        row={}
        for scope,keep in (("all",np.ones(len(data["ids"]),bool)),("without_zara02",data["scene"]!="crowds_zara02")):
            row[scope]={}
            for g,group in mask.items():
                m=group&keep
                if m.any():
                    row[scope][g]=dict(n=int(m.sum()),**{k:float(v[k][m].mean())
                                            for k in ("minFDE","minADE","J",*(f"{e}_ES" for e in ES_NAMES))})
        table[str(a)]=row
    return table


def write_ranking(name,path,y,w,dev,summary,family,fit_available):
    result=dict(**errors(path,y),**fields(*energy(path,y,w)))
    output=ROOT/f"P2_windows_{name}.npz"
    save_npz(output,**result,ids=dev["ids"],track_id=dev["track"],observation=dev["obs"],
             maneuver=dev["label"],reference_bin=dev["bin"],scene=dev["scene"])
    return dict(model_family=family,K=path.shape[1],fit_forecasts_available=fit_available,
                groups=summary.table(result)),output.name


def b0(seed,data,archive,files,summary,half,draws):
    fit,dev=data["fit"],data["dev"]
    name=f"E1C_E64_s{seed}"
    pf,wf=old_paths(name,"fit",fit,files)
    pd,wd=old_paths(name,"dev",dev,files)
    yf,yd=fit["target_rel"],dev["target_rel"]
    fit_grid=grid_measure(pf,yf,wf)
    dev_grid=grid_measure(pd,yd,wd)
    widths=archive["runs"]["B0"][str(seed)]["t12"]["hpd"]["equal"]["fitted_on_other_half"]
    h=np.asarray([widths[k]["global"]["chosen"] for k in "AB"])[half]
    h0=hpd(pd,yd,h,draws)
    sealed=CAL/f"windows/B0_s{seed}.npz"
    verify(sealed,read(CAL/"DONE.json")["files"][f"windows/B0_s{seed}.npz"],files)
    a=arrays(sealed)
    np.testing.assert_array_equal(a["ids"],dev["ids"])
    np.testing.assert_array_equal(h0,a["t12__equal__u__global"]<=.9)
    for k in ("minADE","minFDE"):
        np.testing.assert_allclose(dev_grid[1.][k],a[f"t12__{k}"],atol=1e-12,rtol=0)
    base=dict(**dev_grid[1.],HPD90=h0)
    prepared={np.ones(len(pd),dtype=np.float64).tobytes():base}
    pd_s=scatter(pd,wd)

    def evaluate(alpha):
        key=alpha.tobytes()
        if key not in prepared:
            x=scaled(pd,alpha)
            prepared[key]=dict(**errors(x,yd),**fields(*energy(x,yd,wd,pd_s*alpha[:,None])),
                               HPD90=hpd(x,yd,h,draws))
        return prepared[key]

    l1=select(fit_grid,fit["obs"],"minFDE")
    l1a=select(fit_grid,fit["obs"],"minADE")
    l1fine=select(fit_grid,fit["bin"],"minFDE")
    l2=select(fit_grid,fit["label"],"minFDE")
    v3=GRID[np.stack([dev_grid[float(a)]["minFDE"] for a in GRID],axis=1).argmin(1)]
    l3=dict(zip(GRID,[int((v3==a).sum()) for a in GRID]))
    aniso=fit_aniso(pf,fit)
    px=eval_aniso(pd,dev,aniso)
    x_aniso=dict(**errors(px,yd),**fields(*energy(px,yd,wd)),HPD90=hpd(px,yd,h,draws))
    levels={}
    base_p1={key:base[key] for key in ("minADE","minFDE","J","spread","HPD90")}
    l1_values=evaluate(apply_scale(l1,dev["obs"]))
    for label,choice,value in (
            ("L1_coarse",l1,l1_values),
            ("L1_fine",l1fine,evaluate(apply_scale(l1fine,dev["bin"]))),
            ("L1_coarse_ADE",l1a,evaluate(apply_scale(l1a,dev["obs"]))),
            ("L2_future",l2,evaluate(apply_scale(l2,dev["label"]))),
            ("L3_window_oracle",{str(k):v for k,v in l3.items()},evaluate(v3)),
            ("L1_anisotropic",aniso,x_aniso)):
        levels[label]=dict(selection=choice,metrics=summary.table({k:value[k] for k in base_p1},base_p1))
    levels["L1_anisotropic"]["versus_L1_coarse"]=summary.table({"minFDE":x_aniso["minFDE"]},{"minFDE":l1_values["minFDE"]})
    prefs={}
    for bins,labels in (("coarse","obs"),("fine","bin")):
        p={}
        for es in ES_NAMES:
            choice=select(fit_grid,fit[labels],es+"_ES")
            out=evaluate(apply_scale(choice,dev[labels]))
            subset=["minADE","minFDE","J","HPD90","spread",*(es+"_"+k for k in ("ES","A","S"))]
            p[es]=dict(alpha_ES=choice,alpha_FDE=(l1 if bins=="coarse" else l1fine),
                       metrics=summary.table({k:out[k] for k in subset},{k:base[k] for k in subset}))
        prefs[bins]=p
    ranking,out=write_ranking(f"B0_s{seed}",pd,yd,wd,dev,summary,"B0",True)
    return (dict(fit=curves(fit_grid,fit),dev=curves(dev_grid,dev)),levels,prefs,ranking,out)


def rank_test(runs):
    result={}
    present=next(iter(runs.values()))["groups"]
    for scope in ("all_dev","without_zara02"):
        group_rows={}
        for group in present[scope]:
            by_family={}
            rows=[]
            for name,item in runs.items():
                table=item["groups"][scope][group]
                vals={k:float(v["estimate"]) for k,v in table.items() if k!="population"}
                by_family.setdefault(item["model_family"],[]).append(vals)
                rows.append(vals)
            family={key:{m:dict(mean=float(np.mean([row[m] for row in e])),
                                 sample_sd=float(np.std([row[m] for row in e],ddof=1)) if len(e)>1 else None)
                         for m in e[0]} for key,e in by_family.items()}
            tests={}
            for es in ES_NAMES:
                k=es+"_ES"
                xy=[(v[k]["mean"],v["minFDE"]["mean"]) for v in family.values()]
                pairs=[(a-c)*(b-d) for (a,b),(c,d) in itertools.combinations(xy,2)]
                nonzero=[p for p in pairs if p]
                tests[es]=dict(family_spearman=float(spearmanr(*zip(*xy)).statistic),
                               instance_spearman=float(spearmanr([a[k] for a in rows],[a["minFDE"] for a in rows]).statistic),
                               reversed_fraction=float(np.mean(np.asarray(nonzero)<0)),
                               comparable_family_pairs=len(nonzero),exact_ties=len(pairs)-len(nonzero))
            group_rows[group]=dict(families=family,correlations=tests)
        result[scope]=group_rows
    return result


def gate_test(levels,ranks,prefs):
    outcomes={}
    for s,level in levels.items():
        def test(table,g,threshold):
            a,b=[table[scope][g] for scope in ("all_dev","without_zara02")]
            return a["relative_FDE_gain"]>=threshold and a["delta_minFDE"]["ci95"][1]<0 and b["relative_FDE_gain"]>0
        l1=level["L1_coarse"]["metrics"]
        an=level["L1_anisotropic"]["versus_L1_coarse"]
        l2=level["L2_future"]["metrics"]
        e=prefs[s]["coarse"]["path_weighted"]
        a=e["alpha_ES"]
        fde={scope:{g:e["metrics"][scope][g]["relative_FDE_gain"] for g in ("overall","hard")}
             for scope in ("all_dev","without_zara02")}
        outcomes[s]=dict(A1=bool(test(l1,"overall",.02) or test(l1,"hard",.03)),
                         A2=bool(test(an,"overall",.01)),A3=l2["all_dev"]["overall"]["relative_FDE_gain"]<.02,
                         A3_without_zara02=l2["without_zara02"]["overall"]["relative_FDE_gain"]<.02,
                         E3=a["stationary"]<=.8 and a["other"]>=a["stationary"]+.2-1e-10,
                         E4=fde["all_dev"]["overall"]>=-.01 and fde["all_dev"]["hard"]>=-.02,
                         es_FDE_gain=fde)
    by_scope={}
    for scope,rows in ranks.items():
        table=rows["overall"]
        k="path_weighted_ES"
        test=table["correlations"]["path_weighted"]
        base=table["families"]["B0"][k]
        over=[name for name,values in table["families"].items() if name!="B0" and
              base["mean"]-values[k]["mean"]>2*base["sample_sd"]]
        comp={g:float(np.mean([v["es_FDE_gain"][scope][g] for v in outcomes.values()])) for g in ("overall","hard")}
        by_scope[scope]=dict(E1=test["family_spearman"]<.8 or test["reversed_fraction"]>=.2,E2=bool(over),
                             E2_families=over,E4=comp["overall"]>=-.01 and comp["hard"]>=-.02,E4_mean_FDE_gains=comp)
    flags={k:sum(v[k] for v in outcomes.values())>=4 for k in ("A1","A2","A3","E3")}
    flags.update({k:by_scope["all_dev"][k] for k in ("E1","E2","E4")})
    return dict(flags=flags,per_seed=outcomes,ranking_and_E4_by_scope=by_scope,
                A3_without_zara02=sum(v["A3_without_zara02"] for v in outcomes.values())>=4)


def render(result,levels,prefs,ranks):
    passing=[k for k,v in result["flags"].items() if v]
    lines=["# EXP-ADAPT-ES-PRECHECK","",
           "达到预设门槛的标记为 "+("、".join(passing) or "无")+"。","",
           "## P1 按情境调整宽度","",
           "| seed | 粗分组 α | L1 总体 FDE 改善 | 难组改善 | 去掉 zara02 后总体改善 | L2 总体改善 | L3 总体改善 |",
           "|---:|---|---:|---:|---:|---:|---:|"]
    for seed,v in levels.items():
        q=v["L1_coarse"]
        gain=lambda l,scope,g: v[l]["metrics"][scope][g]["relative_FDE_gain"]
        lines.append(f"| {seed} | {q['selection']} | {gain('L1_coarse','all_dev','overall'):+.2%} | "
                     f"{gain('L1_coarse','all_dev','hard'):+.2%} | {gain('L1_coarse','without_zara02','overall'):+.2%} | "
                     f"{gain('L2_future','all_dev','overall'):+.2%} | {gain('L3_window_oracle','all_dev','overall'):+.2%} |")
    lines+=["","所有分组、配对轨迹区间和各向异性结果见 P1_levels.json。L2 使用未来动作标签，L3 直接用 dev 真值逐窗选择。",
            "", "## P2 Energy score",""]
    for scope in ("all_dev","without_zara02"):
        g=ranks[scope]["overall"]["correlations"]["path_weighted"]
        lines.append(f"- {scope} 的模型族 Spearman 为 {g['family_spearman']:.3f}，排序相反占 {g['reversed_fraction']:.1%}。")
    lines+=["","| seed | 粗分组 α_ES | 总体 FDE 改善 | 难组 FDE 改善 |",
            "|---:|---|---:|---:|"]
    for s,v in prefs.items():
        a=v["coarse"]["path_weighted"]
        m=a["metrics"]["all_dev"]
        lines.append(f"| {s} | {a['alpha_ES']} | {m['overall']['relative_FDE_gain']:+.2%} | {m['hard']['relative_FDE_gain']:+.2%} |")
    lines+=["","完整 ES 排序、A/S 分解及敏感性分析见 P2_ranking.json 和 P2_es_alpha.json。"
            "缩放时概率固定。除 L2 的未来标签与 L3 的单窗上限外，推理所需分组仅看观测。"
            "主要 ES 使用展平 24 维轨迹和米单位。本诊断限于重复使用的 C0 dev 和固定候选的缩放。"
            "结论不足以替代重新训练与独立测试。"]
    return "\n".join(lines)+"\n"


def main():
    start=time.monotonic()
    if (ROOT/"DONE.json").exists():
        assert all(sha(ROOT/p)==h for p,h in read(ROOT/"DONE.json")["files"].items())
        return
    status("CPU_PREFLIGHT")
    scale_path=ROOT/"SCALE_EQUIVALENCE.json"
    scale_check=read(scale_path)
    assert scale_check["status"]=="PASS" and scale_check["n"]==200 and max(scale_check["errors_m"].values())<=1e-4
    files={}
    for p,h in scale_check["files"].items():
        verify(Path(p),h,files)
    for p in (*ROOT.glob("*.py"),ROOT/"PROTOCOL.md",ROOT/"DESIGN.md",*ROOT.glob("standalone/*.ps1")):
        files[str(p)]=sha(p)
    data,archive,cuts=prepare_data(files)
    dev=data["dev"]
    energy_check=check_energy()
    kp,kw,qcheck=quantizer(data,archive,files)
    cached_y=probe_target(dev,files)
    half,half_pop=split_halves(dev["ids"])
    rng=np.random.default_rng(SEED)
    shape=(len(half),DRAW_COUNT)
    draws=(rng.random(shape),rng.standard_normal((*shape,2)))
    summary=Summary(groups(dev),dev["track"],dev["scene"]!="crowds_zara02")
    pre=dict(scale=scale_check,quantizer=qcheck,energy_check=energy_check,cuts=cuts,HPD_halves=half_pop)
    write(ROOT/"PREFLIGHT.json",dict(**pre,files=files))
    ranking={}
    windows=[]
    curves_by_seed,levels,prefs={},{},{}
    for s in SEEDS:
        status("B0_WIDTH_GRID",seed=s,completed=len(levels),total=len(SEEDS))
        c,l,p,r,out=b0(s,data,archive,files,summary,half,draws)
        curves_by_seed[str(s)],levels[str(s)],prefs[str(s)]=c,l,p
        ranking[f"B0_s{s}"]=r
        windows.append(out)
        print(f"B0 {s} P1/P2 finished",flush=True)
        for key,obj in (("P1_alpha_curves",curves_by_seed),("P1_levels",levels),("P2_es_alpha",prefs)):
            write(ROOT/(key+".json"),obj)
    modes=[(arm,s) for arm in LEGACY for s in (42,43,44)]+[(arm,s) for arm in PROBE_ARMS for s in (0,1,2)]
    for i,(arm,seed) in enumerate(modes):
        name=f"{arm}_s{seed}"
        status("P2_RANKING",run=name,completed=i,total=len(modes)+2)
        if arm in LEGACY:
            p,w=old_paths(name,"dev",dev,files)
            fit_present=(OLD/"runs"/name/"fit/predictions.npz").exists()
            y=dev["target_rel"]
        else:
            p,w=probe_paths(arm,seed,dev,cached_y,files)
            fit_present=(PROBE/"runs"/name/"fit_predictions.npz").exists()
            y=cached_y
        ranking[name],out=write_ranking(name,p,y,w,dev,summary,arm,fit_present)
        windows.append(out)
    for name,p,w in (("K-means",kp,kw),("CV",(dev["cv"]-dev["last"])[:,None],np.ones((len(dev["ids"]),1)))):
        ranking[name],out=write_ranking(name,p,dev["target_rel"],w,dev,summary,name,True)
        windows.append(out)
    families=rank_test(ranking)
    result=gate_test(levels,families,prefs)
    write(ROOT/"P2_ranking.json",dict(runs=ranking,family_comparisons=families))
    write(ROOT/"PREFLIGHT.json",dict(**pre,files=files))
    write(ROOT/"RESULT.json",result)
    (ROOT/"REPORT.md").write_text(render(result,levels,prefs,families),encoding="utf-8")
    output_files=[*LABELS,*windows,"SCALE_EQUIVALENCE.json"]
    write(ROOT/"DONE.json",dict(status="SUCCESS",elapsed_s=time.monotonic()-start,
                               files={k:sha(ROOT/k) for k in output_files}))
    status("DONE",B0_seeds=5,ranked_instances=len(ranking))


if __name__=="__main__":
    try:
        main()
    except Exception:
        status("FAILED",traceback=traceback.format_exc())
        raise
