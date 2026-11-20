"""Verify all ES decompositions, group rankings and saved decision gates."""

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT/"snapshot"
ES = ("path_weighted","path_equal","endpoint_weighted","endpoint_equal")
SCOPES = ("all_dev","without_zara02")


def read(p):
    return json.loads(p.read_text("utf-8-sig"))


def ranks(values):
    _, ix, counts = np.unique(values,return_inverse=True,return_counts=True)
    return (np.cumsum(counts)-.5*counts+.5)[ix]


def gates(levels,prefs,curves,scoped):
    seed_gates = {}
    for seed,v in levels.items():
        table = prefs[seed]["coarse"]["path_weighted"]
        fit = curves[seed]["fit"]
        for g in ("stationary","steady","other"):
            for key, saved in (("minFDE",v["L1_coarse"]["selection"][g]),("minADE",v["L1_coarse_ADE"]["selection"][g])):
                choice = min(fit,key=lambda a:fit[a]["all"][g][key])
                assert float(choice)==saved
        for variant in ES:
            for mode, prefix in (("coarse",""),("fine","bin_")):
                entry = prefs[seed][mode][variant]
                for g,setting in entry["alpha_ES"].items():
                    a = min(fit,key=lambda s:fit[s]["all"][prefix+g][variant+"_ES"])
                    assert float(a)==setting
                assert entry["alpha_FDE"]==v["L1_"+mode]["selection"]
        l1 = v["L1_coarse"]["metrics"]
        l2 = v["L2_future"]["metrics"]
        aniso = v["L1_anisotropic"]["versus_L1_coarse"]

        def pass_gain(result,g,limit):
            target = result["all_dev"][g]
            return (target["relative_FDE_gain"]>=limit and target["delta_minFDE"]["ci95"][1]<0
                    and result["without_zara02"][g]["relative_FDE_gain"]>0)

        a = table["alpha_ES"]
        gain = {scope:{g:table["metrics"][scope][g]["relative_FDE_gain"] for g in ("overall","hard")}
                for scope in SCOPES}
        seed_gates[seed] = dict(A1=pass_gain(l1,"overall",.02) or pass_gain(l1,"hard",.03),
                                A2=pass_gain(aniso,"overall",.01),A3=l2["all_dev"]["overall"]["relative_FDE_gain"]<.02,
                                A3_without_zara02=l2["without_zara02"]["overall"]["relative_FDE_gain"]<.02,
                                E3=a["stationary"]<=.8 and a["other"]-a["stationary"]>=.2-1e-10,
                                E4=gain["all_dev"]["overall"]>=-.01 and gain["all_dev"]["hard"]>=-.02,
                                es_FDE_gain=gain)
    result = {}
    for scope, obj in scoped.items():
        x = obj["overall"]
        test=x["correlations"]["path_weighted"]
        b=x["families"]["B0"]["path_weighted_ES"]
        better=[name for name,v in x["families"].items() if b["mean"]-v["path_weighted_ES"]["mean"]>2*b["sample_sd"]]
        e4 = {g:float(np.mean([v["es_FDE_gain"][scope][g] for v in seed_gates.values()])) for g in ("overall","hard")}
        result[scope]=dict(E1=test["family_spearman"]<.8 or test["reversed_fraction"]>=.2,E2=bool(better),
                           E2_families=better,E4=e4["overall"]>=-.01 and e4["hard"]>=-.02,E4_mean_FDE_gains=e4)
    flags={k:sum(v[k] for v in seed_gates.values())>=4 for k in ("A1","A2","A3","E3")}
    flags.update({k:result["all_dev"][k] for k in ("E1","E2","E4")})
    return dict(flags=flags,per_seed=seed_gates,ranking_and_E4_by_scope=result,
                A3_without_zara02=sum(v["A3_without_zara02"] for v in seed_gates.values())>=4)


def main():
    manifest=read(ROOT/"ARTIFACT_MANIFEST.json")["files"]
    assert set(manifest)=={p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*")
                           if p.is_file() and p.name!="ARTIFACT_MANIFEST.json"}
    for name,h in manifest.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
    ranking=read(SNAPSHOT/"P2_ranking.json")
    count=checks=0
    for name,run in ranking["runs"].items():
        with np.load(SNAPSHOT/f"P2_windows_{name}.npz",allow_pickle=False) as data:
            v={k:data[k] for k in data.files}
        assert len(v["ids"])==4753==len(np.unique(v["ids"]))
        assert all(x.shape==(4753,) for x in v.values())
        for version in ES:
            np.testing.assert_allclose(v[version+"_A"]-v[version+"_S"],v[version+"_ES"],rtol=0,atol=1e-12)
        obs=v["observation"]
        grp={"overall":np.ones(len(obs),bool),**{g:obs==g for g in ("stationary","steady","other")},
             **{f"bin_{g}":v["reference_bin"]==g for g in np.unique(v["reference_bin"])},
             "hard":np.isin(v["maneuver"],("turn","turn_and_speed")),
             "remain":(obs=="stationary")&(v["maneuver"]=="stationary"),
             "start":(obs=="stationary")&(v["maneuver"]=="start")}
        for scope, rows in run["groups"].items():
            for g,row in rows.items():
                m=grp[g] if scope=="all_dev" else grp[g]&(v["scene"]!="crowds_zara02")
                assert row["population"]["windows"]==int(m.sum())
                assert row["population"]["tracks"]==len(np.unique(v["track_id"][m]))
                for k,x in row.items():
                    if k!="population":
                        y=np.median(v[k][m]) if k=="spread" else np.mean(v[k][m])
                        np.testing.assert_allclose(y,x["estimate"],rtol=0,atol=1e-12)
                checks+=1
        count+=len(v["ids"])
    for scope,rows in ranking["family_comparisons"].items():
        for g,x in rows.items():
            family=x["families"]
            for variant in ES:
                m=variant+"_ES"
                for name,record in family.items():
                    group=[r["groups"][scope][g][m]["estimate"] for r in ranking["runs"].values() if r["model_family"]==name]
                    np.testing.assert_allclose(np.mean(group),record[m]["mean"],atol=1e-12,rtol=0)
                xy=[(x[m]["mean"],x["minFDE"]["mean"]) for x in family.values()]
                a,b=zip(*xy)
                t=x["correlations"][variant]
                products=[(a-c)*(b-d) for (a,b),(c,d) in itertools.combinations(xy,2)]
                observed=np.corrcoef(ranks(a),ranks(b))[0,1]
                np.testing.assert_allclose(observed,t["family_spearman"],atol=1e-12,rtol=0)
                p=np.asarray(products)
                np.testing.assert_allclose((p<0).sum()/(p!=0).sum(),t["reversed_fraction"],atol=1e-12,rtol=0)
    levels,prefs,curves=(read(SNAPSHOT/n) for n in ("P1_levels.json","P2_es_alpha.json","P1_alpha_curves.json"))
    assert gates(levels,prefs,curves,ranking["family_comparisons"])==read(SNAPSHOT/"RESULT.json")
    print(json.dumps(dict(status="PASS",window_records=count,group_summaries=checks,model_instances=len(ranking["runs"]),
                          model_families=len(ranking["family_comparisons"]["all_dev"]["overall"]["families"]),
                          fit_seed_plans=len(levels),checked_markers=7)))


if __name__=="__main__":
    main()
