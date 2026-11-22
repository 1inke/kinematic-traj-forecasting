"""Evaluate the registered condition and intervention criteria."""
from itertools import product

from bias_inputs import OBS


def away(estimate, direction):
    low, high = estimate["ci95"]
    return low > 0 if direction > 0 else high < 0


def flatten(d):
    return {**{f"{feature}/{name}":row for feature in OBS for name,row in d["feature_bins"][feature].items()},
            **{f"K_joint/{name}":row for name,row in d["K_joint"].items()}}


def conditional_geom(tables, b0, e1b):
    flat = {k:flatten(tables[k]) for k in (*b0,*e1b)}
    by_model = {}
    for key, group in flat.items():
        row = {}
        for name,g in group.items():
            pop = g["population"]
            if pop["windows"] < 150 or pop["tracks"] < 20:
                continue
            for axis in ("lon","lat"):
                v = g["values"][f"nearest_FDE_{axis}_t12"]
                flag = abs(v["estimate"]) >= .08 and away(v,v["estimate"])
                row[f"{name}/{axis}"] = dict(**v,pass_flag=flag,population=pop)
        by_model[key] = row
    confirmed = []
    for name in sorted({n for k in b0 for n in by_model[k]}):
        group,axis = name.rsplit("/",1)
        for direction in (-1,1):
            seeds = [k for k in b0 if (a:=by_model[k].get(name)) and a["pass_flag"] and direction*a["estimate"]>0]
            holdout = {}
            for k in e1b:
                a = flat[k].get(group)
                if a:
                    holdout[k] = dict(**a["values"][f"nearest_FDE_{axis}_t12"],population=a["population"])
            if len(seeds)>=4 and len(holdout)==2 and all(direction*v["estimate"]>0 for v in holdout.values()):
                confirmed.append(dict(cell=name,sign=direction,B0=seeds,E1B=holdout))
    return dict(pass_flag=bool(confirmed),confirmed_cells=confirmed,
                tested_cells_axes={k:len(v) for k,v in by_model.items()},by_model=by_model)


def check_center(tables,b0,e1b):
    contrast = {k:tables[k]["speed_low_minus_high"]["all"] for k in (*b0,*e1b)}
    key = "weighted_center_lon_t12"
    for rows in contrast.values():
        v = rows[key]
        v["pass_magnitude_and_interval"] = abs(v["estimate"]) >= .2 and away(v,v["estimate"])
    signs = {sign:[k for k in b0 if contrast[k][key]["pass_magnitude_and_interval"] and
                   sign*contrast[k][key]["estimate"]>0] for sign in (-1,1)}
    confirms = [sign for sign, names in signs.items() if len(names)>=4 and
                all(sign*contrast[k][key]["estimate"]>0 for k in e1b)]
    return dict(pass_flag=bool(confirms),confirmed_directions=confirms,by_model=contrast)


def geo_entry(observed,group):
    fields = observed["groups"]
    stats = fields["all_dev"][group]["metrics"]
    effect = stats["minFDE_relative_gain"]["estimate"]
    ci = stats["minFDE_delta"]["ci95"]
    noz = fields["without_zara02"][group]["metrics"]["minFDE_relative_gain"]["estimate"]
    share = observed["recordings"][group]["maximum_contribution"]["minFDE"]["share"]
    gate = effect >= (.01 if group=="overall" else .02) and ci[1] < 0 and noz > 0 and share is not None and share <= .5
    return dict(pass_flag=gate,gain=effect,delta_ci95=ci,gain_without_zara02=noz,max_recording_share=share)


def prob_entry(observed,primary):
    row = observed["groups"]["all_dev"]["overall"]["metrics"]
    secondary = "Top1_FDE" if primary=="path_ES" else "path_ES"
    effect = row[f"{primary}_relative_gain"]["estimate"]
    ci = row[f"{primary}_delta"]["ci95"]
    guard = row[f"{secondary}_relative_gain"]["estimate"]
    threshold = .01 if primary=="path_ES" else .03
    return dict(pass_flag=effect>=threshold and ci[1]<0 and guard>=-.01,
                gain=effect,delta_ci95=ci,other_metric=secondary,other_gain=guard)


def correction(effects,b0,e1b,is_geom):
    options = effects[b0[0]]["dev_crossfit"]
    combinations = [(name,tag) for name,tag in product(options,("overall","hard") if is_geom else ("path_ES","Top1_FDE"))
                    if name.split("/",1)[0] in ("K","K+N")]
    passing, fit_only, matrix = [], [], {}
    extract = geo_entry if is_geom else prob_entry
    for case, tag in combinations:
        sources = {"dev_crossfit":b0, "CALIB":e1b, "fit":b0}
        rows = {split:{k:extract(effects[k][split][case],tag) for k in models} for split,models in sources.items()}
        wins = {split:sum(v["pass_flag"] for v in cells.values()) for split,cells in rows.items()}
        positive = {split:sum(v["gain"]>0 for v in cells.values()) for split,cells in rows.items()}
        pass_flag = (wins["dev_crossfit"]>=4 or wins["CALIB"]==2) and positive["dev_crossfit"]>=4 and positive["CALIB"]==2
        label = f"{case}/{tag}"
        matrix[label] = dict(pass_flag=pass_flag,passed_by_source=wins,positive_by_source=positive,by_model=rows)
        if pass_flag:
            passing.append(label)
        if wins["fit"]>=4 and not pass_flag:
            fit_only.append(label)
    return dict(pass_flag=bool(passing),confirmed_configurations=passing,
                fit_passed_without_confirmation=fit_only, by_configuration=matrix)


def evaluate(tables,geometry,probability):
    b0 = [k for k in tables if k.startswith("B0_")]
    e1b = [k for k in tables if k.startswith("E1B_")]
    g1 = conditional_geom(tables,b0,e1b)
    g2 = correction(geometry,b0,e1b,True)
    p1 = check_center(tables,b0,e1b)
    p2 = correction(probability,b0,e1b,False)
    results = {k:{**v["pit"],"pass_flag":v["pit"]["K_joint"]["weighted"] is not None and
                  v["pit"]["K_joint"]["weighted"]>=.30} for k,v in tables.items()}
    p3 = dict(pass_flag=sum(results[k]["pass_flag"] for k in b0)>=4,by_model=results)
    rules = dict(G1=g1,G2=g2,P1=p1,P2=p2,P3=p3)
    return dict(status="COMPLETE",markers=rules,summary={name:v["pass_flag"] for name,v in rules.items()})


def render(result):
    f = result["summary"]
    lines = ["# EXP-COND-BIAS-DIAG", "", "## 结论", ""]
    if f["G2"]:
        lines.append("指定观测特征支持可复现的几何修正。检查通过判据的特征组与修正形式。")
    elif f["P2"]:
        lines.append("概率修正在预设门槛内有收益。几何修正未达到推进标准。")
    elif f["P1"]:
        lines.append("概率中心随观测速度偏移。本次概率修正未通过预设的收益与保护判据。"
                     "报告保留偏差与评分权衡的证据。")
    else:
        lines.append("两类修正均未达到推进标准。证据限于本次使用的特征、修正形式和开发数据。")
    lines += ["", "| 判据 | 结果 |", "|---|---|"]
    for name,pass_flag in f.items():
        lines.append(f"| {name} | {'达标' if pass_flag else '未达标'} |")
    marks = result["markers"]
    lines += ["", "## 速度相关偏差与分布校准", "",
              "低速组减高速组为第 12 步的纵向有符号偏差，单位为米。区间来自同一组轨迹的配对重抽样。",
              "", "| 模型 | 加权中心低速减高速 | 95% 区间 | Top-1 低速减高速 | 加权 PIT 超界窗口比例 |",
              "|---|---:|---:|---:|---:|"]
    for key, row in marks["P1"]["by_model"].items():
        v = row["weighted_center_lon_t12"]
        lo, hi = v["ci95"]
        top = row["top1_lon_t12"]["estimate"]
        ratio = marks["P3"]["by_model"][key]["K_joint"]["weighted"]
        pit = f"{ratio:.1%}" if ratio is not None else "样本不足"
        lines.append(f"| {key} | {v['estimate']:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {top:+.3f} | {pit} |")
    for name in ("G2", "P2"):
        confirmed = marks[name]["confirmed_configurations"]
        if confirmed:
            lines += ["", f"{name} 通过的配置为 " + "、".join(confirmed) + "。"]
    lines += ["", "逐 seed 结果和判据依据见 [RESULT.json](RESULT.json)。"
              "几何和概率修正保留了配对区间与逐录制留一结果。"
              "查看 [geom_correction.json](geom_correction.json) 和 [prob_correction.json](prob_correction.json)。", "",
              "方向无效窗口及按未来动作分组的偏差仅用于描述。"
              "详见 [descriptive_outcome_groups.json](descriptive_outcome_groups.json)。", "",
              "本实验比较固定候选上的线性平移和两参数概率倾斜，结论覆盖指定修正形式。"
              "dev 已反复用于诊断，E1B 只有两个 seed，数据范围为 C0 ETH 训练池。"
              "更多设置见 [PROTOCOL.md](PROTOCOL.md)，核验见 [PREFLIGHT.json](PREFLIGHT.json)。", ""]
    return "\n".join(lines)

