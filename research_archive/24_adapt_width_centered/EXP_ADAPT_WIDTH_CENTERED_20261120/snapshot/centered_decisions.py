"""Evaluate the preregistered flags and render traceable experiment results."""
from itertools import combinations

import numpy as np

from centered_inputs import OBS, SEEDS

ISO = "isotropic_coarse_FDE"
ORACLE = "oracle_future_FDE"
ES = "coarse_path_weighted_ES"
ANISO_V_ISO = "anisotropic_vs_isotropic_coarse_FDE"


def gate(effects, policy, group, minimum):
    entry = effects[policy]
    groups, recordings = entry["groups"], entry["recordings"]
    row = groups["all_dev"][group]
    excluded = groups.get("without_zara02", {}).get(group, {}).get("relative_FDE_gain")
    share = recordings[group]["maximum_contribution_share"]
    gain = row["relative_FDE_gain"]
    interval = row["delta_minFDE"]["ci95"]
    return dict(pass_flag=bool(gain >= minimum and interval[1] < 0 and excluded is not None and excluded > 0
                               and share is not None and share <= .5),
                relative_gain=gain, delta_m=row["delta_minFDE"]["estimate"], ci95=interval,
                relative_gain_without_zara02=excluded, maximum_contribution_share=share)


def evaluate(selections, effects, descriptors):
    each = {}
    for key, e in effects.items():
        s = selections[key]["policies"]
        c1 = {g: gate(e, ISO, g, cut) for g, cut in (("overall", .02), ("hard", .03))}
        c2 = gate(e, ANISO_V_ISO, "overall", .01)
        c2["long_mean"] = gate(e, ANISO_V_ISO, "long_mean", .01)
        oracle_gain = e[ORACLE]["groups"]["all_dev"]["overall"]["relative_FDE_gain"]
        oracle = {g: a["scale"] for g, a in s[ORACLE]["by_group"].items()}
        es = {g: a["scale"] for g, a in s[ES]["by_group"].items()}
        gap = round(max(es.values())-min(es.values()), 10)
        es_pref = {g: {name: j["by_group"].get(g, {}).get("scale") for name, j in s.items()
                       if name.startswith("coarse_") and name.endswith("_ES")} for g in OBS}
        es_effect = {g: -row["relative_FDE_gain"] for g, row in e[ES]["groups"]["all_dev"].items()}
        flatness = descriptors[key]["flatness"]
        moving = {g: all(abs(flatness[g][str(a)]-1) <= .02 for a in (.8, 1.2)) for g in ("steady", "other")}
        each[key] = dict(C1=dict(pass_flag=any(r["pass_flag"] for r in c1.values()), **c1),
                         C2=c2, C3=dict(pass_flag=oracle_gain < .02, relative_gain=oracle_gain,
                                         scale=oracle, boundary_groups=[g for g, a in oracle.items() if a in (.4, 1.6)]),
                         C4=dict(pass_flag=gap >= .2, gap=gap, scale=es, other_gt_one=es["other"]>1.,
                                 stationary_lt_one=es["stationary"]<1., versions=es_pref,
                                 mean_endpoint_bias_m={g: descriptors[key]["mean_endpoint_bias_dev"][g] for g in OBS}),
                         C5=dict(relative_minFDE_increase=es_effect),
                         C6=dict(pass_flag=all(moving.values()), by_moving_group=moving, ratios=flatness))
    confirmations, directions = {}, {}
    for seed in (42, 43):
        b0, heldout = each[f"B0_s{seed}"], each[f"E1B_s{seed}"]
        confirmations[str(seed)] = {}
        for g in ("overall", "hard"):
            b0_gain, h_gain = (arm["C1"][g]["relative_gain"] for arm in (b0, heldout))
            confirmations[str(seed)][g] = dict(pass_flag=b0_gain > 0 and h_gain > 0 and h_gain >= .5*b0_gain,
                                               B0_gain=b0_gain, E1B_gain=h_gain)
        u, v = b0["C4"]["scale"], heldout["C4"]["scale"]
        contrasts = {f"{b}_minus_{a}": dict(B0=round(u[b]-u[a], 10), E1B=round(v[b]-v[a], 10))
                     for a, b in combinations(OBS, 2)}
        strongest = max(contrasts, key=lambda g: abs(contrasts[g]["B0"]))
        main = contrasts[strongest]
        directions[str(seed)] = dict(pass_flag=main["B0"]*main["E1B"] > 0,
                                      comparison=strongest, contrasts=contrasts)
    b0 = [each[f"B0_s{s}"] for s in SEEDS]
    counts = {c: sum(r[c]["pass_flag"] for r in b0) for c in ("C1", "C2", "C3", "C4", "C6")}
    qualifying_c1 = {g: sum(r["C1"][g]["pass_flag"] for r in b0) for g in ("overall", "hard")}
    heldout = {g: all(r[g]["pass_flag"] for r in confirmations.values()) for g in ("overall", "hard")}
    harm = {g: float(np.mean([r["C5"]["relative_minFDE_increase"][g] for r in b0])) for g in ("overall", "hard")}
    summary = {c: {"pass_flag": n >= 4, "passing_B0": n} for c, n in counts.items()}
    summary["C1"]["passing_B0_by_group"] = qualifying_c1
    summary["C1H"] = dict(pass_flag=any(heldout.values()), pass_by_group=heldout, by_seed=confirmations)
    summary["C4"].update(B0_width_flag=summary["C4"]["pass_flag"],
                         heldout_direction=directions,
                         pass_flag=counts["C4"] >= 4 and all(r["pass_flag"] for r in directions.values()))
    summary["C5"] = dict(pass_flag=harm["overall"] <= .01 and harm["hard"] <= .02,
                         mean_B0_relative_minFDE_increase=harm)
    summary["C6"]["passing_B0_by_group"] = {g: sum(r["C6"]["by_moving_group"][g] for r in b0)
                                            for g in ("steady", "other")}
    return dict(summary=summary, by_arm_seed=each)


def report(result):
    summary = result["summary"]
    flags = {k: r["pass_flag"] for k, r in summary.items()}
    lines = ["# EXP-ADAPT-WIDTH-CENTERED", "", "## 结论", "",
             "缩放固定候选的概率加权均值路径。以下结果来自 C0 dev 上对固定预测的比较。[RESULT.json](RESULT.json)", "",
             "| 标记 | 是否达到预设标准 | 达标的 B0 seed 数 |",
             "|---|---|---|"]
    for name in ("C1", "C1H", "C2", "C3", "C4", "C5", "C6"):
        row = summary[name]
        n = str(row["passing_B0"]) + "/5" if "passing_B0" in row else "—"
        marks = "达到" if row["pass_flag"] else "未达到"
        lines.append(f"| {name} | {marks} | {n} |")
    lines += ["", "## 设计给出的后续判断", ""]
    if flags["C1"] and flags["C1H"]:
        lines.append("观测分组选出的散布调整在主分析和留出选择臂均得到支持。按 C4 和 C5 选择训练目标。")
    elif flags["C1"] and not flags["C1H"]:
        lines.append("主分析达到 minFDE 改善标准，留出臂的复核未达标。保留主分析结果，训练决策仍需独立验证。")
    if flags["C4"] and not flags["C5"]:
        lines.append("ES 偏好的散布随情境变化，伴随超过门槛的 minFDE 退步。训练方案需兼顾两项指标。")
    elif not flags["C1"] and flags["C4"] and flags["C5"]:
        lines.append("ES 支持随情境调节散布，minFDE 退步在预设范围内。按设计由研究者确认是否以校准和锐利度为训练目标。")
    elif not flags["C4"]:
        lines.append("ES 的情境差异未达到预注册标准。" +
                     ("可以进一步研究纵向和横向的散布形状。" if flags["C2"] else "可以进一步诊断候选位置分配。"))
    if flags["C6"]:
        lines.append("两个移动组在 0.8 和 1.2 附近均达到平坦度标准，支持 best-of-K 在这个范围内对散布不敏感。")
    lines += ["", "详细结果见 [RESULT.json](RESULT.json) 和 [dev_effects.json](dev_effects.json)。", "",
              "## 按 seed 的效应", "",
              "| 臂 | 粗分组 minFDE 改善（总体） | 粗分组 minFDE 改善（难组） |"
              " 纵横缩放额外改善（总体） | ES 倍数（静止／稳定移动／其他） |",
              "|---|---:|---:|---:|---|"]
    for key, row in result["by_arm_seed"].items():
        es = " / ".join(f"{row['C4']['scale'][g]:.1f}" for g in OBS)
        lines.append(f"| {key} | {row['C1']['overall']['relative_gain']:+.2%} |"
                     f" {row['C1']['hard']['relative_gain']:+.2%} | {row['C2']['relative_gain']:+.2%} | {es} |")
    lines += ["", "轨迹聚类区间、去掉 zara02、逐录制留一见 [dev_effects.json](dev_effects.json)。"
              "均值终点偏差和高速步比例见 [descriptors.json](descriptors.json)。", "",
              "## 预检与适用范围", "",
              "两组留出模型在 CALIB 上与保存的预测一致。B0 恒等缩放复现归档逐窗指标。"
              "均值路径、散布缩放和各向异性对角线的数值核验见 [PREFLIGHT.json](PREFLIGHT.json)。", "",
              "这些结论来自固定预测的干预，训练可能同时改变候选位置和概率。本次缩放固定概率，"
              "另做等权中心敏感性。C0 dev 已重复用于诊断，留出臂只有两个 seed。结论适用于 C0 ETH 训练池。"
              "HPD90 使用 archive 20 的固定核宽，留出模型借用同 seed 的 B0 核宽。详见 [PROTOCOL.md](PROTOCOL.md)。", ""]
    return "\n".join(lines)

