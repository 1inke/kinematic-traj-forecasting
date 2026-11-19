"""Derive experiment tables and figures from the machine-readable result."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, GRAY = "#28678C", "#C36232", "#91979D"
GROUPS = ("overall", "obs_stationary", "obs_steady", "obs_other")
GROUP_LABELS = ("All windows", "Near-stationary", "Steady", "Other")
OBS_BINS = ("stationary", "steady_low", "steady_mid", "steady_high",
            "other_low", "other_mid", "other_high")
BIN_LABELS = ("Near-stationary", "Steady low", "Steady middle", "Steady high",
              "Other low", "Other middle", "Other high")


def entries(result, model, horizon="t12"):
    return [value[horizon] for value in result["runs"][model].values()]


def hpd_groups(row, weight="equal", variant="global"):
    return row["hpd"][weight]["variants"][variant]["groups"]


def mean(values):
    return float(np.mean(list(values)))


def make_figures(result, directory):
    directory.mkdir(exist_ok=True)
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 9, "legend.fontsize": 7,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .6,
        "svg.fonttype": "none", "pdf.fonttype": 42, "savefig.dpi": 300,
    })
    files = []

    def export(fig, name):
        svg, pdf = directory / f"{name}.svg", directory / f"{name}.pdf"
        png, tiff = directory / f"{name}.png", directory / f"{name}.tiff"
        fig.savefig(svg, format="svg", bbox_inches="tight", facecolor="white")
        fig.savefig(pdf, format="pdf", bbox_inches="tight", facecolor="white")
        fig.savefig(png, format="png", dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(tiff, format="tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"},
                    bbox_inches="tight", facecolor="white")
        for file in (svg, pdf, png, tiff):
            assert file.stat().st_size > 1000, file
            files.append(file)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75), sharex=True, sharey=True, layout="constrained")
    for ax, model, color in zip(axes, ("B0", "E3_self"), (BLUE, ORANGE)):
        limit = []
        for seed, row in result["runs"][model].items():
            bins = row["t12"]["spread_skill"]["equal"]["bins"]
            x = [b["S_m"] for b in bins]
            y = [b["R_m"] for b in bins]
            limit += x + y
            ax.loglog(x, y, marker="o", markersize=2.3, linewidth=.9, alpha=.45, color=color, label=seed)
        ax.loglog([min(limit), max(limit)], [min(limit), max(limit)], "--", color=GRAY, linewidth=.8)
        avg = mean(v["spread_skill"]["equal"]["beta"] for v in entries(result, model))
        ax.set(title=f"{model}, mean slope {avg:.2f}", xlabel="RMS endpoint spread (m)")
    axes[0].set_ylabel("RMS center error (m)")
    export(fig, "spread_skill")

    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.75), sharex=True, sharey=True, layout="constrained")
    levels = result["levels"]
    for ax, group, label in zip(axes.flat, GROUPS, GROUP_LABELS):
        ax.plot([.46, 1], [.46, 1], "--", color=GRAY, linewidth=.8)
        for model, color in (("B0", BLUE), ("E3_self", ORANGE)):
            cover = np.array([hpd_groups(row)[group]["coverage"] for row in entries(result, model)])
            ax.fill_between(levels, cover.min(0), cover.max(0), color=color, alpha=.14, linewidth=0)
            ax.plot(levels, cover.mean(0), "o-", color=color, markersize=3, label=model)
        n = hpd_groups(entries(result, "B0")[0])[group]["n"]
        ax.set(title=f"{label} (n = {n:,})", xlim=(.46, 1), ylim=(0, 1.01))
    axes[0, 0].legend(frameon=False, loc="upper left")
    fig.supxlabel("Nominal HPD probability")
    fig.supylabel("Observed coverage")
    export(fig, "hpd_coverage")

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.75), sharex=True, sharey=True, layout="constrained")
    for ax, model in zip(axes, ("B0", "E3_self")):
        ax.plot([0, 1], [0, 1], "--", color=GRAY, linewidth=.8)
        for variant, color, label in (("raw", BLUE, "Raw"), ("calibrated", ORANGE, "Temperature scaled")):
            for i, row in enumerate(entries(result, model)):
                values = row["probability"]["winner"][variant]["bins"]
                ax.plot([r["mean_p"] for r in values], [r["event_rate"] for r in values],
                        "o-", color=color, alpha=.38, linewidth=.9, markersize=2.3,
                        label=label if i == 0 else None)
        ax.set(title=model, xlabel="Mean predicted probability", xlim=(-.02, 1.02), ylim=(-.02, 1.02))
    axes[0].set_ylabel("Observed FDE winner frequency")
    axes[0].legend(frameon=False, loc="upper left")
    export(fig, "probability_reliability")

    fig, ax = plt.subplots(figsize=(6.3, 2.7), layout="constrained")
    ax.axis("off")
    rows = []
    for name, label in zip(OBS_BINS, BIN_LABELS):
        values = [row["quantizer"]["bin_" + name] for row in entries(result, "B0")]
        base = mean(v["model_minFDE_m"] for v in values)
        ref = values[0]["reference_minFDE_m"]
        excess = mean(v["FDE_relative_excess"] for v in values)
        rows.append([label, f"{values[0]['n']:,}", f"{base:.3f}", f"{ref:.3f}", f"{excess:+.1%}"])
    table = ax.table(cellText=rows, colLabels=["Observation bin", "n", "B0 (m)", "Reference (m)", "B0 excess"],
                     colWidths=[.30, .11, .17, .23, .19], loc="center", cellLoc="right")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.40)
    for (i, column), cell in table.get_celld().items():
        cell.set_linewidth(0)
        cell.set_facecolor("#E8ECF0" if i == 0 else "white" if i % 2 else "#F4F6F8")
        if column == 0:
            cell.get_text().set_ha("left")
    ax.set_title("Minimum final displacement error by observation bin")
    export(fig, "quantizer_comparison")
    return files


def report(result):
    decision = result["decisions"]
    b0 = entries(result, "B0")
    beta = decision["reaction"]
    ratio_lo = mean(r["spread_skill"]["equal"]["low_R_over_S"] for r in b0)
    ratio_hi = mean(r["spread_skill"]["equal"]["high_R_over_S"] for r in b0)
    flags = [name for name in ("EASY_TOO_WIDE", "HARD_TOO_NARROW", "MISALLOCATED_EASY", "CENTER_BIAS", "PROB_MISCAL")
             if decision[name]["pass_gate"]]
    flagged = "、".join(flags) if flags else "无"
    lines = [
        "# EXP-SET-CALIB-DIAG 结果",
        "",
        f"**综合结论：`{decision['conclusion']}`。** B0 等权终点散布的平均 β 为 {beta['mean']:.3f}，"
        f"{beta['n_agree']}/5 个 seed 与均值分档一致，判定为 `{beta['label']}`。"
        f"低、高散布箱的平均 R/S 分别为 {ratio_lo:.3f} 和 {ratio_hi:.3f}。触发的附加判定为 {flagged}。",
        "",
        "综合结论按设计表生成。β<1 时，拟合关系中的 R/S 随 S 增加而下降。"
        "容易组、困难组的范围方向需依据下表的覆盖率与经验散布确认。",
        "",
        "## 终点主结果",
        "",
        "| 模型 | seed | 等权 β（95% 窗口重抽样区间） | 加权 β | 等权 HPD90 覆盖 | HPD90 面积 m² | 原始 → 缩放后 ECE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model, seeds in result["runs"].items():
        for seed, run in seeds.items():
            row = run["t12"]
            eq, weighted = (row["spread_skill"][v] for v in ("equal", "weighted"))
            lo, hi = eq["beta_ci95"]
            hp = hpd_groups(row)["overall"]
            raw, temp = (row["probability"]["winner"][v]["ece"] for v in ("raw", "calibrated"))
            lines.append(f"| {model} | {seed} | {eq['beta']:.3f} [{lo:.3f}, {hi:.3f}] | {weighted['beta']:.3f} | "
                         f"{hp['coverage'][2]:.1%} | {hp['mean_HPD90_area_m2']:.3f} | {raw:.4f} → {temp:.4f} |")
    lines += ["", "## B0 的分组覆盖与核宽度敏感性", "",
              "| 分组 | 窗口数 | HPD50 | HPD80 | HPD90 | HPD95 | HPD90 面积 m² | 0.5×h0 的 HPD90 | 2×h0 的 HPD90 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for group in hpd_groups(b0[0]):
        values = [hpd_groups(row)[group] for row in b0]
        cov = [f"{v:.1%}" for v in np.mean([v["coverage"] for v in values], axis=0)]
        area = mean(v["mean_HPD90_area_m2"] for v in values)
        half, double = [mean(hpd_groups(row, variant=mode)[group]["coverage"][2] for row in b0)
                        for mode in ("global_half", "global_double")]
        lines.append(f"| {group} | {values[0]['n']} | {' | '.join(cov)} | {area:.3f} | {half:.1%} | {double:.1%} |")
    lines += ["", "## B0 与量化参照", "",
              "| 观测箱 | n | B0 minADE m | 参照 minADE m | B0 minFDE m | 参照 minFDE m | minFDE 劣势 | 预测/经验散布 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in OBS_BINS:
        values = [row["quantizer"]["bin_" + name] for row in b0]
        get = lambda key: mean(v[key] for v in values)
        errors = [f"{get(key):.3f}" for key in ("model_minADE_m", "reference_minADE_m", "model_minFDE_m",
                                               "reference_minFDE_m")]
        lines.append(f"| {name} | {values[0]['n']} | {' | '.join(errors)} | "
                     f"{get('FDE_relative_excess'):+.1%} | {get('spread_ratio'):.2f} |")
    lines += ["", "## 图和复核", "",
              "- [散布–误差](figures/spread_skill.pdf)。每条线对应一个 seed。灰虚线为等误差与散布。",
              "- [覆盖可靠性](figures/hpd_coverage.pdf)。线为 seed 平均，阴影为各 seed 的最小值到最大值。",
              "- [概率可靠性](figures/probability_reliability.pdf)。每条线对应一个 seed。正温度保持 Top1 候选。",
              "- [量化参照](figures/quantizer_comparison.pdf)。B0 取 5 个 seed 的平均，参照仅在 fit 上拟合。",
              "",
              "完整的分组、权重、带宽、秩次、联合达标及第 6 步结果见 [RESULT.json](RESULT.json)。"
              "输入校验和版本见 [INPUT_MANIFEST.json](INPUT_MANIFEST.json)。",
              "",
              "每窗只有一个真值，因此校准只能在分组层面估计。HPD 覆盖随核宽度变化。"
              "窗口重叠可能导致按窗口重抽样的区间偏窄。分箱量化器是粗观测参照。"
              "C0 dev 曾用于反复诊断，本实验仅覆盖 C0，后续方法需要另外验证。", ""]
    return "\n".join(lines)
