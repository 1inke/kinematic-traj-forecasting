"""Run the sealed C0 width-cause diagnostic on frozen final EMA models."""

import argparse
from collections import defaultdict
from datetime import datetime
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / "NEXT_PHASE_EXEC_20261114"
PREC = ROOT.parent / "EXP_PREC_DIAG_EXEC_20261119"
HISTORY = ROOT.parent / "EXP_PREC_HISTORY_DIAG_EXEC_20261119"
CAL = ROOT.parent / "EXP_SET_CALIB_DIAG_EXEC_20261119"
sys.path[:0] = [str(OLD), str(PREC), str(HISTORY)]

import numpy as np
import torch

from run_prec_diag import arrays, load_prediction, maneuver_labels, read, sha, write
from run_history_diag import observation_masks
from width_model import TERMS, WEIGHTS, checked_terms, gradients, model_for, prediction, replay, rescale, tensors, world_paths
from width_analysis import (ALPHAS, C_ALPHAS, GROUPS, TERM_NAMES, THRESHOLDS, decide, errors, frame,
                            gradient_summary, loss_table, masks, metric_table)

SEEDS = (42, 43, 44, 45, 46)
RNG = 20261120
BATCH = 64
EMAX = 1e-5
F_GROUPS = ("T1", "T2", "T3", "E2_natural", "E2_clamped", "E3_set", "E3_set_nobias")
OUTPUTS = ("PREFLIGHT.json", "A_summary.json", "A_templates.json", "B_scale_slice.json",
           "C_reallocation.json", "D_diversity.json", "E_frame.json", "F_existing_runs.json", "RESULT.json", "REPORT.md")


def status(phase, **kw):
    write(ROOT / "CURRENT.json", dict(at=datetime.now().astimezone().isoformat(), pid=os.getpid(), phase=phase, **kw))


def check(path, expected, provenance):
    value = sha(path)
    assert value == expected, path
    provenance[str(path)] = value
    return value


def sealed(path, receipt, provenance):
    return check(path, read(receipt)["files"][str(path)], provenance)


def as_numpy(x):
    return x.detach().cpu().numpy()


def save_npz(path, **kw):
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as f:
        np.savez_compressed(f, **kw)
    temp.replace(path)


def group_labels(data):
    past = data["past"][:, 0].astype(np.float64)
    by_obs = observation_masks(past)
    future = maneuver_labels(data["past"], data["future"])
    obs = np.select([by_obs["stationary"], by_obs["steady"]], ["stationary", "steady"], default="other")
    group = np.where(obs == "stationary", np.where(future == "stationary", "remain", "start"), obs)
    assert np.isin(future[obs == "stationary"], ["stationary", "start"]).all()
    return group, obs, future


def setup(provenance):
    manifest = read(PREC / "INPUT_MANIFEST.json")["files"]
    files = {key: OLD / f"C0/{key}.npz" for key in ("fit", "dev")}
    for p in files.values():
        check(p, manifest[str(p)], provenance)
    data = {k: arrays(p) for k, p in files.items()}
    groups, obs, future = zip(*(group_labels(data[k]) for k in ("fit", "dev")))
    assert [len(data[k]["ids"]) for k in ("fit", "dev")] == [21719,4753]
    dev_count = {g: int((groups[1] == g).sum()) for g in GROUPS}
    assert list(dev_count.values()) == [942,167,1991,1653], dev_count
    generator = np.random.default_rng(RNG)
    selection = (obs[0] == "stationary")
    for name in ("steady", "other"):
        idx = generator.choice(np.flatnonzero(obs[0] == name), size=3000, replace=False)
        selection[idx] = True
    cal_done = read(CAL / "DONE.json")["files"]
    for seed in SEEDS:
        path = CAL / f"windows/B0_s{seed}.npz"
        check(path, cal_done[f"windows/B0_s{seed}.npz"], provenance)
    for p in [*ROOT.glob("*.py"), ROOT / "PROTOCOL.md", ROOT / "DESIGN.md", *ROOT.glob("standalone/*.ps1")]:
        provenance[str(p)] = sha(p)
    return data, groups, obs, future, selection, dev_count


def load_baseline(seed, data, provenance):
    folder = OLD / f"runs/E1C_E64_s{seed}"
    final = folder / "final.pt"
    sealed(final, folder / "FINAL_LOCK.json", provenance)
    for k in ("fit", "dev"):
        sealed(folder / f"{k}/predictions.npz", folder / "EVALUATION_DONE.json", provenance)
    return final, {k: load_prediction(folder, k, data[k]) for k in ("fit", "dev")}


def check_replay(paths, logits, transform, saved, index, audit):
    err = float(np.max(abs(as_numpy(world_paths(paths.detach(), transform)).astype(np.float64) - saved[0][index])))
    zerr = float(np.max(abs(as_numpy(logits[:, 0])-saved[1][index])))
    assert max(err, zerr) <= EMAX, (index, err, zerr)
    audit["world_path_max_abs"] = max(audit.get("world_path_max_abs", 0.), err)
    audit["logit_max_abs"] = max(audit.get("logit_max_abs", 0.), zerr)


def calculate(seed, data, groups, obs, future, selected, device, provenance, sample_only):
    ckpt, originals = load_baseline(seed, data, provenance)
    model, criterion = model_for(ckpt, device, seed)
    audit = {}
    gd = defaultdict(list)
    fit_loss = {alpha: [] for alpha in ALPHAS}
    dev_scale = {alpha: [] for alpha in ALPHAS}
    dev_c = {name: {alpha: defaultdict(list) for alpha in C_ALPHAS} for name in THRESHOLDS}
    fit_winners, frame_ratios = [], []
    fit_pairs, dev_pairs = [], []
    dev_static = obs[1] == "stationary"
    ref = arrays(CAL / f"windows/B0_s{seed}.npz")
    np.testing.assert_array_equal(ref["ids"], data["dev"]["ids"])
    np.testing.assert_array_equal(ref["observation"], obs[1])
    np.testing.assert_array_equal(ref["maneuver"], future[1])
    for split_i, split in enumerate(("fit", "dev")):
        d = data[split]
        n = len(d["ids"])
        for start in range(0, min(n, BATCH) if sample_only else n, BATCH):
            sl = slice(start, min(n, start+BATCH))
            training = split == "fit"
            a_mask = selected[sl] if training else np.zeros(sl.stop-sl.start, bool)
            stat_mask = obs[split_i][sl] == "stationary"
            tracked = bool(a_mask.any())
            if start % 1024 == 0 and not sample_only:
                status("A_FIT_GRADIENTS" if training else "B_C_DEV", seed=seed, windows=start, total=n)
            p, z, y, tf, capture = replay(model, d, sl, device, tracked, check_original=start == 0)
            f_err = capture.get("original_forward_max_abs", [0., 0.])
            audit["decoder_max_abs"] = max(audit.get("decoder_max_abs", 0.), *f_err)
            check_replay(p, z, tf, originals[split], sl, audit)
            aux = capture["aux"]
            term, winner, eff, pairs = checked_terms(p, y, z, criterion, audit)
            local_active = as_numpy((pairs < .4).to(torch.float64).mean(1))
            (fit_pairs if training else dev_pairs).append(local_active)
            if tracked:
                kept = torch.tensor(a_mask, device=device)
                for k, v in gradients(term, winner, eff, aux, kept, audit).items():
                    gd[k].append(as_numpy(v)[a_mask])
                gd["active_pairs"].append(local_active[a_mask])
            if not stat_mask.any():
                continue
            stat = torch.tensor(stat_mask, device=device)
            with torch.no_grad():
                if training:
                    fit_winners.append(as_numpy(winner[stat]))
                for alpha in ALPHAS:
                    if alpha == 1.:
                        out, logits = p.detach(), z.detach()
                    else:
                        out, logits = rescale(model.decoder, aux, scale_mult=alpha)
                        out, logits = out[:, :1], logits[:, :1]
                    if training:
                        term_a, _, _, _ = checked_terms(out[stat], y[stat], logits[stat], criterion, audit)
                        fit_loss[alpha].append(as_numpy(term_a))
                    else:
                        world = as_numpy(world_paths(out, tf))[stat_mask]
                        dev_scale[alpha].append(errors(world, d["future"][sl][stat_mask]))
                if training:
                    continue
                end = as_numpy(aux[4][:, 0, :, -1])[stat_mask].astype(np.float64)
                center = end - end.mean(1, keepdims=True)
                eig = np.linalg.eigvalsh(np.einsum("nki,nkj->nij", center, center)/20)
                frame_ratios.append(np.divide(eig[:, 1], eig[:, 0], out=np.full(len(eig), np.nan), where=eig[:, 0] > 1e-12))
                if sample_only:
                    continue
                for name in THRESHOLDS:
                    rare = rates < (1/int(name[2:]))
                    for alpha in C_ALPHAS:
                        m = p.new_ones((len(p), 20))
                        m[:, rare] = alpha
                        pk, _ = rescale(model.decoder, aux, cand_mult=m)
                        ww = as_numpy(world_paths(pk, tf))[stat_mask]
                        row = dev_c[name][alpha]
                        row["metrics"].append(errors(ww, d["future"][sl][stat_mask]))
                        target = aux[4][:, 0, rare].detach() * alpha / aux[1][:, 0, None, None].detach()
                        reachable = (target - aux[3][rare].detach()).abs().amax(dim=(2, 3)) <= .75 + 1e-6
                        row["reachable"].append(as_numpy(reachable)[stat_mask])
        if split == "fit" and not sample_only:
            rates = np.bincount(np.concatenate(fit_winners), minlength=20) / sum(len(x) for x in fit_winners)
    if sample_only:
        return audit
    audit["fit_replay_n"], audit["dev_replay_n"] = len(data["fit"]["ids"]), len(data["dev"]["ids"])
    grad = {key: np.concatenate(v) for key, v in gd.items()}
    grad.update(group=groups[0][selected], fit_index=np.flatnonzero(selected), ids=data["fit"]["ids"][selected])
    np.testing.assert_array_equal(grad["winner"][masks(grad["group"])["stationary"]], np.concatenate(fit_winners))
    a_stats, t_stats, diversity, saturation = gradient_summary(grad, rates)
    grad["loser_S"] = grad["S"] - grad["winner_S"]
    pos = np.arange(len(grad["winner"]))[:, None]
    col = np.arange(len(TERM_NAMES))[None, :]
    gn = grad["anchor_grad_norm"].sum(2)
    wn = grad["anchor_grad_norm"][pos, col, grad["winner"][:, None]]
    grad["loser_norm_fraction"] = np.divide(gn - wn, gn, out=np.zeros_like(gn), where=gn > 0)
    save_npz(ROOT / f"A_gradients_s{seed}.npz", **grad, terms=np.asarray(TERM_NAMES), weights=np.asarray(WEIGHTS))
    for key, arr, labels in (("fit", fit_pairs, groups[0]), ("dev", dev_pairs, groups[1])):
        all_active = np.concatenate(arr)
        for name, m in masks(labels).items():
            if m.any():
                diversity[name][f"{key}_active_pairs_all_windows"] = float(all_active[m].mean())
    dev_labels = groups[1][dev_static]
    b_fit = loss_table(fit_loss, groups[0][obs[0] == "stationary"])
    spread_reference = ref["t12__equal__s"][dev_static]
    b_dev = {str(alpha): metric_table(np.concatenate(values), dev_labels) for alpha, values in dev_scale.items()}
    np.testing.assert_allclose(np.concatenate(dev_scale[1.])[:,2], spread_reference, atol=EMAX, rtol=0)
    opt = min(ALPHAS, key=lambda x: b_fit[str(x)]["stationary"]["total"])
    opt_no = min(ALPHAS, key=lambda x: b_fit[str(x)]["stationary"]["no_diversity"])
    c_variants = {}
    for name, rows in dev_c.items():
        output = {}
        reference = ref["t12__reference_minFDE"][dev_static]
        base = metric_table(np.concatenate(rows[1.]["metrics"]), dev_labels, reference)
        for alpha, row in rows.items():
            values = metric_table(np.concatenate(row["metrics"]), dev_labels, reference)
            reach = np.concatenate(row["reachable"])
            for g, r in values.items():
                chosen = masks(dev_labels)[g]
                for key in ("minADE", "minFDE", "mean_spread"):
                    r[f"relative_{key}_gain"] = 1-r[key]/base[g][key]
                r["reachable"] = float(reach[chosen].mean()) if reach.size else None
            output[str(alpha)] = dict(rare_slots=np.flatnonzero(rates < 1/int(name[2:])).tolist(),
                                      reachable=float(reach.mean()) if reach.size else None, groups=values)
        c_variants[name] = output
    eig = np.concatenate(frame_ratios)
    valid = np.isfinite(eig)
    f = dict(median_anisotropy=float(np.median(eig[valid])), undefined=int((~valid).sum()),
             quartiles=np.quantile(eig[valid], [.25,.75]).tolist())
    return (dict(groups=a_stats, saturation=saturation), t_stats,
            dict(fit=b_fit, dev=b_dev, alpha_star=opt, alpha_star_no_diversity=opt_no),
            dict(fit_win_rates=rates.tolist(), variants=c_variants), diversity, f, audit)


@torch.no_grad()
def budget_forecasts(run, data, device, seed, calib_indices, provenance):
    folder = OLD / "runs" / run
    path = folder / "final.pt"
    sealed(path, folder / "FINAL_LOCK.json", provenance)
    snapshot = folder / "calib/predictions.npz"
    sealed(snapshot, folder / "EVALUATION_DONE.json", provenance)
    old = arrays(snapshot)
    model, _ = model_for(path, device, seed)
    fit, dev = data["fit"], data["dev"]
    np.testing.assert_array_equal(fit["ids"][calib_indices], old["ids"])
    audit = {}

    def infer(d, idx):
        p,z,_,tf = prediction(model, *tensors(d, idx, device))
        return as_numpy(world_paths(p,tf)), as_numpy(z[:,0])

    for st in range(0, len(calib_indices), BATCH):
        pred, logit = infer(fit, calib_indices[st:st+BATCH])
        ix = slice(st, st+BATCH)
        error = float(np.max(abs(pred-old["paths"][ix])))
        z_err = float(np.max(abs(logit-old["logits"][ix])))
        assert max(error, z_err) <= EMAX, (run, st, error, z_err)
        audit["calib_world_path_max_abs"] = max(audit.get("calib_world_path_max_abs", 0.), error)
        audit["calib_logit_max_abs"] = max(audit.get("calib_logit_max_abs", 0.), z_err)
    paths = [infer(dev, slice(st,st+BATCH))[0] for st in range(0,len(dev["ids"]),BATCH)]
    return np.concatenate(paths), audit


def existing(data, dev_obs, device, provenance):
    split = OLD / "E1/calib_split.npz"
    provenance[str(split)] = sha(split)
    indices = arrays(split)["calib_index"]
    n = len(data["dev"]["ids"])
    f = {}
    rules = [(f"E1B_E{ep}_s{s}", s, ep) for ep in (8,16,32,64) for s in (42,43)]
    rules += [(f"{arm}_s{s}", s, None) for arm in F_GROUPS for s in (42,43,44)]
    for i, (name, seed, ep) in enumerate(rules):
        status("F_BUDGET_REPLAY" if ep else "F_SAVED", run=name, completed=i, total=len(rules))
        if ep:
            path, audit = budget_forecasts(name, data, device, seed, indices, provenance)
            row = dict(source="NEW_FROZEN_EMA_DEV_FORWARD", budget=ep, archived_calib_replay=audit)
        else:
            folder = OLD / "runs" / name
            sealed(folder / "dev/predictions.npz", folder / "EVALUATION_DONE.json", provenance)
            path, _ = load_prediction(folder,"dev",data["dev"])
            row = dict(source="SEALED_SAVED_DEV_PREDICTIONS")
        endpoints = path[:, :, -1].astype(np.float64)
        spread = np.sqrt(np.square(endpoints-endpoints.mean(1,keepdims=True)).sum(2).mean(1))
        assert len(spread) == n
        for g in ("stationary","steady","other"):
            m = dev_obs == g
            row[g] = dict(n=int(m.sum()), median_spread=float(np.median(spread[m])))
        f[name] = row
    return f


def render(result, A, B, C, D, E, F):
    descriptions = dict(M1="非赢家梯度不足", M2="当前附近的整体缩放损失更低",
                        M2_prime="某个更窄的整体缩放降低目标至少 1%",
                        M3="整体缩放的几何损失由起步组承担", M4="少胜候选内收有可复现的净收益",
                        M5="参数框限制，且少胜候选大量向内饱和", M6="多样性项维持宽度",
                        M7="预算曲线两组变化均在 5% 内")
    keys = [str(s) for s in SEEDS]
    active = [k for k, v in result["markers"].items() if v["active"]]
    lines = ["# EXP-WIDTH-CAUSE-DIAG", "", "触发的预设标记为 " + ("、".join(active) or "无") + "。", ""]
    msg = "少胜候选内收在本次主分析达到预设收益与起步保护条件。" if "M4" in active else "少胜候选内收未达到本次预设的免费再分配判据。"
    if "M4" in active and "M5" in active:
        msg += "主分析同时支持当前参数化的限制。"
    if "M4" in active and "M1" in active:
        msg += "主分析同时显示少胜候选收到的梯度信号很少。"
    if "M2_prime" in active:
        msg += "整体缩放切片支持进一步检查优化与参数耦合。"
    lines += [msg, "", "| 标记 | 判定 | 证据 |", "|---|---|---|"]
    files = ("A_summary.json", "B_scale_slice.json", "B_scale_slice.json", "B_scale_slice.json",
             "C_reallocation.json", "C_reallocation.json 与 A_summary.json", "D_diversity.json", "F_existing_runs.json")
    for (name, v), evidence in zip(result["markers"].items(), files):
        votes = f"（{v['seeds_passing']}/5）" if "seeds_passing" in v else "（两组预算 seed）"
        lines.append(f"| {name} {descriptions[name]} | {'达到' if v['active'] else '未达到'}{votes} | {evidence} |")
    lines += ["", "## 逐 seed 主要数值", "",
              "| seed | remain 非赢家梯度 | coverage 有效候选数 | 最优 α | 去多样性最优 α | 内收 FDE 改善 | 起步 FDE 改善 | 内收可达率 |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key in keys:
        r = result["per_seed"][key]
        accessible = r["M5"]["reachable"]
        acc = f"{accessible:.1%}" if accessible is not None else "无少胜候选"
        lines.append(f"| {key} | {r['M1']['loser_fraction']:.1%} | {r['M1']['coverage_effective_count']:.2f} | "
                     f"{B[key]['alpha_star']:.1f} | {B[key]['alpha_star_no_diversity']:.1f} | "
                     f"{r['M4']['relative_gain']:+.2%} | {r['M4']['start_relative_gain']:+.2%} | {acc} |")
    lines += ["", "## 坐标系与预算", "",
              f"dev 默认朝向 {E['dev']['default_heading']:.1%}，0.03–0.12 m 朝向 {E['dev']['small_displacement_heading']:.1%}。"
              f"起步方向合成长度为 {E['dev']['start_resultant_length']:.3f}。详见 E_frame.json。",
              "E1B 的 8 组预算模型使用旧 EMA 权重补做 dev 前向，并在旧 calib 上回放验证。F 的其余变体读已有预测。",
              "", "## 局限", "",
              "A 是最终 EMA、eval、无增强下的梯度快照，反映方向和相对大小。"
              "B、C 是对既有模型的一维干预。dev 已用于多轮探索。起步组共 167 窗，解读需看逐 seed。"
              "结论限于 C0。方向合成长度接近零也可由对称多峰产生。"]
    return "\n".join(lines) + "\n"


def main(device, sample_only):
    started = time.monotonic()
    if (ROOT / "DONE.json").exists() and not sample_only:
        done = read(ROOT / "DONE.json")
        assert all(sha(ROOT / p) == h for p, h in done["files"].items())
        return
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    provenance = {}
    data, groups, obs, future, selected, counts = setup(provenance)
    pre = dict(seed=RNG, fit_sample_indices=np.flatnonzero(selected).tolist(),
               fit_sample_counts={k:int((obs[0][selected] == k).sum()) for k in ("stationary","steady","other")},
               dev_counts=counts, device=device, checks={})
    if sample_only:
        audit = calculate(42, data, groups, obs, future, selected, device, provenance, True)
        print(dict(audit=audit, sample_counts=pre["fit_sample_counts"]), flush=True)
        return
    status("PREFLIGHT", seeds=list(SEEDS), total=len(SEEDS))
    stores = [{} for _ in range(6)]
    for i, seed in enumerate(SEEDS):
        task = calculate(seed, data, groups, obs, future, selected, device, provenance, False)
        for store, value in zip(stores, task[:6]):
            store[str(seed)] = value
        pre["checks"][str(seed)] = task[-1]
        write(ROOT / "PREFLIGHT.json", dict(**pre, files=provenance))
        status("B0_SEED_COMPLETE", seed=seed, completed=i+1, total=len(SEEDS))
        print(f"B0 seed {seed} complete ({i+1}/5)", flush=True)
    A, T, B, C, D, pred_cov = stores
    E = dict(fit=frame(data["fit"],groups[0]), dev=frame(data["dev"],groups[1], pred_cov))
    for s in SEEDS:
        D[str(s)]["optimum"] = dict(with_diversity=B[str(s)]["alpha_star"], without_diversity=B[str(s)]["alpha_star_no_diversity"])
    F = existing(data, obs[1], device, provenance)
    final = decide(SEEDS, A, B, C, D, F)
    for name, obj in zip(OUTPUTS[:-1], (dict(**pre, files=provenance), A, T, B, C, D, E, F, final)):
        write(ROOT / name, obj)
    (ROOT / "REPORT.md").write_text(render(final, A, B, C, D, E, F), encoding="utf-8")
    files = [*OUTPUTS, *(f"A_gradients_s{k}.npz" for k in SEEDS)]
    write(ROOT / "DONE.json", dict(status="SUCCESS", elapsed_s=time.monotonic()-started, files={p:sha(ROOT / p) for p in files}))
    status("DONE", completed=len(SEEDS), total=len(SEEDS))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--device", default="cuda:0", choices=("cuda:0", "cpu"))
    cli = parser.parse_args()
    try:
        main(cli.device, cli.sample)
    except Exception:
        if not cli.sample:
            status("FAILED", traceback=traceback.format_exc())
        raise
