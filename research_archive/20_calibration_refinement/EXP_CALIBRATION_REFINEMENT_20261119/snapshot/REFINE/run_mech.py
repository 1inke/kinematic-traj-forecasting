"""Zero-training replay and geometry diagnosis of E3_self."""

import argparse
import time

import numpy as np
import torch
from torch.nn import functional as F

from refine_io import OLD, PREC, ROOT, anchored, arrays, finish, finished, npz, populations, read, status, write
from run_prec_diag import load_prediction, quality, run_folder
from baseline_factory import make_model, prediction, world_paths
from set_refine_decoder import attach_set_refiner

FOLDER = ROOT / "MECH_DIAG"
SEEDS = (42, 43, 44)
LIMIT = 1e-5


def check_array(a, b, field):
    error = float(np.max(np.abs(a - b)))
    if not np.isfinite(error) or error > LIMIT:
        raise ValueError(f"REPLAY_MISMATCH {field} max_abs={error:.10g} limit={LIMIT}")
    return error


@torch.no_grad()
def replay(seed, data, device, sample=False):
    directory = run_folder("E3_self", seed)
    ckpt = directory / "final.pt"
    locked = read(directory / "FINAL_LOCK.json")
    ckpt_hash = anchored(ckpt, locked)
    finished_eval = read(directory / "EVALUATION_DONE.json")
    p1_file = directory / "dev/pass1.npz"
    files = {str(ckpt): ckpt_hash, str(p1_file): anchored(p1_file, finished_eval)}
    for model in ("E3_self", "B0"):
        run = run_folder(model, seed)
        path = run / "dev/predictions.npz"
        files[str(path)] = anchored(path, read(run / "EVALUATION_DONE.json"))
    target_paths, target_scores = load_prediction(directory, "dev", data)
    first_saved = arrays(p1_file)
    np.testing.assert_array_equal(first_saved["ids"], data["ids"])
    np.testing.assert_array_equal(first_saved["logits"], target_scores)
    model, _ = make_model(device, seed)
    attach_set_refiner(model, "self").to(device)
    snapshot = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(snapshot["ema"], strict=True)
    del snapshot
    model.eval()
    decoder = model.decoder
    assert decoder.basis_variant == "linear"
    captured = {}

    def save_scale(module, args, value):
        captured["scale"] = F.softplus(value) + .25

    def save_anchors(module, args, value):
        captured["second"] = value[0]
        captured["scale_arg"] = args[2]

    def save_raw(module, args, value):
        captured["raw"] = value

    handles = [decoder.proposal_scale.register_forward_hook(save_scale),
               decoder.refiner.register_forward_hook(save_anchors),
               decoder.refiner.anchor_out.register_forward_hook(save_raw)]
    parts = {}
    errors = {name: 0. for name in ("final_world", "interpolated_world", "scores", "first_world", "residual")}
    n = 64 if sample else len(data["ids"])
    try:
        for st in range(0, n, 64):
            idx = slice(st, min(st + 64, n))
            past = torch.as_tensor(data["past"][idx], device=device)
            valid = torch.as_tensor(data["num_valid"][idx], device=device)
            captured.clear()
            paths, scores, _, transform = prediction(model, past, valid)
            center, origin, rotation = transform
            count, agents = past.shape[:2]
            first = decoder.last_pass1_relative[:, :1]
            second = captured["second"].reshape(count, agents, 20, 3, 2)[:, :1]
            raw = captured["raw"].reshape(count, agents, 20, 3, 2)[:, 0]
            scale = captured["scale"][:, 0]
            assert torch.equal(captured["scale"].flatten(0, 1), captured["scale_arg"])

            def interpolated(anchor):
                return torch.einsum("tp,bakpc->baktc", decoder.curve_W, anchor) - center[:, :1, None]

            def numpy(tensor):
                return tensor.detach().cpu().numpy()

            first_world = numpy(world_paths(interpolated(first), transform))
            final_world = numpy(world_paths(paths, transform))
            interp_world = numpy(world_paths(interpolated(second), transform))
            score = numpy(scores[:, 0])
            pairs = dict(final_world=(final_world, target_paths[idx]),
                         interpolated_world=(interp_world, target_paths[idx]), scores=(score, target_scores[idx]),
                         first_world=(first_world, first_saved["paths"][idx]),
                         residual=(numpy(second[:, 0] - first[:, 0]),
                                   numpy(.30 * scale[:, None, None] * raw.tanh())))
            for key, (a, b) in pairs.items():
                errors[key] = max(errors[key], check_array(a, b, f"s{seed}/{st}/{key}"))
            truth = np.einsum("nti,nij->ntj", data["future"][idx] - numpy(origin)[:, None], numpy(rotation))
            for name, value in dict(first=numpy(first[:, 0]), second=numpy(second[:, 0]),
                                    raw= numpy(raw), scale=numpy(scale), logits=score, y_local=truth,
                                    first_world=first_world, second_world=final_world).items():
                parts.setdefault(name, []).append(value)
            if not sample and st % 512 == 0:
                status("MECH_REPLAY", seed=seed, windows_done=idx.stop, windows_total=n, device=device)
    finally:
        for h in handles:
            h.remove()
    return {k: np.concatenate(v) for k, v in parts.items()}, dict(windows=n, device=device, errors=errors, files=files)


def shape(anchor):
    return np.linalg.norm(anchor[..., 2, :] - 3 * anchor[..., 0, :], axis=-1) + np.linalg.norm(
        anchor[..., 1, :] - 2 * anchor[..., 0, :], axis=-1)


def ratio(a, b):
    return float(a / b) if abs(b) > 1e-12 else None


def summarize(values, data, baseline, masks):
    a, b, raw, scale = [values[k].astype(np.float64) for k in ("first", "second", "raw", "scale")]
    p, q = [quality(values[k].astype(np.float64), data["future"]) for k in ("first_world", "second_world")]
    b0 = quality(baseline, data["future"])
    row = np.arange(len(a))
    first_ix, second_ix = p["FDE"].argmin(1), q["FDE"].argmin(1)
    first_fde, second_fde = p["FDE"][row, first_ix], q["FDE"][row, second_ix]
    same = first_ix == second_ix
    gain = first_fde - second_fde
    endpoint = a[:, :, -1]
    delta = b[:, :, -1] - endpoint
    radial = endpoint - endpoint.mean(1, keepdims=True)
    distance = np.linalg.norm(radial, axis=-1)
    direction = np.divide(radial, distance[..., None], where=distance[..., None] > 0, out=np.zeros_like(radial))
    inward = np.einsum("nkd,nkd->nk", delta, direction)
    projected_scale = np.linalg.norm(direction * scale[:, None], axis=-1)
    normalized_radial = np.divide(inward, projected_scale, where=projected_scale > 0, out=np.zeros_like(inward))
    cuts = np.quantile(distance, [.25, .50, .75])
    quartiles = np.searchsorted(cuts, distance, side="right") + 1
    y_change = np.abs(b[:, :, -1, 1]) - np.abs(endpoint[..., 1])
    abs_delta = np.linalg.norm(delta, axis=-1)
    saturation = np.abs(np.tanh(raw)) > .95
    s1, s2 = shape(a), shape(b)
    truth = values["y_local"]
    target_shape = shape(truth[:, [3, 7, 11]])
    shape_x1, shape_x2 = s1[row, first_ix], s2[row, second_ix]
    gap1, gap2 = abs(shape_x1 - target_shape), abs(shape_x2 - target_shape)
    # Symmetric replacement of the two local error coordinates (two orders).
    err0, err1 = a[row, first_ix, -1] - truth[:, -1], b[row, first_ix, -1] - truth[:, -1]
    mixed_x = np.hypot(err1[:, 0], err0[:, 1])
    mixed_y = np.hypot(err0[:, 0], err1[:, 1])
    local0, local1 = [np.linalg.norm(e, axis=1) for e in (err0, err1)]
    long_gain = np.where(same, .5 * ((local0 - mixed_x) + (mixed_y - local1)), 0.)
    precision = np.where(same, gain, 0.)
    redistribution = gain - precision
    lateral_gain = precision - long_gain
    rank = np.argsort(-values["logits"], axis=1, kind="stable")
    pulled = (quartiles[row, first_ix] == 4) & (inward[row, first_ix] < 0)
    per = dict(ids=data["ids"], first_FDE_winner=first_ix, second_FDE_winner=second_ix, same_winner=same,
               first_minFDE=first_fde, second_minFDE=second_fde, first_minADE=p["ADE"].min(1),
               second_minADE=q["ADE"].min(1), B0_minFDE=b0["FDE"].min(1), B0_minADE=b0["ADE"].min(1),
               endpoint_shift_m=abs_delta, longitudinal_shift_m=delta[..., 0], lateral_shift_m=delta[..., 1],
               radial_shift_m=inward, radial_shift_in_scale=normalized_radial, radial_quartile=quartiles,
               logit_rank=np.argsort(rank, axis=1) + 1, path_shape_first_m=s1, path_shape_second_m=s2,
               path_shape_truth_m=target_shape, saturation_fraction=saturation.mean((2, 3)),
               precision_gain_m=precision, redistribution_gain_m=redistribution,
               precision_longitudinal_m=long_gain, precision_lateral_m=lateral_gain,
               abs_lateral_change_m=y_change, first_winner_outer_and_inward=pulled)

    def candidates(m):
        return dict(candidates=int(m.sum()), mean_longitudinal_m=float(delta[..., 0][m].mean()),
                    mean_lateral_m=float(delta[..., 1][m].mean()), mean_radial_m=float(inward[m].mean()),
                    mean_radial_in_scale=float(normalized_radial[m].mean()), median_norm_m=float(np.median(abs_delta[m])),
                    mean_abs_lateral_change_m=float(y_change[m].mean()), saturation=float(saturation[m].mean()))

    groups = {}
    for name, m in masks.items():
        precis, total = float(precision[m].sum()), float(gain[m].sum())
        windows = {key: float(per[key][m].mean()) for key in
                   ("first_minADE", "first_minFDE", "second_minADE", "second_minFDE", "B0_minADE", "B0_minFDE")}
        harmed = m & (gain < 0)
        all_c = np.broadcast_to(m[:, None], distance.shape)
        groups[name] = dict(
            n=int(m.sum()), **windows, same_winner_fraction=float(same[m].mean()),
            total_FDE_gain_sum_m=total, precision_gain_sum_m=precis,
            redistribution_gain_sum_m=float(redistribution[m].sum()), precision_share=ratio(precis, total),
            redistribution_share=ratio(float(redistribution[m].sum()), total),
            longitudinal_share_of_precision=ratio(float(long_gain[m].sum()), precis),
            lateral_share_of_precision=ratio(float(lateral_gain[m].sum()), precis),
            precision_longitudinal_gain_sum_m=float(long_gain[m].sum()),
            precision_lateral_gain_sum_m=float(lateral_gain[m].sum()),
            B0_median_minFDE_m=float(np.median(per["B0_minFDE"][m])),
            shape=dict(first_winner_mean_L_m=float(shape_x1[m].mean()), second_winner_mean_L_m=float(shape_x2[m].mean()),
                       truth_mean_L_m=float(target_shape[m].mean()), first_mean_gap_m=float(gap1[m].mean()),
                       second_mean_gap_m=float(gap2[m].mean()),
                       relative_gap_reduction=ratio(gap1[m].sum() - gap2[m].sum(), gap1[m].sum()),
                       same_first_winner_second_mean_gap_m=float(abs(s2[row, first_ix] - target_shape)[m].mean())),
            candidates=candidates(all_c), outer_candidates=candidates(all_c & (quartiles == 4))
                if (all_c & (quartiles == 4)).any() else None,
            harmed_windows=int(harmed.sum()), harmed_first_winner_outer_inward=ratio((pulled & harmed).sum(), harmed.sum()),
            scale_axis_median=np.median(scale[m], axis=0).tolist(),
        )
    return dict(groups=groups, radial_cuts_m=cuts.tolist(),
                quartiles={str(i): candidates(quartiles == i) for i in range(1, 5)},
                logit_ranks={str(i): candidates(per["logit_rank"] == i) for i in range(1, 21)}), per


def mark(rows):
    checks = {}
    for seed, values in rows.items():
        g = values["groups"]
        steady, static, turns, all_ = (g[k] for k in ("steady", "obs_stationary", "turn", "overall"))
        checks[seed] = dict(
            STEADY_PRECISION=[steady["precision_share"], steady["longitudinal_share_of_precision"],
                              steady["total_FDE_gain_sum_m"], steady["precision_gain_sum_m"]],
            SHAPE_REGULARIZED=[steady["shape"]["relative_gap_reduction"]],
            LOWSPEED_OVERSHIFT=[static["candidates"]["median_norm_m"], static["B0_median_minFDE_m"]],
            TAIL_CONTRACTION=[turns["outer_candidates"]["mean_radial_in_scale"], turns["harmed_first_winner_outer_inward"]],
            STRAIGHTENING=[-turns["candidates"]["mean_abs_lateral_change_m"]],
            DIVISION_OF_LABOR=[all_["first_minFDE"] / all_["B0_minFDE"] - 1],
        )
    predicates = dict(STEADY_PRECISION=lambda a, b, c, d: a >= .6 and b >= .6 and c > 0 and d > 0,
                      SHAPE_REGULARIZED=lambda a: a >= .20, LOWSPEED_OVERSHIFT=lambda a, b: a >= b,
                      TAIL_CONTRACTION=lambda a, b: a <= -.05 and b >= .5,
                      STRAIGHTENING=lambda a: a >= .03, DIVISION_OF_LABOR=lambda a: a >= .02)
    out = {}
    for name, rule in predicates.items():
        seed_values = {s: values[name] for s, values in checks.items()}
        eligible = [v for v in seed_values.values() if all(x is not None for x in v)]
        votes = sum(rule(*v) for v in eligible)
        avg = [float(np.mean(col)) if col else None
               for col in ([v[i] for v in seed_values.values() if v[i] is not None]
                           for i in range(len(next(iter(seed_values.values())))))]
        out[name] = dict(active=votes >= 2, agreeing_seeds=votes, mean_statistics=avg, seed_values=seed_values)
    return out


def main(device, sample):
    torch.set_num_threads(4 if device == "cpu" else 2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    FOLDER.mkdir(exist_ok=True)
    if not sample and finished(FOLDER):
        return
    start = time.monotonic()
    source = OLD / "C0/dev.npz"
    data_hash = anchored(source, read(PREC / "INPUT_MANIFEST.json"))
    data = arrays(source)
    masks, labels, obs = populations(data)
    results, files = {}, []
    for seed in (SEEDS[:1] if sample else SEEDS):
        print(f"MECH E3_self seed {seed} ({device})", flush=True)
        observed, audit = replay(seed, data, device, sample)
        print(audit, flush=True)
        if sample:
            return
        baseline, _ = load_prediction(run_folder("B0", seed), "dev", data)
        stats, per = summarize(observed, data, baseline, masks)
        stats["replay"] = audit
        results[str(seed)] = stats
        path = FOLDER / f"windows/E3_self_s{seed}.npz"
        npz(path, **per, group=data["group"], maneuver=labels, observation=obs)
        files.append(path)
    markers = mark(results)
    write(FOLDER / "RESULT.json", dict(experiment="EXP-REFINE-MECH-DIAG", dev_hash=data_hash,
                                       windows=len(data["ids"]), seeds=results, markers=markers))
    lines = ["# EXP-REFINE-MECH-DIAG", "",
             "触发的描述性标记为 " + ("、".join(k for k, v in markers.items() if v["active"]) or "无") + "。", "",
             "| 标记 | 满足种子数 | 三个种子的平均统计量（口径见 PROTOCOL） |",
             "|---|---:|---|"]
    for name, entry in markers.items():
        lines.append(f"| {name} | {entry['agreeing_seeds']}/3 | {entry['mean_statistics']} |")
    lines += ["", "| seed | 分组 | n | B0 minFDE m | 第一遍 minFDE m | 第二遍 minFDE m | 同赢家精度占比 | 转向横向位移变化 m |",
              "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for seed, per_seed in results.items():
        for name, g in per_seed["groups"].items():
            share = g["precision_share"]
            share_text = f"{share:.1%}" if share is not None else "未定义"
            lines.append(f"| {seed} | {name} | {g['n']} | {g['B0_minFDE']:.4f} | {g['first_minFDE']:.4f} | "
                         f"{g['second_minFDE']:.4f} | {share_text} | {g['candidates']['mean_abs_lateral_change_m']:+.4f} |")
    lines += ["", "完整的 minADE、形状、尺度、饱和度、径向四分位和评分排名见 RESULT.json。"
              "这些标记描述已训练模型的行为。"]
    (FOLDER / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    finish(FOLDER, files + [FOLDER / "RESULT.json", FOLDER / "REPORT.md"], seconds=time.monotonic() - start)
    status("MECH_DONE", seeds_complete=3, seeds_total=3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda:0"))
    parser.add_argument("--sample", action="store_true")
    arg = parser.parse_args()
    main(arg.device, arg.sample)
