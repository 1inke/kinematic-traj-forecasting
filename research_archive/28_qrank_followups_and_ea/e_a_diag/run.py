"""E-A pilot: rebuild the archived neighborhoods, then replay frozen B0."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
WORKSPACE = PROJECT / "local/workspace"
OUT = HERE / "results"
DATA = Path.home() / "Desktop/DRSE/ASTRA/datasets/eth_ucy"
RECORDINGS = {"eth": ("biwi_eth",), "univ": ("students001", "students003")}
PAST, FUTURE = 8, 12
DT, RADIUS, MAX_NEIGHBORS = 0.4, 5.0, 8
REPLAY_TOLERANCE = 5e-4
METRICS = ("minADE", "minFDE", "Top1_FDE", "brier_minFDE", "ES")
sys.path[:0] = [str(PROJECT / "local"), str(PROJECT), str(WORKSPACE)]
from cloud_handoff import digest, read, write
from frozen_replay.run import drive_index, verify_sources
from review_download import Archive, REVIEW


def status(phase, **extra):
    write(OUT / "CURRENT.json", dict(status=phase, at=datetime.now(timezone.utc).isoformat(), **extra))
    print(phase, extra, flush=True)


def source(fold):
    name = f"B0_{fold}_s42"
    run = REVIEW / "stage2/runs" / name
    receipt = read(run / "RESULT.json")
    contract = receipt["contract"]
    if (receipt["status"], receipt["epochs"], contract["arm"], contract["fold"], contract["seed"]) != (
        "PASS", 64, "B0", fold, 42
    ):
        raise ValueError(f"Invalid training receipt for {name}")
    return name, run, receipt


def load_inputs(fold):
    _, run, receipt = source(fold)
    test = WORKSPACE / f"fivefold/{fold}/test.npz"
    official = run / "dev_metrics.npz"
    audit = read(WORKSPACE / "fivefold/source_audit.json")["folds"][fold]
    processed = DATA / "processed_data_diverse"
    paths = {key: processed / f"{fold}_{key}.npy" for key in ("data_test", "num_test")}
    hashes = {str(p): digest(p) for p in (test, official, *paths.values())}
    expected = {test: receipt["contract"]["data"][test.name], official: receipt["artifacts"][official.name]}
    expected.update({p: audit["approved_source_arrays"][key] for key, p in paths.items()})
    if any(hashes[str(p)] != sha for p, sha in expected.items()):
        raise ValueError(f"Source checksum mismatch for {fold}")
    with np.load(test, allow_pickle=False) as bundle:
        arrays = {k: bundle[k] for k in ("ids", "past", "future", "num_valid")}
    archive = np.load(paths["data_test"], mmap_mode="r", allow_pickle=False)
    counts = np.load(paths["num_test"], allow_pickle=False)
    if (len(counts) != audit["splits"]["test"]["windows"]
            or len(np.unique(arrays["ids"])) != len(counts)
            or not np.array_equal(counts, arrays["num_valid"])
            or not np.array_equal(archive[:, :, :PAST, -2:], arrays["past"])
            or not np.array_equal(archive[:, 0, PAST:, -2:], arrays["future"])):
        raise ValueError(f"QRANK and archived preprocessing disagree for {fold}")
    return arrays, archive, official, hashes


def load_label(path):
    rows = np.loadtxt(path, usecols=(0, 1, 13, 15), dtype=np.float32)
    if not np.isfinite(rows).all() or not np.array_equal(rows[:, :2], rows[:, :2].astype(int)):
        raise ValueError(f"Invalid labels: {path}")
    by_frame, by_track = {}, {}
    for f, ped, x, y in rows:
        frame, identity = int(f), int(ped)
        track = by_track.setdefault(identity, {})
        if frame in track:
            raise ValueError(f"Duplicate observation in {path}: {frame}, {identity}")
        track[frame] = (x, y)
        by_frame.setdefault(frame, []).append(identity)
    return by_frame, by_track


def history(by_track, ped, frame, steps):
    track = by_track[ped]
    span = range(frame - steps + 1, frame + 1)
    if all(t in track for t in span):
        return np.array([track[t] for t in span], dtype=np.float32)
    return None


def frame_windows(by_frame, by_track):
    for frame, peds in sorted(by_frame.items()):
        histories, futures, recent = {}, {}, {}
        for ped in peds:
            if (short := history(by_track, ped, frame, 2)) is None:
                continue
            recent[ped] = short
            if (obs := history(by_track, ped, frame, PAST)) is None:
                continue
            histories[ped] = obs
            track = by_track[ped]
            horizon = range(frame + 1, frame + FUTURE + 1)
            if all(t in track for t in horizon):
                futures[ped] = np.array([track[t] for t in horizon], dtype=np.float32)
        if futures:
            yield frame, histories, futures, recent


def geometry(obs, focal):
    xy = obs[:, -1].astype(np.float64)
    velocity = (xy - obs[:, -2]) / DT
    offset, rel_v = xy - xy[focal], velocity - velocity[focal]
    speed2 = np.einsum("ij,ij->i", rel_v, rel_v)
    dot = np.einsum("ij,ij->i", offset, rel_v)
    closest_time = np.divide(-dot, speed2, out=np.zeros_like(dot), where=speed2 > 1e-12)
    time = closest_time.clip(0.0, FUTURE * DT)
    distance = np.linalg.norm(offset, axis=1)
    dca = np.linalg.norm(offset + time[:, None] * rel_v, axis=1)
    return distance, dca, time, closest_time > 0


def describe(stats):
    result = {k: float(values.mean()) for k, values in stats.items()}
    conflict_count = int(stats["proxy_6m_conflicts"].sum())
    omitted = int(stats["proxy_6m_omitted"].sum())
    result.update(proxy_6m_conflicts_total=conflict_count, proxy_6m_omitted_total=omitted,
                  proxy_6m_omitted_share=omitted / max(conflict_count, 1),
                  proxy_6m_conflict_share=conflict_count / max(int(stats["proxy_6m_neighbors"].sum()), 1),
                  windows=int(len(stats["archived_neighbors"])))
    return result


def rebuild(fold, arrays, archive, hashes):
    count = len(arrays["ids"])
    distance_past = np.zeros_like(arrays["past"])
    risk_past = np.zeros_like(arrays["past"])
    causal_count = np.empty_like(arrays["num_valid"])
    tracks, frames, records, rows = [], [], [], []
    row = 0
    for recording in RECORDINGS[fold]:
        label = DATA / "labels" / fold / f"{recording}.txt"
        hashes[str(label)] = digest(label)
        by_frame, by_track = load_label(label)
        for frame, histories, futures, recent in frame_windows(by_frame, by_track):
            ids = np.array(list(histories), dtype=int)
            orig_ids = np.array(list(futures), dtype=int)
            position = {ped: idx for idx, ped in enumerate(ids)}
            observed = np.stack(list(histories.values()))
            archived_obs = np.stack([histories[ped] for ped in orig_ids])
            archived_fut = np.stack(list(futures.values()))
            recent_ids = np.array(list(recent), dtype=int)
            recent_idx = {ped: idx for idx, ped in enumerate(recent_ids)}
            recent_obs = np.stack(list(recent.values()))
            for orig_focal, ped in enumerate(orig_ids):
                focal = position[ped]
                archived_dist, *_ = geometry(archived_obs, orig_focal)
                if len(orig_ids) <= MAX_NEIGHBORS:
                    archived_selected = np.r_[orig_focal, np.flatnonzero(np.arange(len(orig_ids)) != orig_focal)]
                else:
                    archived_selected = np.argsort(archived_dist)[:MAX_NEIGHBORS + 1]
                    archived_selected = archived_selected[archived_dist[archived_selected] < RADIUS]
                n = len(archived_selected)
                frame_tag = -1 if recording == "students001" else frame
                if (row >= count or arrays["num_valid"][row] != n
                        or archive[row, 0, PAST - 1, 1] != frame_tag
                        or not np.array_equal(archived_obs[archived_selected], archive[row, :n, :PAST, -2:])
                        or not np.array_equal(archived_fut[archived_selected], archive[row, :n, PAST:, -2:])):
                    raise ValueError(f"B0 window mismatch at {fold}/{recording}/{frame}/{ped} row {row}")
                archived_peds = orig_ids[archived_selected[1:]]
                dist, dca, time, approaching = geometry(observed, focal)
                nonself = np.arange(len(ids)) != focal
                nearest = np.lexsort((ids, dist))
                nearest = nearest[nonself[nearest]]
                k = len(nearest) if len(ids) <= MAX_NEIGHBORS else min(
                    MAX_NEIGHBORS, np.count_nonzero(nonself & (dist < RADIUS))
                )
                first = nearest[:k]
                conflict = (dca < 1.0) & nonself
                risk_order = np.lexsort((ids, dist, time, dca))
                risk_order = risk_order[conflict[risk_order]]
                chosen = np.r_[risk_order, nearest[~conflict[nearest]]][:k]
                second = nearest[np.isin(nearest, chosen)]
                causal_count[row] = k + 1
                distance_past[row, :k + 1] = observed[np.r_[focal, first]]
                risk_past[row, :k + 1] = observed[np.r_[focal, second]]

                rd, r_dca, _, _ = geometry(recent_obs, recent_idx[ped])
                near = (rd < 6.0) & (recent_ids != ped)
                proxy = near & (r_dca < 1.0)
                omitted_proxy = proxy & ~np.isin(recent_ids, archived_peds)
                omitted = conflict & ~np.isin(ids, archived_peds)
                restored = second[omitted[second]]
                extra_restored = restored[~np.isin(restored, first)]
                rows.append(dict(
                    scene_people=len(by_frame[frame]), observed_candidates=len(nearest),
                    archived_neighbors=n - 1, causal_neighbors=k,
                    proxy_6m_neighbors=near.sum(), proxy_6m_conflicts=proxy.sum(),
                    proxy_6m_omitted=omitted_proxy.sum(), proxy_6m_any_omitted=omitted_proxy.any(),
                    proxy_6m_no_conflict_05=not (near & (r_dca < 0.5)).any(),
                    proxy_6m_no_conflict_10=not proxy.any(),
                    proxy_6m_no_conflict_20=not (near & (r_dca < 2.0)).any(),
                    proxy_5m_over_cap=np.count_nonzero(near & (rd < RADIUS)) > MAX_NEIGHBORS,
                    archived_eligible_people=len(orig_ids),
                    proxy_6m_omitted_no_complete_history=(omitted_proxy & ~np.isin(recent_ids, ids)).sum(),
                    proxy_6m_omitted_no_complete_future=(omitted_proxy & np.isin(recent_ids, ids)
                                                         & ~np.isin(recent_ids, orig_ids)).sum(),
                    proxy_6m_omitted_after_eligible=(omitted_proxy & np.isin(recent_ids, orig_ids)).sum(),
                    causal_no_conflict=not conflict.any(),
                    causal_any_archived_omission=omitted.any(),
                    causal_approaching_omission=(omitted & approaching).any(),
                    risk_restored_archived=len(restored), risk_exclusive_restored=len(extra_restored),
                    distance_any_restored=omitted[first].any(), risk_any_restored=len(restored) > 0,
                    changed=not np.array_equal(first, second),
                ))
                tracks.append(f"{recording}:{ped}")
                records.append(recording)
                frames.append(frame)
                row += 1
    if row != count:
        raise ValueError(f"Rebuilt {row} of {count} B0 windows in {fold}")
    stats = {key: np.array([r[key] for r in rows]) for key in rows[0]}
    dest = OUT / fold
    dest.mkdir(exist_ok=True)
    audit = dict(status="PASS", fold=fold, source_sha256=hashes, overall=describe(stats),
                 by_recording={rec: describe({k: v[np.array(records) == rec] for k, v in stats.items()})
                               for rec in RECORDINGS[fold]})
    write(dest / "AUDIT.json", audit)
    return audit, stats, np.array(tracks), np.array(frames), distance_past, risk_past, causal_count


def evaluate(fold, arrays, official, built):
    import torch
    from advisor_eval import forecast_metrics, safe_npz
    from advisor_train import core
    from qrank_eval import infer
    from qrank_decisions import paired

    name, _, receipt = source(fold)
    weight = OUT / "weights" / f"{name}.pt"
    checksum = receipt["artifacts"]["final_ema.pt"]
    if digest(weight) != checksum:
        raise ValueError(f"Checkpoint checksum mismatch for {fold}")
    _, stats, tracks, frames, distance_past, risk_past, causal_count = built
    torch.set_num_threads(1)
    model, _ = core.make_model("cuda:0", 42)
    model.load_state_dict(torch.load(weight, map_location="cpu", weights_only=True), strict=True)
    model.requires_grad_(False)
    arms = dict(archived=arrays, distance=dict(arrays, past=distance_past, num_valid=causal_count),
                conflict=dict(arrays, past=risk_past, num_valid=causal_count))
    scores = {}
    replay_delta = {}
    for arm, data in arms.items():
        status("INFERENCE", fold=fold, arm=arm)
        pred = infer(core, model, data)
        paths, logits = pred["layer_paths"], pred["layer_logits"]
        if paths.shape != (len(tracks), 1, 20, FUTURE, 2) or logits.shape != (len(tracks), 1, 20):
            raise ValueError(f"Unexpected candidate shape for {fold}/{arm}")
        scores[arm] = forecast_metrics(paths[:, 0], logits[:, 0], arrays["future"])
        if any(not np.isfinite(scores[arm][key]).all() for key in METRICS):
            raise ValueError(f"Non-finite metrics for {fold}/{arm}")
        if arm == "archived":
            with np.load(official, allow_pickle=False) as prior:
                if not np.array_equal(prior["ids"], arrays["ids"]):
                    raise ValueError("Original test IDs differ")
                replay_delta = {key: float(np.abs(scores[arm][key] - prior[key]).mean()) for key in METRICS}
            if any(x > REPLAY_TOLERANCE for x in replay_delta.values()):
                raise ValueError(f"Archived replay mismatch for {fold}: {replay_delta}")
    if any(not np.allclose(scores["distance"][key][~stats["changed"]],
                           scores["conflict"][key][~stats["changed"]], atol=1e-6, rtol=0)
           for key in METRICS):
        raise ValueError(f"Unchanged neighborhoods behave differently for {fold}")

    groups = dict(overall=np.ones(len(tracks), dtype=bool),
                  archived_omitted=stats["causal_any_archived_omission"],
                  risk_exclusive_restore=stats["risk_exclusive_restored"] > 0,
                  no_causal_conflict=stats["causal_no_conflict"], changed=stats["changed"])
    grouped = {}
    for group, mask in groups.items():
        if mask.any():
            grouped[group] = dict(
                windows=int(mask.sum()), tracks=int(len(np.unique(tracks[mask]))),
                mean={arm: {key: float(values[key][mask].mean()) for key in METRICS}
                      for arm, values in scores.items()},
                conflict_vs_distance={key: paired(scores["conflict"][key][mask],
                                                  scores["distance"][key][mask], tracks[mask])
                                      for key in METRICS},
            )
    dest = OUT / fold
    saved = {f"{arm}_{key}": values[key] for arm, values in scores.items() for key in METRICS}
    sha = safe_npz(dest / "metrics.npz", ids=arrays["ids"], tracks=tracks, frames=frames, **stats, **saved)
    result = dict(status="PASS", fold=fold, seed=42, checkpoint_sha256=checksum,
                  audit_sha256=digest(dest / "AUDIT.json"), metrics_sha256=sha,
                  replay_mean_abs_error=replay_delta, groups=grouped, gpu=torch.cuda.get_device_name(0))
    write(dest / "RESULT.json", result)
    return result


def main(folds, audit_only):
    OUT.mkdir(exist_ok=True)
    status("VERIFYING", folds=folds)
    verify_sources()
    inputs, rebuilt = {}, {}
    for fold in folds:
        status("AUDITING", fold=fold)
        arrays, archive, official, hashes = load_inputs(fold)
        rebuilt[fold] = rebuild(fold, arrays, archive, hashes)
        inputs[fold] = (arrays, official)
    if audit_only:
        status("AUDIT_COMPLETE", folds=folds)
        return

    status("CHECKPOINTS", folds=folds)
    checkpoints = {}
    for fold in folds:
        name, _, receipt = source(fold)
        checkpoints[name] = receipt["artifacts"]["final_ema.pt"]
    pending = [name for name, sha in checkpoints.items()
               if not (path := OUT / "weights" / f"{name}.pt").is_file() or digest(path) != sha]
    if pending:
        drive = Archive()
        weights = drive_index(drive, pending)
        for name in pending:
            drive.download(weights[name], OUT / "weights" / f"{name}.pt", checkpoints[name])
    results = {fold: evaluate(fold, *inputs[fold], rebuilt[fold]) for fold in folds}
    summary = dict(status="COMPLETE", code_sha256=digest(Path(__file__)), training_runs=0, weight_updates=0,
                   paired_bootstrap="2,000 resamples clustered by recording and pedestrian track",
                   design=dict(observation_frames=PAST, horizon_s=FUTURE * DT, distance_radius_m=RADIUS,
                               audit_radius_m=6.0, conflict_dca_m=1.0, max_neighbors=MAX_NEIGHBORS,
                               causal_arms="Same complete past-only pool, same count and distance slot ordering."),
                   audits={fold: rebuilt[fold][0] for fold in folds}, inference=results)
    write(OUT / "SUMMARY.json", summary)
    status("COMPLETE", folds=folds, summary_sha256=digest(OUT / "SUMMARY.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", choices=tuple(RECORDINGS))
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    try:
        main([args.fold] if args.fold else list(RECORDINGS), args.audit_only)
    except Exception as error:
        status("FAILED", error=str(error))
        raise
