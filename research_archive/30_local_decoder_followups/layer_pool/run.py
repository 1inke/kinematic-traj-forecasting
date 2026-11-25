"""Evaluate cross-layer candidate coverage with a fixed forecast budget."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
WORKSPACE = PROJECT / "local/workspace"
OUT = HERE / "results"
sys.path[:0] = [str(PROJECT), str(PROJECT / "refine_pred"), str(PROJECT / "local"), str(WORKSPACE)]
from cloud_handoff import digest, read, write
from refine_pred import run as parent
from analysis import summarize_group


BUDGETS = (4, 8)
METHODS = ("final", "first", "conf4", "cover4", "conf8", "cover8", "pool40")
GROUPS = {"all": (), "turn": ("turn", "turn_and_speed"), "deceleration": ("deceleration",), "steady": ("steady",)}


def status(state, **fields):
    write(OUT / "CURRENT.json", dict(status=state, at=datetime.now(timezone.utc).isoformat(), **fields))
    print(state, fields, flush=True)


def source_path(name):
    return OUT / "runs" / name / "METRICS.npz"


def signature(name):
    return dict(code=digest(Path(__file__)), design=digest(HERE / "README.md"),
                ref=digest(parent.metric_path(name).with_name("RESULT.json")))


def complete(name):
    array = source_path(name)
    stamp = array.with_name("RESULT.json")
    if not array.is_file() or not stamp.is_file():
        return False
    info = read(stamp)
    return info["binding"] == signature(name) and info["metrics_sha256"] == digest(array)


def selection(paths, logits):
    """Rank 20+20 candidates using only predicted trajectories and scores."""
    endpoints = paths[:, :, -1]
    distances = np.linalg.norm(endpoints[:, :, None] - endpoints[:, None], axis=-1)
    batch = np.arange(len(paths))
    first_order = np.argsort(-logits[:, 0], axis=1, kind="stable")
    final_order = np.argsort(-logits[:, 1], axis=1, kind="stable") + 20
    selections = {}
    for budget in BUDGETS:
        retain = final_order[:, :20 - budget]
        closest = distances[batch[:, None], retain].min(axis=1)
        chosen = np.zeros((len(paths), 20), dtype=bool)
        recover = []
        for _ in range(budget):
            marginal = np.maximum(closest[:, None] - distances[:, :20], 0).sum(axis=-1)
            marginal[chosen] = -np.inf
            pick = marginal.argmax(axis=1)
            chosen[batch, pick] = True
            recover.append(pick)
            closest = np.minimum(closest, distances[batch, pick])
        selections[f"conf{budget}"] = np.column_stack((retain, first_order[:, :budget]))
        selections[f"cover{budget}"] = np.column_stack((retain, *recover))
    for candidates in selections.values():
        ordered = np.sort(candidates, axis=1)
        if candidates.shape != (len(paths), 20) or not np.array_equal(candidates[:, 0], final_order[:, 0]):
            raise ValueError("Fixed budget or protected Top-1 failed")
        if (np.diff(ordered, axis=1) == 0).any():
            raise ValueError("Repeated slot in a mixed pool")
    return selections, final_order[:, 0]


def evaluate(paths, logits, truth):
    values = {}
    for start in range(0, len(paths), 192):
        chunk = slice(start, start + 192)
        p = np.concatenate((paths[chunk, 0], paths[chunk, 3]), axis=1)
        chosen, top = selection(p, logits[chunk][:, (0, 3)])
        error = np.linalg.norm(p - truth[chunk, None], axis=-1)
        ade, fde = error.mean(axis=-1), error[:, :, -1]
        good = (ade <= .2) & (fde <= .4)
        hits = {"first": slice(0, 20), "final": slice(20, 40), "pool40": slice(None)}
        batch = np.arange(len(p))
        for name, take in {**hits, **chosen}.items():
            if name in chosen:
                a, f, g = (x[batch[:, None], take] for x in (ade, fde, good))
                values.setdefault(f"{name}_slots", []).append(take.astype(np.uint8))
            else:
                a, f, g = ade[:, take], fde[:, take], good[:, take]
            values.setdefault(f"{name}_minADE", []).append(a.min(axis=1))
            values.setdefault(f"{name}_minFDE", []).append(f.min(axis=1))
            values.setdefault(f"{name}_joint", []).append(g.any(axis=1))
        values.setdefault("final_top1_fde", []).append(fde[batch, top])
    return {name: np.concatenate(chunks) for name, chunks in values.items()}


def replay(name):
    import torch
    from advisor_train import core, load_data
    from qrank_eval import infer
    from qrank_model import attach

    if not parent.completed(name):
        raise ValueError(f"Invalid REFINE-PRED reference for {name}")
    source, _ = parent.formal(name)
    arm, fold, seed = parent.RUNS[name]
    folder, test = parent.folders(fold)
    checkpoint = parent.weight_path(name)
    if checkpoint is None or digest(test) != source["contract"]["data"]["test.npz"]:
        raise ValueError(f"Missing verified local inputs for {name}")
    data = load_data(test, read(folder / "TRACKS.json")["windows"])
    torch.set_num_threads(1)
    device = "cuda:0"
    model, _ = core.make_model(device, seed)
    attach(model, arm)
    model.decoder.use_kin = source["contract"]["use_kin"]
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.requires_grad_(False)
    pred = infer(core, model, {key: data[key] for key in ("ids", "past", "num_valid")})
    scores = evaluate(pred["layer_paths"], pred["layer_logits"], data["future"])
    with np.load(parent.metric_path(name), allow_pickle=False) as prior:
        np.testing.assert_array_equal(prior["ids"], data["ids"])
        drift = {f"{layer}_{metric}": float(np.abs(scores[f"{layer}_{metric}"] - prior[f"{layer}_{metric}"]).max())
                 for layer in ("first", "final") for metric in ("minFDE", "minADE")}
        drift["top1"] = float(np.abs(scores["final_top1_fde"] - prior["final_Top1_FDE"]).max())
        scores.update({key: prior[key] for key in ("ids", "tracks", "recording")})
    if not all(np.isfinite(x) and x <= .0005 for x in drift.values()):
        raise ValueError(f"Replay disagrees with prior layers for {name}: {drift}")
    path = source_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    parent.persist_npz(path, **scores)
    write(path.with_name("RESULT.json"), dict(status="PASS", binding=signature(name), metrics_sha256=digest(path),
                                             max_abs_drift=drift, windows=len(data["ids"]), weight_updates=0))


def load(path):
    with np.load(path, allow_pickle=False) as obj:
        return {key: obj[key] for key in obj.files}


def summarize(fold, names):
    rows = [load(source_path(name)) for name in names]
    cohort = load(parent.folders(fold)[0] / "LABELS.npz")
    for row in rows:
        np.testing.assert_array_equal(row["ids"], cohort["ids"])
        np.testing.assert_array_equal(row["tracks"], rows[0]["tracks"])
    report = {}
    base = {metric: np.mean([row[f"final_{metric}"] for row in rows], axis=0) for metric in ("minADE", "minFDE", "joint")}
    tracks = rows[0]["tracks"]
    for group, labels in GROUPS.items():
        mask = np.isin(cohort["maneuver"], labels) if labels else np.ones(len(tracks), dtype=bool)
        if not mask.any():
            continue
        block = {}
        for method in METHODS:
            entry = {}
            for metric in ("minADE", "minFDE"):
                delta = np.mean([row[f"{method}_{metric}"] for row in rows], axis=0) - base[metric]
                entry[metric] = summarize_group(base[metric], delta, tracks, mask)
            joint = [row[f"{method}_joint"][mask] for row in rows]
            old = [row["final_joint"][mask] for row in rows]
            entry["joint"] = dict(baseline=float(np.mean(old)), mean=float(np.mean(joint)),
                                  rescue=float(np.mean([a & ~b for a, b in zip(joint, old)])),
                                  loss=float(np.mean([~a & b for a, b in zip(joint, old)])))
            block[method] = entry
        report[group] = block
    return report


def coordinate():
    OUT.mkdir(exist_ok=True)
    status("PREFLIGHT", total=len(parent.RUNS))
    parent.previous.verify_sources()
    prior = read(parent.OUT / "CURRENT.json")
    if prior["status"] != "COMPLETE" or digest(parent.OUT / "SUMMARY.json") != prior["summary_sha256"]:
        raise ValueError("The prior five-fold analysis is incomplete")
    for name in parent.RUNS:
        if not parent.completed(name) or parent.weight_path(name) is None:
            raise ValueError(f"Missing bound replay inputs for {name}")
    pending = [name for name in parent.RUNS if not complete(name)]
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")

    def run_one(name):
        folder = source_path(name).parent
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / "worker.log").open("a", encoding="utf-8") as log:
            subprocess.run([sys.executable, "-u", "-B", str(Path(__file__)), "--run", name], check=True,
                           env=env, stdout=log, stderr=subprocess.STDOUT)
        if not complete(name):
            raise ValueError(f"Missing completed receipt for {name}")
        return name

    count = len(parent.RUNS) - len(pending)
    status("REPLAYING", completed=count, total=len(parent.RUNS), workers=2)
    with ThreadPoolExecutor(max_workers=2) as pool:
        for task in as_completed([pool.submit(run_one, name) for name in pending]):
            name = task.result()
            count += 1
            status("REPLAYING", completed=count, total=len(parent.RUNS), last_verified=name)
    status("SUMMARIZING", completed=count, total=len(parent.RUNS))
    results = {}
    for fold, seeds in parent.FOLDS.items():
        results[fold] = {arm: summarize(fold, [f"{arm}_{fold}_s{seed}" for seed in seeds]) for arm in parent.ARMS}
    rel = {fold: arms["O-conf+E0"]["all"]["cover4"]["minFDE"]["relative"] for fold, arms in results.items()}
    opportunity = sum(x <= -.02 for x in rel.values()) >= 3 and rel["eth"] <= .01
    decision = "FIXED_K_OPPORTUNITY" if opportunity else "NO_FIXED_K_SIGNAL"
    output = dict(status="COMPLETE", runs=count, weight_updates=0, decision=decision, e0_cover4_vs_final_relative_fde=rel,
                  results=results, prior_summary_sha256=prior["summary_sha256"])
    write(OUT / "SUMMARY.json", output)
    lines = ["# EXP-LAYER-POOL", "", f"Opportunity screen: {decision}", "",
             "| Fold | Cover 16+4 minFDE change | Conf 16+4 | Cover 12+8 | Pool40 ceiling |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for fold, arms in results.items():
        e0 = arms["O-conf+E0"]["all"]
        changes = " | ".join(f"{e0[m]['minFDE']['relative']:+.2%}" for m in ("cover4", "conf4", "cover8", "pool40"))
        lines.append(f"| {fold.upper()} | {changes} |")
    lines.extend(("", "E0, relative to all 20 final-layer candidates. Positive values indicate increased error.",
                  "SUMMARY.json includes QFL, maneuver groups, ADE, joint recall, and the paired intervals."))
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    status("COMPLETE", completed=count, total=len(parent.RUNS), decision=decision, summary_sha256=digest(OUT / "SUMMARY.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", choices=tuple(parent.RUNS))
    args = parser.parse_args()
    try:
        if args.run:
            replay(args.run)
        else:
            coordinate()
    except Exception as error:
        if not args.run:
            status("FAILED", error=str(error))
        raise
