"""Run the preregistered REFINE-PRED frozen replays on the five sealed folds."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from analysis import ARMS, CONFIRMATION, EXPLORATORY, decide, fold_report
from measure import geometry, observed, vector_norm

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
WORKSPACE = PROJECT / "local/workspace"
OUT = HERE / "results"
sys.path[:0] = [str(PROJECT), str(PROJECT / "local"), str(WORKSPACE)]
from cloud_handoff import digest, read, write
from frozen_replay import run as previous
from review_download import Archive
from tracks import audit_tracks

FOLDS = {name: tuple(range(42, 47 if name == "eth" else 45)) for name in (*EXPLORATORY, *CONFIRMATION)}
RUNS = {f"{arm}_{fold}_s{seed}": (arm, fold, seed)
        for fold, seeds in FOLDS.items() for seed in seeds for arm in ARMS}
TOL = 5e-4


class InvalidReplay(ValueError):
    """A preregistered replication or manipulation check failed."""


def log_status(phase, **info):
    write(OUT / "CURRENT.json", dict(status=phase, at=datetime.now(timezone.utc).isoformat(), **info))
    print(phase, info, flush=True)


def lock():
    dependencies = [HERE / name for name in ("run.py", "analysis.py", "measure.py", "tracks.py", "DESIGN.md")]
    dependencies += [PROJECT / "frozen_replay/run.py", PROJECT / "e_a_diag/run.py"]
    return {p.relative_to(PROJECT).as_posix(): digest(p) for p in dependencies}


def formal(name):
    arm, fold, seed = RUNS[name]
    path = previous.source_receipt(name)
    source = read(path)
    config = source["contract"]
    if (source["status"], source["epochs"], config["arm"], config["fold"], config["seed"]) != (
        "PASS", 64, arm, fold, seed
    ):
        raise InvalidReplay(f"Incorrect formal receipt for {name}")
    return source, path


def folders(fold):
    return OUT / "tracks" / fold, WORKSPACE / "fivefold" / fold / "test.npz"


def metric_path(name):
    return OUT / "runs" / name / "metrics.npz"


def binding(name):
    source, source_path = formal(name)
    _, fold, _ = RUNS[name]
    folder, _ = folders(fold)
    bound = dict(receipt=digest(source_path), weights=source["artifacts"]["final_ema.pt"],
                 track_audit=digest(folder / "TRACKS.json"), labels=digest(folder / "LABELS.npz"),
                 sources=read(OUT / "SOURCE_LOCK.json")["files"])
    if name in previous.RUNS:
        bound["archive28"] = digest(previous.OUT / "runs" / name / "RESULT.json")
    return bound


def completed(name):
    path = metric_path(name)
    receipt = path.with_name("RESULT.json")
    if not path.is_file() or not receipt.is_file():
        return False
    saved = read(receipt)
    return saved["status"] == "PASS" and saved["binding"] == binding(name) and saved["metrics_sha256"] == digest(path)


def persist_npz(path, **arrays):
    temp = path.with_name(path.stem + ".part.npz")
    np.savez_compressed(temp, **arrays)
    temp.replace(path)


def prepare_population(fold):
    from eth_speed_math import maneuver_labels

    folder, test = folders(fold)
    for name, (_, model_fold, _) in RUNS.items():
        if fold == model_fold and formal(name)[0]["contract"]["data"]["test.npz"] != digest(test):
            raise InvalidReplay(f"Fold data differ for {name}")
    audit = audit_tracks(fold, test, WORKSPACE / "fivefold/source_audit.json", folder)
    with np.load(test, allow_pickle=False) as data:
        ids = data["ids"]
        labels = maneuver_labels(data["past"], data["future"])
    dest = folder / "LABELS.npz"
    if dest.is_file():
        with np.load(dest, allow_pickle=False) as saved:
            if np.array_equal(saved["ids"], ids) and np.array_equal(saved["maneuver"], labels):
                return audit
    persist_npz(dest, ids=ids, maneuver=labels)
    return audit


def weight_path(name):
    for path in (previous.OUT / "weights" / f"{name}.pt", PROJECT / "local/results" / name / "final_ema.pt",
                 OUT / "weights" / f"{name}.pt"):
        if path.is_file() and digest(path) == formal(name)[0]["artifacts"]["final_ema.pt"]:
            return path
    return None


def fetch_weights(run_names):
    missing = [name for name in run_names if weight_path(name) is None]
    if not missing:
        return
    archive = Archive()
    files = previous.drive_index(archive, missing)

    def download(name):
        sha = formal(name)[0]["artifacts"]["final_ema.pt"]
        archive.download(files[name], OUT / "weights" / f"{name}.pt", sha)

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(download, missing))


def replay(name):
    import torch
    from advisor_train import core, load_data
    from eth_speed_math import observed_speed
    from order_numpy import conf_order, risk_order
    from qrank_eval import infer
    from qrank_model import attach

    if read(OUT / "SOURCE_LOCK.json")["files"] != lock():
        raise InvalidReplay("Analysis and source lock differ")
    source, path = formal(name)
    evidence = path.with_name("dev_metrics.npz")
    if digest(evidence) != source["artifacts"]["dev_metrics.npz"]:
        raise InvalidReplay(f"Invalid archived metrics for {name}")
    arm, fold, seed = RUNS[name]
    folder, test = folders(fold)
    audit = read(folder / "TRACKS.json")
    if digest(test) != source["contract"]["data"]["test.npz"] or digest(folder / "TRACKS.npz") != audit["track_sha256"]:
        raise InvalidReplay(f"Source or trajectory binding differs for {name}")
    data = load_data(test, audit["windows"])
    covariates, direction = observed(data["past"][:, 0], observed_speed)
    with np.load(folder / "TRACKS.npz", allow_pickle=False) as mapped:
        np.testing.assert_array_equal(data["ids"], mapped["ids"])
        covariates.update(tracks=mapped["tracks"], recording=mapped["recording"], ids=data["ids"])
    checkpoint = weight_path(name)
    if checkpoint is None:
        raise InvalidReplay(f"No verified checkpoint for {name}")
    torch.set_num_threads(1)
    model, _ = core.make_model("cuda:0", seed)
    attach(model, arm)
    model.decoder.use_kin = source["contract"]["use_kin"]
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.requires_grad_(False)
    inputs_only = {k: data[k] for k in ("ids", "past", "num_valid")}
    pred = infer(core, model, inputs_only)
    paths, logits = pred["layer_paths"], pred["layer_logits"]
    n = len(data["ids"])
    if paths.shape != (n, 4, 20, 12, 2) or logits.shape != (n, 4, 20) or not (
        np.isfinite(paths).all() and np.isfinite(logits).all()
    ):
        raise InvalidReplay(f"Unexpected predictions for {name}")
    movement = vector_norm(paths[:, 3, :, -1] - paths[:, 0, :, -1])
    movement_median = float(np.median(movement))
    if movement_median < .02:
        raise InvalidReplay(f"Candidate endpoint movement is only {movement_median:g} m for {name}")
    fields = dict(covariates, movement=np.median(movement, axis=1))
    drift = {}
    layers = {}
    met = ("minFDE", "minADE", "Top1_FDE", "brier_minFDE")
    for layer in range(4):
        letters = {0: "first", 3: "final"}
        record = geometry(paths[:, layer], data["future"], direction, logits[:, layer],
                          compare_order=layer in letters, ordering=(conf_order, risk_order))
        for key in met:
            drift[f"layer{layer}.{key}"] = abs(float(record[key].mean()) - source["layers"][layer]["overall"][key])
        if layer in letters:
            label = letters[layer]
            layers[label] = record
            fields.update({f"{label}_{k}": record[k] for k in (*met, "spread_long", "spread_lat", "entropy", "best_slot")})
            for axis in ("long", "lat"):
                fields[f"{label}_best_error_{axis}_sq"] = record[f"best_error_{axis}"] ** 2
    fields["best_slot_switch"] = layers["first"]["best_slot"] != layers["final"]["best_slot"]
    with np.load(evidence, allow_pickle=False) as old:
        np.testing.assert_array_equal(data["ids"], old["ids"])
        formal_mean_abs = float(np.abs(layers["final"]["minFDE"] - old["minFDE"]).mean())
        drift["archive27.final.mean_absolute_minFDE"] = formal_mean_abs
        formal_max_abs = float(np.abs(layers["final"]["minFDE"] - old["minFDE"]).max())
        np.testing.assert_array_equal(fields["X1"] > 1.3, old["fast"])
        np.testing.assert_array_equal(fields["observed_steady"], old["steady"])
    if fold in EXPLORATORY:
        if not previous.completed(name):
            raise InvalidReplay(f"Archive 28 is incomplete for {name}")
        with np.load(previous.OUT / "runs" / name / "metrics.npz", allow_pickle=False) as old:
            np.testing.assert_array_equal(data["ids"], old["ids"])
            for layer, record in layers.items():
                drift[f"archive28.{layer}.minFDE"] = float(np.abs(record["minFDE"] - old[f"{layer}_conf/minFDE"]).max())
    if any(not np.isfinite(x) or x > TOL for x in drift.values()):
        raise InvalidReplay(f"Replay differs by more than {TOL:g} m for {name}: {drift}")
    dest = metric_path(name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    persist_npz(dest, **fields)
    receipt = dict(status="PASS", binding=binding(name), metrics_sha256=digest(dest), windows=n,
                   layers_reordered_oracle_exact=[0, 3], endpoint_movement_median_m=movement_median,
                   replay_drift=drift, archive27_final_max_abs_minFDE_descriptive=formal_max_abs,
                   device=torch.cuda.get_device_name(0), weight_updates=0)
    write(dest.with_name("RESULT.json"), receipt)


def execute(folds, phase):
    names = [name for name, (_, fold, _) in RUNS.items() if fold in folds]
    pending = [name for name in names if not completed(name)]
    count = len(names) - len(pending)
    log_status(phase, folds=folds, completed_stage=count, stage_total=len(names), total=len(RUNS))
    fetch_weights(pending)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")

    def run(name):
        dest = metric_path(name).parent
        dest.mkdir(parents=True, exist_ok=True)
        with (dest / "worker.log").open("a", encoding="utf-8") as stream:
            process = subprocess.run([sys.executable, "-u", "-B", str(Path(__file__)), "--run", name],
                                     env=env, stdout=stream, stderr=subprocess.STDOUT)
        if process.returncode == 7:
            raise InvalidReplay(f"Replication check failed in {name}; see {dest / 'worker.log'}")
        if process.returncode or not completed(name):
            raise RuntimeError(f"Replay process failed in {name}; see {dest / 'worker.log'}")
        return name

    with ThreadPoolExecutor(max_workers=2) as pool:
        tasks = [pool.submit(run, name) for name in pending]
        for item in as_completed(tasks):
            try:
                name = item.result()
            except Exception:
                for todo in tasks:
                    todo.cancel()
                raise
            count += 1
            log_status(phase, folds=folds, completed_stage=count, stage_total=len(names), last_verified=name, total=len(RUNS))


def aggregate_fold(fold, arm, seeds):
    folder, _ = folders(fold)
    with np.load(folder / "LABELS.npz", allow_pickle=False) as data:
        ids, labels = data["ids"], data["maneuver"]
    rows = []
    for seed in seeds:
        name = f"{arm}_{fold}_s{seed}"
        if not completed(name):
            raise InvalidReplay(f"Missing verified row metrics for {name}")
        with np.load(metric_path(name), allow_pickle=False) as metrics:
            np.testing.assert_array_equal(ids, metrics["ids"])
            rows.append({key: metrics[key] for key in metrics.files})
    return fold_report(rows, labels, fold, seeds)


def analyze(folds):
    result = {}
    for fold in folds:
        result[fold] = {}
        for arm in ARMS:
            print("ANALYZING", fold, arm, flush=True)
            result[fold][arm] = aggregate_fold(fold, arm, FOLDS[fold])
    return result


def verify_preview(preview):
    expected = {
        "eth": ((.4067, -.1212, .0330, .1846, .664, .670), (.2628, -.0878, .0204, .0995, .620, .695)),
        "zara01": ((.1987, -.1892, -.1179, -.1597, .469, .578), (.1094, -.2140, -.1439, -.2026, .453, .588)),
    }
    checks = {}
    for fold, definitions in expected.items():
        checks[fold] = {}
        for arm, targets in zip(ARMS, definitions):
            data = preview[fold][arm]
            earlier = []
            for seed in FOLDS[fold]:
                name = f"{arm}_{fold}_s{seed}"
                path = previous.OUT / "runs" / name / "metrics.npz"
                with np.load(path, allow_pickle=False) as row:
                    earlier.append({k: row[k] for k in ("maneuver", "observed_speed", "steady", "fast",
                                                         "first_conf/minFDE", "final_conf/minFDE")})
            values = earlier[0]
            b = np.mean([r["first_conf/minFDE"] for r in earlier], axis=0)
            d = np.mean([r["final_conf/minFDE"] - r["first_conf/minFDE"] for r in earlier], axis=0)
            labels = values["maneuver"]
            harm = d > .10

            def paired_auc(score):
                compare = score[harm, None] - score[None, ~harm]
                return float(((compare > 0) + .5 * (compare == 0)).mean())

            masks = dict(deceleration=labels == "deceleration", steady=labels == "steady",
                         overall=np.ones(len(b), bool), observed_fast=values["fast"],
                         observed_steady=values["steady"], stop=labels == "stop", turn=labels == "turn")
            archive = {k: float(d[m].sum() / b[m].sum()) for k, m in masks.items()}
            archive["X1_auc"] = paired_auc(values["observed_speed"])
            archive["future_deceleration_auc"] = paired_auc((labels == "deceleration").astype(int))
            reproduced = {key: data["groups"][key]["relative"] for key in masks}
            reproduced["X1_auc"] = data["proxies"]["X1"]["auc"]
            reproduced["future_deceleration_auc"] = data["future_deceleration_auc"]["auc"]
            np.testing.assert_allclose(list(reproduced.values()), list(archive.values()), atol=1e-5, rtol=0)
            document = dict(zip(("deceleration", "steady", "overall", "observed_fast", "X1_auc", "future_deceleration_auc"),
                                targets))
            differences = {k: val - document[k] for k, val in archive.items() if k in document}
            discrepancies = {k: v for k, v in differences.items() if abs(v) > (5.1e-4 if k.endswith("_auc") else 5.1e-5)}
            checks[fold][arm] = dict(replay=reproduced, independently_recomputed_archive28=archive,
                                     document_section8=document, beyond_document_rounding=discrepancies)
    return checks


def render_report(summary):
    decision = summary["decision"]
    rows = ["# EXP-REFINE-PRED", "",
            f"**Decision: {decision['outcome']}**. C1={decision['C1']}, C2={decision['C2']}, C3={decision['C3']}.",
            "", "Frozen QRANK models, 34 replays, zero training. Positive minFDE changes denote higher error.", "",
            "| Fold | Scope | E0 deceleration change (95% CI) | E0 steady change | QFL deceleration change | E0 longitudinal ratio, decel / steady |",
            "| --- | --- | --- | --- | --- | --- |"]
    for fold, report in summary["folds"].items():
        e0, qfl = report[ARMS[0]]["groups"], report[ARMS[1]]["groups"]
        d, s = (e0[k] for k in ("deceleration", "steady"))
        interval = ", ".join(f"{100 * x:+.2f}%" for x in d["ci95"])
        rows.append(f"| {fold.upper()} | {'confirmation' if fold in CONFIRMATION else 'exploratory'} | "
                    f"{d['relative']:+.2%} [{interval}], {d['windows']} windows / {d['tracks']} tracks"
                    f"{' (small)' if d['small'] else ''} | {s['relative']:+.2%} | {qfl['deceleration']['relative']:+.2%} | "
                    f"{d['coverage']['long']['ratio_median']:.3f} / {s['coverage']['long']['ratio_median']:.3f} |")
    rows += ["", "## Observation proxy AUROC (E0)", "", "| Proxy | ETH | ZARA01 | HOTEL | ZARA02 | UNIV | Pass |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for proxy, gate in decision["c2_by_proxy"].items():
        vals = [summary["folds"][f][ARMS[0]]["proxies"][proxy]["auc"] for f in FOLDS]
        rows.append(f"| {proxy} | " + " | ".join(f"{v:.3f}" for v in vals) + f" | {gate['passed']} |")
    rows += ["", "The JSON also reports secondary proxies, all maneuver labels, seed consistency, speed-matched strata,"
             " dependence on the largest contributing tracks, UNIV recording sensitivity, and ZARA02 without seed 44.",
             "", "PREFLIGHT.json records differences between the printed exploratory AUROCs and recomputation from archive 28.",
             "", "Changes compare jointly trained layer outputs. They describe associations within frozen checkpoints."]
    (OUT / "REPORT.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def coordinate():
    OUT.mkdir(exist_ok=True)
    previous.verify_sources()
    seal = lock()
    write(OUT / "SOURCE_LOCK.json", dict(files=seal))
    log_status("TRACK_AUDIT", total=len(RUNS))
    audit = {fold: prepare_population(fold) for fold in FOLDS}
    execute(EXPLORATORY, "EXPLORATORY_REPLAY")
    log_status("PREVIEW_CHECK", completed=16, total=len(RUNS))
    exploratory = analyze(EXPLORATORY)
    checks = verify_preview(exploratory)
    if lock() != seal:
        raise InvalidReplay("Source code changed during preflight")
    write(OUT / "PREFLIGHT.json", dict(status="PASS", archive28_replays=16, preview=checks, tracks=audit, sources=seal))
    execute(CONFIRMATION, "CONFIRMATION_REPLAY")
    log_status("FINAL_ANALYSIS", completed=len(RUNS), total=len(RUNS))
    reports = dict(exploratory, **analyze(CONFIRMATION))
    alternate = {arm: aggregate_fold("zara02", arm, (42, 43)) for arm in ARMS}
    if lock() != seal:
        raise InvalidReplay("Source code changed before decision")
    result = dict(status="COMPLETE", decision=decide(reports), inference_runs=len(RUNS), weight_updates=0,
                  folds=reports, tracks=audit, zara02_without_local_s44=alternate, preflight=digest(OUT / "PREFLIGHT.json"),
                  source_lock=seal, interpretation=dict(
                      small_group="Fewer than 30 windows or 10 tracks are excluded from the registered rules.",
                      tertiles="Boundary ties stay together. If both quantiles are equal, the common value is in the middle.",
                      secondary_auc_direction="Higher initial longitudinal spread (X5) and entropy (X6) are nominated for harm.",
                      shrinkage="Average each layer's spread over seeds in each window, then take the median of the window ratios.",
                      concentration="Top ten by absolute contribution for concentration; remove ten largest signed contributors in sensitivity.",
                      recording="UNIV recorder-specific AUROC is a sensitivity result; the preregistration gives no numeric threshold for disagreement.",
                  ))
    write(OUT / "SUMMARY.json", result)
    render_report(result)
    log_status("COMPLETE", completed=len(RUNS), total=len(RUNS), outcome=result["decision"]["outcome"],
               summary_sha256=digest(OUT / "SUMMARY.json"), report_sha256=digest(OUT / "REPORT.md"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", choices=tuple(RUNS))
    args = parser.parse_args()
    try:
        if args.run:
            replay(args.run)
        else:
            coordinate()
    except (InvalidReplay, AssertionError) as error:
        if not args.run:
            log_status("INVALID", error=str(error))
        print(error, file=sys.stderr)
        sys.exit(7)
    except Exception as error:
        if not args.run:
            log_status("FAILED", error=str(error))
        raise
