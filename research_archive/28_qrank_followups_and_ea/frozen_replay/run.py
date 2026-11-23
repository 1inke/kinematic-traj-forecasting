"""Replay identical QRANK weights with two decoder layers and two selectors."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
WORKSPACE = PROJECT / "local/workspace"
OUT = HERE / "results"
sys.path.insert(0, str(PROJECT / "local"))
from cloud_handoff import digest, read, write
from review_download import Archive, FOLDER, REVIEW, names


FOLDS = {"eth": (42, 43, 44, 45, 46), "zara01": (42, 43, 44)}
ARMS = ("O-conf+QFL", "O-conf+E0")
RUNS = {f"{arm}_{fold}_s{seed}": (arm, fold, seed)
        for fold, seeds in FOLDS.items() for seed in seeds for arm in ARMS}
METRICS = ("minADE", "minFDE", "Top1_FDE", "brier_minFDE", "minFDE3", "minFDE5")
PAIRS = (("final_conf", "first_conf"), ("first_risk", "first_conf"), ("final_risk", "final_conf"),
         ("fast_first_conf", "final_conf"))
TOLERANCE = 5e-4


def now():
    return datetime.now(timezone.utc).isoformat()


def source_receipt(name):
    return REVIEW / "stage2/runs" / name / "RESULT.json"


def bound(name):
    arm, fold, seed = RUNS[name]
    r = read(source_receipt(name))
    if (r["status"], r["epochs"], r["contract"]["arm"], r["contract"]["fold"], r["contract"]["seed"]) != (
            "PASS", 64, arm, fold, seed):
        raise ValueError(f"Invalid formal result for {name}")
    return r, {"source_result": digest(source_receipt(name)), "checkpoint": r["artifacts"]["final_ema.pt"],
               "replay_code": digest(Path(__file__))}


def completed(name):
    result = OUT / "runs" / name / "RESULT.json"
    metrics = result.parent / "metrics.npz"
    if not result.is_file() or not metrics.is_file():
        return False
    saved = read(result)
    _, binding = bound(name)
    return saved["status"] == "PASS" and saved["binding"] == binding and digest(metrics) == saved["metrics_sha256"]


def drive_index(archive, pending):
    stage = names(archive.children([FOLDER]))["stage2"]
    runs = names(archive.children([stage["id"]]))["runs"]
    dirs = {r["id"]: r["name"] for r in archive.children([runs["id"]], "and mimeType='application/vnd.google-apps.folder'")
            if r["name"] in pending}
    weights = {dirs[w["parents"][0]]: w for w in archive.children(list(dirs), "and name='final_ema.pt'")}
    if set(weights) != set(pending):
        raise ValueError(f"Missing Drive checkpoints: {set(pending) - set(weights)}")
    return weights


def verify_sources():
    code = read(REVIEW / "cloud/RUNTIME.json")["code"]
    required = ("qrank_model.py", "qrank_eval.py", "advisor_train.py", "advisor_eval.py", "order_numpy.py", "eth_speed_math.py")
    for name in required:
        if digest(WORKSPACE / name) != code[name]:
            raise ValueError(f"Source differs from the formal study: {name}")
    manifest = WORKSPACE / "baseline_sources.manifest.json"
    if digest(manifest) != read(REVIEW / "cloud/RUNTIME.json")["data"]["baseline_sources"]:
        raise ValueError("Baseline source manifest differs from the formal study")
    for name, sha in read(manifest)["members"].items():
        if digest(WORKSPACE / name) != sha:
            raise ValueError(f"Baseline source differs from manifest: {name}")


def replay(name):
    import numpy as np
    import torch

    sys.path.insert(0, str(WORKSPACE))
    from advisor_eval import forecast_metrics, groups
    from advisor_train import core, inputs, load_data
    from eth_speed_math import maneuver_labels, observed_speed
    from order_numpy import conf_order, risk_order, softmax
    from qrank_eval import infer
    from qrank_model import attach

    arm, fold, seed = RUNS[name]
    reference, binding = bound(name)
    target = OUT / "runs" / name
    checkpoint = OUT / "weights" / f"{name}.pt"
    if digest(checkpoint) != binding["checkpoint"]:
        raise ValueError(f"Checkpoint differs from its receipt: {name}")
    test, size = inputs(fold)[1]
    if digest(test) != reference["contract"]["data"][test.name]:
        raise ValueError(f"Evaluation split differs from the formal run: {name}")
    original = source_receipt(name).with_name("dev_metrics.npz")
    if digest(original) != reference["artifacts"][original.name]:
        raise ValueError(f"Formal metrics differ from their receipt: {name}")

    data = load_data(test, size)
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for the local two-worker replay")
    model, _ = core.make_model("cuda:0", seed)
    attach(model, arm)
    model.decoder.use_kin = reference["contract"]["use_kin"]
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.requires_grad_(False)
    pred = infer(core, model, data)
    if pred["layer_paths"].shape != (size, 4, 20, 12, 2) or pred["layer_logits"].shape != (size, 4, 20):
        raise ValueError("Replay must retain four layers of 20 candidates per window")

    masks = groups(data)
    speed = observed_speed(data["past"][:, 0])
    labels = maneuver_labels(data["past"], data["future"])
    fast = masks["fast"] = speed > 1.3
    masks.update(nonfast=~fast, fast_steady=fast & (labels == "steady"),
                 fast_deceleration=fast & np.isin(labels, ("deceleration", "stop")),
                 fast_turn=fast & np.isin(labels, ("turn", "turn_and_speed")))
    conditions = {}
    error_delta = {}
    row = np.arange(size)
    for layer, index in (("first", 0), ("final", 3)):
        paths = pred["layer_paths"][:, index]
        logits = pred["layer_logits"][:, index]
        err = np.linalg.norm(paths - data["future"][:, None], axis=-1)
        fde = err[:, :, -1]
        common = forecast_metrics(paths, logits, data["future"])
        orders = {"conf": conf_order(logits), "risk": risk_order(paths[:, :, -1], softmax(logits))}
        for selector, order in orders.items():
            chosen = np.take_along_axis(fde, order, axis=1)
            conditions[f"{layer}_{selector}"] = {m: common[m] for m in ("minADE", "minFDE", "brier_minFDE")}
            conditions[f"{layer}_{selector}"].update(Top1_FDE=chosen[:, 0], minFDE3=chosen[:, :3].min(1),
                                                     minFDE5=chosen[:, :5].min(1))
        if not np.array_equal(orders["conf"][:, 0], logits.argmax(1)) or not np.array_equal(
                conditions[f"{layer}_conf"]["Top1_FDE"], fde[row, logits.argmax(1)]):
            raise ValueError("Confidence selection differs from the recorded definition")
        saved_layer = reference["layers"][index]["overall"]
        for metric in ("minADE", "minFDE", "Top1_FDE", "brier_minFDE"):
            error_delta[f"{layer}.{metric}"] = abs(conditions[f"{layer}_conf"][metric].mean() - saved_layer[metric])
    with np.load(original, allow_pickle=False) as verified:
        for key in ("ids", "fast", "hard", "overall"):
            got = data["ids"] if key == "ids" else masks[key]
            if not np.array_equal(got, verified[key]):
                raise ValueError(f"Formal window/mask differs: {key}")
        for metric in ("minADE", "minFDE", "Top1_FDE", "brier_minFDE"):
            error_delta[f"final_npz.{metric}"] = abs((conditions["final_conf"][metric] - verified[metric]).mean())
    if not all(np.isfinite(x) and x <= TOLERANCE for x in error_delta.values()):
        raise ValueError(f"Replay disagrees with the original metrics beyond {TOLERANCE:g}: {error_delta}")

    conditions["fast_first_conf"] = {m: np.where(fast, conditions["first_conf"][m], conditions["final_conf"][m])
                                     for m in METRICS}
    arrays = {f"{condition}/{metric}": value for condition, metrics in conditions.items() for metric, value in metrics.items()}
    target.mkdir(parents=True, exist_ok=True)
    part = target / "metrics.part.npz"
    result = target / "metrics.npz"
    np.savez_compressed(part, ids=data["ids"], observed_speed=speed, maneuver=labels, **masks, **arrays)
    part.replace(result)
    write(target / "RESULT.json", dict(status="PASS", name=name, binding=binding, windows=size,
                                       metrics_sha256=digest(result), replay_difference=error_delta,
                                       tolerance=TOLERANCE, gpu=torch.cuda.get_device_name(0), at=now()))
    print(f"PASS {name}, {size} windows, max reference drift {max(error_delta.values()):.6g}", flush=True)


def analyze():
    import numpy as np

    sys.path.insert(0, str(WORKSPACE))
    from qrank_decisions import short_compare

    result = {}
    labels = ("overall", "fast", "nonfast", "fast_steady", "fast_deceleration", "fast_turn", "hard", "stationary")
    for fold, seeds in FOLDS.items():
        result[fold] = {}
        for arm in ARMS:
            cohorts = []
            for seed in seeds:
                name = f"{arm}_{fold}_s{seed}"
                if not completed(name):
                    raise ValueError(f"Missing verified replay: {name}")
                with np.load(OUT / "runs" / name / "metrics.npz", allow_pickle=False) as z:
                    cohorts.append({k: z[k] for k in z.files})
            result[fold][arm] = {}
            for group in labels:
                mask = cohorts[0][group]
                if not mask.any():
                    continue
                groups = {"windows": int(mask.sum()), "small_group": bool(mask.sum() < 20), "conditions": {}, "paired": {}}
                for condition in dict.fromkeys(c for pair in PAIRS for c in pair):
                    groups["conditions"][condition] = {m: float(np.mean([r[f"{condition}/{m}"][mask].mean() for r in cohorts]))
                                                       for m in METRICS}
                for tried, reference in PAIRS:
                    def arm_rows(condition):
                        return [{"ids": r["ids"], group: r[group], **{m: r[f"{condition}/{m}"] for m in METRICS}}
                                for r in cohorts]
                    a, b = arm_rows(tried), arm_rows(reference)
                    groups["paired"][f"{tried}_vs_{reference}"] = {m: short_compare(a, b, m, group) for m in METRICS}
                result[fold][arm][group] = groups
    write(OUT / "SUMMARY.json", dict(status="COMPLETE", folds=FOLDS, conditions=PAIRS, report=result,
                                     fixed_candidate_count=20, interpretation="Exploratory replay; future labels are evaluation-only"))


def coordinate():
    OUT.mkdir(exist_ok=True)
    verify_sources()
    pending = [name for name in RUNS if not completed(name)]
    write(OUT / "CURRENT.json", dict(status="PREPARING", completed=len(RUNS) - len(pending), total=len(RUNS), parallel=2, at=now()))
    archive = Archive() if pending else None
    remote = drive_index(archive, pending) if pending else {}
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

    def handle(name):
        _, binding = bound(name)
        archive.download(remote[name], OUT / "weights" / f"{name}.pt", binding["checkpoint"])
        dest = OUT / "runs" / name
        dest.mkdir(parents=True, exist_ok=True)
        with (dest / "worker.log").open("a", encoding="utf-8") as stream:
            subprocess.run([sys.executable, "-u", "-B", str(Path(__file__)), "--run", name],
                           env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        if not completed(name):
            raise ValueError(f"Worker did not publish a verified replay: {name}")
        return name

    count = len(RUNS) - len(pending)
    with ThreadPoolExecutor(max_workers=2) as pool:
        write(OUT / "CURRENT.json", dict(status="RUNNING", completed=count, total=len(RUNS), parallel=2, at=now()))
        futures = [pool.submit(handle, name) for name in pending]
        for future in as_completed(futures):
            try:
                print("COMPLETE", future.result(), flush=True)
            except Exception:
                for waiting in futures:
                    waiting.cancel()
                raise
            count += 1
            write(OUT / "CURRENT.json", dict(status="RUNNING", completed=count, total=len(RUNS), parallel=2, at=now()))
    analyze()
    write(OUT / "CURRENT.json", dict(status="COMPLETE", completed=count, total=len(RUNS), parallel=2,
                                    summary_sha256=digest(OUT / "SUMMARY.json"), at=now()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", choices=tuple(RUNS))
    args = parser.parse_args()
    if args.run:
        replay(args.run)
    else:
        try:
            coordinate()
        except Exception as error:
            write(OUT / "FAILURE.json", dict(status="FAILED", error=str(error), at=now()))
            write(OUT / "CURRENT.json", dict(status="FAILED", error=str(error), total=len(RUNS), parallel=2, at=now()))
            raise
