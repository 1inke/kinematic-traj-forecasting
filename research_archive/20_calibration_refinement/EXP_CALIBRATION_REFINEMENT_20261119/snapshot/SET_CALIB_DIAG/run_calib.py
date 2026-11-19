"""Execute the registered C0 calibration diagnostic without model inference."""

import argparse
from datetime import datetime
from importlib.metadata import version
import os
import sys
import time
import traceback

import numpy as np
from scipy.special import log_softmax

from calib_data import (ARMS, BINS, OBS, ROOT, STEPS, load_prediction, prepare, read, run_folder,
                        sha, window_save)
from calib_metrics import DRAW_COUNT, LEVELS, SEED, WEIGHTS, coverage, moments, probability, spread_skill
from calib_report import make_figures, report
from run_prec_diag import write

N_RUNS = sum(map(len, ARMS.values()))


def status(phase, **fields):
    write(ROOT / "CURRENT.json", dict(at=datetime.now().astimezone().isoformat(),
                                      pid=os.getpid(), phase=phase, **fields))


def comparison(a, f, ref_a, ref_f, spread, prepared, step):
    out = {}
    for group, mask in prepared["masks"].items():
        qa, qf = float(ref_a[mask].mean()), float(ref_f[mask].mean())
        ma, mf = float(a[mask].mean()), float(f[mask].mean())
        row = dict(n=int(mask.sum()), model_minADE_m=ma, model_minFDE_m=mf,
                   reference_minADE_m=qa, reference_minFDE_m=qf, FDE_relative_excess=mf/qf-1 if qf else None)
        if group.startswith("bin_"):
            empirical = prepared["fit_bins"][group[4:]][f"t{step}_empirical_spread_m"]
            rms = float(np.sqrt(np.square(spread[mask]).mean()))
            row.update(fit_empirical_spread_m=empirical, model_rms_spread_m=rms,
                       spread_ratio=rms/empirical if empirical else None)
        out[group] = row
    return out


def evaluate(model, seed, data, draws, on_progress):
    fit, dev = data["fit"], data["dev"]
    args = (run_folder(model, seed),)
    fit_paths, fit_logits = load_prediction(*args, "fit", fit)
    paths, logits = load_prediction(*args, "dev", dev)
    fit_logp = log_softmax(fit_logits.astype(np.float64), axis=1)
    logp = log_softmax(logits.astype(np.float64), axis=1)
    fit_weights = (np.full_like(fit_logp, 1 / 20), np.exp(fit_logp))
    dev_log_weights = (np.full_like(logp, -np.log(20)), logp)
    score_distance = np.linalg.norm(paths.astype(np.float64) - dev["future"][:, None], axis=-1)
    windows = dict(ids=dev["ids"], half=data["half"], observation=data["obs"], maneuver=data["labels"],
                   group=dev["group"], reference_bin=np.asarray(BINS)[data["dev_bin"]])
    result = {}
    for step in STEPS:
        tag, idx = f"t{step}", step - 1
        endpoint = paths[:, :, idx].astype(np.float64)
        target = dev["future"][:, idx].astype(np.float64)
        spreads, hpd = {}, {}
        for name, p_fit, log_w in zip(WEIGHTS, fit_weights, dev_log_weights):
            fit_s, _ = moments(fit_paths[:, :, idx].astype(np.float64), fit["future"][:, idx], p_fit)
            s, err = moments(endpoint, target, np.exp(log_w))
            if name == "equal":
                equal_s = s
            spreads[name] = spread_skill(fit_s, s, err, data["heading"], data["masks"])
            windows[f"{tag}__{name}__s"] = s
            windows[f"{tag}__{name}__r"] = np.linalg.norm(err, axis=1)

            def variant(mode):
                on_progress(step, name, mode)

            hpd[name], mc = coverage(endpoint, target, log_w, equal_s, data["half"], data["masks"],
                                      *draws, variant)
            windows.update({f"{tag}__{name}__{k}": v for k, v in mc.items()})
        ade, fde = score_distance[..., :step].mean(-1), score_distance[..., idx]
        min_a, min_f = ade.min(1), fde.min(1)
        ref_a, ref_f = data["ref"][step]
        for key, values in (("minADE", min_a), ("minFDE", min_f), ("reference_minADE", ref_a),
                            ("reference_minFDE", ref_f)):
            windows[f"{tag}__{key}"] = values
        result[tag] = dict(spread_skill=spreads, hpd=hpd,
                           probability=probability(logits, fde, (ade <= .2) & (fde <= .4), data["half"]),
                           quantizer=comparison(min_a, min_f, ref_a, ref_f, equal_s, data, step))
    return result, windows


def decisions(results):
    rows = [v["t12"] for v in results["B0"].values()]
    spreads = [v["spread_skill"]["equal"] for v in rows]
    betas = [v["beta"] for v in spreads]
    coverage90 = [v["hpd"]["equal"]["variants"]["global"]["groups"] for v in rows]
    near = [v["obs_stationary"]["coverage"][2] for v in coverage90]
    other = [v["obs_other"]["coverage"][2] for v in coverage90]
    overall = [v["overall"]["coverage"][2] for v in coverage90]
    ratios = [v["quantizer"]["bin_stationary"]["spread_ratio"] for v in rows]
    assert all(v is not None for v in ratios)
    bias = [sum(v["B_over_R"] is not None and v["B_over_R"] > .5 for v in entry["bins"])
            for entry in spreads]

    def gate(values, test):
        mean, n = float(np.mean(values)), sum(bool(test(v)) for v in values)
        return dict(seed_values=values, mean=mean, n_agree=n, pass_gate=bool(test(mean) and n >= 4))

    def reaction(beta):
        return "UNDER_RESPONSIVE" if beta < .8 else "OVER_RESPONSIVE" if beta > 1.25 else "RESPONSIVE"

    beta_mean = float(np.mean(betas))
    beta_label = reaction(beta_mean)
    beta_agree = sum(reaction(v) == beta_label for v in betas)
    reaction_label = beta_label if beta_agree >= 4 else "INCONCLUSIVE"
    easy_agree = sum(p > .95 or r > 2 for p, r in zip(near, ratios))
    easy_mean_gate = bool(np.mean(near) > .95 or np.mean(ratios) > 2)
    easy_wide = easy_mean_gate and easy_agree >= 4
    hard_narrow = gate(other, lambda v: v < .85)
    range_ok = gate(overall, lambda v: .85 <= v <= .95)
    center_bias = gate(bias, lambda v: v >= 5)
    allocation = {name: gate([v["quantizer"]["bin_" + name]["FDE_relative_excess"] for v in rows],
                             lambda x: x >= .05) for name in BINS[:4]}
    misallocated = any(v["pass_gate"] for v in allocation.values())
    raw = [v["probability"]["winner"]["raw"]["ece"] for v in rows]
    calibrated = [v["probability"]["winner"]["calibrated"]["ece"] for v in rows]
    reduction = [1 - b / a if a > 0 else 0. for a, b in zip(raw, calibrated)]
    ece_n = sum(a > .05 and r > .5 for a, r in zip(raw, reduction))
    prob_marker = bool(np.mean(raw) > .05 and np.mean(reduction) > .5 and ece_n >= 4)

    conclusion = "MIXED"
    if reaction_label == "UNDER_RESPONSIVE" and (easy_wide or misallocated):
        conclusion = "STRUCTURED_MISCALIBRATION"
    elif reaction_label == "RESPONSIVE" and range_ok["pass_gate"] and misallocated:
        conclusion = "RANGE_OK_PLACEMENT_BAD"
    elif center_bias["pass_gate"]:
        conclusion = "CENTER_BIAS"
    elif reaction_label == "RESPONSIVE" and range_ok["pass_gate"] and not misallocated:
        conclusion = "RANGE_OK"
    return dict(conclusion=conclusion, horizon=12,
                reaction=dict(label=reaction_label, mean_label=beta_label, mean=beta_mean,
                              seed_values=betas, n_agree=beta_agree),
                EASY_TOO_WIDE=dict(pass_gate=easy_wide, n_agree=easy_agree, mean_gate=easy_mean_gate,
                                   coverage90=near, spread_ratio=ratios),
                HARD_TOO_NARROW=hard_narrow, overall_HPD90_range=range_ok,
                MISALLOCATED_EASY=dict(pass_gate=misallocated, bins=allocation),
                CENTER_BIAS=center_bias,
                PROB_MISCAL=dict(pass_gate=prob_marker, n_agree=ece_n, raw=raw,
                                 calibrated=calibrated, relative_drop=reduction))


def main(preflight_only):
    if not preflight_only and (ROOT / "DONE.json").exists():
        seal = read(ROOT / "DONE.json")
        assert seal["status"] == "SUCCESS"
        for name, digest in seal["files"].items():
            assert sha(ROOT / name) == digest, name
        status("done", completed=N_RUNS, total=N_RUNS)
        return
    start = time.perf_counter()
    status("input_preflight", completed=0, total=N_RUNS)
    data = prepare(lambda phase: status(phase, completed=0, total=N_RUNS))
    write(ROOT / "INPUT_MANIFEST.json", dict(files=data["inputs"], versions={
        p: version(p) for p in ("numpy", "scipy", "scikit-learn", "threadpoolctl", "matplotlib")}))
    write(ROOT / "PREFLIGHT_PASS.json", data["preflight"])
    status("preflight_pass", completed=0, total=N_RUNS)
    if preflight_only:
        print(data["preflight"], flush=True)
        return

    window_dir, checkpoint_dir = ROOT / "windows", ROOT / "checkpoints"
    window_dir.mkdir(exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    shape = (len(data["dev"]["ids"]), DRAW_COUNT)
    draws = (rng.random(shape), rng.standard_normal((*shape, 2)))
    runs = {}
    completed = 0
    window_files = []
    input_digest = sha(ROOT / "INPUT_MANIFEST.json")
    for model, seeds in ARMS.items():
        runs[model] = {}
        for seed in seeds:
            dest = window_dir / f"{model}_s{seed}.npz"
            checkpoint = checkpoint_dir / f"{model}_s{seed}.json"
            record = read(checkpoint) if checkpoint.exists() else {}
            if (record.get("input_digest") == input_digest and dest.exists()
                    and sha(dest) == record["window_digest"]):
                entry = record["result"]
            else:
                def on_progress(step, weights, variant):
                    status("hpd_monte_carlo", model=model, seed=seed, step=step, weights=weights,
                           variant=variant, completed=completed, total=N_RUNS)

                entry, window = evaluate(model, seed, data, draws, on_progress)
                window_save(dest, window)
                write(checkpoint, dict(result=entry, window_digest=sha(dest), input_digest=input_digest))
            runs[model][str(seed)] = entry
            window_files.append(dest)
            completed += 1
            status("seed_complete", model=model, seed=seed, completed=completed, total=N_RUNS)
            print(f"{model} seed {seed}: completed {completed}/{N_RUNS}", flush=True)
    result = dict(experiment="EXP-SET-CALIB-DIAG", random_seed=SEED, n_draws_per_window=DRAW_COUNT,
                  levels=LEVELS.tolist(), preflight=data["preflight"], observation_bins=data["fit_bins"],
                  observation_speed_cuts_mps=data["cuts"], runs=runs, decisions=decisions(runs))
    status("report", completed=completed, total=N_RUNS)
    write(ROOT / "RESULT.json", result)
    figures = make_figures(result, ROOT / "figures")
    (ROOT / "REPORT.md").write_text(report(result), encoding="utf-8")
    files = [ROOT / f for f in ("INPUT_MANIFEST.json", "PREFLIGHT_PASS.json", "RESULT.json", "REPORT.md")]
    files += window_files + figures
    write(ROOT / "DONE.json", dict(status="SUCCESS", elapsed_s=time.perf_counter() - start,
                                   files={p.relative_to(ROOT).as_posix(): sha(p) for p in files}))
    status("done", completed=completed, total=N_RUNS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    cli = parser.parse_args()
    try:
        main(cli.preflight_only)
    except Exception:
        error = traceback.format_exc()
        status("FAILED", error=error)
        print(error, file=sys.stderr, flush=True)
        raise SystemExit(1)
