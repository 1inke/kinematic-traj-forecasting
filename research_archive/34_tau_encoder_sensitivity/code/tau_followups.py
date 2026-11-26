"""Train and evaluate the registered EXP-TAU follow-up cells."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

from followup_geometry import learned_top_p, matched_distance
from tau_run import (ORIGINAL_MODEL, SEEDS, checkpoint, core, data_paths, digest,
                     forecast_metrics, latest, load_data, safe_npz, write_json)
import tau_geometry


E1 = ("tau", "distance", "full", "top_p")
E2 = ("tau", "distance", "top_p")
E3 = ("base", "state", "sparse", "both")
E5 = ("r06_t24", "r06_t30", "r06_t48", "r02_t24", "r02_t30", "r02_t48")
ANCHOR = "tau_100"
SETUPS = {
    **{f"{arm}_{pct}": dict(arm=arm, pct=pct, cutoff=2.4, radius=.6)
       for arm in E1 for pct in ((100,) if arm == "full" else (25, 50, 100))},
    **{f"encoder_{arm}": dict(arm=arm, pct=100) for arm in E3},
    **{f"sens_{name}": dict(arm="tau", pct=100, cutoff=int(name[-2:])/10,
                            radius=int(name[1:3])/10) for name in E5 if name != "r06_t24"},
}
BATCHES = {
    "selection": [f"{arm}_100" for arm in E1]
                 + [f"{arm}_{pct}" for pct in (25, 50) for arm in E2],
    "encoder-sensitivity": [f"encoder_{arm}" for arm in E3]
                           + [ANCHOR] + [f"sens_{name}" for name in E5 if name != "r06_t24"],
}


def read(path):
    return json.loads(path.read_bytes())


def make_model(name, device, seed):
    setting = SETUPS[name]
    if name.startswith("encoder_"):
        namespace = ORIGINAL_MODEL.__globals__
        original = namespace["FrontEncoderASTRA"]

        def paired_constructor(cfg, variant):
            base = original(cfg, "f0_baseline")
            donor = original(cfg, "f3_factorized")
            arm = setting["arm"]
            base.use_self_state = arm in ("state", "both")
            if base.use_self_state:
                base.robust_state = donor.robust_state
                base.self_state_gate = donor.self_state_gate
            if arm in ("sparse", "both"):
                base.agent_aware_trans_encoder = donor.agent_aware_trans_encoder
            return base

        namespace["FrontEncoderASTRA"] = paired_constructor
        try:
            return ORIGINAL_MODEL(device, seed)
        finally:
            namespace["FrontEncoderASTRA"] = original

    model, cfg = ORIGINAL_MODEL(device, seed)
    arm = setting["arm"]
    tau_geometry.attach(model, "T2" if arm in ("tau", "distance") else "T3", 2.)
    model.graph_builder.tau_cutoff = setting["cutoff"]
    model.agent_aware_trans_encoder.tau_bias_radius = setting["radius"]
    if arm in ("tau", "distance"):
        model.graph_builder.tau_radius = setting["radius"]
        model.graph_builder.tau_followup_selector = arm
    else:
        model.graph_builder.mode = "full"
        if arm == "top_p":
            model.agent_aware_trans_encoder.tau_followup_top_p = .75
    return model, cfg


def subset_indices(count, seed, fraction):
    if fraction == 100:
        return np.arange(count)
    rng = np.random.default_rng(np.random.SeedSequence([20261126, seed]))
    return np.sort(rng.permutation(count)[:round(count * fraction / 100)])


def make_trainer(fit, idx, name, device, seed, epochs, batch):
    core.make_model = lambda device, seed: make_model(name, device, seed)
    try:
        return core.Trainer(fit, idx, batch=batch, epochs=epochs, device=device,
                            seed=seed, order_seed=seed, arm="B0")
    finally:
        core.make_model = ORIGINAL_MODEL


def tracks_for(ids):
    return np.array(["|".join(window.split("|")[:2]) for window in ids])


def validate_result(out, name, seed, ids):
    result_path, metrics_path = out/"RESULT.json", out/"METRICS.npz"
    if not result_path.is_file() or not metrics_path.is_file():
        return False
    result = read(result_path)
    contract = result["contract"]
    assert (contract["name"], contract["seed"]) == (name, seed)
    assert digest(metrics_path) == result["metrics_sha256"]
    with np.load(metrics_path, allow_pickle=False) as metrics:
        np.testing.assert_array_equal(metrics["ids"], ids)
        np.testing.assert_array_equal(metrics["tracks"], tracks_for(ids))
    return True


def import_prior(prior, out, data, seed, original_assets, protocol):
    source = prior/"runs"/f"C0_T2_s{seed}"
    report = read(source/"RESULT.json")
    trained = read(source/"TRAINED.json")
    prior_contract = report["contract"]
    fit_path, fit_count = data_paths(data, "C0", False)
    dev_path, dev_count = data_paths(data, "C0", True)
    assert prior_contract == trained["contract"] and trained["updates"] == 21760
    assert prior_contract["assets"] == original_assets and prior_contract["fit_sha256"] == digest(fit_path)
    assert (prior_contract["arm"], prior_contract["fold"], prior_contract["seed"]) == ("T2", "C0", seed)
    assert (prior_contract["power"], prior_contract["epochs"], prior_contract["batch"]) == (2., 64, 64)
    assert prior_contract["gpu"] == "Tesla T4"
    metric_path = source/"METRICS.npz"
    assert digest(metric_path) == report["metrics_sha256"]
    dev = load_data(dev_path, dev_count)
    tracks = tracks_for(dev["ids"])
    with np.load(metric_path, allow_pickle=False) as metrics:
        np.testing.assert_array_equal(metrics["ids"], dev["ids"])
        np.testing.assert_array_equal(metrics["tracks"], tracks)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(metric_path, out/"METRICS.npz")
    report["kaggle_original_result_sha256"] = digest(source/"RESULT.json")
    report["contract"] = dict(name=ANCHOR, seed=seed, settings=SETUPS[ANCHOR],
                              source="EXP_TAU_KAGGLE_v2", fit_sha256=digest(fit_path),
                              fit_windows=fit_count, epochs=64, batch=64,
                              protocol=protocol, original=prior_contract)
    write_json(out/"RESULT.json", report)
    print(f"REUSED {out.name}", flush=True)


def run(args):
    fit_path, fit_count = data_paths(args.data, "C0", False)
    dev = load_data(*data_paths(args.data, "C0", True))
    if not args.smoke and validate_result(args.out, args.name, args.seed, dev["ids"]):
        assert read(args.out/"RESULT.json")["contract"]["protocol"] == args.protocol
        return
    fit = load_data(fit_path, fit_count)
    idx = subset_indices(fit_count, args.seed, SETUPS[args.name]["pct"])
    if args.smoke:
        idx = idx[:16]
        dev = {key: value[:8] for key, value in dev.items()}
    args.out.mkdir(parents=True, exist_ok=True)
    epochs, batch = (2, 8) if args.smoke else (64, 64)
    device = "cpu" if args.cpu else "cuda:0"
    gpu = "CPU" if args.cpu else torch.cuda.get_device_name(0)
    assert args.smoke or gpu == "Tesla T4" or "L4" in gpu.split(), gpu
    contract = dict(name=args.name, seed=args.seed, settings=SETUPS[args.name],
                    fit_windows=len(idx), fit_sha256=digest(fit_path),
                    fit_indices_sha256=sha256(idx.astype("<i8").tobytes()).hexdigest(),
                    protocol=args.protocol, epochs=epochs, batch=batch, gpu=gpu,
                    torch=str(torch.__version__), cuda=torch.version.cuda)
    if (args.out/"CONTRACT.json").is_file():
        assert read(args.out/"CONTRACT.json") == contract
    else:
        write_json(args.out/"CONTRACT.json", contract)
    trainer = make_trainer(fit, idx, args.name, device, args.seed, epochs, batch)
    start = latest(args.out, core, trainer, contract)
    for epoch in range(start + 1, epochs + 1):
        if time.time() >= args.deadline:
            if trainer.epoch and not (args.out/f"epoch_{trainer.epoch:03d}.json").is_file():
                checkpoint(args.out, core, trainer, contract)
            return
        now = time.monotonic()
        for _ in range(trainer.steps_per_epoch):
            trainer.step()
        if epoch % 8 == 0 or epoch in (1, epochs):
            checkpoint(args.out, core, trainer, contract)
            for old in sorted(args.out.glob("epoch_???.json"))[:-2]:
                old.unlink()
                old.with_suffix(".pt").unlink(missing_ok=True)
        status = dict(epoch=epoch, updates=trainer.updates, seconds=time.monotonic()-now)
        write_json(args.out/"CURRENT.json", status)
        print(f"TRAIN {args.out.name} {json.dumps(status)}", flush=True)
    model = trainer.ema.module
    paths, logits = core.infer(model, dev)
    assert np.isfinite(paths).all() and np.isfinite(logits).all()
    values = forecast_metrics(paths, logits, dev["future"])
    tracks = tracks_for(dev["ids"])
    with torch.no_grad():
        p, n = (torch.as_tensor(dev[key][:64], device=device) for key in ("past", "num_valid"))
        _, _, graph, *_ = core.prepare(p)
        context = model.graph_builder.build_context(graph.transpose(1, 2), n)
        valid_edges = (context.adjacency & ~torch.eye(graph.shape[1], device=device, dtype=torch.bool))
        core.prediction(model, p, n)
        encoder = model.agent_aware_trans_encoder
        top_p_allowed = getattr(encoder, "tau_followup_allowed", None)
        graph_degree = float(valid_edges.sum() / n.sum())
        attention_degree = None if top_p_allowed is None else float(
            (top_p_allowed & ~torch.eye(graph.shape[1], device=device, dtype=torch.bool)).sum() / n.sum())
    bias = getattr(encoder, "tau_beta", None)
    receipt = dict(contract=contract, windows=len(tracks), tracks=len(np.unique(tracks)),
                   updates=trainer.updates, parameters=sum(p.numel() for p in model.parameters()),
                   metrics={key: float(value.mean()) for key, value in values.items()},
                   bias=None if bias is None else bias.detach().cpu().tolist(),
                   first_64_graph_degree=graph_degree, first_64_pruned_attention_degree=attention_degree,
                   checkpoint_sha256=digest(args.out/f"epoch_{epochs:03d}.pt"))
    receipt["metrics_sha256"] = safe_npz(args.out/"METRICS.npz", ids=dev["ids"], tracks=tracks, **values)
    write_json(args.out/"RESULT.json", receipt)
    print(f"EVAL {args.out.name} minFDE={receipt['metrics']['minFDE']:.6f}", flush=True)


def preflight(args):
    device = "cpu" if args.cpu else "cuda:0"
    fit = load_data(*data_paths(args.data, "C0", False))
    p, n = (torch.as_tensor(fit[k][:16], device=device) for k in ("past", "num_valid"))
    _, _, graph, *_ = core.prepare(p)
    models = {name: make_model(name, device, 42)[0].eval()
              for name in (ANCHOR, "distance_100", "full_100", "top_p_100", "sens_r02_t48")}
    previous, _ = ORIGINAL_MODEL(device, 42)
    tau_geometry.attach(previous, "T2", 2.)
    with torch.no_grad():
        pred = core.prediction(previous.eval(), p, n)
        current = core.prediction(models[ANCHOR], p, n)
        for l, r in zip(pred[:2], current[:2]):
            torch.testing.assert_close(l, r, atol=0, rtol=0)
        def adj(model):
            return model.graph_builder.build_context(graph.transpose(1, 2), n).adjacency
        a, d, f, rad = (adj(models[name]) for name in (ANCHOR, "distance_100", "full_100", "sens_r02_t48"))
        assert torch.equal(a.sum(-1), d.sum(-1))
        assert (a != d).any() and (a != rad).any()
        assert (f.sum(-1) >= a.sum(-1)).all()
        top = models["top_p_100"]
        core.prediction(top, p, n)
        chosen = top.agent_aware_trans_encoder.tau_followup_allowed
        assert not (chosen & ~f).any() and (chosen != f).any()
        synthetic_scores = torch.tensor([0., 8., -6.], device=device).view(1, 1, 1, 3, 1).expand(1, 1, 3, 3, 1)
        synthetic_allowed = torch.ones((1,3,3), device=device, dtype=torch.bool)
        assert learned_top_p(synthetic_scores, synthetic_allowed, .75)[0,0].tolist() == [True, True, False]
    assert set(subset_indices(len(fit["ids"]),42,25)) < set(subset_indices(len(fit["ids"]),42,50))
    for name in ("top_p_100", "encoder_base", "encoder_state", "encoder_sparse", "encoder_both", "sens_r02_t48"):
        trainer = make_trainer(fit, np.arange(16), name, device, 42, 2, 8)
        trainer.step()
        trainer.step()
        if name in ("top_p_100", "sens_r02_t48"):
            assert any(k.endswith("tau_beta") for k in trainer.connected)
        if name.startswith("encoder"):
            assert any(k.startswith("robust_state.") for k in trainer.connected) == (
                name in ("encoder_state","encoder_both"))
    print(f"FOLLOWUP_PREFLIGHT_PASS device={device}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("train", "preflight"))
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--name", choices=SETUPS, default=ANCHOR)
    p.add_argument("--seed", type=int, choices=SEEDS, default=42)
    p.add_argument("--protocol", default="local")
    p.add_argument("--deadline", type=float, default=float("inf"))
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(1)
    (run if args.command == "train" else preflight)(args)


if __name__ == "__main__":
    main()
