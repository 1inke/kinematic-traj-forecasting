"""Unified train/evaluate/preflight runner for Encoder C1/C2 and Decoder Cascade+Cover studies."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "qoder_work/NEXT_PHASE_EXEC_20261114"
for _p in (ROOT, BASE, ROOT / "ASTRA_ACCURACY_Q20", ROOT / "FRONT_ENCODER_SERIES", ROOT / "HIERARCHICAL_DECODER_SERIES"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import common  # noqa: E402
common.ASTRA = ROOT / "ASTRA_ACCURACY_Q20"
from prepare_inputs import digest  # noqa: E402
from run_b0 import checkpoint, latest, load_data, publish, write_json  # noqa: E402
from advisor_eval import forecast_metrics, groups as eval_groups, safe_npz  # noqa: E402
from order_numpy import conf_order, softmax  # noqa: E402
import train_core as core  # noqa: E402

from encoder_c1c2 import (  # noqa: E402
    ENCODER_ARMS,
    configure_encoder,
    encoder_diagnostics,
    lagged_relation_features,
)
from decoder_cascade_cover import (  # noqa: E402
    DECODER_ARMS,
    THETA_D,
    attach_decoder,
    cascade_decoder_loss,
    cover_order_numpy,
    cover_order_torch,
)

SEEDS = (42, 43, 44)
DT = 0.4
ORIGINAL_MAKE_MODEL = core.make_model


def data_paths(data_root: Path, fold: str, evaluation: bool) -> tuple[Path, int]:
    if fold == "C0":
        split, count = ("dev", 4753) if evaluation else ("fit", 21719)
        return data_root / "qoder_work/NEXT_PHASE_EXEC_20261114/C0" / f"{split}.npz", count
    audit = json.loads((data_root / "fivefold/source_audit.json").read_bytes())
    assert audit["recording_isolation_passed"]
    split = "test" if evaluation else "train"
    return data_root / f"fivefold/{fold}/{split}.npz", audit["folds"][fold]["splits"][split]["windows"]


def tracks_for(ids: np.ndarray) -> np.ndarray:
    return np.array(["|".join(str(w).split("|")[:2]) for w in ids])


def recordings_for(ids: np.ndarray) -> np.ndarray:
    return np.array([str(w).split("|", 1)[0] for w in ids])


def fast_mask(past_full: np.ndarray) -> np.ndarray:
    p = np.asarray(past_full, dtype=np.float64)[:, 0]
    return np.linalg.norm(np.diff(p, axis=1), axis=-1).mean(1) / DT > 1.3


def build_model_for(study: str, arm: str, device: str, seed: int):
    model, cfg = ORIGINAL_MAKE_MODEL(device, seed)
    if study == "encoder":
        configure_encoder(model, cfg, arm, seed)
    elif study == "decoder":
        attach_decoder(model, cfg, arm)
    else:
        raise ValueError(f"Unknown study: {study}")
    return model, cfg


class UnifiedTrainer(core.Trainer):
    def __init__(self, data, study: str, arm: str, seed: int, device: str, epochs: int, batch: int = 64):
        def factory(dev, sd):
            return build_model_for(study, arm, dev, sd)

        core.make_model = factory
        try:
            super().__init__(
                data,
                np.arange(len(data["ids"])),
                batch=batch,
                epochs=epochs,
                device=device,
                seed=seed,
                order_seed=seed,
                arm="B0",
            )
        finally:
            core.make_model = ORIGINAL_MAKE_MODEL

        self.study = study
        self.exp_arm = arm
        # Ensure optimizer includes any newly attached/unfrozen trainable params and excludes frozen ones (E1)
        self.ema = core.ModelEMA(self.model, 0.995)
        self.optimizer, self.scheduler = core.optimizer_scheduler(self.model, self.steps_per_epoch, epochs)

    def step(self):
        if self.order is None or self.cursor == len(self.order):
            if self.epoch >= self.epochs:
                raise StopIteration
            self.order = self.indices[torch.randperm(len(self.indices), generator=self.generator).numpy()]
            self.cursor, self.epoch = 0, self.epoch + 1

        idx = self.order[self.cursor : self.cursor + self.batch]
        self.model.train()
        p = torch.as_tensor(self.data["past"][idx], device=self.device)
        y = torch.as_tensor(self.data["future"][idx], device=self.device)
        n = torch.as_tensor(self.data["num_valid"][idx], device=self.device)
        out, logits, target, _ = core.prediction(self.model, p, n, y, augment=True)

        if self.study == "encoder" or self.exp_arm == "M0":
            loss, detail = core.loss_value(out, target, logits, self.loss_fn)
            extra_info = {}
        else:
            loss, extra_info = cascade_decoder_loss(core, self.model, self.exp_arm, target, self.loss_fn)
            detail = dict(regression=extra_info["regression"], classification=extra_info["classification"])

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss in {self.study}/{self.exp_arm} at epoch {self.epoch}")

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if not torch.isfinite(param.grad).all():
                    raise FloatingPointError(f"Nonfinite grad in {name}")
                if param.grad.abs().any():
                    self.connected.add(name)
        trainable = [param for param in self.model.parameters() if param.requires_grad]
        grad = torch.nn.utils.clip_grad_norm_(trainable, 5.0, error_if_nonfinite=True)
        self.optimizer.step()
        self.ema.update(self.model)
        self.scheduler.step()
        self.updates += 1
        self.cursor += len(idx)
        self.batch_orders.append(self.data["ids"][idx].tolist())

        row = dict(
            epoch=self.epoch,
            update=self.updates,
            cursor=self.cursor,
            loss=float(loss.detach()),
            regression=float(detail["regression"]),
            classification=float(detail["classification"]),
            grad_norm=float(grad),
            lr=self.optimizer.param_groups[0]["lr"],
        )
        if "l0_pos_window" in extra_info:
            row["l0_pos_sum"] = float(extra_info["l0_pos_window"].sum())
            row["l0_match_sum"] = float(extra_info["l0_match_window"].sum())
            row["l0_count"] = int(len(extra_info["l0_pos_window"]))
        if "l1_match_rate" in extra_info:
            row["l1_match_rate"] = extra_info["l1_match_rate"]
        if "ld_match_rate" in extra_info:
            row["ld_match_rate"] = extra_info["ld_match_rate"]
        self.history.append(row)
        return row


def infer_full(model, data, batch: int = 64) -> dict[str, np.ndarray]:
    """Run deterministic EMA inference, capturing L0 & final paths/logits and optional disc order."""
    model.eval()
    device = next(model.parameters()).device
    dec = model.decoder
    is_cascade = hasattr(dec, "disc_out") and hasattr(dec, "outs")
    chunks: dict[str, list[np.ndarray]] = {}

    with torch.no_grad():
        for start in range(0, len(data["ids"]), batch):
            sl = slice(start, start + batch)
            p = torch.as_tensor(data["past"][sl], device=device)
            n = torch.as_tensor(data["num_valid"][sl], device=device)
            out, logits, _, tf = core.prediction(model, p, n)
            paths = core.world_paths(out, tf).double().cpu().numpy()
            logits_np = logits[:, 0].cpu().numpy()
            chunks.setdefault("paths", []).append(paths)
            chunks.setdefault("logits", []).append(logits_np)
            if is_cascade:
                l0_paths = core.world_paths(dec.outs[0]["paths"][:, None], tf).double().cpu().numpy()
                l0_logits = dec.outs[0]["logits"].cpu().numpy()
                chunks.setdefault("l0_paths", []).append(l0_paths)
                chunks.setdefault("l0_logits", []).append(l0_logits)
                if dec.disc_out is not None:
                    chunks.setdefault("disc_order", []).append(dec.disc_out["order"].cpu().numpy().astype(np.uint8))
                    chunks.setdefault("geom_logits", []).append(dec.disc_out["geom_logits"].cpu().numpy())
                    diff_yd = float((dec.outs[-1]["paths"] - dec.disc_out["paths"]).abs().max())
                    chunks.setdefault("mc_b_max_diff", []).append(np.array([diff_yd], dtype=np.float64))

    return {k: np.concatenate(v, axis=0) for k, v in chunks.items()}


def fit_val_subset(fit: dict, size: int = 2500) -> dict:
    """Deterministic track-isolated validation subset of C0 fit for selecting tau_out."""
    tracks = tracks_for(fit["ids"])
    uniq_tracks, inv = np.unique(tracks, return_inverse=True)
    rng = np.random.default_rng(np.random.SeedSequence(20261127))
    perm = rng.permutation(len(uniq_tracks))
    chosen = []
    total = 0
    for t_id in perm:
        idxs = np.flatnonzero(inv == t_id)
        chosen.append(idxs)
        total += len(idxs)
        if total >= size:
            break
    sel = np.sort(np.concatenate(chosen))
    return {k: v[sel] for k, v in fit.items()}


def effective_good_clusters(endpoints: np.ndarray, logits: np.ndarray, good_mask: np.ndarray, radius: float = 0.2) -> np.ndarray:
    """GT-free greedy deduplication count of distinct good candidate modes per window."""
    N, K, _ = endpoints.shape
    counts = np.zeros(N, dtype=np.int32)
    order_all = np.argsort(-logits, axis=1, kind="stable")
    for i in range(N):
        g = good_mask[i]
        if not g.any():
            continue
        chosen = []
        for idx in order_all[i]:
            if not g[idx]:
                continue
            pt = endpoints[i, idx]
            if all(np.linalg.norm(pt - endpoints[i, prev]) > radius for prev in chosen):
                chosen.append(idx)
        counts[i] = len(chosen)
    return counts


def compute_kinematics(paths: np.ndarray, past_full: np.ndarray) -> dict[str, np.ndarray]:
    """Compute per-window speed p95, acceleration mean/p95, and jerk mean/p95 over KxT."""
    p0 = np.asarray(past_full, dtype=np.float64)[:, 0]  # [N, 8, 2]
    v_obs = np.diff(p0[:, -3:], axis=1) / DT           # [N, 2, 2]: steps -2->-1, -1->0
    a_obs_last = (v_obs[:, 1:2] - v_obs[:, 0:1]) / DT  # [N, 1, 2]

    N, K, T, _ = paths.shape
    seq = np.concatenate([np.broadcast_to(p0[:, None, -1:], (N, K, 1, 2)), paths], axis=2)
    v = np.diff(seq, axis=2) / DT                      # [N, K, 12, 2]
    v_ext = np.concatenate([np.broadcast_to(v_obs[:, None, 1:2], (N, K, 1, 2)), v], axis=2)
    a = np.diff(v_ext, axis=2) / DT                    # [N, K, 12, 2]
    a_ext = np.concatenate([np.broadcast_to(a_obs_last[:, None], (N, K, 1, 2)), a], axis=2)
    j = np.diff(a_ext, axis=2) / DT                    # [N, K, 12, 2]

    spd = np.linalg.norm(v, axis=-1).reshape(N, -1)
    acc = np.linalg.norm(a, axis=-1).reshape(N, -1)
    jrk = np.linalg.norm(j, axis=-1).reshape(N, -1)
    return dict(
        speed_p95=np.quantile(spd, 0.95, axis=1),
        acc_mean=acc.mean(axis=1),
        acc_p95=np.quantile(acc, 0.95, axis=1),
        jerk_mean=jrk.mean(axis=1),
        jerk_p95=np.quantile(jrk, 0.95, axis=1),
    )


def compute_all_metrics(paths: np.ndarray, logits: np.ndarray, dev: dict, study: str,
                        fit_val_preds: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
                        l0_paths: np.ndarray | None = None) -> tuple[dict[str, np.ndarray], dict]:
    """Compute GEMINI.md full metrics, subgroup masks, and decoder ordering metrics."""
    future = dev["future"].astype(np.float64)
    past = dev["past"]
    base = forecast_metrics(paths, logits, future)

    err = np.linalg.norm(paths - future[:, None], axis=-1)  # [N, K, 12]
    ade = err.mean(axis=2)                                  # [N, K]
    fde = err[:, :, -1]                                     # [N, K]
    best = fde.argmin(axis=1)
    rows = np.arange(len(paths))

    good_cand = (ade <= 0.2) & (fde <= 0.4)
    j_cov = good_cand.any(axis=1).astype(np.float64)
    j_cont = np.maximum(ade / 0.2, fde / 0.4).min(axis=1)
    eff_clusters = effective_good_clusters(paths[:, :, -1], logits, good_cand, radius=0.2).astype(np.float64)

    best_err = err[rows, best]  # [N, 12]
    early_ade = best_err[:, 0:4].mean(axis=1)
    mid_ade = best_err[:, 4:8].mean(axis=1)
    late_ade = best_err[:, 8:12].mean(axis=1)

    kin = compute_kinematics(paths, past)

    # Pairwise endpoint spread (for R1 early signal in decoder study)
    end = paths[:, :, -1]
    pair_dist = np.linalg.norm(end[:, :, None] - end[:, None, :], axis=-1)
    triu_i, triu_j = np.triu_indices(end.shape[1], 1)
    endpoint_spread_m = pair_dist[:, triu_i, triu_j].mean(axis=1)

    metrics = {
        **base,
        "J_cov": j_cov,
        "J_cont": j_cont,
        "eff_good_clusters": eff_clusters,
        "early_ADE": early_ade,
        "mid_ADE": mid_ade,
        "late_ADE": late_ade,
        "endpoint_spread_m": endpoint_spread_m,
        **kin,
    }

    extra_info = {}
    if study == "decoder":
        tau_out = 1.0
        tau_val_table = {}
        if fit_val_preds is not None:
            fv_paths, fv_logits, fv_future = fit_val_preds
            fv_fde = np.linalg.norm(fv_paths[:, :, -1] - fv_future[:, None, -1], axis=-1)
            fv_end = fv_paths[:, :, -1]
            fv_rows = np.arange(len(fv_paths))
            best_score = float("inf")
            for tau in (0.5, 1.0, 2.0):
                ord_fv, _ = cover_order_numpy(fv_end, softmax(fv_logits / tau), first_argmax=True, logits=fv_logits)
                fde3 = float(np.take_along_axis(fv_fde, ord_fv[:, :3], axis=1).min(axis=1).mean())
                fde5 = float(np.take_along_axis(fv_fde, ord_fv[:, :5], axis=1).min(axis=1).mean())
                sc = 0.5 * (fde3 + fde5)
                tau_val_table[str(tau)] = dict(minFDE3=fde3, minFDE5=fde5, mean_3_5=sc)
                if sc < best_score:
                    best_score = sc
                    tau_out = tau
        extra_info["tau_out"] = float(tau_out)
        extra_info["tau_out_fit_val"] = tau_val_table

        orders = {
            "conf": conf_order(logits),
            "riskp": cover_order_numpy(end, softmax(logits), first_argmax=False)[0],
            "cov_t05": cover_order_numpy(end, softmax(logits / 0.5), first_argmax=True, logits=logits)[0],
            "cov_t10": cover_order_numpy(end, softmax(logits / 1.0), first_argmax=True, logits=logits)[0],
            "cov_t20": cover_order_numpy(end, softmax(logits / 2.0), first_argmax=True, logits=logits)[0],
        }
        orders["cov_sel"] = orders[f"cov_t{int(round(tau_out * 10)):02d}"]

        for oname, ord_arr in orders.items():
            for k in (1, 3, 5, 20):
                metrics[f"{oname}_minADE{k}"] = np.take_along_axis(ade, ord_arr[:, :k], axis=1).min(axis=1)
                metrics[f"{oname}_minFDE{k}"] = np.take_along_axis(fde, ord_arr[:, :k], axis=1).min(axis=1)
        extra_info["cov_sel_order"] = orders["cov_sel"].astype(np.uint8)
        extra_info["conf_order"] = orders["conf"].astype(np.uint8)

        if l0_paths is not None:
            l0_err = np.linalg.norm(l0_paths - future[:, None], axis=-1)
            l0_ade = l0_err.mean(axis=2)
            l0_fde = l0_err[:, :, -1]
            metrics["L0_minFDE"] = l0_fde.min(axis=1)
            metrics["L0_J_cov"] = ((l0_ade <= 0.2) & (l0_fde <= 0.4)).any(axis=1).astype(np.float64)
            # Net rescue vs M0/L0
            metrics["L1_vs_L0_rescue_jcov"] = ((metrics["L0_J_cov"] < 0.5) & (j_cov > 0.5)).astype(np.float64)
            metrics["L1_vs_L0_lost_jcov"] = ((metrics["L0_J_cov"] > 0.5) & (j_cov < 0.5)).astype(np.float64)

    return metrics, extra_info


def l0_match_rates(l0_paths: np.ndarray, future: np.ndarray, past_full: np.ndarray, theta0: float) -> dict:
    err = np.linalg.norm(l0_paths - future[:, None], axis=-1)  # [N, K, 12]
    steps = np.arange(1, err.shape[-1] + 1, dtype=np.float64) / err.shape[-1]
    match_any = (err <= float(theta0) * steps).all(axis=-1).any(axis=1)
    fast = fast_mask(past_full)
    return dict(
        theta0=float(theta0),
        windows=int(len(match_any)),
        overall_match_rate=float(match_any.mean()),
        fast_windows=int(fast.sum()),
        fast_match_rate=float(match_any[fast].mean()) if fast.any() else None,
    )


def assert_runtime_device(args) -> str:
    if args.cpu or args.smoke:
        return "cpu" if args.cpu else "cuda:0"
    assert torch.cuda.is_available(), "CUDA required"
    if args.local_gpu:
        return "cuda:0"
    gpu = torch.cuda.get_device_name(0)
    if "T4" not in gpu:
        raise RuntimeError(f"Expected Tesla T4 in Kaggle run, got {gpu}")
    return "cuda:0"


def preflight(args) -> None:
    device = "cpu" if args.cpu else "cuda:0"
    if not args.cpu and not args.local_gpu:
        assert torch.cuda.is_available() and "T4" in torch.cuda.get_device_name(0), torch.cuda.get_device_name(0)

    fit_path, fit_n = data_paths(args.data, "C0", False)
    dev_path, dev_n = data_paths(args.data, "C0", True)
    fit = load_data(fit_path, fit_n)
    dev = load_data(dev_path, dev_n)
    small_fit = {k: v[:16] for k, v in fit.items()}
    small_dev = {k: v[:16] for k, v in dev.items()}
    p, f, n = (torch.as_tensor(small_fit[k], device=device) for k in ("past", "future", "num_valid"))

    arms = ENCODER_ARMS if args.study == "encoder" else DECODER_ARMS
    report = {}

    if args.study == "encoder":
        # 1. Parameter counts, V0-c equivalence, MC-1, MC-3, E2 == f4, E4 final step == E3 final step
        baseline = UnifiedTrainer(small_fit, "encoder", "E0", 42, device, 2, batch=8)
        baseline.model.eval()
        with torch.no_grad():
            o0 = core.prediction(baseline.model, p, n)

        init_hashes = {}
        for arm in ENCODER_ARMS:
            t = UnifiedTrainer(small_fit, "encoder", arm, 42, device, 2, batch=8)
            t.model.eval()
            total_p = sum(x.numel() for x in t.model.parameters())
            train_p = sum(x.numel() for x in t.model.parameters() if x.requires_grad)
            with torch.no_grad():
                o = core.prediction(t.model, p, n)
                if arm in ("E3", "E4"):
                    diff_p = float((o[0] - o0[0]).abs().max())
                    diff_z = float((o[1] - o0[1]).abs().max())
                    assert diff_p <= 1e-6 and diff_z <= 1e-6, (arm, diff_p, diff_z)
            # Run 2 training steps and verify finiteness + gradient flow
            t.step()
            t.step()
            diag, _ = encoder_diagnostics(core, t.model, small_dev, arm, device)
            if arm == "E1":
                assert diag["mc1_pass"], diag
                # MC-1 keys/shapes identical to E0
                assert list(t.model.state_dict().keys()) == list(baseline.model.state_dict().keys())
                for k in baseline.model.state_dict():
                    assert t.model.state_dict()[k].shape == baseline.model.state_dict()[k].shape
            if arm in ("E5", "E6"):
                assert diag["mc3_pass"], (arm, diag)
            if arm in ("E3", "E4"):
                assert any(name.endswith("physical_time_bias.weights") for name in t.connected), arm
            report[arm] = dict(total_params=total_p, trainable_params=train_p, diag=diag)
            del t

        # Check E4 final step == contemporaneous final step
        enc = baseline.model.agent_aware_trans_encoder
        rc, _, _ = enc.relation_features(p[:, :9])
        rl = lagged_relation_features(enc, p[:, :9])
        torch.testing.assert_close(rc[:, :, :, -1, :], rl[:, :, :, -1, :], atol=0, rtol=0)

        # Check exact determinism & resume on E3
        args.out.mkdir(parents=True, exist_ok=True)
        res_dir = args.out / "resume_check"
        res_dir.mkdir(parents=True, exist_ok=True)
        t_a = UnifiedTrainer(small_fit, "encoder", "E3", 42, device, 2, batch=8)
        for _ in range(t_a.steps_per_epoch):
            t_a.step()
        contract = dict(preflight="encoder_E3")
        checkpoint(res_dir, core, t_a, contract)
        for _ in range(t_a.steps_per_epoch):
            t_a.step()
        wanted = t_a.snapshot(contract)
        t_b = UnifiedTrainer(small_fit, "encoder", "E3", 42, device, 2, batch=8)
        assert latest(res_dir, core, t_b, contract) == 1
        for _ in range(t_b.steps_per_epoch):
            t_b.step()
        core.exact(wanted, t_b.snapshot(contract), "PREFLIGHT_ENCODER_RESUME")
    else:
        # Decoder preflight: cover_order parity, parameter counts, MC-b, MC-d, exact resume on JM4
        rng = np.random.default_rng(20261127)
        end_np = rng.normal(size=(64, 20, 2))
        z_np = rng.normal(size=(64, 20))
        w_np = np.exp(z_np - z_np.max(1, keepdims=True))
        w_np /= w_np.sum(1, keepdims=True)
        pi_t, d_t = cover_order_torch(torch.tensor(end_np, device=device), torch.tensor(w_np, device=device), True, torch.tensor(z_np, device=device))
        pi_n, d_n = cover_order_numpy(end_np, w_np, True, z_np)
        np.testing.assert_array_equal(pi_t.cpu().numpy(), pi_n)
        np.testing.assert_allclose(d_t.cpu().numpy(), d_n, atol=1e-12)

        counts = {}
        for arm in DECODER_ARMS:
            t = UnifiedTrainer(small_fit, "decoder", arm, 42, device, 2, batch=8)
            total_p = sum(x.numel() for x in t.model.parameters())
            counts[arm] = total_p
            t.step()
            t.step()
            t.model.eval()
            out_dict = infer_full(t.model, small_dev, batch=8)
            assert np.isfinite(out_dict["paths"]).all() and np.isfinite(out_dict["logits"]).all()
            if "mc_b_max_diff" in out_dict:
                assert float(np.max(out_dict["mc_b_max_diff"])) == 0.0, arm
            report[arm] = dict(total_params=total_p)
            del t

        assert counts["M4"] <= 400000 and counts["JM4"] <= 400000, counts
        m6_diff = abs(counts["M6"] - counts["M4"]) / counts["M4"]
        assert m6_diff <= 0.02, f"MC-d failed: {m6_diff}"
        report["mc_d_m6_vs_m4_diff"] = m6_diff

        # Exact resume check on JM4
        args.out.mkdir(parents=True, exist_ok=True)
        res_dir = args.out / "resume_check"
        res_dir.mkdir(parents=True, exist_ok=True)
        t_a = UnifiedTrainer(small_fit, "decoder", "JM4", 42, device, 2, batch=8)
        for _ in range(t_a.steps_per_epoch):
            t_a.step()
        contract = dict(preflight="decoder_JM4")
        checkpoint(res_dir, core, t_a, contract)
        for _ in range(t_a.steps_per_epoch):
            t_a.step()
        wanted = t_a.snapshot(contract)
        t_b = UnifiedTrainer(small_fit, "decoder", "JM4", 42, device, 2, batch=8)
        assert latest(res_dir, core, t_b, contract) == 1
        for _ in range(t_b.steps_per_epoch):
            t_b.step()
        core.exact(wanted, t_b.snapshot(contract), "PREFLIGHT_DECODER_RESUME")

    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "PREFLIGHT.json", dict(status="PASS", study=args.study, device=device, report=report))
    print(f"PREFLIGHT_PASS study={args.study} device={device}", flush=True)


def train(args) -> None:
    device = assert_runtime_device(args)
    fit_path, fit_n = data_paths(args.data, "C0", False)
    fit = load_data(fit_path, fit_n)
    if args.smoke:
        fit = {k: v[:16] for k, v in fit.items()}
        epochs, batch = 2, 8
    else:
        epochs, batch = 64, 64

    gpu_name = "CPU" if device == "cpu" else torch.cuda.get_device_name(0)
    contract = dict(
        study=args.study,
        arm=args.arm,
        seed=args.seed,
        order_seed=args.seed,
        fold="C0",
        epochs=epochs,
        batch=batch,
        fit_sha256=digest(fit_path),
        fit_windows=len(fit["ids"]),
        protocol=args.protocol,
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        gpu=gpu_name,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    spec = args.out / "CONTRACT.json"
    if spec.exists():
        assert json.loads(spec.read_bytes()) == contract, f"Contract mismatch at {spec}"
    else:
        write_json(spec, contract)

    trained_file = args.out / "TRAINED.json"
    t = UnifiedTrainer(fit, args.study, args.arm, args.seed, device, epochs, batch=batch)
    start = latest(args.out, core, t, contract)
    if trained_file.exists():
        rec = json.loads(trained_file.read_bytes())
        assert rec["contract"] == contract and start == epochs
        assert digest(args.out / f"epoch_{epochs:03d}.pt") == rec["checkpoint_sha256"]
        return

    for epoch in range(start + 1, epochs + 1):
        if time.time() >= args.deadline:
            if t.epoch > start and not (args.out / f"epoch_{t.epoch:03d}.json").exists():
                checkpoint(args.out, core, t, contract)
            print(f"PAUSED {args.out.name} at {t.epoch}/{epochs}", flush=True)
            return
        now = time.monotonic()
        for _ in range(t.steps_per_epoch):
            t.step()
        if epoch % 8 == 0 or epoch in (1, epochs):
            checkpoint(args.out, core, t, contract)
            keep_n = 1 if epoch == epochs else 2
            for old in sorted(args.out.glob("epoch_???.json"))[:-keep_n]:
                old.unlink()
                old.with_suffix(".pt").unlink(missing_ok=True)
        epoch_rows = t.history[-t.steps_per_epoch :]
        status = dict(
            epoch=epoch,
            updates=t.updates,
            seconds=time.monotonic() - now,
            mean_loss=float(np.mean([r["loss"] for r in epoch_rows])),
            grad_norm=float(np.mean([r["grad_norm"] for r in epoch_rows])),
        )
        if "l0_pos_sum" in epoch_rows[0]:
            cnt = sum(r["l0_count"] for r in epoch_rows)
            status["l0_pos_mean"] = sum(r["l0_pos_sum"] for r in epoch_rows) / max(cnt, 1)
            status["l0_match_rate"] = sum(r["l0_match_sum"] for r in epoch_rows) / max(cnt, 1)
            status["l1_match_rate"] = float(np.mean([r["l1_match_rate"] for r in epoch_rows]))
            if "ld_match_rate" in epoch_rows[0]:
                status["ld_match_rate"] = float(np.mean([r["ld_match_rate"] for r in epoch_rows]))
        write_json(args.out / "CURRENT.json", status)
        print(f"TRAIN {args.out.name} {json.dumps(status)}", flush=True)

    # Save training log
    log_path = args.out / "training_log.jsonl"
    log_path.write_text(
        "".join(json.dumps(row) + "\n" for row in t.history[:: max(t.steps_per_epoch, 1)]),
        encoding="utf-8",
    )
    last_epoch_summary = json.loads((args.out / "CURRENT.json").read_bytes())
    write_json(
        trained_file,
        dict(
            contract=contract,
            updates=t.updates,
            parameters=sum(p.numel() for p in t.model.parameters()),
            trainable_parameters=sum(p.numel() for p in t.model.parameters() if p.requires_grad),
            final_epoch=last_epoch_summary,
            checkpoint_sha256=digest(args.out / f"epoch_{epochs:03d}.pt"),
        ),
    )


def evaluate(args) -> None:
    device = assert_runtime_device(args)
    trained = json.loads((args.out / "TRAINED.json").read_bytes())
    contract = trained["contract"]
    epochs = int(contract["epochs"])
    ckpt_path = args.out / f"epoch_{epochs:03d}.pt"
    assert digest(ckpt_path) == trained["checkpoint_sha256"]

    fit_path, fit_n = data_paths(args.data, "C0", False)
    dev_path, dev_n = data_paths(args.data, "C0", True)
    fit = load_data(fit_path, fit_n)
    dev = load_data(dev_path, dev_n)
    if args.smoke:
        fit = {k: v[:16] for k, v in fit.items()}
        dev = {k: v[:12] for k, v in dev.items()}

    model, _ = build_model_for(args.study, args.arm, device, args.seed)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["ema"], strict=True)

    out_dev = infer_full(model, dev)
    paths, logits = out_dev["paths"], out_dev["logits"]
    assert np.isfinite(paths).all() and np.isfinite(logits).all(), f"Nonfinite eval outputs in {args.out.name}"

    fit_val_preds = None
    mc_decoder = {}
    if args.study == "decoder":
        fv = fit if args.smoke else fit_val_subset(fit, size=2500)
        fv_out = infer_full(model, fv)
        fit_val_preds = (fv_out["paths"], fv_out["logits"], fv["future"].astype(np.float64))

        # MC-a diagnostics (L0 match rate on C0 fit_val, C0 dev, and ETH test v>1.3 m/s)
        theta0 = 0.5 if args.arm in ("M0", "M1", "M6") else 1.0
        l0_dev = out_dev.get("l0_paths", paths)
        l0_fv = fv_out.get("l0_paths", fv_out["paths"])
        eth_path, eth_n = data_paths(args.data, "eth", True)
        eth_test = load_data(eth_path, eth_n)
        if args.smoke:
            eth_test = {k: v[:12] for k, v in eth_test.items()}
        eth_out = infer_full(model, eth_test)
        l0_eth = eth_out.get("l0_paths", eth_out["paths"])

        mc_decoder["MC_a"] = dict(
            train_final_l0_pos_mean=trained["final_epoch"].get("l0_pos_mean", 1.0),
            train_final_l0_match_rate=trained["final_epoch"].get("l0_match_rate"),
            c0_fit_val=l0_match_rates(l0_fv, fv["future"].astype(np.float64), fv["past"], theta0),
            c0_dev=l0_match_rates(l0_dev, dev["future"].astype(np.float64), dev["past"], theta0),
            eth_test_l0=l0_match_rates(l0_eth, eth_test["future"].astype(np.float64), eth_test["past"], theta0),
        )
        if "mc_b_max_diff" in out_dev:
            mc_b = float(np.max(out_dev["mc_b_max_diff"]))
            assert mc_b == 0.0, f"MC-b violated: {mc_b}"
            mc_decoder["MC_b_max_yd_diff"] = mc_b
            mc_decoder["MC_b_pass"] = mc_b == 0.0

    metrics, extra_info = compute_all_metrics(
        paths, logits, dev, args.study, fit_val_preds=fit_val_preds, l0_paths=out_dev.get("l0_paths")
    )

    diagnostics = {}
    if args.study == "encoder":
        e0_bundle = None
        if args.e0_out is not None and (args.e0_out / "attn_argmax.npz").is_file():
            with np.load(args.e0_out / "attn_argmax.npz", allow_pickle=False) as z:
                e0_bundle = {
                    k: torch.from_numpy(z[k])
                    for k in ("joint", "marg_t", "pair_t", "row_mask", "pair_mask")
                }
        diag, attn_bundle = encoder_diagnostics(core, model, dev, args.arm, device, e0_attn_argmax=e0_bundle)
        if args.arm == "E1":
            assert diag["mc1_pass"], diag
        if args.arm in ("E5", "E6"):
            assert diag["mc3_pass"], diag
        diagnostics = diag
        if attn_bundle is not None and args.arm in ("E0", "E3", "E4"):
            safe_npz(
                args.out / "attn_argmax.npz",
                joint=attn_bundle["joint"].numpy().astype(np.uint8),
                marg_t=attn_bundle["marg_t"].numpy().astype(np.uint8),
                pair_t=attn_bundle["pair_t"].numpy().astype(np.uint8),
                row_mask=attn_bundle["row_mask"].numpy(),
                pair_mask=attn_bundle["pair_mask"].numpy(),
            )
    else:
        diagnostics = mc_decoder
        diagnostics["tau_out"] = extra_info.get("tau_out")
        diagnostics["tau_out_fit_val"] = extra_info.get("tau_out_fit_val")

    masks = eval_groups(dev)
    masks["fast"] = fast_mask(dev["past"])
    tracks = tracks_for(dev["ids"])
    recordings = recordings_for(dev["ids"])

    npz_payload = dict(ids=dev["ids"], tracks=tracks, recordings=recordings, **masks, **metrics)
    if args.study == "decoder":
        npz_payload["cov_sel_order"] = extra_info["cov_sel_order"]
        npz_payload["conf_order"] = extra_info["conf_order"]
        if "disc_order" in out_dev:
            npz_payload["disc_order"] = out_dev["disc_order"]

    metrics_sha = safe_npz(args.out / "METRICS.npz", **npz_payload)

    # Save final EMA state cleanly
    with tempfile.TemporaryDirectory(prefix="final_ema_") as tmp:
        p_ema = Path(tmp) / "final_ema.pt"
        torch.save(core.cpu_tree(state["ema"]), p_ema)
        ema_sha = publish(p_ema, args.out / "final_ema.pt")

    subgroup_summary = {
        g: dict(
            windows=int(m.sum()),
            **{k: float(v[m].mean()) for k, v in metrics.items()},
        )
        for g, m in masks.items()
        if m.any()
    }

    result = dict(
        status="PASS",
        contract=contract,
        windows=len(dev["ids"]),
        tracks=len(np.unique(tracks)),
        updates=trained["updates"],
        parameters=trained["parameters"],
        trainable_parameters=trained["trainable_parameters"],
        inference_parameters=trained["parameters"],
        metrics=subgroup_summary["overall"],
        subgroups=subgroup_summary,
        diagnostics=diagnostics,
        metrics_sha256=metrics_sha,
        final_ema_sha256=ema_sha,
    )
    write_json(args.out / "RESULT.json", result)
    print(
        f"EVAL {args.out.name} minFDE={result['metrics']['minFDE']:.6f} "
        f"Top1_FDE={result['metrics']['Top1_FDE']:.6f} J_cov={result['metrics']['J_cov']:.4f}",
        flush=True,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("preflight", "train", "evaluate"))
    p.add_argument("--study", choices=("encoder", "decoder"), required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--arm", default="E0")
    p.add_argument("--seed", type=int, choices=(42, 43, 44), default=42)
    p.add_argument("--protocol", default="local")
    p.add_argument("--deadline", type=float, default=float("inf"))
    p.add_argument("--e0-out", type=Path, default=None)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--local-gpu", action="store_true")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    valid_arms = ENCODER_ARMS if args.study == "encoder" else DECODER_ARMS
    if args.command != "preflight" and args.arm not in valid_arms:
        raise ValueError(f"Arm {args.arm} not valid for study {args.study}: {valid_arms}")

    torch.set_num_threads(1)
    globals()[args.command](args)


if __name__ == "__main__":
    main()
