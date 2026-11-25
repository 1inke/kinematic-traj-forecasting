"""Execute the registered code-path checks before formal training."""
import argparse
from contextlib import contextmanager
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from advisor_train import Trainer as ArchiveTrainer
from advisor_train import core, inputs, load_data
import baseline_factory
from order_numpy import risk_order, softmax
from prefix_loss import ARMS, KS, coverage_order, prefix_loss
from prefix_train import PrefixTrainer, subset
from run_b0 import write_json


def model_input(data, n, device):
    return [torch.as_tensor(data[k][:n], device=device) for k in ("past", "future", "num_valid")]


@contextmanager
def record_inputs():
    received = []
    original = baseline_factory.augment_trajectories

    def record(*args, **kwargs):
        values = original(*args, **kwargs)
        received.append(tuple(x.detach().cpu().clone() for x in values))
        return values

    with patch.object(baseline_factory, "augment_trajectories", record):
        yield received


def latency(data, count=60):
    single = subset(data, 1)
    past, _, tokens = model_input(single, 1, "cpu")
    trainer = PrefixTrainer(single, "A", 42, "cpu", 2)
    model = trainer.model.eval()
    records = {}
    with torch.no_grad():
        for label in ("conf", "riskp"):
            for _ in range(8):
                path, logits, _, _ = core.prediction(model, past, tokens)
            samples = []
            for _ in range(count):
                began = time.perf_counter()
                path, logits, _, _ = core.prediction(model, past, tokens)
                end, scores = path[:, 0, :, -1], logits[:, 0].double()
                if label == "riskp":
                    coverage_order(end, scores.softmax(-1))
                else:
                    torch.argsort(-scores, dim=-1, stable=True)
                samples.append(1000*(time.perf_counter()-began))
            records[label] = float(np.median(samples))
    return {"ms": records, "flag_over_12_3_ms": records["riskp"] > 12.3, "repetitions": count}


def quick(device, fit, report):
    cpu = "cpu"
    smoke = {}
    for arm in ARMS:
        t = PrefixTrainer(subset(fit, 8), arm, 42, cpu, 2)
        rows = [t.step(), t.step()]
        assert t.epoch == 2 and all(np.isfinite(r["loss"]) for r in rows)
        for r in rows:
            assert all(0 <= r[f"distinct{k}"] <= 1 for k in KS) if arm != "A" else True
        smoke[arm] = rows
    report["PF-0"] = smoke

    t = PrefixTrainer(subset(fit, 64), "A", 42, device, 2)
    past, future, tokens = model_input(fit, 64, device)
    paths, logits, target, _ = core.prediction(t.model, past, tokens, future)
    g = np.random.default_rng(20261125)
    end = np.concatenate([g.normal(size=(2000, 20, 2)), paths[:, 0, :, -1].detach().cpu().numpy()])
    z = np.concatenate([g.normal(size=(2000, 20)), logits[:, 0].detach().cpu().numpy()])
    actual = coverage_order(torch.as_tensor(end, device=device), torch.as_tensor(softmax(z), device=device)).cpu().numpy()
    assert np.array_equal(actual, risk_order(end, softmax(z)))
    report["PF-1"] = {"synthetic": 2000, "real": 64, "match": 1.0}
    seed = 42
    exact, _ = core.loss_value(paths, target, logits, t.loss_fn)
    test, _ = t.objective(paths, target, logits)
    assert torch.equal(exact, test)
    paired = subset(fit, 128)
    reference = ArchiveTrainer(paired, "B0", seed, device, 25)
    for _ in range(50):
        reference.step()
    t = PrefixTrainer(paired, "A", seed, device, 25)
    for _ in range(50):
        t.step()
    core.exact(reference.model.state_dict(), t.model.state_dict(), "PF2 model")
    core.exact(reference.ema.module.state_dict(), t.ema.module.state_dict(), "PF2 EMA")
    report["PF-2"] = {"steps": 50, "device": device, "bitwise_equal": True}

    recorded = None
    for arm in ARMS:
        t = PrefixTrainer(paired, arm, seed, device, 5)
        with record_inputs() as history:
            for _ in range(10):
                t.step()
        if recorded is None:
            recorded = history
        else:
            core.exact(recorded, history, f"PF3 augmentation {arm}")
    report["PF-3"] = {"arms": list(ARMS), "batches": 10, "bitwise_equal": True}

    t = PrefixTrainer(subset(fit, 64), "N-cov", 42, device, 2)
    paths, z, target, _ = core.prediction(t.model, past, tokens, future)
    extra, _ = prefix_loss("N-cov", paths, target, z, t.loss_fn)
    extra.backward()
    def norm(params):
        return sum(float(p.grad.detach().norm()) for p in params if p.grad is not None)

    decoder = t.model.decoder
    score = norm(decoder.score_head.parameters())
    proposal = norm(decoder.proposal_delta.parameters()) + norm(decoder.proposal_scale.parameters())
    assert score == 0 and proposal > 0
    report["PF-5"] = {"score_head_grad_norm": score, "proposal_grad_norm": proposal}
    report["PF-LAT"] = latency(fit)


def execute(out, device):
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    if device != "cpu":
        assert torch.cuda.is_available() and "L4" in torch.cuda.get_device_name().split()
    report = {"status": "RUNNING", "device": device, "torch": str(torch.__version__)}
    try:
        path, size = inputs("C0")[0]
        quick(device, load_data(path, size), report)
        report["status"] = "PASS"
    except BaseException as error:
        report.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(out, report)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--out", required=True, type=Path)
    cli.add_argument("--device", default="cpu")
    options = cli.parse_args()
    execute(options.out, options.device)
