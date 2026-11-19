"""Fixed 20-epoch input ablations using the sealed archive-18 B0 features."""

import hashlib
import time

import numpy as np
import torch
from torch import nn

from refine_io import DEC, OLD, PREC, ROOT, anchored, arrays, finish, finished, npz, populations, read
from refine_io import save_torch, seal, sha, status, verify, write
from dec_cache import load
from dec_model import InfoRefiner, make_basis, predict as predict_a1
from dec_train import Trainer as ArchiveTrainer
from dec_analysis import window_metrics
from probe_model import baseline_loss
from probe_train import cpu_tree

FOLDER = ROOT / "ABL"
ARMS = ("A1", "R-centroid", "R-mode", "R-geom", "R-lowspeed")
SEEDS = (0, 1, 2)
COUNTS = dict(zip(ARMS, (7019, 7187, 6707, 5483, 7019)))
THR = .005
DEVICE = "cuda:0"


class InputProjection(nn.Linear):
    def __init__(self, arm, original):
        width = {"R-centroid": 84, "R-mode": 64, "R-geom": 13}[arm]
        super().__init__(width, 24, device=original.weight.device)
        self.arm = arm
        with torch.no_grad():
            self.bias.copy_(original.bias)
            if arm == "R-centroid":
                self.weight[:, :77].copy_(original.weight)
                self.weight[:, 77:].zero_()
            else:
                self.weight.copy_(original.weight[:, :64] if arm == "R-mode" else original.weight[:, 64:])

    def forward(self, a1_inputs):
        if self.arm == "R-mode":
            features = a1_inputs[..., :64]
        elif self.arm == "R-geom":
            features = a1_inputs[..., 64:]
        else:
            norm_anchors = a1_inputs[..., 64:70]
            centered = norm_anchors - norm_anchors.mean(dim=1, keepdim=True)
            distance = torch.linalg.vector_norm(centered[..., -2:], dim=-1, keepdim=True)
            features = torch.cat((a1_inputs, centered, distance), dim=-1)
        return super().forward(features)


class Variant(InfoRefiner):
    def __init__(self, arm, control):
        super().__init__("A1")
        self.to(control.inp[0].weight.device)
        self.load_state_dict(control.state_dict(), strict=True)
        self.arm = arm
        if arm != "R-lowspeed":
            self.inp[0] = InputProjection(arm, self.inp[0])

    def forward(self, mode, relative, scale, gate):
        hidden = self.stem(mode, relative, scale)
        raw = self.anchor_out(hidden).reshape_as(relative)
        bound = .30 * scale[:, None, None]
        if self.arm == "R-lowspeed":
            bound = bound * gate[:, None, None, None]
        return relative + bound * torch.tanh(raw), self.score_out(hidden).squeeze(-1)


def forecast(model, head, cache, index):
    if model.arm == "A1":
        return predict_a1(model, head, cache, index)
    mode, relative, scale, gate = [cache[k][index] for k in ("mode", "relative", "scale", "gate")]
    anchors, scores = model(mode, relative, scale, gate)
    logits = head(torch.cat((mode, anchors.flatten(-2)), dim=-1)).squeeze(-1) + scores
    paths = torch.einsum("tp,nkpc->nktc", model.basis, anchors)
    return paths, logits, anchors - relative


class VariantTrainer(ArchiveTrainer):
    def __init__(self, arm, seed, cache, head, criterion):
        super().__init__("A1", seed, cache, head, criterion, DEVICE)
        self.arm = arm
        self.model = Variant(arm, self.model)
        # The inherited AdamW and scheduler are fresh. Replace only their parameter list.
        assert len(self.opt.param_groups) == 1 and not self.opt.state
        self.opt.param_groups[0]["params"] = list(self.model.parameters())

    def epoch_step(self):
        start = time.monotonic()
        order = torch.randperm(len(self.cache["y"]), generator=self.generator)
        digest = hashlib.sha256(order.numpy().astype("<i8").tobytes()).hexdigest()
        self.model.train()
        total = 0.
        for idx in order.split(self.batch):
            idx = idx.to(self.device)
            paths, logits, _ = forecast(self.model, self.head, self.cache, idx)
            loss, _ = baseline_loss(paths, self.cache["y"][idx], logits, self.criterion)
            assert torch.isfinite(loss), "Nonfinite loss"
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            grads = [p.grad.flatten() for p in self.model.parameters() if p.grad is not None]
            assert torch.isfinite(torch.cat(grads)).all(), "Nonfinite gradient"
            self.opt.step()
            self.scheduler.step()
            self.updates += 1
            total += float(loss.detach()) * len(idx)
        self.epoch += 1
        for name, p in self.model.named_parameters():
            if p.grad is not None and bool(p.grad.abs().any()):
                self.connected.add(name)
        row = dict(epoch=self.epoch, updates=self.updates, loss=total / len(order), order_hash=digest,
                   seconds=time.monotonic()-start, lr=self.opt.param_groups[0]["lr"])
        self.history.append(row)
        return row


def speed(past):
    delta = np.diff(past[:, 0].astype(np.float64), axis=1)
    return (np.hypot(delta[..., 0], delta[..., 1]) / .4).mean(1)


def setup():
    data, head, criterion = load("cpu")
    head = head.to(DEVICE)
    criterion = criterion.to(DEVICE)
    reference = read(PREC / "INPUT_MANIFEST.json")
    pre = dict(c0={}, cache=read(DEC / "cache/DONE.json")["files"], archive={})
    source = read(DEC / "SOURCE_MANIFEST.json")
    for directory, names in ((DEC / "code", ("dec_io.py", "dec_cache.py", "dec_model.py", "dec_train.py", "dec_analysis.py")),
                             (DEC.parent / "EXP_SET_INTERACT_PROBE_EXEC_20261117/code",
                              ("probe_model.py", "probe_train.py", "probe_cache.py", "probe_io.py"))):
        for name in names:
            file = directory / name
            pre["archive"][str(file)] = anchored(file, source)
    vref = None
    for split in ("fit", "dev"):
        path = OLD / f"C0/{split}.npz"
        pre["c0"][str(path)] = anchored(path, reference)
        c0 = arrays(path)
        meta = data[split]["meta"]
        np.testing.assert_array_equal(c0["ids"], meta["ids"])
        np.testing.assert_array_equal(c0["group"], meta["group"])
        v = speed(c0["past"])
        if split == "fit":
            vref = float(np.median(v))
            assert vref > 0
        cache = data[split]["tensors"]
        data[split]["tensors"] = {k: cache[k].to(DEVICE) for k in ("mode", "relative", "scale", "logits", "y")}
        data[split]["tensors"]["gate"] = torch.as_tensor(np.clip(v/vref, .1, 1.), dtype=torch.float32, device=DEVICE)
    masks, labels, obs = populations(c0)
    np.testing.assert_array_equal(labels, data["dev"]["meta"]["labels"])
    pre.update(v_ref_mps=vref, windows={s: len(data[s]["meta"]["ids"]) for s in ("fit", "dev")})
    return data, head, criterion, masks, obs, pre


@torch.no_grad()
def evaluate(model, head, cache):
    model.eval()
    parts = []
    raw = {}

    def capture(module, args, value):
        raw["saturated"] = (torch.tanh(value).abs() > .95).float().mean((1, 2))

    hook = model.anchor_out.register_forward_hook(capture)
    try:
        for lo in range(0, len(cache["y"]), 256):
            i = slice(lo, lo + 256)
            pred = forecast(model, head, cache, i)
            assert all(torch.isfinite(p).all() for p in pred)
            parts.append(tuple(p.cpu().numpy() for p in (*pred, raw["saturated"])))
    finally:
        hook.remove()
    paths, logits, shifts, sat = (np.concatenate(v) for v in zip(*parts))
    w = window_metrics(paths, logits, shifts, cache["y"].cpu().numpy())
    w["saturation_fraction"] = sat
    return w, dict(paths=paths, logits=logits, shift=shifts)


def refs(seed):
    directory = DEC / f"runs/A1_s{seed}"
    verify(read(directory / "DONE.json")["files"])
    return directory


def compare_replay(prediction, metrics, ids, archive):
    old = arrays(archive / "predictions_local.npz")
    np.testing.assert_array_equal(ids, old["ids"])
    for key, val in prediction.items():
        np.testing.assert_array_equal(val, old[key], err_msg=f"{archive.name} {key}")
    truth = arrays(archive / "window_metrics.npz")
    np.testing.assert_array_equal(ids, truth["ids"])
    for key in ("minADE", "minFDE", "top1FDE", "mean_shift"):
        np.testing.assert_array_equal(metrics[key], truth[key], err_msg=f"{archive.name} {key}")


def preflight(data, head, criterion, pre):
    dev, fit = data["dev"], data["fit"]
    controls = {}
    for seed in SEEDS:
        archive = refs(seed)
        trainer = ArchiveTrainer("A1", seed, fit["tensors"], head, criterion, DEVICE)
        expected = torch.load(archive / "initial.pt", map_location="cpu", weights_only=False)
        for name, val in trainer.model.state_dict().items():
            assert torch.equal(val.cpu(), expected[name]), name
        saved = torch.load(archive / "final.pt", map_location="cpu", weights_only=False)
        trainer.model.load_state_dict(saved["model"], strict=True)
        w, predictions = evaluate(trainer.model, head, dev["tensors"])
        compare_replay(predictions, w, dev["meta"]["ids"], archive)
        controls[str(seed)] = "BITWISE_EQUAL"
    cache = fit["tensors"]
    for arm in ARMS:
        trainer = (ArchiveTrainer(arm, 0, cache, head, criterion, DEVICE) if arm == "A1"
                   else VariantTrainer(arm, 0, cache, head, criterion))
        assert sum(p.numel() for p in trainer.model.parameters()) == COUNTS[arm]
        trial = slice(0, 256)
        with torch.no_grad():
            paths, logits, shift = forecast(trainer.model, head, cache, trial)
            base = torch.einsum("tp,nkpc->nktc", make_basis().to(DEVICE), cache["relative"][trial])
            assert torch.equal(paths, base) and not bool(shift.any()), arm
            assert float((logits - cache["logits"][trial]).abs().max()) <= 1e-4, arm
    write(FOLDER / "PREFLIGHT_PASS.json", dict(**pre, A1_archived_final_replay=controls, parameters=COUNTS,
                                               training_recipe=dict(epochs=20, batch=256, paired_order=True)))


def train(arm, seed, data, head, criterion, masks, obs, contract, completed):
    out = FOLDER / f"runs/{arm}_s{seed}"
    if finished(out):
        return read(out / "RESULT.json")
    tr = (ArchiveTrainer(arm, seed, data["fit"]["tensors"], head, criterion, DEVICE) if arm == "A1"
          else VariantTrainer(arm, seed, data["fit"]["tensors"], head, criterion))
    archive = DEC / f"runs/A1_s{seed}"
    initial = cpu_tree(tr.model.state_dict())
    expected_init = torch.load(archive / "initial.pt", map_location="cpu", weights_only=False)
    for key, val in initial.items():
        if key != "inp.0.weight" or arm in ("A1", "R-lowspeed"):
            assert torch.equal(val, expected_init[key]), key
    save_torch(out / "initial.pt", initial)
    receipts = sorted((out / "checkpoints").glob("epoch*.json"))
    if receipts:
        record = read(receipts[-1])
        verify(record["files"])
        snapshot = torch.load(next(iter(record["files"])), map_location="cpu", weights_only=False)
        tr.restore(snapshot, contract)
    expected_history = read(archive / "TRAINING.json")["history"]
    assert len(expected_history) == 20
    for i in range(tr.epoch, 20):
        status("ABL_TRAIN", arm=arm, seed=seed, epoch=i+1, epochs=20, completed=completed, total=15)
        row = tr.epoch_step()
        assert expected_history[i]["order_hash"] == row["order_hash"], (arm, seed, i)
        if arm == "A1":
            assert row["loss"] == expected_history[i]["loss"], (arm, seed, i, row["loss"], expected_history[i]["loss"])
            if i == 0:
                receipt = read(archive / "checkpoints/epoch01.json")
                verify(receipt["files"])
                old = torch.load(next(iter(receipt["files"])), map_location="cpu", weights_only=False)
                for key, val in tr.model.state_dict().items():
                    assert torch.equal(val.cpu(), old["model"][key]), key
        checkpoint = out / f"checkpoints/epoch{i+1:02d}.pt"
        save_torch(checkpoint, tr.snapshot(contract))
        write(checkpoint.with_suffix(".json"), dict(epoch=i+1, files=seal([checkpoint])))
        write(out / "TRAINING.json", dict(arm=arm, seed=seed, epochs=tr.epoch, updates=tr.updates,
                                          history=tr.history, connected=sorted(tr.connected)))
        print(f"{arm} s{seed} epoch={i+1}/20 loss={row['loss']:.9f} seconds={row['seconds']:.3f}", flush=True)
        if (ROOT / "PAUSE.request").exists():
            status("PAUSED", arm=arm, seed=seed, completed=completed, epoch=tr.epoch)
            return None
    status("ABL_EVALUATE", arm=arm, seed=seed, completed=completed, total=15)
    dev, archived_match = data["dev"], None
    w, predictions = evaluate(tr.model, head, dev["tensors"])
    meta = dev["meta"]
    if arm == "A1":
        old = torch.load(archive / "final.pt", map_location="cpu", weights_only=False)
        for key, val in tr.model.state_dict().items():
            assert torch.equal(val.cpu(), old["model"][key]), (seed, key)
        compare_replay(predictions, w, meta["ids"], archive)
        archived_match = "BITWISE_EQUAL"
    save_torch(out / "final.pt", tr.snapshot(contract))
    rows = {name: dict(n=int(m.sum()), **{k: float(v[m].mean()) for k, v in w.items()})
            for name, m in masks.items()}
    npz(out / "window_metrics.npz", **w, ids=meta["ids"], group=meta["group"], maneuver=meta["labels"],
        observation=obs, speed_gate=dev["tensors"]["gate"].cpu().numpy())
    result = dict(arm=arm, seed=seed, epochs=20, updates=tr.updates, clean=rows, parameter_count=COUNTS[arm],
                  parameter_diff_vs_A1=COUNTS[arm]/COUNTS["A1"]-1, archived_A1_exact=archived_match,
                  training_seconds=sum(v["seconds"] for v in tr.history))
    write(out / "RESULT.json", result)
    files = [out / name for name in ("initial.pt", "final.pt", "RESULT.json", "TRAINING.json", "window_metrics.npz")]
    finish(out, files + sorted((out / "checkpoints").glob("epoch*.json")))
    return result


def decision(runs):
    def paired(arm, group):
        rows = {}
        for metric in ("minADE", "minFDE"):
            values = [1-runs[arm][str(s)]["clean"][group][metric]/runs["A1"][str(s)]["clean"][group][metric]
                      for s in SEEDS]
            mean = float(np.mean(values))
            pos, neg = sum(v >= 0 for v in values), sum(v <= 0 for v in values)
            same = pos if mean > 0 else neg if mean < 0 else max(pos, neg)
            rows[metric] = dict(mean=mean, values=values, same_direction=same,
                                within_band=abs(mean) <= THR and sum(abs(v) <= THR for v in values) >= 2 and same >= 2,
                                worse=mean <= -THR and neg >= 2, better=mean >= THR and pos >= 2)
        return rows

    pairs = {arm: {group: paired(arm, group) for group in runs["A1"]["0"]["clean"]}
             for arm in ARMS[1:]}
    mode, geom, centroid = (
        pairs["R-mode"]["overall"]["minFDE"], pairs["R-geom"]["overall"]["minFDE"],
        pairs["R-centroid"]["overall"]["minFDE"])
    low = pairs["R-lowspeed"]
    static, overall = low["stationary"]["minFDE"], low["overall"]["minFDE"]
    return dict(percent_threshold=100*THR, paired=pairs, decision=dict(
        E1_capacity=mode["within_band"], E2_geometry=mode["worse"] and geom["within_band"],
        mode_and_geometry_needed=mode["worse"] and geom["worse"],
        E4_centroid_improves=centroid["better"], centroid_within_band=centroid["within_band"],
        S1_lowspeed=(static["mean"] >= .05 and sum(x > 0 for x in static["values"]) >= 2
                     and overall["mean"] >= -THR and sum(x >= -THR for x in overall["values"]) >= 2)))


def main():
    if finished(FOLDER):
        return
    assert finished(ROOT / "MECH_DIAG"), "MECH-DIAG must finish before ABL"
    start = time.monotonic()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available()
    status("ABL_PREFLIGHT", completed=0, total=15)
    data, head, criterion, masks, obs, pre = setup()
    preflight(data, head, criterion, pre)
    head_before = cpu_tree(head.state_dict())
    contract = dict(protocol=sha(ROOT / "PROTOCOL.md"), source=sha(ROOT / "run_abl.py"),
                    cache=sha(DEC / "cache/DONE.json"), speed_reference_mps=pre["v_ref_mps"],
                    epochs=20, batch=256, device=DEVICE)
    runs, completed = {}, 0
    for arm in ARMS:
        runs[arm] = {}
        for seed in SEEDS:
            result = train(arm, seed, data, head, criterion, masks, obs, contract, completed)
            if result is None:
                return
            runs[arm][str(seed)] = result
            completed += 1
    assert all(torch.equal(v.cpu(), head_before[k]) for k, v in head.state_dict().items())
    decisions = decision(runs)
    write(FOLDER / "RESULT.json", dict(experiment="EXP-REFINE-ABL", contract=contract, runs=runs, **decisions))
    lines = ["# EXP-REFINE-ABL", "",
             "触发的判定为 " + ("、".join(k for k, v in decisions["decision"].items() if v) or "无") + "。", "",
             "A1 三个 seed 的重训结果与 archive 18 逐位一致。保留各臂指定输入，参数量差异列于下表。",
             "", "| 实验臂 | 参数量 | 与 A1 参数量差异 | 总体 minADE 改善 | 总体 minFDE 改善 | 静止 minFDE 改善 |",
             "|---|---:|---:|---:|---:|---:|"]
    for arm, groups in decisions["paired"].items():
        values = [groups[group][metric]["mean"] for group, metric in
                  (("overall", "minADE"), ("overall", "minFDE"), ("stationary", "minFDE"))]
        lines.append(f"| {arm} | {COUNTS[arm]} | {COUNTS[arm]/COUNTS['A1']-1:+.2%} | "
                     + " | ".join(f"{v:+.2%}" for v in values) + " |")
    lines += ["", "RESULT.json 留存了配对种子、各组误差、Top1 FDE、修正量和饱和比例。",
              "输入信息和首层可训练参数量随部分干预一起变化，结论限于固定 B0 缓存的机制诊断。"]
    (FOLDER / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    run_receipts = [FOLDER / f"runs/{arm}_s{seed}/DONE.json" for arm in ARMS for seed in SEEDS]
    finish(FOLDER, [FOLDER / k for k in ("RESULT.json", "REPORT.md", "PREFLIGHT_PASS.json")] + run_receipts,
           seconds=time.monotonic()-start)
    status("DONE", completed=15, total=15)


if __name__ == "__main__":
    main()
