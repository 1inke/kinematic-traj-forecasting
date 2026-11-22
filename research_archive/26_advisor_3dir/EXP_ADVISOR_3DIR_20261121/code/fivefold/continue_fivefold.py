"""Run the remaining admitted five-fold jobs after importing a local handoff."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import time

from advisor_study import FOLDS, SEEDS, Deadline, confirm, queue, read, run_path, status, usable
from advisor_train import inputs, torch
from prepare_inputs import digest
from run_b0 import publish, verify_inputs, write_json


ROOT = Path(__file__).resolve().parent
LOCAL_TRAINER = "c9c58a9b22d42147fb645ac27b6fe111a5b0ff72eac0644cc60202b00b49e13a"
CLOUD_TRAINER = "7768beb8b2749d52edff7e5617ac76f62025acf2912f797d3832d7ec7f2ae52f"


def contract(arm, fold, seed):
    return dict(arm=arm, fold=fold, seed=seed, order_seed=seed, epochs=64, batch=64,
                device="cuda:0", torch=str(torch.__version__), cuda=torch.version.cuda,
                gradient_clip=5, training_inputs={file.name: digest(file) for file, _ in inputs(fold)},
                source={name: digest(ROOT / name) for name in
                        ("advisor_model.py", "advisor_train.py", "advisor_eval.py")},
                baseline_source=digest(ROOT / "baseline_sources.manifest.json"))


def check_local(saved, target):
    expected = dict(target, torch="2.11.0+cu128", cuda="12.8",
                    gpu_name="NVIDIA GeForce RTX 5070 Laptop GPU",
                    source=dict(target["source"], **{"advisor_train.py": LOCAL_TRAINER}))
    if saved != expected:
        raise ValueError(f"Unexpected local contract for {target['arm']} {target['fold']} s{target['seed']}")


def checkpoint(src, epoch, target):
    path = src / f"epoch_{epoch:03d}.pt"
    receipt = read(path.with_suffix(".json"))
    if receipt["epoch"] != epoch or digest(path) != receipt["sha256"]:
        raise ValueError(f"Invalid local checkpoint receipt: {path}")
    saved = read(src / "CONTRACT.json")
    check_local(saved, target)
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["contract"] != saved or state["epoch"] != epoch or state["updates"] != receipt["updates"]:
        raise ValueError(f"Invalid local checkpoint state: {path}")
    return state, receipt


def assign_checkpoint(src, dst, epoch, target):
    migrated = dst / "FROM_LOCAL.json"
    path = dst / f"epoch_{epoch:03d}.pt"
    if migrated.is_file():
        if read(migrated)["epoch"] != epoch or read(dst / "CONTRACT.json") != target:
            raise ValueError(f"Checkpoint migration or runtime changed for {dst.name}")
        receipt_path = path.with_suffix(".json")
        if path.is_file() and receipt_path.is_file() and digest(path) == read(receipt_path)["sha256"]:
            return
    state, receipt = checkpoint(src, epoch, target)
    dst.mkdir(parents=True, exist_ok=True)
    spec = dst / "CONTRACT.json"
    if spec.is_file() and read(spec) != target:
        raise ValueError(f"Existing runtime contract differs for {dst.name}")
    write_json(spec, target)
    state["contract"] = target
    with tempfile.TemporaryDirectory(prefix="from_local_") as temp:
        local = Path(temp) / path.name
        torch.save(state, local)
        checksum = publish(local, path)
    write_json(path.with_suffix(".json"), dict(sha256=checksum, epoch=epoch, updates=state["updates"]))
    write_json(migrated, dict(epoch=epoch, source=receipt, source_contract=read(src / "CONTRACT.json")))


def admit(transfer):
    hashes = verify_inputs(ROOT)
    fp = ROOT / "fivefold_inputs.manifest.json"
    manifest = read(transfer / "HANDOFF.json")
    if manifest["inputs"] != hashes or manifest["fivefold_manifest"] != digest(fp):
        raise ValueError("Handoff and Colab inputs differ")
    for name, checksum in read(fp)["members"].items():
        if digest(ROOT / name) != checksum:
            raise ValueError(f"Five-fold data changed: {name}")
    if digest(ROOT / "advisor_train.py") != CLOUD_TRAINER:
        raise ValueError("Unexpected Colab trainer source")
    admission = read(transfer / "ADMISSION.json")
    selected = admission["admitted"]
    if admission["status"] != "LOCKED" or admission["always"] != ["B0"]:
        raise ValueError("The local five-fold admission is not locked")
    if selected != {"R2": "minFDE"}:
        raise ValueError(f"Unexpected admission for this handoff: {selected}")
    return manifest, admission


def pending(out, transfer, manifest, selected, check=False):
    queued = []
    imports = manifest["runs"]
    keys = set()
    for fold in FOLDS:
        for arm in ("B0", *selected):
            for seed in SEEDS:
                dst = run_path(out, arm, fold, seed)
                keys.add(dst.name)
                entry = imports.get(dst.name, {})
                src = transfer / "runs" / dst.name
                target = contract(arm, fold, seed)
                if entry.get("completed"):
                    r = read(src / "RESULT.json")
                    check_local(r["contract"], target)
                    if r["status"] != "PASS" or r["epochs"] != 64 or not r["artifacts"]:
                        raise ValueError(f"Invalid imported result: {dst.name}")
                    if not check:
                        dst.mkdir(parents=True, exist_ok=True)
                        local = dst / "RESULT.json"
                        if local.is_file() and read(local) != r:
                            raise ValueError(f"Existing results conflict for {dst.name}")
                        write_json(local, r)
                    continue
                if entry.get("checkpoint"):
                    epoch = entry["checkpoint"]
                    if check:
                        checkpoint(src, epoch, target)
                    elif not usable(dst, deep=True):
                        assign_checkpoint(src, dst, epoch, target)
                queued.append((arm, fold, seed, dst))
    if set(imports) - keys:
        raise ValueError("The handoff contains runs outside the admitted experiment")
    return queued


def memory_cap(transfer, requested):
    cost = read(transfer / "L4_COST.json")["arms"]
    free, total = torch.cuda.mem_get_info()
    peak = max(cost[a]["L4_peak_reserved_bytes"] for a in ("B0", "R2"))
    # The original L4 preflight passed six lanes; include a CUDA context allowance.
    headroom = max(0, min(free - 2 * 1024**3, int(.85 * total)))
    safe = headroom // (peak + 768 * 1024**2)
    if safe < 1:
        raise RuntimeError("Insufficient free GPU memory for a training lane")
    return min(requested, 6, safe)


def work(out, transfer, maximum, hours, check):
    torch.set_num_threads(1)
    manifest, admission = admit(transfer)
    selected = admission["admitted"]
    if check:
        todo = pending(out, transfer, manifest, selected, check=True)
        print(f"Verified handoff: {30 - len(todo)} completed, {len(todo)} to run", flush=True)
        return
    if not torch.cuda.is_available() or "L4" not in torch.cuda.get_device_name(0).split():
        raise RuntimeError("This continuation requires an NVIDIA L4 GPU")
    cap = memory_cap(transfer, maximum)
    (out / "stage2").mkdir(parents=True, exist_ok=True)
    (out / "stage1").mkdir(exist_ok=True)
    write_json(out / "stage1/RESULT.json", read(transfer / "C0_RESULT.json"))
    write_json(out / "stage2/ADMISSION.json", admission)
    todo = pending(out, transfer, manifest, selected)
    write_json(out / "HANDOFF.json", manifest)
    write_json(out / "L4_RUNTIME.json", dict(max_parallel=cap, torch=str(torch.__version__),
                                            cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                                            c0_completed=33, local_fivefold_completed=30 - len(todo),
                                            at_utc=datetime.now(timezone.utc).isoformat()))
    print(f"Prepared {len(todo)} five-fold jobs; GPU parallelism: {cap}", flush=True)
    start = time.monotonic()
    end = start + hours * 3600
    finished = queue(out, "STAGE2_CLOUD", todo, cap, end, end + 90)
    if finished["failed"]:
        raise RuntimeError(f"Five-fold jobs failed: {finished['failed']}")
    result = confirm(out, selected)
    rows = ["# EXP-ADVISOR-3DIR 五折续跑", "",
            f"复用本机已完成的 {30 - len(todo)} 组五折结果。Colab 队列覆盖 {len(todo)} 组。",
            "本机保留已完成组的原始预测和权重，Drive 保存结果回执及其散列。",
            "C0 筛选和五折验证跨设备运行。各组环境见运行回执，续跑起点见 FROM_LOCAL.json。", ""]
    for arm, value in result.items():
        rows.append(f"- {arm} 五折通过 = {value['F-PASS']}，ETH 改善 {value['F-ETH']:+.2%}。")
    with tempfile.TemporaryDirectory(prefix="advisor_report_") as temp:
        path = Path(temp) / "REPORT.md"
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        publish(path, out / path.name)
    names = ("stage1/RESULT.json", "stage2/ADMISSION.json", "stage2/RESULT.json",
             "L4_RUNTIME.json", "HANDOFF.json", "REPORT.md")
    write_json(out / "DONE.json", dict(status="PASS", stage1=33, stage2=30,
                                       files={name: digest(out / name) for name in names}))
    status(out, "DONE", stage1=33, stage2=30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--transfer", type=Path, required=True)
    parser.add_argument("--parallel", type=int, choices=range(1, 7), default=6)
    parser.add_argument("--hours", type=float, default=8)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        work(args.out, args.transfer, args.parallel, args.hours, args.check)
    except Deadline:
        status(args.out, "TIME_LIMIT_RESUMABLE")
        raise SystemExit(75)
