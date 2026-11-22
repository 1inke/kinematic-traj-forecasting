"""Export a verified local snapshot and its specific Colab launcher."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

from launch_fivefold import prepare


HERE = Path(__file__).resolve().parent
COLAB = HERE.parent
LOCAL = COLAB.parent / "local"
INPUTS = COLAB / "upload"
SEALED_SHA = "5a2952fb93dccd1f5c6a17a3a1b85c64786eaf734f9214df6900355fd39df4e8"
sys.path.insert(0, str(COLAB))
from build_starter import code, markdown
from prepare_inputs import digest


def read(path):
    return json.loads(path.read_bytes())


def take_snapshot():
    results = LOCAL / "results"
    files = {}
    for destination, source in (
        ("ADMISSION.json", "stage2/ADMISSION.json"),
        ("C0_RESULT.json", "stage1/RESULT.json"),
        ("L4_COST.json", "cloud/COST.json"),
    ):
        files[destination] = (results / source).read_bytes()
    if len(list((results / "stage1/runs").glob("*/RESULT.json"))) != 33:
        raise ValueError("Incomplete C0 screening")
    records = {}
    for run in sorted((results / "stage2/runs").iterdir()):
        if not run.is_dir():
            continue
        result = run / "RESULT.json"
        if result.is_file():
            content = result.read_bytes()
            value = json.loads(content)
            if value["status"] != "PASS" or not value["artifacts"]:
                raise ValueError(f"Invalid local run: {result}")
            for name, checksum in value["artifacts"].items():
                if Path(name).name != name or digest(run / name) != checksum:
                    raise ValueError(f"Local artifact verification failed: {run.name}/{name}")
            files[f"runs/{run.name}/RESULT.json"] = content
            records[run.name] = dict(completed=True)
            continue
        for receipt in sorted(run.glob("epoch_???.json"), reverse=True):
            content = receipt.read_bytes()
            row = json.loads(content)
            weights = receipt.with_suffix(".pt")
            if not weights.is_file():
                continue
            state = weights.read_bytes()
            if sha256(state).hexdigest() != row["sha256"]:
                continue
            files[f"runs/{run.name}/{receipt.name}"] = content
            files[f"runs/{run.name}/{weights.name}"] = state
            files[f"runs/{run.name}/CONTRACT.json"] = (run / "CONTRACT.json").read_bytes()
            records[run.name] = dict(completed=False, checkpoint=row["epoch"])
            break
    hashes = LOCAL / "workspace"
    meta = dict(runs=records, inputs={name: digest(hashes / f"{name}.manifest.json")
                                     for name in ("c0_inputs", "baseline_sources")},
                fivefold_manifest=digest(hashes / "fivefold_inputs.manifest.json"),
                exported_at_utc=datetime.now(timezone.utc).isoformat())
    files["HANDOFF.json"] = json.dumps(meta, ensure_ascii=False, indent=2).encode()
    return files, meta


def build(destination):
    destination.mkdir(parents=True, exist_ok=True)
    snapshot, meta = take_snapshot()
    tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archive = destination / f"advisor_fivefold_handoff_{tag}.zip"
    notebook = destination / f"Advisor_Fivefold_{tag}.ipynb"
    output = f"ADVISOR_FIVEFOLD_{tag}"
    original = INPUTS / "advisor_study_resume.zip"
    if digest(original) != SEALED_SHA:
        raise ValueError("Original Colab bundle differs")
    with zipfile.ZipFile(original) as runner:
        sources = {name: runner.read(name) for name in runner.namelist()}
    files = dict(sources, **{"continue_fivefold.py": (HERE / "continue_fivefold.py").read_bytes()})
    files.update({f"transfer/{name}": content for name, content in snapshot.items()})
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as bundle:
        for name, content in files.items():
            if name.endswith(".py"):
                compile(content, name, "exec")
            bundle.writestr(name, content)
        bundle.writestr("MANIFEST.json", json.dumps({k: sha256(v).hexdigest() for k, v in files.items()}))
    checksum = digest(archive)
    raw = (HERE / "launch_fivefold.py").read_text(encoding="utf-8")
    entry = (f'# @title 启动 / 继续剩余五折（L4，最多 6 路） {{ display-mode: "form" }}\n'
             f"{raw}\ncolab_main({archive.name!r}, {checksum!r}, {output!r})\n")
    compile(entry, notebook.name, "exec")
    cell = code(entry)
    cell["metadata"]["cellView"] = "form"
    document = dict(nbformat=4, nbformat_minor=4, cells=[
        markdown("# EXP-ADVISOR-3DIR 五折续跑\n\n"
                 "点击下面单元左侧的 ▶，如出现 Drive 授权提示，使用保存实验的账号授权。"
                 "已完成的组会直接复用，剩余组最多同时跑 6 路。"
                 f"日志和结果写入 Drive 的 `{output}`，结束后自动请求释放 L4。"),
        cell,
    ], metadata=dict(accelerator="GPU", colab=dict(provenance=[], gpuType="L4"),
                     kernelspec=dict(display_name="Python 3", language="python", name="python3"),
                     language_info=dict(name="python")))
    notebook.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    with tempfile.TemporaryDirectory(prefix="fivefold_handoff_") as temp:
        temp = Path(temp)
        drive = temp / "drive"
        drive.mkdir()
        for name in ("COLAB_INPUTS.json", "c0_inputs.zip", "baseline_sources.zip", "fivefold_inputs.zip"):
            shutil.copyfile(INPUTS / name, drive / name)
        shutil.copyfile(archive, drive / archive.name)
        work = temp / "workspace"
        prepare(drive, work, archive.name, checksum)
        subprocess.run([sys.executable, "-B", str(work / "continue_fivefold.py"), "--check",
                        "--out", str(temp / "out"), "--transfer", str(work / "transfer")], check=True)

    receipt = dict(archive=str(archive.resolve()), sha256=checksum, notebook=str(notebook.resolve()),
                   output=output, handoff=meta)
    (destination / "EXPORT.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    build(parser.parse_args().destination)
