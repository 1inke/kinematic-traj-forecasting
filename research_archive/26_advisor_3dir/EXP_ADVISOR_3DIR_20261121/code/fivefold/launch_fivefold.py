"""Embedded in the one-cell Colab notebook for the five-fold handoff."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import traceback
import zipfile


def prepare(drive_root, workspace, archive_name, archive_sha):
    archive_path = drive_root / archive_name
    if sha256(archive_path.read_bytes()).hexdigest() != archive_sha:
        raise ValueError("Handoff archive checksum differs")
    workspace.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        files = json.loads(archive.read("MANIFEST.json"))
        if set(archive.namelist()) != set(files) | {"MANIFEST.json"}:
            raise ValueError("Unexpected handoff archive members")
        for name, checksum in files.items():
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or "\\" in name:
                raise ValueError(f"Invalid handoff path: {name}")
            content = archive.read(name)
            if sha256(content).hexdigest() != checksum:
                raise ValueError(f"Handoff member checksum differs: {name}")
            target = workspace.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    subprocess.run([sys.executable, "-B", str(workspace / "prepare_inputs.py"),
                    "--drive", str(drive_root), "--workspace", str(workspace), "--fivefold"], check=True)


def colab_main(archive_name, archive_sha, output_name):
    from google.colab import drive, runtime
    import torch

    mounted = False
    output = None
    workspace = Path("/content/advisor_fivefold")
    state = dict(status="FAILED", started_at_utc=datetime.now(timezone.utc).isoformat())
    try:
        if not torch.cuda.is_available() or "L4" not in torch.cuda.get_device_name(0).split():
            raise RuntimeError("请在「更改运行时类型」中选择 L4 GPU，然后执行本单元。")
        drive_root = Path("/content/drive/MyDrive/EXP_ADVISOR_3DIR_L4_20261121")
        if not drive_root.is_dir():
            drive.mount("/content/drive")
        mounted = True
        output = drive_root / output_name
        prepare(drive_root, workspace, archive_name, archive_sha)
        output.mkdir(exist_ok=True)
        print(f"L4 已就绪，正在加载断点和运行队列。进度目录 {output.name}", flush=True)
        with (output / "CLOUD_FIVEFOLD.log").open("a", encoding="utf-8") as log:
            code = subprocess.run(
                [sys.executable, "-u", "-B", str(workspace / "continue_fivefold.py"),
                 "--out", str(output), "--transfer", str(workspace / "transfer"),
                 "--parallel", "6", "--hours", "8"], stdout=log, stderr=subprocess.STDOUT).returncode
        state["exit_code"] = code
        if code == 0 and (output / "DONE.json").is_file():
            state["status"] = "COMPLETE"
        elif code == 75:
            state["status"] = "TIME_LIMIT_RESUMABLE"
    except BaseException:
        state["detail"] = traceback.format_exc()
        traceback.print_exc()
        raise
    finally:
        try:
            if output is not None and output.is_dir() and (workspace / "run_b0.py").is_file():
                sys.path.insert(0, str(workspace))
                from run_b0 import write_json

                state["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
                write_json(output / "CLOUD_RUN_STATUS.json", state)
        finally:
            try:
                if mounted:
                    subprocess.run([sys.executable, "-c",
                                    "from google.colab import drive; drive.flush_and_unmount(timeout_ms=120000)"],
                                   check=True, timeout=150)
            finally:
                print(f"请求释放 L4。运行状态 {state['status']}", flush=True)
                runtime.unassign()
