"""Recover interrupted Drive media downloads before resuming the sealed run."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
import re
import subprocess
import sys
from time import sleep
from urllib.error import URLError
from urllib.request import Request, urlopen

import run
from cloud_handoff import digest, read, write
from review_download import Archive, DRIVE, requests
from frozen_replay.run import drive_index


CHUNK = 256 * 1024
CONTENT_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+)")


def byte_range(url, headers, start):
    headers = dict(headers, Range=f"bytes={start}-{start + CHUNK - 1}", **{"Accept-Encoding": "identity"})
    for attempt in range(6):
        try:
            with urlopen(Request(url, headers=headers), timeout=12) as response:
                payload = response.read()
                if response.status == 200 and start == 0:
                    return payload, len(payload)
                match = CONTENT_RANGE.fullmatch(response.headers.get("Content-Range", ""))
                if response.status != 206 or not match:
                    raise ValueError("Drive response has no valid byte range")
                first, last, total = map(int, match.groups())
                if first != start or last != min(start + CHUNK, total) - 1 or len(payload) != last - first + 1:
                    raise ValueError("Drive response does not match the requested range")
                return payload, total
        except (OSError, URLError, ValueError) as error:
            if attempt == 5:
                raise RuntimeError(f"Drive media transfer failed at offset {start}: {type(error).__name__}") from None
            sleep(min(attempt + 1, 3))


def materialize(name, file, headers):
    path = run.OUT / "weights" / f"{name}.pt"
    if run.weight_path(name):
        return name
    want = run.formal(name)[0]["artifacts"]["final_ema.pt"]
    url = f"{DRIVE}/{file['id']}?alt=media"
    partial = path.parent / "partial" / name
    partial.mkdir(parents=True, exist_ok=True)
    manifest = partial / "MEDIA.json"
    if not manifest.is_file():
        first, size = byte_range(url, headers, 0)
        (partial / "0.bin").write_bytes(first)
        write(manifest, dict(size=size, checksum=want))
    manifest_data = read(manifest)
    if manifest_data["checksum"] != want:
        raise ValueError(f"Partial cache belongs to different weights for {name}")
    size = manifest_data["size"]
    byte_zero = (partial / "0.bin").read_bytes()
    if len(byte_zero) == size:
        whole = byte_zero
    else:
        chunks = []
        for offset in range(0, size, CHUNK):
            part = partial / f"{offset}.bin"
            expected = min(CHUNK, size - offset)
            if not part.is_file() or part.stat().st_size != expected:
                data, check_size = byte_range(url, headers, offset)
                if check_size != size:
                    raise ValueError(f"Drive media size changed for {name}")
                tmp = part.with_suffix(".part")
                tmp.write_bytes(data)
                tmp.replace(part)
            chunks.append(part.read_bytes())
        whole = b"".join(chunks)
    if sha256(whole).hexdigest() != want:
        raise ValueError(f"Drive checkpoint fails SHA-256 for {name}")
    dest = path.with_suffix(".range-download")
    dest.write_bytes(whole)
    dest.replace(path)
    if digest(path) != want:
        raise ValueError(f"Written checkpoint fails SHA-256 for {name}")
    return name


def replay_cached(confirmation):
    names = [name for name in confirmation if not run.completed(name) and run.weight_path(name)]
    if not names:
        return
    count = len(run.RUNS) - len(confirmation) + sum(run.completed(name) for name in confirmation)
    run.log_status("REPLAYING_CACHED_WEIGHTS", verified_replays=count, queued=len(names), total=len(run.RUNS))

    def evaluate(name):
        folder = run.metric_path(name).parent
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / "worker.log").open("a", encoding="utf-8") as log:
            process = subprocess.run([sys.executable, "-B", "-u", str(run.HERE / "run.py"), "--run", name],
                                     stdout=log, stderr=subprocess.STDOUT)
        if process.returncode == 7:
            raise run.InvalidReplay(f"Replication check failed for {name}; see worker.log")
        if process.returncode or not run.completed(name):
            raise RuntimeError(f"Could not complete {name}; see worker.log")
        return name

    with ThreadPoolExecutor(max_workers=2) as pool:
        for task in as_completed([pool.submit(evaluate, name) for name in names]):
            name = task.result()
            count += 1
            run.log_status("REPLAYING_CACHED_WEIGHTS", verified_replays=count, last_verified=name, total=len(run.RUNS))


def download_phase(pending, attempt):
    catalog = run.OUT / "weights" / "DRIVE_MEDIA.json"
    archive = Archive()
    if catalog.is_file():
        files = read(catalog)
    else:
        files = {}
    missing_entries = [name for name in pending if name not in files]
    if missing_entries:
        files.update(drive_index(archive, missing_entries))
        catalog.parent.mkdir(parents=True, exist_ok=True)
        write(catalog, files)
    run.log_status("CACHING_CONFIRMATION_WEIGHTS", missing_weights=len(pending), attempt=attempt, total=len(run.RUNS))
    with ThreadPoolExecutor(max_workers=4) as pool:
        tasks = {pool.submit(materialize, name, files[name], archive.headers): name for name in pending}
        for task in as_completed(tasks):
            try:
                print("CACHED", task.result(), flush=True)
            except (OSError, ValueError, RuntimeError) as error:
                if run.weight_path(tasks[task]) is None:
                    print("RETRY", tasks[task], error, flush=True)


def resume():
    check = run.OUT / "PREFLIGHT.json"
    if check.is_file() and read(check)["sources"] == run.lock():
        confirmation = [name for name, (_, fold, _) in run.RUNS.items() if fold in run.CONFIRMATION]
        for attempt in range(1, 4):
            replay_cached(confirmation)
            pending = [name for name in confirmation if not run.completed(name) and not run.weight_path(name)]
            if not pending:
                break
            run.log_status("CHECKPOINT_METADATA", missing_weights=len(pending), attempt=attempt, total=len(run.RUNS))
            try:
                download_phase(pending, attempt)
            except (OSError, RuntimeError, requests.RequestException, subprocess.SubprocessError) as error:
                print("TRANSFER RETRY", attempt, type(error).__name__, flush=True)
        replay_cached(confirmation)
        pending = [name for name in confirmation if not run.completed(name)]
        if pending:
            run.log_status("WAITING_FOR_CHECKPOINTS", verified_replays=len(run.RUNS) - len(pending),
                           missing_runs=pending, total=len(run.RUNS))
            return 2
    return subprocess.call([sys.executable, "-B", "-u", str(run.HERE / "run.py")])


if __name__ == "__main__":
    try:
        sys.exit(resume())
    except run.InvalidReplay as error:
        run.log_status("INVALID", error=str(error))
        raise
    except Exception as error:
        run.log_status("FAILED", error=str(error))
        raise
