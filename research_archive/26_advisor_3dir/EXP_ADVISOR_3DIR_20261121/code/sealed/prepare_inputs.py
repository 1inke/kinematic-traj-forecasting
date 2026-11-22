"""Copy sealed experiment inputs from a mounted Drive and verify their contents."""
import argparse
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import shutil
import zipfile

BUNDLES = ("c0_inputs.zip", "baseline_sources.zip", "fivefold_inputs.zip")


def digest(path):
    with path.open("rb") as f:
        h = sha256()
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def unpack(source, destination, expected):
    destination = Path(destination)
    for name, wanted in expected.items():
        rel = PurePosixPath(name)
        if rel.is_absolute() or ".." in rel.parts or "\\" in name:
            raise ValueError(f"Invalid archive member: {name}")
        target = destination.joinpath(*rel.parts)
        content = source.read(name)
        if sha256(content).hexdigest() != wanted:
            raise ValueError(f"Archive member checksum mismatch: {name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and digest(target) == wanted:
            continue
        temp = target.with_suffix(target.suffix + ".part")
        temp.write_bytes(content)
        temp.replace(target)


def prepare(drive, workspace, fivefold=False):
    drive, workspace = Path(drive), Path(workspace)
    manifest = json.loads((drive / "COLAB_INPUTS.json").read_text(encoding="utf-8-sig"))
    pinned = {x["file"]: x for x in manifest["bundles"]}
    selected = BUNDLES if fivefold else BUNDLES[:2]
    archives = workspace / "_archives"
    archives.mkdir(parents=True, exist_ok=True)
    for name in selected:
        item = pinned[name]
        source, local = drive / name, archives / name
        if not local.exists() or digest(local) != item["sha256"]:
            part = local.with_suffix(".part")
            shutil.copyfile(source, part)
            if part.stat().st_size != item["size"] or digest(part) != item["sha256"]:
                raise ValueError(f"Copied archive checksum mismatch: {name}")
            part.replace(local)
        manifest_name = local.stem + ".manifest.json"
        with zipfile.ZipFile(local) as archive:
            info = archive.read(manifest_name)
            members = json.loads(info)["members"]
            if set(archive.namelist()) != (set(members) | {manifest_name}):
                raise ValueError(f"Archive file list mismatch: {name}")
            unpack(archive, workspace, members)
            (workspace / manifest_name).write_bytes(info)
        print(f"Verified and unpacked {name}", flush=True)
    return {name: pinned[name]["sha256"] for name in selected}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--fivefold", action="store_true")
    args = parser.parse_args()
    prepare(args.drive, args.workspace, args.fivefold)

