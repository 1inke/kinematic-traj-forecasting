"""Prepare the sealed study locally and run a one-GPU, one-worker queue."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import os
from pathlib import Path
import shutil
import sys
import zipfile


HERE = Path(__file__).resolve().parent
COLAB = HERE.parent / "colab"
UPLOAD = COLAB / "upload"
WORKSPACE = HERE / "workspace"
OUT = HERE / "results"
SEALED_SHA = "5a2952fb93dccd1f5c6a17a3a1b85c64786eaf734f9214df6900355fd39df4e8"
LOCAL_SOURCES = {
    "advisor_train.py": COLAB / "advisor/advisor_train.py",
    "advisor_study.py": COLAB / "advisor/advisor_study.py",
    "run_b0.py": COLAB / "run_b0.py",
}


def prepare():
    bundle = UPLOAD / "advisor_study_resume.zip"
    if sha256(bundle.read_bytes()).hexdigest() != SEALED_SHA:
        raise ValueError("Cloud study bundle failed verification")
    WORKSPACE.mkdir(exist_ok=True)
    with zipfile.ZipFile(bundle) as archive:
        for name in archive.namelist():
            if Path(name).name != name or name in (".", ".."):
                raise ValueError(f"Unexpected study bundle member: {name}")
            (WORKSPACE / name).write_bytes(archive.read(name))
    for name, source in LOCAL_SOURCES.items():
        shutil.copyfile(source, WORKSPACE / name)
    sys.path.insert(0, str(WORKSPACE))
    from prepare_inputs import prepare as unpack_inputs

    unpack_inputs(UPLOAD, WORKSPACE, fivefold=True)


def preflight():
    import advisor_study as study

    torch = study.torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable on the local GPU")
    hashes = study.verify_inputs(WORKSPACE)
    fivefold = WORKSPACE / "fivefold_inputs.manifest.json"
    for name, checksum in study.read(fivefold)["members"].items():
        if study.digest(WORKSPACE / name) != checksum:
            raise ValueError(f"Five-fold input failed verification: {name}")
    cloud = study.read(OUT / "cloud/PREFLIGHT.json")
    binding = cloud["binding"]
    if cloud["status"] != "PASS" or binding["inputs"] != hashes or binding["fivefold_manifest"] != study.digest(fivefold):
        raise ValueError("Input protocol differs from the passed cloud preflight")
    for name, checksum in binding["code"].items():
        if name not in LOCAL_SOURCES and study.digest(WORKSPACE / name) != checksum:
            raise ValueError(f"Study protocol source differs from the cloud preflight: {name}")
    checks = dict(status="PASS", local_gpu=torch.cuda.get_device_name(0), max_parallel=1,
                  local_torch=str(torch.__version__), local_cuda=torch.version.cuda,
                  cloud_torch="2.14.1+cu130", cloud_cuda="13.0", cloud_gpu="NVIDIA L4",
                  cloud_bundle_sha256=SEALED_SHA, checks=study.deterministic(),
                  gpu=study.structural("cuda:0", windows=16, latency=False),
                  at_utc=datetime.now(timezone.utc).isoformat())
    study.write_json(OUT / "LOCAL_RUNTIME.json", checks)


def verify_restore():
    """Restore an imported state and run one real update outside the result directories."""
    from advisor_study import read, write_json
    from advisor_train import Trainer, core, inputs, load_data, restore
    from prepare_inputs import digest

    for path in sorted((OUT / "stage1/runs").glob("*/epoch_???.json"), reverse=True):
        run = path.parent
        if (run / "RESULT.json").exists() or not path.with_suffix(".pt").exists():
            continue
        cloud = read(run / "CONTRACT.json")
        fit_path, windows = inputs(cloud["fold"])[0]
        fit = load_data(fit_path, windows)
        trainer = Trainer(fit, cloud["arm"], cloud["seed"], "cuda:0", cloud["epochs"])
        torch = core.torch
        contract = dict(cloud, torch=str(torch.__version__), cuda=torch.version.cuda,
                        gpu_name=torch.cuda.get_device_name(0),
                        source=dict(cloud["source"], **{"advisor_train.py": digest(WORKSPACE / "advisor_train.py")}))
        epoch = restore(run, trainer, contract, local_gpu=True)
        if epoch == 0 or epoch >= cloud["epochs"]:
            continue
        write_json(OUT / "RESTORE_CHECK.json", dict(run=run.name, restored_epoch=epoch, next_batch=trainer.step()))
        print(f"Verified cloud restore and next update for {run.name} after epoch {epoch}", flush=True)
        return
    print("No imported partial checkpoint available for the restore probe", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--sensitivity", action="store_true")
    args = parser.parse_args()
    os.environ["ADVISOR_LOCAL_GPU"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "1"
    prepare()
    import advisor_study as study

    study.status(OUT,"LOCAL_PREFLIGHT")
    try:
        preflight()
        if args.check:
            verify_restore()
            return
        study.run_phases(OUT, cap=1, admit_until=float("inf"), deadline=float("inf"),
                         sensitivity=args.sensitivity)
    except Exception as error:
        study.status(OUT, "FAILED", error=str(error))
        raise


if __name__ == "__main__":
    main()
