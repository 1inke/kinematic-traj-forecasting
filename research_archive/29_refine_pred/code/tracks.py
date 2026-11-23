"""Bind the sealed evaluation rows to anonymized recording/track clusters."""

from pathlib import Path

import numpy as np

from cloud_handoff import digest, read, write
from e_a_diag.run import DATA, frame_windows, load_label


RECORDINGS = {
    "eth": ("biwi_eth",),
    "zara01": ("crowds_zara01",),
    "hotel": ("biwi_hotel",),
    "zara02": ("crowds_zara02",),
    "univ": ("students001", "students003"),
}


def audit_tracks(fold, test, source_manifest, destination):
    audit = read(source_manifest)["folds"][fold]
    processed = DATA / "processed_data_diverse"
    historical = processed / f"{fold}_data_test.npy"
    count_path = processed / f"{fold}_num_test.npy"
    labels = {name: DATA / f"labels/{fold}/{name}.txt" for name in RECORDINGS[fold]}
    hashes = {p.name: digest(p) for p in (test, historical, count_path, *labels.values())}
    expected = audit["approved_source_arrays"]
    if hashes[historical.name] != expected["data_test"] or hashes[count_path.name] != expected["num_test"]:
        raise ValueError(f"Processed source hash differs for {fold}")
    cached, receipt = destination / "TRACKS.npz", destination / "TRACKS.json"
    if receipt.exists() and cached.exists():
        prior = read(receipt)
        if prior["sources"] == hashes and prior["code"] == digest(Path(__file__)) and prior["track_sha256"] == digest(cached):
            return prior

    archive = np.load(historical, mmap_mode="r", allow_pickle=False)
    counts = np.load(count_path, allow_pickle=False)
    with np.load(test, allow_pickle=False) as sealed:
        ids = sealed["ids"]
        if (len(np.unique(ids)) != audit["splits"]["test"]["windows"]
                or not np.array_equal(counts, sealed["num_valid"])
                or not np.array_equal(archive[:, :, :8, -2:], sealed["past"])
                or not np.array_equal(archive[:, 0, 8:, -2:], sealed["future"])):
            raise ValueError(f"Historical and sealed inputs disagree for {fold}")

    clusters, recordings = [], []
    row = 0
    for recording, label in labels.items():
        for frame, histories, futures, _ in frame_windows(*load_label(label)):
            peds = np.array(list(futures))
            obs = np.stack([histories[ped] for ped in peds])
            fut = np.stack(list(futures.values()))
            tag = -1 if recording == "students001" else frame
            for i, ped in enumerate(peds):
                if len(peds) < 9:
                    chosen = np.r_[i, np.flatnonzero(np.arange(len(peds)) != i)]
                else:
                    xy = obs[:, -1].astype(np.float64)
                    distances = np.linalg.norm(xy - xy[i], axis=1)
                    nearest = np.argsort(distances)[:9]
                    chosen = nearest[distances[nearest] < 5]
                n = len(chosen)
                if (row >= len(ids) or counts[row] != n or archive[row, 0, 7, 1] != tag
                        or not np.array_equal(obs[chosen], archive[row, :n, :8, -2:])
                        or not np.array_equal(fut[chosen], archive[row, :n, 8:, -2:])):
                    raise ValueError(f"Historical window did not reconstruct: {fold}/{recording}, row {row}")
                clusters.append((recording, int(ped)))
                recordings.append(recording)
                row += 1
    if row != len(ids):
        raise ValueError(f"Reconstructed {row}/{len(ids)} windows for {fold}")
    mapping = {track: index for index, track in enumerate(sorted(set(clusters)))}
    tracks = np.array([mapping[track] for track in clusters], dtype=np.int32)
    destination.mkdir(parents=True, exist_ok=True)
    temporary = cached.with_name("TRACKS.part.npz")
    np.savez_compressed(temporary, ids=ids, tracks=tracks, recording=np.array(recordings))
    temporary.replace(cached)
    result = dict(status="PASS", windows=row, trajectories=len(mapping), sources=hashes, code=digest(Path(__file__)),
                  track_sha256=digest(cached),
                  by_recording={name: recordings.count(name) for name in labels})
    write(receipt, result)
    return result
