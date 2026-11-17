"""Read-only label/array lineage and timing audit. No torch or inference."""
from __future__ import annotations

import ast
from collections import Counter, defaultdict
import hashlib
from pathlib import Path

import numpy as np

from speed_io import read, sha, write


def signature(x):
    return hashlib.blake2b(np.ascontiguousarray(x, np.float32).tobytes(), digest_size=20).digest()


def numeric_assignments(path):
    """Read literal assignments without executing external preprocessing scripts."""
    tree = ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    found = defaultdict(list)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                name = target.id if isinstance(target, ast.Name) else target.attr if isinstance(target, ast.Attribute) else ''
                found[name].append(node.value.value)
    return dict(found)


def raw_index(files):
    index = defaultdict(set); summaries = []
    for file in files:
        rows = np.genfromtxt(file, dtype=str)
        frames = rows[:, 0].astype(np.float32).astype(np.int64)
        ids = rows[:, 1].astype(np.float32).astype(np.int64)
        xy = rows[:, [13, 15]].astype(np.float32)
        conflicts = 0
        gaps = Counter()
        for identity in np.unique(ids):
            mask = ids == identity
            track = {}
            for frame, point in zip(frames[mask], xy[mask]):
                if frame in track and not np.array_equal(point, track[frame]):
                    conflicts += 1
                track[int(frame)] = point
            ordered = sorted(track)
            gaps.update(np.diff(ordered).tolist())
            for start in ordered:
                if all(start + j in track for j in range(20)):
                    sig = signature(np.stack([track[start + j] for j in range(20)]))
                    index[sig].add(file.stem.removesuffix('_train'))
        summaries.append(dict(file=str(file), sha256=sha(file), n_rows=len(rows),
                              frame_min=int(frames.min()), frame_max=int(frames.max()),
                              track_label_frame_gap_histogram=dict(gaps), conflicting_duplicate_rows=conflicts))
    return index, summaries


def run_audit(c, root, contract_sha):
    root = Path(root); old = read(c['recording_audit'])
    code = Path(c['astra_dir'])
    assignments = {name: numeric_assignments(code/name) for name in
                   ('data/eth.py', 'data/process_eth.py', 'utils/video2images.py')}
    code_checks = dict(frame_skip_one=assignments['data/eth.py'].get('frame_skip') == [1, 1],
                       preprocessing_scale_one=assignments['data/process_eth.py'].get('scale') == [1],
                       video_frame_interval_ten=assignments['utils/video2images.py'].get('frame_interval') == [10])
    # Videos establish metadata consistency with the extraction code, not the
    # original annotation-to-video alignment or a new physical calibration.
    import cv2
    videos = []
    for file in sorted(Path(c['video_dir']).glob('*.avi')):
        cap = cv2.VideoCapture(str(file))
        fps = float(cap.get(cv2.CAP_PROP_FPS)); count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
        videos.append(dict(file=str(file), fps=fps, frame_count=count,
                           implied_dt=10/fps if fps > 0 else None))
    video_ok = len(videos) >= 5 and all(v['fps'] > 0 and abs(v['implied_dt']-.4) < 1e-6 for v in videos)
    populations = []; raw_summaries = []; checks = []
    lineage = {row['fold']: row for row in old['recording_lineage']}
    # All five TEST folds compare sites; ETH/HOTEL TRAIN arrays also checked in full.
    for fold in ('eth', 'hotel', 'univ', 'zara01', 'zara02'):
        for split in (('train', 'test') if fold in ('eth', 'hotel') else ('test',)):
            names = lineage[fold][split + '_recordings']
            suffix = '_train' if split == 'train' else ''
            label_dir = Path(c['data_dir']).parent/'labels'/fold
            files = [label_dir/(name+suffix+'.txt') for name in names]
            index, summaries = raw_index(files); raw_summaries.extend(summaries)
            raw = np.load(Path(c['data_dir'])/f'{fold}_data_{split}.npy', mmap_mode='r')
            counts = Counter(); unmatched = []; ambiguous = 0
            frame_increments = Counter()
            for i in range(len(raw)):
                matches = index.get(signature(raw[i, 0, :, -2:]), set())
                if not matches:
                    unmatched.append(i)
                elif len(matches) > 1:
                    ambiguous += 1
                else:
                    counts.update(matches)
                frame = raw[i, 0, :, 1]
                if np.all(frame == -1):
                    frame_increments.update({'sentinel_minus_one': 1})
                else:
                    frame_increments.update({str(float(g)): int(n) for g, n in zip(*np.unique(np.diff(frame), return_counts=True))})
            ok = not unmatched and not ambiguous and not any(x['conflicting_duplicate_rows'] for x in summaries)
            ok = ok and not any(k not in ('1.0', 'sentinel_minus_one') for k in frame_increments)
            checks.append(ok)
            populations.append(dict(fold=fold, split=split, n=len(raw), passed=ok,
                                    coordinate_match='EXACT_FLOAT32_FULL_20_FOCAL_POINTS',
                                    unmatched=len(unmatched), first_unmatched=unmatched[:10],
                                    ambiguous_recordings=ambiguous, assigned_recordings=dict(counts),
                                    processed_frame_increment_histogram=dict(frame_increments),
                                    raw_window_label_increment=1))
    result = dict(input_contract_sha256=contract_sha, pipeline_consistent=all(code_checks.values()) and video_ok and all(checks),
                  code_checks=code_checks, source_literal_assignments=assignments,
                  videos=videos, populations=populations, raw_label_summaries=raw_summaries,
                  protocol_dt_seconds=.4, protocol_coordinate_unit='metre',
                  physical_calibration_independently_verified=False,
                  annotation_video_alignment_independently_verified=False,
                  conclusion_scope='No extra coordinate scaling or focal frame skipping in this processing chain if PASS; upstream physical calibration remains an assumption.')
    write(root/'DIAG'/'PREPROCESSING_AUDIT.json', result)
    if not result['pipeline_consistent']:
        raise ValueError('PREPROCESSING_AUDIT_FAILED: diagnose before model claims')
    return result
