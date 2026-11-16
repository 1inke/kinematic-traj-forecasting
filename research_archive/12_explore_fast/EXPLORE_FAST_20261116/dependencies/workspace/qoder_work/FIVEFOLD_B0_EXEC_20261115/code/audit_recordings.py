"""Human-confirmed recording-disjoint audit; original scene-ID failure is retained."""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fivefold_common import FOLDS, now, read_json, sha256, write_json_once


def signature(xy):
    return hashlib.blake2b(np.ascontiguousarray(xy, dtype=np.float32).tobytes(), digest_size=20).digest()


def raw_index(files):
    index = defaultdict(set)
    for file in files:
        recording = file.stem.removesuffix('_train')
        table = np.genfromtxt(file, dtype=str)
        frames = table[:, 0].astype(np.float32).astype(np.int64)
        ids = table[:, 1].astype(np.float32).astype(np.int64)
        xy = table[:, [13, 15]].astype(np.float32)
        for identity in np.unique(ids):
            chosen = ids == identity
            track = dict(zip(frames[chosen].tolist(), xy[chosen]))
            for start in sorted(track):
                if all(start + step in track for step in range(20)):
                    index[signature(np.stack([track[start + step] for step in range(20)]))].add(recording)
    return index


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--astra', type=Path, required=True)
    p.add_argument('--original-audit', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    original = read_json(args.original_audit)
    # Extract only the reviewed split function; importing eth.py would load torch.
    split_source = args.astra / 'data/eth.py'
    tree = ast.parse(split_source.read_text(encoding='utf-8-sig'))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'get_ethucy_split')
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(split_source), 'exec'), namespace)
    by_fold = {row['fold']: row for row in original['folds']}
    raw_hashes = {}
    lineage = []
    passed = True
    for fold in FOLDS:
        train_names, _, test_names = namespace['get_ethucy_split'](fold)
        expected_train = {name.removesuffix('_train') for name in train_names}
        expected_test = set(test_names)
        assert not expected_train & expected_test
        labels = args.astra / 'datasets/eth_ucy/labels' / fold
        training_files = [labels / (name + '.txt') for name in train_names]
        testing_files = [labels / (name + '.txt') for name in test_names]
        for file in training_files + testing_files:
            raw_hashes[str(file.resolve())] = sha256(file)
        train_index, test_index = raw_index(training_files), raw_index(testing_files)
        result = dict(fold=fold, train_recordings=sorted(expected_train), test_recordings=sorted(expected_test),
                      shared_coarse_scene_ids=sorted(set(by_fold[fold]['train_scene_ids']) & set(by_fold[fold]['test_scene_ids'])),
                      populations={})
        for split, correct, opposite in [('train', train_index, test_index), ('test', test_index, train_index)]:
            path = args.astra / f'datasets/eth_ucy/processed_data_diverse/{fold}_data_{split}.npy'
            if sha256(path) != by_fold[fold]['sha256']['data_' + split]:
                raise ValueError('DATA_CHANGED_SINCE_F0')
            data = np.load(path, mmap_mode='r')
            assigned = Counter()
            unmatched, ambiguous, opposite_matches = [], 0, 0
            for row in range(len(data)):
                sig = signature(data[row, 0, :, -2:])
                matches = correct.get(sig, set())
                if not matches:
                    unmatched.append(row)
                elif len(matches) > 1:
                    ambiguous += 1
                else:
                    assigned.update(matches)
                if sig in opposite:
                    opposite_matches += 1
            result['populations'][split] = dict(samples=len(data), unique_recording_match=dict(assigned),
                unmatched_count=len(unmatched), first_unmatched_rows=unmatched[:20],
                ambiguous_recording_count=ambiguous, opposite_split_focal_path_matches=opposite_matches)
            if unmatched or ambiguous or opposite_matches:
                passed = False
        other_failures = [failure for failure in by_fold[fold]['failures'] if failure != 'A3_scene_overlap']
        result['remaining_original_failures'] = other_failures
        if other_failures:
            passed = False
        lineage.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    report = dict(original)
    report.update(at=now(), stage='F0_recording_lineage_audit', passed=passed,
        original_audit_sha256=sha256(args.original_audit),
        original_audit_retained=True, recording_lineage=lineage,
        protocol_amendment=dict(human_choice='Keep original arrays; verify recording-disjoint splits before proceeding',
            replaces='A3 strict disjointness of coarse scene IDs only',
            unchanged='All original arrays, scene_tail splitting, model/loss/budgets/seeds and other F0 checks',
            limitation='Recording-disjoint is not physical-location-disjoint; scene IDs include homography aliases'),
        raw_label_sha256=raw_hashes, split_source_sha256=sha256(split_source),
        test_read_scope='Fixed F0 integrity/lineage audit only; no model test metrics or budget selection')
    for row in report['folds']:
        row['coarse_scene_original_passed'] = row['passed']
        row['coarse_scene_original_failures'] = row['failures']
        row['failures'] = [value for value in row['failures'] if value != 'A3_scene_overlap']
        lineage_row = next(value for value in lineage if value['fold'] == row['fold'])
        lineage_ok = all(not value['unmatched_count'] and not value['ambiguous_recording_count'] and
                         not value['opposite_split_focal_path_matches'] for value in lineage_row['populations'].values())
        row['passed'] = not row['failures'] and lineage_ok
    write_json_once(args.out, report)
    print(json.dumps(dict(status='PASS' if passed else 'FAIL', output=str(args.out))))
    raise SystemExit(0 if passed else 1)


if __name__ == '__main__':
    main()
