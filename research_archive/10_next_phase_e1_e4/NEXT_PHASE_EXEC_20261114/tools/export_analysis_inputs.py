"""Publish byte-preserved evidence plus compact saved-output analysis inputs.

Reads local immutable results; never imports a model, trains, or changes the source.
Run only when rebuilding the publication package, with the original experiment root.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
TEXT_AND_FIGURES = {'.py', '.json', '.md', '.ps1', '.cmd', '.xml', '.csv', '.yaml', '.yml', '.svg', '.png', '.pdf'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    project = source.parents[1]
    old = project / 'qoder_work/E2E_MOTION_RESEARCH_RESET_EXEC_20261102'
    original_seal = json.loads((source / 'FINAL_OUTPUT_SEAL.json').read_text(encoding='utf-8'))['files']
    source_seal = json.loads((source / 'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))['files']
    assert json.loads((source / 'standalone_recovery_20261114/RESULT.json').read_text())['status'] == 'SUCCESS'
    copied = []

    def copy_file(path, relative, anchor):
        destination = PACKAGE / relative
        actual = digest(path)
        expected = original_seal.get(str(path)) or source_seal.get(str(path))
        if expected and expected != actual:
            raise ValueError(f'Original seal mismatch: {path.name}')
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copyfile(path, destination)
        assert digest(destination) == actual
        if not any(row['path'] == relative.as_posix() for row in copied):
            copied.append(dict(path=relative.as_posix(), source=anchor, sha256=actual,
                               bytes=destination.stat().st_size, original_seal_checked=bool(expected)))

    for path in sorted(source.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in TEXT_AND_FIGURES:
            continue
        relative = path.relative_to(source)
        if any(part in {'resume', 'runtime_cache', '__pycache__', 'logs'} for part in relative.parts):
            continue
        copy_file(path, Path('snapshot') / relative, 'experiment/' + relative.as_posix())

    # The actual source dependencies are recorded by the scientific source seal.
    # Raw labels, processed inputs and fitted weights are deliberately not copied.
    for name in source_seal:
        path = Path(name)
        if path.is_relative_to(source) or path.suffix.lower() not in TEXT_AND_FIGURES:
            continue
        if path.is_relative_to(project):
            relative = Path('dependencies/project') / path.relative_to(project)
        elif 'ASTRA_ACCURACY_Q20' in path.parts:
            index = path.parts.index('ASTRA_ACCURACY_Q20')
            relative = Path('dependencies') / Path(*path.parts[index:])
        else:
            raise ValueError(f'Unmapped source dependency: {path.name}')
        copy_file(path, relative, 'sealed_dependency/' + relative.relative_to('dependencies').as_posix())
    copy_file(project / 'src/f16/eval.py', Path('dependencies/project/src/f16/eval.py'), 'project/src/f16/eval.py')
    copy_file(project / 'NEXT_PHASE_RESULTS_REVIEW_20261115.md', Path('RESULTS_REVIEW_ZH.md'), 'project/NEXT_PHASE_RESULTS_REVIEW_20261115.md')

    # Reuse the exact complete-link implementation used by the original bootstrap.
    spec = importlib.util.spec_from_file_location('publication_original_eval', project / 'src/f16/eval.py')
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    populations = []
    candidates = [(run.name, split.name, split) for run in sorted((source / 'runs').iterdir())
                  if run.is_dir() for split in sorted(run.iterdir())
                  if split.is_dir() and (split / 'predictions.npz').exists()]
    candidates += [('OLD42', split, old / 'populations' / split) for split in ('fit', 'dev')]
    for run, split, directory in candidates:
        metric_path = directory / 'window_metrics.npz'
        prediction_path = directory / 'predictions.npz'
        with np.load(prediction_path, allow_pickle=False) as z:
            ids = z['ids']
            if 'groups' in z.files:
                groups = z['groups']
            elif 'group' in z.files:
                groups = z['group']
            else:
                data_split = 'fit' if split == 'calib' else split
                with np.load(source / 'C0' / f'{data_split}.npz', allow_pickle=False) as data:
                    if split == 'calib':
                        with np.load(source / 'E1/calib_split.npz', allow_pickle=False) as indices:
                            chosen = indices['calib_index']
                        assert np.array_equal(ids, data['ids'][chosen])
                        groups = data['group'][chosen]
                        future = data['future'][chosen]
                    else:
                        assert np.array_equal(ids, data['ids'])
                        groups = data['group']
            logits = z['logits']
            paths = z['paths'] if split in ('dev', 'calib') else None
        if metric_path.exists():
            with np.load(metric_path, allow_pickle=False) as z:
                window = {name: z[name] for name in z.files}
        else:
            assert split == 'calib'
            window = evaluator.errors(paths, future)
        n = len(ids)
        assert len(set(ids.tolist())) == n and set(np.unique(groups)).issubset({0, 1, 2})
        top = logits.argmax(1)
        rows = np.arange(n)
        assert np.array_equal(window['good'].any(1), window['coverage'])
        output = dict(ids=ids, groups=groups.astype(np.int8), coverage=window['coverage'],
                      minADE=window['ADE'].min(1), minFDE=window['FDE'].min(1),
                      minJ=window['J'].min(1), good=window['good'],
                      Q_winner=window['Q_winner'].astype(np.int8), J_winner=window['J_winner'].astype(np.int8),
                      top1=top.astype(np.int8), top1_good=window['good'][rows, top],
                      top1_ADE=window['ADE'][rows, top], top1_FDE=window['FDE'][rows, top],
                      Q_ADE=window['Q_ADE'], Q_FDE=window['Q_FDE'],
                      Q_segments=window['Q_segments'], J_segments=window['J_segments'])
        if split == 'dev':
            cc = evaluator.clusters(paths, window['good'])
            output.update(distinct=cc[:, 0], effective_good=cc[:, 1])
        aggregate_path = directory / ('metrics.json' if split != 'calib' else 'summary_v2.json')
        metrics = json.loads(aggregate_path.read_text(encoding='utf-8'))
        for index, group in enumerate(('overall', 'stable', 'transition', 'change')):
            mask = np.ones(n, bool) if index == 0 else groups == index - 1
            for key, field in [('coverage', 'coverage'), ('minADE', 'minADE'), ('minFDE', 'minFDE')]:
                assert abs(float(output[key][mask].mean()) - metrics[group][field]) < 2e-12, (run, split, group, field)
            if 'effective_good' in output:
                assert abs(float(output['effective_good'][mask].mean()) - metrics[group]['effective_good']) < 2e-12
        relative = Path('analysis_inputs') / run / (split + '.npz')
        destination = PACKAGE / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            with np.load(destination, allow_pickle=False) as existing:
                assert set(existing.files) == set(output)
                assert all(np.array_equal(existing[key], value) for key, value in output.items())
        else:
            np.savez_compressed(destination, **output)
        populations.append(dict(run=run, split=split, path=relative.as_posix(), windows=n,
                                sha256=digest(destination), bytes=destination.stat().st_size,
                                source_window_metrics_sha256=digest(metric_path) if metric_path.exists() else None,
                                source_predictions_sha256=digest(prediction_path),
                                effective_good_available=split == 'dev'))
        # Include the legacy reference's aggregate evidence used by E1A.
        if run == 'OLD42':
            for path in sorted(directory.iterdir()):
                if path.is_file() and path.suffix in {'.json', '.csv'}:
                    copy_file(path, Path('snapshot/legacy_reference/OLD42') / split / path.name,
                              'legacy_reference/OLD42/' + split + '/' + path.name)
        print(json.dumps(dict(exported=run, split=split, windows=n), ensure_ascii=False), flush=True)

    write_json(PACKAGE / 'analysis_inputs/INDEX.json', dict(
        schema_version=1, populations=populations, groups={'0': 'stable', '1': 'transition', '2': 'change'},
        note='Derived saved-output statistics, not model predictions. Float64 metrics are retained without quantization. '
             'Dev contains exact complete-link cluster counts for all bootstrap endpoints. '
             'FIT/CALIB cluster and motion aggregates remain in snapshot metrics.json; raw paths are omitted.'))
    write_json(PACKAGE / 'COPY_PROVENANCE.json', dict(files=copied, source_files_byte_preserved=True,
        exclusions=['all .pt model/optimizer/resume weights', 'raw labels and processed observations/GT',
                    'full candidate path arrays and first-pass arrays', 'runtime caches and logs',
                    'per-epoch resume JSON duplicates', 'local .memory, credentials and unrelated workspace changes']))
    versions = {}
    for distribution in ('numpy', 'scipy', 'torch', 'psutil', 'matplotlib'):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    write_json(PACKAGE / 'PUBLICATION_ENVIRONMENT.json', dict(
        captured_at='publication, not a substitute for sealed original preflight',
        python=sys.version, packages=versions))
    print(json.dumps(dict(status='EXPORTED', source_files=len(copied), populations=len(populations))))


if __name__ == '__main__':
    main()
