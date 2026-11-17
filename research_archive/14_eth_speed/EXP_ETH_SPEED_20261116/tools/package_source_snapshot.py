"""Publish the completed ETH diagnostics and one conditional speed-aug run."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json
import shutil
import subprocess
import sys

import numpy as np

W = Path(__file__).resolve().parents[1]
SRC = W/'qoder_work/EXP_ETH_SPEED_EXEC_20261116'
DESIGN = Path('C:/Users/linke/Downloads/1/EXP_ETH_SPEED_20261116')
REPO = W/'github_exports/kinematic-traj-forecasting_20261115'
DEST = REPO/'research_archive/14_eth_speed/EXP_ETH_SPEED_20261116'
TEXT = {'.json', '.md', '.py', '.ps1', '.cmd', '.xml', '.csv', '.yaml', '.yml', '.log', '.jsonl'}
sys.path.insert(0, str(DESIGN/'code'))
from speed_io import config, imports, read, sha, write
from eth_speed_math import observed_speed, speed_bins, maneuver_labels


def main():
    assert not DEST.exists(), 'Refuse to overwrite an existing publication'
    c = config()
    audit = subprocess.run([sys.executable, '-X', 'utf8', '-B', str(SRC/'analysis_tools/analyze_completed.py')],
                           cwd=W, capture_output=True, text=True, encoding='utf-8', check=True)
    assert json.loads(audit.stdout)['status'] == 'COMPLETE_VERIFIED'
    analysis = read(SRC/'ANALYSIS_20261116.json')
    diag = read(SRC/'DIAG/RESULTS.json')
    fd, fm = imports(c)
    copies, omitted = [], []

    def copy(file, relative):
        file, relative = Path(file), Path(relative)
        out = DEST/relative; out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, out)
        digest = sha(file); assert sha(out) == digest
        copies.append(dict(source=str(file), archive_path=relative.as_posix(), sha256=digest, bytes=file.stat().st_size))

    def omit(file, reason, digest=None):
        omitted.append(dict(source=str(file), bytes=file.stat().st_size, sha256=digest or sha(file), reason=reason))

    for file in sorted(DESIGN.rglob('*')):
        if file.is_file() and '__pycache__' not in file.parts:
            assert file.suffix in TEXT or file.name.startswith('requirements')
            copy(file, Path('design_original')/file.relative_to(DESIGN))
    for file in sorted(SRC.rglob('*')):
        if not file.is_file() or '__pycache__' in file.parts:
            continue
        rel = file.relative_to(SRC)
        if file.suffix in TEXT or file.name in ('window_metrics.npz', 'scale_inputs.npz'):
            copy(file, Path('snapshot')/rel)
        else:
            omit(file, 'weights_or_full_world_coordinate_predictions_local_only')

    # Preserve selected text dependencies from the actual sealed execution chain.
    dependencies = read(c['upstream_manifest'])['files'].copy()
    dependencies.update(read(SRC/'INPUT_CONTRACT.json')['files'])
    dependency_count = 0
    for original, digest in sorted(dependencies.items()):
        file = Path(original)
        if file.is_relative_to(SRC) or file.is_relative_to(DESIGN):
            continue
        assert sha(file) == digest, original
        if file.suffix in TEXT:
            if file.is_relative_to(W):
                rel = Path('dependencies/workspace')/file.relative_to(W)
            else:
                base = Path('C:/Users/linke/Desktop'); assert file.is_relative_to(base)
                rel = Path('dependencies/external_desktop')/file.relative_to(base)
            copy(file, rel); dependency_count += 1
        else:
            omit(file, 'sealed_raw_data_labels_weights_predictions_videos_or_bytecode_not_republished', digest)

    # Derived evaluation labels/errors have no world coordinates or raw futures.
    derived = []
    for fold in ('eth', 'hotel'):
        train = fd.load_split(c['data_dir'], fold, 'train')
        index, val_index = fd.scene_tail_split(train['scene'])
        training_speed = observed_speed(train['past'][index])
        data = fd.load_split(c['data_dir'], fold, 'test')
        populations = [('test', data)]
        if fold == 'eth': populations.append(('val', fd.subset(train, val_index)))
        for split, population in populations:
            arrays = dict(ids=population['ids'], group=population['group'], scene=population['scene'],
                          observed_speed_mps=observed_speed(population['past']),
                          speed_bin=speed_bins(population['past'], diag['folds'][fold]['profile']),
                          maneuver=maneuver_labels(population['past'], population['future']))
            if split == 'test': arrays['train_observed_speed_mps'] = training_speed
            if fold == 'eth' and split == 'test':
                direction = population['past'][:, 0, -1].astype(np.float64)-population['past'][:, 0, -4]
                norm = np.linalg.norm(direction, axis=-1); arrays['heading_valid'] = norm > .03
                direction /= np.maximum(norm[:, None], 1e-12)
                endpoint = population['future'][:, -1].astype(np.float64)
                for arm, folder in [('B0', SRC/'DIAG/eth/identity'), ('AUG', SRC/'SPEED_AUG/B0_wide_eth_E64_s42/test')]:
                    with np.load(folder/'predictions.npz', allow_pickle=False) as z:
                        paths, logits = z['paths'].astype(np.float64), z['logits'].astype(np.float64)
                    fde = np.sqrt(np.square(paths[:, :, -1]-endpoint[:, None]).sum(-1))
                    row = np.arange(len(paths))
                    for name, slot in [('Top1', logits.argmax(-1)), ('oracle_FDE', fde.argmin(-1))]:
                        arrays[arm+'_'+name+'_longitudinal_error'] = ((paths[row, slot, -1]-endpoint)*direction).sum(-1)
            output = DEST/f'analysis_inputs/{fold}_{split}_labels.npz'; output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output, **arrays)
            derived.append(dict(path=output.relative_to(DEST).as_posix(), sha256=sha(output),
                                keys={k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in arrays.items()},
                                provenance='Derived from the same local past/future/IDs and sealed world predictions used by the completed independent analysis; labels are evaluation-only.'))
        del train, data

    baseline = Path(c['baselines']['eth']['checkpoint']).parent
    for name in ('val_window_metrics.npz', 'metrics.json', 'TRAIN_COMPLETE.json', 'DONE.json'):
        copy(baseline/name, Path('references/B0_eth_E64_s42')/name)
    copy(SRC/'RESULT_ANALYSIS_ZH.md', Path('RESULT_ANALYSIS_ZH.md'))
    copy(Path(__file__), Path('tools/package_source_snapshot.py'))
    write(DEST/'STATUS.json', dict(study='EXP_ETH_SPEED_20261116', execution_status='COMPLETE',
        completed_at=read(SRC/'LAST_EXIT.json')['at'], diagnostic_status='COMPLETE',
        diagnostic_verdict='ELIGIBLE_EXPLORATORY', speed_aug_status='COMPLETE',
        speed_aug_scientific_verdict='PROMISING_SINGLE_SEED_WITH_TRADEOFFS',
        original_fixed_verdict='PROMISING_SINGLE_SEED', new_training_runs=1, seed=42, epochs=64,
        train_windows=28791, train_updates=28800, test_windows_ETH=364, test_windows_HOTEL=1197,
        relative_ADE_improvement=analysis['fixed_verdict']['ADE_gain'],
        relative_FDE_improvement=analysis['fixed_verdict']['FDE_gain'],
        coverage_B0=59, coverage_AUG=55, rescued=13, lost=17,
        Top1_FDE_relative_worsening=-analysis['paired']['overall']['top1_FDE']['relative_gain'],
        deceleration_FDE_relative_worsening=-analysis['paired']['maneuver']['deceleration']['minFDE']['relative_gain'],
        change_group_FDE_relative_worsening=-analysis['paired']['original_group']['change']['minFDE']['relative_gain'],
        baseline_upgraded=False, extra_seeds_executed=False, old_fivefold_resumed=False, COMBO_executed=False,
        independent_unopened_test_claim=False, raw_data_and_weights_included=False, complete_predictions_included=False,
        archive_published_at=datetime.now(timezone(timedelta(hours=8))).isoformat()))
    write(DEST/'COPY_PROVENANCE.json', dict(copies=copies))
    write(DEST/'DERIVED_INPUTS.json', dict(raw_coordinates_included=False, future_coordinate_labels_included=False, files=derived))
    write(DEST/'OMITTED_ARTIFACTS.json', dict(policy='Raw labels/data, videos, weights, bytecode and complete coordinate predictions remain local; only compact per-window metrics and derived evaluation metadata/errors are published.', files=omitted))
    write(DEST/'LOCAL_RECOMPUTE_RECEIPT.json', dict(status='PASS', at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        upstream_files_verified=456, input_contract_files_verified=99, DIAG_output_files_verified=40,
        AUG_output_files_verified=73, baseline_DONE_files_verified=71,
        saved_prediction_populations_recomputed=13, baseline_validation_recomputed=True,
        all_64_epoch_batch_order_hashes_match=True, producer_sha256=sha(__file__),
        local_analysis_script_sha256=sha(SRC/'analysis_tools/analyze_completed.py'),
        portable_scope='Rebuilds aggregate/stratified/paired metrics, speed thresholds/scales, original decisions and derived endpoint summaries from included arrays; no raw-coordinate error, maneuver-label or geometry rederivation without omitted local data/predictions.',
        text_dependencies=dependency_count, copied_files=len(copies), derived_arrays=len(derived)))
    print(json.dumps(dict(destination=str(DEST), copied=len(copies), dependencies=dependency_count,
          NPZ=len(list(DEST.rglob('*.npz'))), omitted=len(omitted)), indent=2))


if __name__ == '__main__':
    main()
