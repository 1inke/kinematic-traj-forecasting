"""Explicit stage runner. Help/defaults never start an experiment.

Production preflight/final-release receipts must be supplied; this script does
not fabricate them from passing synthetic tests or launch background workers.
"""
import argparse
import importlib.util
from pathlib import Path
import traceback
from io_contract import (load_npz, validate_observed, aligned_gt, sha, json_read,
    json_new, npz_new, verify_manifest, checked_seal, seal, code_digest, finish_stage)


def resource_guard(path, estimate):
    if estimate <= 0:
        raise ValueError('inventory must come from measured preflight, not available RAM')
    spec = importlib.util.spec_from_file_location('registered_resource_guard', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    g = module.Guard(estimate)
    # This is an in-process runner, not a lightweight pre-Popen coordinator.
    g.check('runtime', force=True)
    return g


def require_release(path):
    m = verify_manifest(path)
    if m.get('status') != 'ALL_REGISTERED_FINALS_SEALED' or m.get('code_digest') != code_digest():
        raise ValueError('hold/inner remain closed: valid final release is required')
    if m.get('all_executable_branches_accounted_for') is not True:
        raise ValueError('release must account for END and TIME, including evidence-backed STOP')
    return m


def preflight_receipt(path, binding, inventory_bytes, device):
    m = verify_manifest(path)
    if m.get('status') != 'PASS_REAL_DATA_GPU_RESUME_AND_TINY':
        raise ValueError('formal training is blocked pending actual full preflight')
    if m.get('binding') != binding:
        raise ValueError('preflight source/data/statistics/label binding mismatch')
    if m.get('inventory_bytes') != inventory_bytes or m.get('device') != device:
        raise ValueError('preflight measured inventory/device mismatch')
    required = ('source_preservation', 'independent_128_actions', 'actual_gpu_gradients',
                'both_513_subset_2epoch_exact_resume', 'tiny_active_mse_drop_90pct',
                'bad_source_output_rejected', 'measured_inventory_and_benchmark')
    if any(m.get('checks', {}).get(k) is not True for k in required):
        raise ValueError('incomplete real preflight; synthetic tests do not suffice')


def population_contract(observed, population):
    """Reject accidental relabeling of held inputs as train before GT is opened.

    Counts are a tripwire, not an identity proof; exact input SHA must ALSO be
    registered by the real input audit/source manifest before production.
    """
    expected = {'train': (3103, 721), 'val': (993, 256),
                'hold': (3825, 1023), 'inner': (1856, 497)}
    z = load_npz(observed, ('sample_id', 'group'))
    n, change = expected[population]
    if len(z['sample_id']) != n or int((z['group'] == 2).sum()) != change:
        raise ValueError('registered population/count mismatch; cannot bypass held release by relabeling')


def execute(args, g):
    import numpy as np
    out = Path(args.out).resolve()
    if args.command == 'prepare':
        from input_audit import prepare
        return prepare(args.observed, args.gt, args.split, out)
    population_contract(args.observed, args.population)
    if args.population in ('hold', 'inner'):
        require_release(args.release)
    if args.command == 'masks':
        from build_teacher import write_masks
        return write_masks(args.observed, out, args.branch, g.check)
    if args.command == 'labels':
        from build_teacher import write_labels
        return write_labels(args.observed, args.gt, args.masks, out, g.check)
    x = load_npz(args.observed)
    validate_observed(x)
    if args.command == 'baseline':
        from inference import predict_paths
        from build_teacher import read_targets
        if args.baseline == 'identity':
            values = np.zeros((len(x['sample_id']), 20, 2), np.float64)
            training_label_sha = None
        else:
            if not args.train_observed or not args.labels:
                raise ValueError('mean baseline requires registered train observations and train labels')
            train_x = load_npz(args.train_observed)
            validate_observed(train_x)
            teacher = checked_seal(args.labels)
            if (len(train_x['sample_id']) != 3103 or teacher['branch'] != args.branch or
                    teacher['observed_sha'] != sha(args.train_observed)):
                raise ValueError('mean baseline train label contract mismatch')
            mean = read_targets(args.labels, train_x['sample_id']).astype(np.float64).mean(axis=(0, 1))
            values = np.broadcast_to(mean, (len(x['sample_id']), 20, 2)).copy()
            training_label_sha = sha(Path(args.labels)/'done.json')
        result = predict_paths(x['past'], x['native'], values, args.branch, g.check)
        out.mkdir(parents=True, exist_ok=False)
        npz_new(out/'predictions.npz', sample_id=x['sample_id'], **result)
        seal(out, [out/'predictions.npz'], stage='PREDICTIONS_SAVED', branch=args.branch,
             baseline=args.baseline, training_label_sha=training_label_sha,
             observed_sha=sha(args.observed), no_GT_forward=True)
        return
    if args.command == 'fit':
        if args.population != 'train':
            raise ValueError('training must use registered train only')
        from features import transform
        from build_teacher import read_targets
        from train import Engine, checkpoint_new, checkpoint_load_trusted
        receipt = checked_seal(args.labels)
        if receipt['branch'] != args.branch or receipt['observed_sha'] != sha(args.observed):
            raise ValueError('teacher branch/input binding mismatch')
        opportunity = json_read(Path(args.labels)/'opportunity.json')
        if opportunity['change_N'] != 721 or not opportunity['opportunity_pass']:
            raise ValueError('P0 train opportunity gate not passed')
        stats = load_npz(args.statistics)
        if len(x['sample_id']) != 3103 or not np.array_equal(stats['train_sample_id'], x['sample_id']):
            raise ValueError('not registered train/statistics')
        prepared = Path(args.statistics).resolve().parent
        prep = checked_seal(prepared)
        if (prep.get('stage') != 'TRAIN_VAL_PREPARED' or
                prep['files'].get('X_train.npz') != sha(args.observed) or
                prep['files'].get('statistics.npz') != sha(args.statistics)):
            raise ValueError('statistics/observations must belong to verified original split preparation')
        audit = json_read(prepared/'INPUT_AUDIT.json')
        binding = dict(code_digest=code_digest(), observed_sha=sha(args.observed),
            stats_sha=sha(args.statistics), teacher_done_sha=sha(Path(args.labels)/'done.json'),
            preparation_done_sha=sha(prepared/'done.json'), split_sha=audit['split_sha'],
            source_lock_sha=sha(args.source_lock), branch=args.branch)
        preflight_receipt(args.preflight, binding, args.inventory_bytes, args.device)
        contract = dict(**binding, arm=args.arm, seed=args.seed,
                        preflight_sha=sha(args.preflight), resource_inventory=args.inventory_bytes,
                        training_protocol=dict(network='40-64-32-2 GELU/tanh', parameters=4770,
                            objective='mean_window(mean_slot(mean_coordinate_squared_parameter_error))',
                            optimizer='AdamW', lr=.001, weight_decay=.0001, clip_norm=5.,
                            epochs=8, batch_windows=32, early_stop=False, train_windows=3103,
                            action_grid='END49' if args.branch == 'END' else 'TIME441',
                            motion_and_junction_limit=1.01, dt=.4,
                            identity_tie_first=True, G_mask_after_normalization=True))
        # Digest of standardized tensors is also checked by Engine on resume.
        engine = Engine(transform(x['past'], x['native'], stats, args.arm),
                        read_targets(args.labels, x['sample_id']), contract,
                        seed=args.seed, device=args.device)
        if args.resume:
            if not args.resume_sha or sha(args.resume) != args.resume_sha:
                raise ValueError('resume checkpoint must have an explicit trusted SHA')
            engine.restore(checkpoint_load_trusted(args.resume))
        out.mkdir(parents=True, exist_ok=False)
        files = []
        last_epoch = engine.epoch
        while not engine.complete:
            g.check()
            engine.step()
            if engine.epoch != last_epoch:
                name = out/f'epoch_{engine.epoch:02d}.pt'
                checkpoint_new(name, engine.snapshot())
                files.append(name)
                last_epoch = engine.epoch
        checkpoint_new(out/'checkpoint.pt', engine.snapshot())
        json_new(out/'history.json', engine.history)
        json_new(out/'run_manifest.json', dict(contract=contract, epochs=8, updates=engine.updates,
                 parameter_count=4770, optimizer='AdamW', lr=.001, weight_decay=.0001,
                 grad_clip=5., batch_windows=32, status='TRAINED_REQUIRES_SAVED_AUDIT'))
        files += [out/'checkpoint.pt', out/'history.json', out/'run_manifest.json']
        seal(out, files, stage='FINAL8_MODEL_SEALED', contract=contract)
        return
    if args.command == 'predict':
        from train import checkpoint_load_trusted, infer
        from model import ParameterRegressor
        from features import transform
        from inference import predict_paths
        model_receipt = checked_seal(args.model_dir)
        ck = Path(args.model_dir)/'checkpoint.pt'
        state = checkpoint_load_trusted(ck)
        if state['epoch'] != 8 or state['contract'] != model_receipt['contract']:
            raise ValueError('not a final8 bound model')
        contract = state['contract']
        if contract['stats_sha'] != sha(args.statistics) or contract['branch'] != args.branch:
            raise ValueError('model/statistics/branch mismatch')
        model = ParameterRegressor().to(args.device)
        model.load_state_dict(state['model'], strict=True)
        values = infer(model, transform(x['past'], x['native'], load_npz(args.statistics),
                       contract['arm']), args.device, g.check)
        result = predict_paths(x['past'], x['native'], values, args.branch, g.check)
        out.mkdir(parents=True, exist_ok=False)
        npz_new(out/'predictions.npz', sample_id=x['sample_id'], **result)
        seal(out, [out/'predictions.npz'], stage='PREDICTIONS_SAVED', branch=args.branch,
             observed_sha=sha(args.observed), checkpoint_sha=sha(ck),
             model_contract=contract, no_GT_forward=True)
        return
    if args.command in ('evaluate', 'audit'):
        receipt = checked_seal(args.prediction_dir)
        if receipt['observed_sha'] != sha(args.observed) or receipt['branch'] != args.branch:
            raise ValueError('prediction binding mismatch')
        saved = load_npz(Path(args.prediction_dir)/'predictions.npz')
        if not np.array_equal(saved['sample_id'], x['sample_id']):
            raise ValueError('prediction IDs mismatch')
        out.mkdir(parents=True, exist_ok=False)
        if args.command == 'audit':
            from audit_saved import check_saved
            report = check_saved(x, saved, args.branch, check=g.check)
            json_new(out/'SCALAR_AUDIT.json', report)
            seal(out, [out/'SCALAR_AUDIT.json'], stage='SCALAR_AUDIT_ONLY',
                 observed_sha=sha(args.observed),
                 prediction_done_sha=sha(Path(args.prediction_dir)/'done.json'))
            return
        from evaluate import evaluate_population
        from build_teacher import read_targets
        gt = aligned_gt(x, load_npz(args.gt))
        teacher_receipt = checked_seal(args.labels)
        if (teacher_receipt['observed_sha'] != sha(args.observed) or
                teacher_receipt['gt_sha'] != sha(args.gt) or teacher_receipt['branch'] != args.branch):
            raise ValueError('evaluation teacher population binding mismatch')
        targets = read_targets(args.labels, x['sample_id']).astype(np.float64)
        label_chunks = [load_npz(f, ('best_J', 'identity_J'))
                        for f in sorted(Path(args.labels).glob('[0-9]*.npz'))]
        best_j = np.concatenate([z['best_J'] for z in label_chunks])
        identity_j = np.concatenate([z['identity_J'] for z in label_chunks])
        for port in ('raw', 'guarded'):
            samples, rows = evaluate_population(x, gt, saved[port+'_paths'], g.check)
            samples['parameter_mse'] = ((saved['continuous_param'].astype(float)-targets)**2).mean(axis=(1, 2))
            samples['regret'] = (samples['J']-best_j).mean(axis=1)
            samples['identity_regret'] = (identity_j-best_j).mean(axis=1)
            if port == 'guarded' and np.min(samples['J']-best_j) < -2e-12:
                raise AssertionError('guarded action better than exhaustive safe oracle: audit implementation')
            npz_new(out/f'{port}_samples.npz', sample_id=x['sample_id'], **samples)
            json_new(out/f'{port}_groups.json', rows)
        seal(out, list(out.glob('*')), stage='METRICS_REQUIRE_COMPLETION_REVIEW',
             observed_sha=sha(args.observed), gt_sha=sha(args.gt),
             prediction_done_sha=sha(Path(args.prediction_dir)/'done.json'),
             branch=args.branch, model_contract=receipt.get('model_contract'),
             baseline=receipt.get('baseline'))
        return
    raise ValueError('unimplemented command')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('prepare', 'masks', 'labels', 'fit', 'predict', 'baseline', 'evaluate', 'audit'))
    p.add_argument('--run', action='store_true', help='explicitly execute; otherwise show plan only')
    p.add_argument('--observed', required=True)
    p.add_argument('--out', required=True, help='new output directory; never overwrite')
    p.add_argument('--source-lock', required=True, help='manifest with absolute source files + original SHA')
    p.add_argument('--guard', required=True, help='registered resource_guard_v3.py')
    p.add_argument('--inventory-bytes', type=int, required=True, help='measured inventory, not hard ceiling')
    p.add_argument('--population', choices=('train', 'val', 'hold', 'inner'), default='train')
    p.add_argument('--branch', choices=('END', 'TIME'), default='END')
    for key in ('gt', 'split', 'masks', 'labels', 'statistics', 'preflight', 'release',
                'model-dir', 'prediction-dir', 'resume', 'resume-sha', 'train-observed'):
        p.add_argument('--'+key)
    p.add_argument('--baseline', choices=('identity', 'train_mean'), default='identity')
    p.add_argument('--arm', choices=('G', 'H'), default='H')
    p.add_argument('--seed', type=int, choices=(42, 43, 44), default=42)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    if not args.run:
        print('PLAN_ONLY: no data read, no output written, no training started.')
        return
    required = {'prepare': ('gt', 'split'), 'labels': ('gt', 'masks'),
                'fit': ('labels', 'statistics', 'preflight'),
                'predict': ('model_dir', 'statistics'), 'evaluate': ('gt', 'prediction_dir', 'labels'),
                'audit': ('prediction_dir',)}.get(args.command, ())
    if any(getattr(args, name) is None for name in required):
        p.error('missing command-specific paths: '+', '.join(required))
    if args.population in ('hold', 'inner') and not args.release:
        p.error('held populations require --release')
    if Path(args.out).exists():
        p.error('output already exists; preserve it and choose an explicit new version')
    lock = verify_manifest(args.source_lock)
    locked = {str((Path(name) if Path(name).is_absolute() else Path(args.source_lock).resolve().parent/name).resolve())
              for name in lock['files']}
    needed = list(Path(__file__).resolve().parent.glob('*.py'))+[Path(args.guard).resolve()]
    if any(str(q.resolve()) not in locked for q in needed):
        p.error('source lock must include all code modules and exact resource adapter')
    g = resource_guard(args.guard, args.inventory_bytes)
    try:
        execute(args, g)
        verify_manifest(args.source_lock)
        json_new(Path(args.out)/'RESOURCE_AND_SOURCE_FINAL.json', dict(
            final=g.check(force=True), observations=g.observations,
            source_lock_sha=sha(args.source_lock), source_unchanged=True))
        finish_stage(args.out, 'RESOURCE_AND_SOURCE_FINAL.json')
    except BaseException as exc:
        failure = Path(args.out)/'FAILURE.json'
        if not failure.exists():
            json_new(failure, dict(error=str(exc), traceback=traceback.format_exc(),
                                  no_automatic_retry=True))
        raise


if __name__ == '__main__':
    main()
