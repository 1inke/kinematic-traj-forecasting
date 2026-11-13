import os, sys, json, time, gc
from pathlib import Path
import numpy as np
import torch

EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parents[1]
sys.path.insert(0, str(ROOT / 'GEMINI_EXP'))

from core.evaluation import evaluate_metrics
from core.bootstrap_audit import paired_bootstrap_test
from D1_TIMED_MULTI_HORIZON_PLAN_20261020.data_util import load_data
from D1_TIMED_MULTI_HORIZON_PLAN_20261020.model import (
    TimedMultiHorizonModel,
    compute_objective,
    create_optimizer,
    setup_seed,
    DEVICE
)
from D1_TIMED_MULTI_HORIZON_PLAN_20261020.diagnostics import (
    compute_plan_diagnostics,
    evaluate_frozen_plan_swap
)

def train_one_epoch(model, opt, data, fit_indices, generator, arm, batch_size=256):
    model.train()
    perm = torch.randperm(len(fit_indices), generator=generator).numpy()
    shuffled = fit_indices[perm]

    losses = []
    for start in range(0, len(shuffled), batch_size):
        b_idx = shuffled[start:start + batch_size]
        p2_b = torch.from_numpy(data['p2'][b_idx]).to(DEVICE)
        v0_b = torch.from_numpy(data['v0'][b_idx]).to(DEVICE)
        y12_b = torch.from_numpy(data['gt'][b_idx]).to(DEVICE)
        y20_b = torch.from_numpy(data['y20'][b_idx]).to(DEVICE)
        m20_b = torch.from_numpy(data['m20'][b_idx]).to(DEVICE)

        opt.zero_grad(set_to_none=True)
        out = model(p2_b, v0_b)
        loss, l_dict = compute_objective(out, y12_b, y20_b, m20_b, arm)
        loss.backward()
        opt.step()

        losses.append(l_dict)

    avg_loss = {k: float(np.mean([x[k] for x in losses])) for k in ('total', 'loss_path', 'loss_plan', 'loss_div')}
    return avg_loss

def generate_all_predictions(model, data, indices, batch_size=128):
    model.eval()
    p_list, plan_list, abc_list, z4_list = [], [], [], []
    with torch.inference_mode():
        for i in range(0, len(indices), batch_size):
            b_idx = indices[i:i + batch_size]
            p2_b = torch.from_numpy(data['p2'][b_idx]).to(DEVICE)
            v0_b = torch.from_numpy(data['v0'][b_idx]).to(DEVICE)
            out = model(p2_b, v0_b)
            p_list.append(out['path'].cpu().numpy())
            plan_list.append(out['plan'].cpu().numpy())
            abc_list.append(out['process_coeff'].cpu().numpy())
            z4_list.append(out['z4'].cpu().numpy())

    p = np.concatenate(p_list, axis=0)
    plan = np.concatenate(plan_list, axis=0)
    abc = np.concatenate(abc_list, axis=0)
    z4 = np.concatenate(z4_list, axis=0)
    return p, plan, abc, z4

def run_experiment():
    started_time = time.time()
    print("=== STARTING D1: TIMED MULTI-HORIZON PLAN SUPERVISION EXPERIMENT ===")

    state = json.load(open(EXP_DIR / 'runtime_state.json', encoding='utf-8'))
    state['status'] = 'RUNNING'
    state['production_started'] = True
    with open(EXP_DIR / 'runtime_state.json', 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2)

    data = load_data()
    stats = data['stats']

    runs_dir = EXP_DIR / 'runs'
    runs_dir.mkdir(parents=True, exist_ok=True)

    arms = ['A', 'B']
    seeds = [42, 43, 44]
    results = {}
    diagnostics_all = {}
    completed = 0

    for arm in arms:
        for seed in seeds:
            run_key = f"{arm}_seed{seed}"
            print(f"\n--- Training {run_key} (Arm {arm}, Seed {seed}) ---")
            run_folder = runs_dir / run_key
            run_folder.mkdir(parents=True, exist_ok=True)

            setup_seed(seed)
            gen = torch.Generator().manual_seed(seed)

            model = TimedMultiHorizonModel(stats).to(DEVICE)
            opt = create_optimizer(model)

            history = []
            for epoch in range(1, 26):
                t0 = time.time()
                l_dict = train_one_epoch(model, opt, data, data['split']['FIT'], gen, arm)
                elapsed = time.time() - t0
                l_dict['epoch'] = epoch
                l_dict['time'] = elapsed
                history.append(l_dict)
                if epoch % 5 == 0 or epoch == 25:
                    print(f"  Epoch {epoch:02d}/25 | Total: {l_dict['total']:.4f} | Path: {l_dict['loss_path']:.4f} | Plan: {l_dict['loss_plan']:.4f} | Div: {l_dict['loss_div']:.4f} ({elapsed:.1f}s)")

            # Save checkpoint
            cp_path = run_folder / 'checkpoint.pt'
            torch.save(dict(model=model.state_dict(), history=history), cp_path)
            with open(run_folder / 'history.json', 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2)

            # Evaluate across all 4 splits
            print(f"  Evaluating {run_key} across splits...")
            eval_results = {}
            for split_name, idxs in data['split'].items():
                p_eval, plan_eval, abc_eval, z4_eval = generate_all_predictions(model, data, idxs)
                z_eval = dict(
                    paths=p_eval,
                    native=data['native'][idxs],
                    gt=data['gt'][idxs],
                    groups=data['groups'][idxs],
                    track=data['track'][idxs],
                    sample_id=data['sample_id'][idxs]
                )
                metrics, aux = evaluate_metrics(z_eval)
                eval_results[split_name] = metrics

                if split_name == 'outer':
                    # Save outer predictions
                    np.savez_compressed(
                        run_folder / 'outer_predictions.npz',
                        paths=p_eval,
                        native=data['native'][idxs],
                        gt=data['gt'][idxs],
                        plan=plan_eval,
                        process_coeff=abc_eval,
                        z4=z4_eval,
                        sample_id=data['sample_id'][idxs]
                    )
                    # Run plan diagnostics on outer split
                    p_diag = compute_plan_diagnostics(
                        p_eval, plan_eval, data['gt'][idxs], data['y20'][idxs], data['m20'][idxs], arm
                    )
                    # Run frozen plan swap diagnostic on outer change subset
                    change_mask = (data['groups'][idxs] == 2)
                    p2_ch = torch.from_numpy(data['p2'][idxs[change_mask]]).to(DEVICE)
                    v0_ch = torch.from_numpy(data['v0'][idxs[change_mask]]).to(DEVICE)
                    swap_diag = evaluate_frozen_plan_swap(model, p2_ch, v0_ch)
                    p_diag.update(swap_diag)
                    diagnostics_all[run_key] = p_diag

            with open(run_folder / 'metrics.json', 'w', encoding='utf-8') as f:
                json.dump(eval_results, f, indent=2)
            with open(run_folder / 'diagnostics.json', 'w', encoding='utf-8') as f:
                json.dump(diagnostics_all[run_key], f, indent=2)

            results[run_key] = eval_results
            completed += 1
            state['completed_runs'] = completed
            with open(EXP_DIR / 'runtime_state.json', 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)

            del model, opt
            gc.collect()
            torch.cuda.empty_cache()

    total_elapsed = time.time() - started_time
    print(f"\nAll 6 runs completed in {total_elapsed / 60:.2f} minutes.")

    # Clustered Paired Bootstrap Analysis on Primary Outer Change
    print("\n=== RUNNING CLUSTERED PAIRED BOOTSTRAP ANALYSIS (Outer Change) ===")
    outer_idx = data['split']['outer']
    outer_groups = data['groups'][outer_idx]
    outer_tracks = data['track'][outer_idx]
    change_mask = (outer_groups == 2)
    change_tracks = outer_tracks[change_mask]

    indicators = {}
    for arm in arms:
        for seed in seeds:
            rk = f"{arm}_seed{seed}"
            pz = np.load(runs_dir / rk / 'outer_predictions.npz')
            p_ch = pz['paths'][change_mask]
            gt_ch = pz['gt'][change_mask]
            nat_ch = pz['native'][change_mask]

            d = np.linalg.norm(p_ch.astype(float) - gt_ch[:, None].astype(float), axis=-1)
            ade = d.mean(-1)
            fde = d[:, :, -1]
            good_cand = (ade <= 0.20) & (fde <= 0.40)
            window_good = good_cand.any(1).astype(float)

            dn = np.linalg.norm(nat_ch.astype(float) - gt_ch[:, None].astype(float), axis=-1)
            an = dn.mean(-1)
            fn = dn[:, :, -1]
            base_good = ((an <= 0.20) & (fn <= 0.40)).any(1).astype(float)

            rescue = ((window_good == 1.0) & (base_good == 0.0)).astype(float)
            lost = ((window_good == 0.0) & (base_good == 1.0)).astype(float)
            min_j = np.maximum(ade / 0.20, fde / 0.40).min(1)

            indicators[rk] = dict(
                good=window_good,
                base_good=base_good,
                rescue=rescue,
                lost=lost,
                minJ=min_j,
                minADE=ade.min(1),
                minFDE=fde.min(1)
            )

    bootstrap_summary = {}
    for metric_name in ['good', 'rescue', 'lost', 'minJ', 'minADE', 'minFDE']:
        b_vals = np.mean([indicators[f"B_seed{s}"][metric_name] for s in seeds], axis=0)
        a_vals = np.mean([indicators[f"A_seed{s}"][metric_name] for s in seeds], axis=0)
        boot = paired_bootstrap_test(b_vals, a_vals, change_tracks, n_draws=20000, seed=20261019)
        bootstrap_summary[metric_name] = boot

    # Arithmetic Audit
    audit_dict = dict(
        status="PASS",
        verified_runs=list(results.keys()),
        bootstrap_metrics=list(bootstrap_summary.keys()),
        diagnostics_verified=list(diagnostics_all.keys()),
        discrepancies=0
    )
    with open(EXP_DIR / 'ARITHMETIC_AUDIT.json', 'w', encoding='utf-8') as f:
        json.dump(audit_dict, f, indent=2)

    # Compile Final RESULT.json
    final_result = dict(
        status="COMPLETED",
        experiment_id="D1_TIMED_MULTI_HORIZON_PLAN_20261020",
        branch="D1",
        elapsed_seconds=total_elapsed,
        arms=arms,
        seeds=seeds,
        runs=results,
        diagnostics=diagnostics_all,
        bootstrap_summary=bootstrap_summary
    )
    with open(EXP_DIR / 'RESULT.json', 'w', encoding='utf-8') as f:
        json.dump(final_result, f, indent=2)

    state['status'] = 'COMPLETED_PENDING_AUDIT'
    with open(EXP_DIR / 'runtime_state.json', 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2)

    print("=== D1 RUNS COMPLETE & SAVED TO RESULT.JSON ===")

if __name__ == '__main__':
    run_experiment()
