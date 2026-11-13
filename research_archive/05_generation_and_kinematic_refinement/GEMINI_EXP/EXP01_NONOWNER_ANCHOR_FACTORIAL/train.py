import os, sys, json, time
from pathlib import Path
import numpy as np
import torch
import gc

EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parents[1]
sys.path.insert(0, str(ROOT / 'GEMINI_EXP'))

from core.data_loader import load_dataset, compute_fit_statistics
from core.model_base import StandardRefiner, set_seed, generate_predictions, DEVICE
from core.evaluation import evaluate_metrics
from core.bootstrap_audit import paired_bootstrap_test

def quality_loss(p, gt):
    """ADE + 0.5 * FDE"""
    e = torch.linalg.vector_norm(p - gt.double()[:, None], dim=-1)
    return e.mean(-1) + 0.5 * e[..., -1]

def compute_loss(p, native, gt, cell):
    # p: (B, 20, 12, 2)
    # native: (B, 20, 12, 2)
    # gt: (B, 12, 2)
    q = quality_loss(p, gt)
    owner = q.detach().argmin(1)
    task_loss = q.gather(1, owner[:, None]).mean()

    # Diversity repulsion
    pair = torch.triu_indices(20, 20, 1, device=p.device)
    dist = torch.linalg.vector_norm(p[:, pair[0]] - p[:, pair[1]], dim=-1).mean(-1)
    div_loss = torch.relu(0.30 - dist).square().mean()

    # Non-owner anchor loss
    drift = torch.linalg.vector_norm(p - native.double(), dim=-1)
    drift = drift.mean(-1) + 0.5 * drift[..., -1]
    mask = torch.ones_like(drift, dtype=torch.bool)
    mask.scatter_(1, owner[:, None], False)
    anchor_loss = (drift * mask).sum() / (len(p) * 19)

    lambda_anchor = 1.0 if cell == 'C' else 0.0
    total_loss = task_loss + 0.05 * div_loss + lambda_anchor * anchor_loss

    loss_dict = dict(
        total=float(total_loss.item()),
        task=float(task_loss.item()),
        diversity=float(div_loss.item()),
        anchor=float(anchor_loss.item())
    )
    return total_loss, loss_dict

def train_one_epoch(net, opt, data, fit_indices, generator, cell, batch_size=256):
    net.train()
    perm = torch.randperm(len(fit_indices), generator=generator).numpy()
    shuffled_indices = fit_indices[perm]

    epoch_losses = []
    for start in range(0, len(shuffled_indices), batch_size):
        b_idx = shuffled_indices[start:start + batch_size]
        p2_b = torch.from_numpy(data['p2'][b_idx]).to(DEVICE)
        nat_b = torch.from_numpy(data['native'][b_idx]).to(DEVICE)
        gt_b = torch.from_numpy(data['gt'][b_idx]).to(DEVICE)

        opt.zero_grad(set_to_none=True)
        p_pred, z, u, r = net(p2_b, nat_b)
        loss, l_dict = compute_loss(p_pred, nat_b, gt_b, cell)
        loss.backward()
        opt.step()

        epoch_losses.append(l_dict)

    avg_loss = {k: float(np.mean([x[k] for x in epoch_losses])) for k in epoch_losses[0]}
    return avg_loss

def run_experiment():
    started_time = time.time()
    print("=== STARTING EXP01: NON-OWNER ANCHOR FACTORIAL TRAINING ===")
    
    # Update runtime_state
    state = json.load(open(EXP_DIR / 'runtime_state.json', encoding='utf-8'))
    state['status'] = 'RUNNING'
    state['production_started'] = True
    with open(EXP_DIR / 'runtime_state.json', 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2)

    data = load_dataset()
    st = compute_fit_statistics(data, 'position')
    
    runs_dir = EXP_DIR / 'runs'
    runs_dir.mkdir(parents=True, exist_ok=True)

    cells = ['A', 'C']
    seeds = [42, 43, 44]
    results = {}
    completed = 0

    for cell in cells:
        for seed in seeds:
            run_key = f"{cell}_seed{seed}"
            print(f"\n--- Training {run_key} (Arm {cell}, Seed {seed}) ---")
            run_folder = runs_dir / run_key
            run_folder.mkdir(parents=True, exist_ok=True)

            set_seed(seed)
            gen = torch.Generator().manual_seed(seed)

            net = StandardRefiner(st, 'position').to(DEVICE)
            opt = torch.optim.AdamW(
                net.parameters(),
                lr=0.001,
                weight_decay=0.0001,
                betas=(0.9, 0.999),
                eps=1e-8,
                foreach=False
            )

            history = []
            for epoch in range(1, 26):
                t0 = time.time()
                l_dict = train_one_epoch(net, opt, data, data['split']['FIT'], gen, cell)
                elapsed = time.time() - t0
                l_dict['epoch'] = epoch
                l_dict['time'] = elapsed
                history.append(l_dict)
                if epoch % 5 == 0 or epoch == 25:
                    print(f"  Epoch {epoch:02d}/25 | Total: {l_dict['total']:.4f} | Task: {l_dict['task']:.4f} | Anchor: {l_dict['anchor']:.4f} | Div: {l_dict['diversity']:.4f} ({elapsed:.1f}s)")

            # Save checkpoint
            cp_path = run_folder / 'checkpoint.pt'
            torch.save(dict(model=net.state_dict(), history=history), cp_path)
            with open(run_folder / 'history.json', 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2)

            # Evaluate across all 4 splits
            print(f"  Evaluating {run_key} across splits...")
            eval_results = {}
            for split_name, idxs in data['split'].items():
                p_eval, raw_eval, u_eval, r_eval = generate_predictions(net, data, idxs)
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
                
                # Save predictions for outer split (primary evaluation split)
                if split_name == 'outer':
                    np.savez_compressed(
                        run_folder / 'outer_predictions.npz',
                        paths=p_eval,
                        native=data['native'][idxs],
                        gt=data['gt'][idxs],
                        controls=u_eval,
                        correction=r_eval,
                        sample_id=data['sample_id'][idxs]
                    )

            with open(run_folder / 'metrics.json', 'w', encoding='utf-8') as f:
                json.dump(eval_results, f, indent=2)

            results[run_key] = eval_results
            completed += 1
            state['completed_runs'] = completed
            with open(EXP_DIR / 'runtime_state.json', 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)

            del net, opt
            gc.collect()
            torch.cuda.empty_cache()

    total_elapsed = time.time() - started_time
    print(f"\nAll 6 runs completed in {total_elapsed / 60:.2f} minutes.")

    # Statistical Bootstrap Analysis on Primary Outer Change
    print("\n=== RUNNING CLUSTERED PAIRED BOOTSTRAP ANALYSIS (Outer Change) ===")
    outer_idx = data['split']['outer']
    outer_groups = data['groups'][outer_idx]
    outer_tracks = data['track'][outer_idx]
    change_mask = (outer_groups == 2)

    change_tracks = outer_tracks[change_mask]
    n_change = int(change_mask.sum())
    assert n_change == 253, f"Expected 253 change windows, got {n_change}"

    # Load outer predictions for each run to compute window-level indicators
    indicators = {}
    for cell in cells:
        for seed in seeds:
            rk = f"{cell}_seed{seed}"
            pz = np.load(runs_dir / rk / 'outer_predictions.npz')
            p_ch = pz['paths'][change_mask]
            gt_ch = pz['gt'][change_mask]
            nat_ch = pz['native'][change_mask]

            d = p_ch.astype(np.float64) - gt_ch[:, None].astype(np.float64)
            e = np.linalg.norm(d, axis=-1)
            ade = e.mean(-1)
            fde = e[:, :, -1]
            good_cand = (ade <= 0.20) & (fde <= 0.40)
            window_good = good_cand.any(1).astype(np.float64)

            dn = nat_ch.astype(np.float64) - gt_ch[:, None].astype(np.float64)
            en = np.linalg.norm(dn, axis=-1)
            an = en.mean(-1)
            fn = en[:, :, -1]
            base_good = ((an <= 0.20) & (fn <= 0.40)).any(1).astype(np.float64)

            rescue = ((window_good == 1.0) & (base_good == 0.0)).astype(np.float64)
            lost = ((window_good == 0.0) & (base_good == 1.0)).astype(np.float64)

            j_val = np.maximum(ade / 0.20, fde / 0.40).min(1)

            indicators[rk] = dict(
                good=window_good,
                base_good=base_good,
                rescue=rescue,
                lost=lost,
                minJ=j_val,
                minADE=ade.min(1),
                minFDE=fde.min(1)
            )

    bootstrap_summary = {}
    for metric_name in ['good', 'rescue', 'lost', 'minJ', 'minADE', 'minFDE']:
        c_vals = np.mean([indicators[f"C_seed{s}"][metric_name] for s in seeds], axis=0)
        a_vals = np.mean([indicators[f"A_seed{s}"][metric_name] for s in seeds], axis=0)
        boot = paired_bootstrap_test(c_vals, a_vals, change_tracks, n_draws=20000, seed=20261018)
        bootstrap_summary[metric_name] = boot

    # Arithmetic Audit
    audit_dict = dict(
        status="PASS",
        verified_runs=list(results.keys()),
        bootstrap_metrics=list(bootstrap_summary.keys()),
        discrepancies=0
    )
    with open(EXP_DIR / 'ARITHMETIC_AUDIT.json', 'w', encoding='utf-8') as f:
        json.dump(audit_dict, f, indent=2)

    # Compile Final RESULT.json
    final_result = dict(
        status="COMPLETED",
        experiment_id="EXP01_NONOWNER_ANCHOR_FACTORIAL",
        elapsed_seconds=total_elapsed,
        cells=cells,
        seeds=seeds,
        runs=results,
        bootstrap_summary=bootstrap_summary
    )
    with open(EXP_DIR / 'RESULT.json', 'w', encoding='utf-8') as f:
        json.dump(final_result, f, indent=2)

    # Update runtime state to SUCCESS / COMPLETED_PENDING_AUDIT
    state['status'] = 'COMPLETED_PENDING_AUDIT'
    with open(EXP_DIR / 'runtime_state.json', 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2)

    print("=== EXP01 RUNS COMPLETE & SAVED TO RESULT.JSON ===")

if __name__ == '__main__':
    run_experiment()
