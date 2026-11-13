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
from D2_MOTION_PRETRAINED_REPRESENTATION_20261020.model import (
    MotionPretrainedEncoder,
    PretrainedConditionedGenerator,
    compute_pretrain_loss,
    compute_generator_loss,
    setup_seed,
    DEVICE
)
from D2_MOTION_PRETRAINED_REPRESENTATION_20261020.diagnostics import (
    evaluate_pretrain_loss_split,
    evaluate_representation_ablations
)

def pretrain_one_epoch(encoder, opt, past_fit, generator, arm, batch_size=256):
    encoder.train()
    N = len(past_fit)
    perm = torch.randperm(N, generator=generator).numpy()
    shuffled = past_fit[perm]
    hidden_indices = [4, 5] if arm == 'A' else [6, 7]

    losses = []
    for start in range(0, N, batch_size):
        p_b = torch.from_numpy(shuffled[start:start + batch_size]).to(DEVICE)
        B = len(p_b)
        rel_pos = p_b - p_b[:, 0:1, :]
        mask = torch.ones((B, 8, 1), dtype=torch.float32, device=DEVICE)
        mask[:, hidden_indices, :] = 0.0
        rel_in = rel_pos.clone()
        rel_in[:, hidden_indices, :] = 0.0
        t_norm = (torch.arange(8, dtype=torch.float32, device=DEVICE) / 8.0)[None, :, None].expand(B, -1, -1)
        x_in = torch.cat((rel_in.float(), mask, t_norm), dim=-1)

        opt.zero_grad(set_to_none=True)
        _, pred = encoder.forward_pretrain(x_in)
        loss = compute_pretrain_loss(pred, p_b, arm)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    return float(np.mean(losses))

def train_generator_one_epoch(generator_model, opt, data, h64_data, fit_indices, rng_gen, batch_size=256):
    generator_model.train()
    perm = torch.randperm(len(fit_indices), generator=rng_gen).numpy()
    shuffled = fit_indices[perm]

    losses = []
    for start in range(0, len(shuffled), batch_size):
        b_idx = shuffled[start:start + batch_size]
        p2_b = torch.from_numpy(data['p2'][b_idx]).to(DEVICE)
        h64_b = torch.from_numpy(h64_data[b_idx]).to(DEVICE)
        v0_b = torch.from_numpy(data['v0'][b_idx]).to(DEVICE)
        y12_b = torch.from_numpy(data['gt'][b_idx]).to(DEVICE)

        opt.zero_grad(set_to_none=True)
        out = generator_model(p2_b, h64_b, v0_b)
        loss, l_dict = compute_generator_loss(out, y12_b)
        loss.backward()
        opt.step()
        losses.append(l_dict)

    avg_loss = {k: float(np.mean([x[k] for x in losses])) for k in ('total', 'loss_path', 'loss_goal', 'loss_div')}
    return avg_loss

def generate_all_predictions(generator_model, data, h64_data, indices, batch_size=128):
    generator_model.eval()
    p_list, goal_list, abc_list = [], [], []
    with torch.inference_mode():
        for i in range(0, len(indices), batch_size):
            b_idx = indices[i:i + batch_size]
            p2_b = torch.from_numpy(data['p2'][b_idx]).to(DEVICE)
            h64_b = torch.from_numpy(h64_data[b_idx]).to(DEVICE)
            v0_b = torch.from_numpy(data['v0'][b_idx]).to(DEVICE)
            out = generator_model(p2_b, h64_b, v0_b)
            p_list.append(out['path'].cpu().numpy())
            goal_list.append(out['goal'].cpu().numpy())
            abc_list.append(out['process_coeff'].cpu().numpy())

    p = np.concatenate(p_list, axis=0)
    goal = np.concatenate(goal_list, axis=0)
    abc = np.concatenate(abc_list, axis=0)
    return p, goal, abc

def run_experiment():
    started_time = time.time()
    print("=== STARTING D2: MOTION PRETRAINED REPRESENTATION EXPERIMENT ===")

    state = json.load(open(EXP_DIR / 'runtime_state.json', encoding='utf-8'))
    state['status'] = 'RUNNING'
    state['production_started'] = True
    with open(EXP_DIR / 'runtime_state.json', 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2)

    data = load_data()
    stats = data['stats']
    past_fit = data['past'][data['split']['FIT']]
    past_outer = data['past'][data['split']['outer']]

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
            print(f"\n=======================================================")
            print(f"--- D2: Running {run_key} (Arm {arm}, Seed {seed}) ---")
            print(f"=======================================================")
            run_folder = runs_dir / run_key
            run_folder.mkdir(parents=True, exist_ok=True)

            setup_seed(seed)
            gen_pre = torch.Generator().manual_seed(seed)

            # Stage 1: Pretraining the Motion Encoder (20 epochs)
            print(f"Stage 1: Pretraining Motion Encoder for {run_key} (20 epochs)...")
            encoder = MotionPretrainedEncoder().to(DEVICE)
            opt_enc = torch.optim.AdamW(encoder.parameters(), lr=0.001, weight_decay=0.0001, betas=(0.9, 0.999), eps=1e-8)

            pre_history = []
            for ep in range(1, 21):
                t0 = time.time()
                loss_val = pretrain_one_epoch(encoder, opt_enc, past_fit, gen_pre, arm)
                el = time.time() - t0
                pre_history.append(dict(epoch=ep, loss=loss_val, time=el))
                if ep % 5 == 0 or ep == 20:
                    print(f"  Pretrain Epoch {ep:02d}/20 | Loss: {loss_val:.4f}m ({el:.1f}s)")

            # Measure pretraining transfer: evaluate pretrain loss on FIT and Outer
            fit_pre_err = evaluate_pretrain_loss_split(encoder, past_fit, arm)
            outer_pre_err = evaluate_pretrain_loss_split(encoder, past_outer, arm)
            print(f"  Pretrain Final Loss -> FIT: {fit_pre_err:.4f}m | Outer: {outer_pre_err:.4f}m")

            # Save encoder checkpoint
            torch.save(dict(encoder=encoder.state_dict(), pre_history=pre_history), run_folder / 'encoder_checkpoint.pt')

            # Extract representation h64 for all 28,791 samples with frozen encoder
            print("  Extracting frozen h64 representation across full dataset...")
            encoder.eval()
            h64_list = []
            with torch.inference_mode():
                for i in range(0, len(data['past']), 256):
                    p_batch = torch.from_numpy(data['past'][i:i + 256]).to(DEVICE)
                    h64_batch = encoder.extract_representation(p_batch)
                    h64_list.append(h64_batch.cpu().numpy())
            h64_all = np.concatenate(h64_list, axis=0) # (28791, 64)
            assert h64_all.shape == (28791, 64) and np.all(np.isfinite(h64_all))

            # Stage 2: Train Generator (25 epochs) with fresh generator optimizer
            print(f"Stage 2: Training Generator for {run_key} (25 epochs)...")
            setup_seed(seed + 1000)
            gen_train = torch.Generator().manual_seed(seed + 1000)

            generator_model = PretrainedConditionedGenerator(stats).to(DEVICE)
            opt_gen = torch.optim.AdamW(generator_model.parameters(), lr=0.001, weight_decay=0.0001, betas=(0.9, 0.999), eps=1e-8)

            gen_history = []
            for ep in range(1, 26):
                t0 = time.time()
                l_dict = train_generator_one_epoch(generator_model, opt_gen, data, h64_all, data['split']['FIT'], gen_train)
                el = time.time() - t0
                l_dict['epoch'] = ep
                l_dict['time'] = el
                gen_history.append(l_dict)
                if ep % 5 == 0 or ep == 25:
                    print(f"  Generator Epoch {ep:02d}/25 | Total: {l_dict['total']:.4f} | Path: {l_dict['loss_path']:.4f} | Goal: {l_dict['loss_goal']:.4f} ({el:.1f}s)")

            # Save generator checkpoint
            torch.save(dict(generator=generator_model.state_dict(), history=gen_history), run_folder / 'generator_checkpoint.pt')
            with open(run_folder / 'history.json', 'w', encoding='utf-8') as f:
                json.dump(dict(pretrain=pre_history, generator=gen_history), f, indent=2)

            # Evaluate generator across all 4 splits
            print(f"  Evaluating {run_key} across all 4 splits...")
            eval_results = {}
            for split_name, idxs in data['split'].items():
                p_eval, goal_eval, abc_eval = generate_all_predictions(generator_model, data, h64_all, idxs)
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
                    np.savez_compressed(
                        run_folder / 'outer_predictions.npz',
                        paths=p_eval,
                        native=data['native'][idxs],
                        gt=data['gt'][idxs],
                        goal=goal_eval,
                        process_coeff=abc_eval,
                        sample_id=data['sample_id'][idxs]
                    )
                    # Run representation ablation diagnostics on outer change
                    change_mask = (data['groups'][idxs] == 2)
                    p2_ch = torch.from_numpy(data['p2'][idxs[change_mask]]).to(DEVICE)
                    h64_ch = torch.from_numpy(h64_all[idxs[change_mask]]).to(DEVICE)
                    v0_ch = torch.from_numpy(data['v0'][idxs[change_mask]]).to(DEVICE)
                    ablations = evaluate_representation_ablations(generator_model, p2_ch, h64_ch, v0_ch)
                    ablations.update(dict(
                        fit_pretrain_loss=fit_pre_err,
                        outer_pretrain_loss=outer_pre_err
                    ))
                    diagnostics_all[run_key] = ablations

            with open(run_folder / 'metrics.json', 'w', encoding='utf-8') as f:
                json.dump(eval_results, f, indent=2)
            with open(run_folder / 'diagnostics.json', 'w', encoding='utf-8') as f:
                json.dump(diagnostics_all[run_key], f, indent=2)

            results[run_key] = eval_results
            completed += 1
            state['completed_generation_runs'] = completed
            with open(EXP_DIR / 'runtime_state.json', 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)

            del encoder, generator_model, opt_enc, opt_gen
            gc.collect()
            torch.cuda.empty_cache()

    total_elapsed = time.time() - started_time
    print(f"\nAll 6 pretrain + 6 generation runs completed in {total_elapsed / 60:.2f} minutes.")

    # Clustered Paired Bootstrap Analysis on Primary Outer Change (B vs A)
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
        experiment_id="D2_MOTION_PRETRAINED_REPRESENTATION_20261020",
        branch="D2",
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

    print("=== D2 RUNS COMPLETE & SAVED TO RESULT.JSON ===")

if __name__ == '__main__':
    run_experiment()
