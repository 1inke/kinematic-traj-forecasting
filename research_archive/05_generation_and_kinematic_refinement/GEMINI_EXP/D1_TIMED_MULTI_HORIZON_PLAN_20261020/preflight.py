import os, sys, json
from pathlib import Path
import numpy as np
import torch
import psutil

EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parents[1]
sys.path.insert(0, str(ROOT / 'GEMINI_EXP'))

from D1_TIMED_MULTI_HORIZON_PLAN_20261020.data_util import load_data
from D1_TIMED_MULTI_HORIZON_PLAN_20261020.model import TimedMultiHorizonModel, compute_objective, create_optimizer, setup_seed, DEVICE

def run_preflight():
    print("=== D1 TIMED MULTI-HORIZON PLAN PREFLIGHT AUDIT ===")

    # 1. Load data
    data = load_data()
    splits = data['split']
    for s1 in splits:
        for s2 in splits:
            if s1 != s2:
                overlap = set(splits[s1]).intersection(set(splits[s2]))
                assert len(overlap) == 0, f"Leakage detected between {s1} and {s2}!"
    print("Audit 1: Disjoint splits confirmed (Zero leakage).")

    # 2. Shape and type audit
    assert data['p2'].shape == (28791, 96)
    assert data['v0'].shape == (28791, 2)
    assert data['gt'].shape == (28791, 12, 2)
    assert data['y20'].shape == (28791, 2)
    assert data['m20'].shape == (28791,) and data['m20'].dtype == bool
    assert np.all(np.isfinite(data['p2'])) and np.all(np.isfinite(data['v0']))
    print("Audit 2: Shapes, types, and finites verified across all 28,791 samples.")

    # 3. Model construction and parameter audit
    stats = data['stats']
    model = TimedMultiHorizonModel(stats).to(DEVICE)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable_params == 32910, f"Expected 32,910 parameters, got {trainable_params}"
    print(f"Audit 3: Parameter count verified: {trainable_params} trainable parameters.")

    # 4. Zero-initialization and baseline geometry check
    assert torch.all(model.plan_head[-1].weight == 0) and torch.all(model.plan_head[-1].bias == 0)
    assert torch.all(model.process_head[-1].weight == 0) and torch.all(model.process_head[-1].bias == 0)
    
    p2_dummy = torch.from_numpy(data['p2'][:8]).to(DEVICE)
    v0_dummy = torch.from_numpy(data['v0'][:8]).to(DEVICE)
    with torch.no_grad():
        out_init = model(p2_dummy, v0_dummy)
    
    # Path should start exactly at 4.8 * s * v0 + s^2 * centers12
    s = (torch.arange(1, 13, dtype=torch.float64) / 12.0).to(DEVICE)[None, None, :, None]
    v_exp = v0_dummy.double()[:, None, None, :]
    c_exp = model.centers12[None, :, None, :]
    expected_init = 4.8 * s * v_exp + s.square() * c_exp
    max_err = float(torch.max(torch.abs(out_init['path'] - expected_init)))
    assert max_err < 1e-12, f"Initial path mismatch: max_err={max_err}"
    print("Audit 4: Zero-init baseline matches exact physical polynomial geometry.")

    # 5. Target scaling audit for both Arm A and Arm B
    y12_dummy = torch.from_numpy(data['gt'][:8]).to(DEVICE)
    y20_dummy = torch.from_numpy(data['y20'][:8]).to(DEVICE)
    m20_dummy = torch.from_numpy(data['m20'][:8]).to(DEVICE)

    loss_A, d_A = compute_objective(out_init, y12_dummy, y20_dummy, m20_dummy, arm='A')
    loss_B, d_B = compute_objective(out_init, y12_dummy, y20_dummy, m20_dummy, arm='B')
    assert torch.isfinite(loss_A) and torch.isfinite(loss_B)
    print(f"Audit 5: Loss computation verified (Init loss A: {loss_A.item():.4f}, B: {loss_B.item():.4f}).")

    # 6. Real gradient step verification
    opt = create_optimizer(model)
    opt.zero_grad()
    out_train = model(p2_dummy, v0_dummy)
    loss_train, _ = compute_objective(out_train, y12_dummy, y20_dummy, m20_dummy, arm='B')
    loss_train.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), f"Invalid gradient on {name}"
    opt.step()
    print("Audit 6: Real backward and optimization step executed cleanly.")

    # 7. System resource check
    mem = psutil.virtual_memory()
    avail_gb = mem.available / (1024**3)
    assert avail_gb >= 6.0, f"Insufficient RAM: {avail_gb:.2f} GB"
    print(f"Audit 7: System memory check passed ({avail_gb:.2f} GB available).")

    preflight_dict = dict(
        status="PASS",
        split_sizes={k: len(v) for k, v in splits.items()},
        trainable_parameters=trainable_params,
        init_baseline_error=max_err,
        available_memory_gb=avail_gb
    )
    with open(EXP_DIR / 'PREFLIGHT.json', 'w', encoding='utf-8') as f:
        json.dump(preflight_dict, f, indent=2)

    state = json.load(open(EXP_DIR / 'runtime_state.json', encoding='utf-8'))
    state['status'] = 'PREFLIGHT_PASSED'
    state['preflight_status'] = 'PASS'
    with open(EXP_DIR / 'runtime_state.json', 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2)

    print("=== D1 PREFLIGHT AUDIT PASSED ===")
    return True

if __name__ == '__main__':
    run_preflight()
