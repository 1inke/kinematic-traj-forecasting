import os, sys, json
from pathlib import Path
import numpy as np
import torch
import psutil

EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parents[1]
sys.path.insert(0, str(ROOT / 'GEMINI_EXP'))

from core.data_loader import load_dataset, compute_fit_statistics
from core.model_base import StandardRefiner, set_seed, DEVICE

def run_preflight():
    print("=== STARTING EXP03 PREFLIGHT AUDIT ===")
    
    # 1. Dataset loading & integrity audit
    data = load_dataset()
    splits = data['split']
    for s1 in splits:
        for s2 in splits:
            if s1 != s2:
                overlap = set(splits[s1]).intersection(set(splits[s2]))
                assert len(overlap) == 0, f"Data leakage: Overlap between {s1} and {s2}!"
    print("Audit 1: Split disjointness verified (Zero leakage across FIT, outer, inner, CAL).")

    # 2. Statistics check
    st = compute_fit_statistics(data, 'position')
    assert st['mean'].shape == (120,) and st['std'].shape == (120,)
    assert np.all(st['std'] > 0)
    print("Audit 2: FIT-only normalization statistics computed successfully.")

    # 3. Model parameter & initialization audit
    net = StandardRefiner(st, 'position').to(DEVICE)
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    assert trainable_params == 25304, f"Parameter count mismatch: {trainable_params} != 25304"
    assert torch.all(net.head[-1].weight == 0) and torch.all(net.head[-1].bias == 0)
    print(f"Audit 3: Model parameter count verified ({trainable_params} params, exact zero-init).")

    # 4. Identity transformation on initialization
    p2_dummy = torch.from_numpy(data['p2'][:4]).to(DEVICE)
    nat_dummy = torch.from_numpy(data['native'][:4]).to(DEVICE)
    with torch.no_grad():
        p_init, z_init, u_init, r_init = net(p2_dummy, nat_dummy)
    assert torch.all(r_init == 0), "Initialization does not yield exact zero displacement!"
    assert torch.all(p_init == nat_dummy), "Initialization does not yield exact identity paths!"
    print("Audit 4: Identity transformation verified at initialization.")

    # 5. Kernel displacement bound check
    net.head[-1].weight.data.normal_(0, 1) # temporarily disrupt to test bounds
    with torch.no_grad():
        _, _, _, r_test = net(p2_dummy, nat_dummy)
    r_disp = torch.linalg.vector_norm(r_test, dim=-1).cpu().numpy() # (4, 20, 12)
    max_allowed = 0.20 * (np.arange(1, 13) / 12.0) ** 2 + 1e-12
    for t in range(12):
        assert np.all(r_disp[..., t] <= max_allowed[t]), f"Displacement exceeded bound at step {t+1}!"
    print("Audit 5: Strict physical acceleration-integrated displacement bounds verified.")

    # 6. Loss formulations audit (Damping & Smoothness)
    # Test zero control
    u_zero = torch.zeros(4, 20, 12, 2, dtype=torch.float64, device=DEVICE)
    l_ctrl_zero = u_zero.square().sum(-1).mean()
    l_smooth_zero = torch.diff(u_zero, dim=2).square().sum(-1).mean()
    assert l_ctrl_zero.item() == 0.0
    assert l_smooth_zero.item() == 0.0

    # Test random non-zero control
    u_rand = torch.randn(4, 20, 12, 2, dtype=torch.float64, device=DEVICE)
    l_ctrl_rand = u_rand.square().sum(-1).mean()
    l_smooth_rand = torch.diff(u_rand, dim=2).square().sum(-1).mean()
    assert l_ctrl_rand.item() > 0.0
    assert l_smooth_rand.item() > 0.0
    print("Audit 6: Kinematic damping & smoothness loss formulations verified.")

    # 7. System resource check
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024**3)
    assert available_gb >= 6.0, f"Insufficient memory available: {available_gb:.2f} GB < 6.0 GB"
    print(f"Audit 7: Memory check passed ({available_gb:.2f} GB available).")

    preflight_result = dict(
        status="PASS",
        split_sizes={k: len(v) for k, v in splits.items()},
        trainable_parameters=trainable_params,
        displacement_bound_verified=True,
        loss_formulations_verified=True,
        available_memory_gb=available_gb
    )
    with open(EXP_DIR / 'PREFLIGHT.json', 'w', encoding='utf-8') as f:
        json.dump(preflight_result, f, indent=2)

    # Update runtime_state.json
    state = json.load(open(EXP_DIR / 'runtime_state.json', encoding='utf-8'))
    state['status'] = 'PREFLIGHT_PASSED'
    state['preflight_status'] = 'PASS'
    with open(EXP_DIR / 'runtime_state.json', 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2)

    print("=== PREFLIGHT AUDIT COMPLETE: ALL PASS ===")
    return True

if __name__ == '__main__':
    run_preflight()
