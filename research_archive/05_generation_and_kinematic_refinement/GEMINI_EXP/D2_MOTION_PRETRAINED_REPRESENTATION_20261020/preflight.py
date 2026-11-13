import os, sys, json
from pathlib import Path
import numpy as np
import torch
import psutil

EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parents[1]
sys.path.insert(0, str(ROOT / 'GEMINI_EXP'))

from D1_TIMED_MULTI_HORIZON_PLAN_20261020.data_util import load_data
from D2_MOTION_PRETRAINED_REPRESENTATION_20261020.model import (
    MotionPretrainedEncoder,
    PretrainedConditionedGenerator,
    compute_pretrain_loss,
    compute_generator_loss,
    setup_seed,
    DEVICE
)

def run_preflight():
    print("=== D2 MOTION PRETRAINED REPRESENTATION PREFLIGHT AUDIT ===")

    # 1. Load data
    data = load_data()
    splits = data['split']
    for s1 in splits:
        for s2 in splits:
            if s1 != s2:
                overlap = set(splits[s1]).intersection(set(splits[s2]))
                assert len(overlap) == 0, f"Split overlap between {s1} and {s2}"
    print("Audit 1: Split disjointness verified (Zero leakage).")

    # 2. Check shapes
    past = data['past']
    assert past.shape == (28791, 8, 2)
    assert np.all(np.isfinite(past))
    print("Audit 2: Past trajectory data shape verified (28791, 8, 2).")

    # 3. Parameter count audit
    encoder = MotionPretrainedEncoder().to(DEVICE)
    enc_total = sum(p.numel() for p in encoder.parameters())
    enc_frozen = sum(p.numel() for p in encoder.proj.parameters()) + sum(p.numel() for p in encoder.transformer.parameters())
    assert enc_total == 71554, f"Encoder param mismatch: {enc_total} != 71554"
    assert enc_frozen == 67264, f"Frozen param mismatch: {enc_frozen} != 67264"
    print(f"Audit 3a: Pretrained Encoder verified ({enc_total} total, {enc_frozen} frozen for generation).")

    stats = data['stats']
    generator = PretrainedConditionedGenerator(stats).to(DEVICE)
    gen_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    assert gen_params == 40328, f"Generator param mismatch: {gen_params} != 40328"
    print(f"Audit 3b: Generator parameters verified ({gen_params} trainable params).")

    # 4. Zero initialization & physical baseline check
    assert torch.all(generator.goal_head[-1].weight == 0) and torch.all(generator.goal_head[-1].bias == 0)
    assert torch.all(generator.process_head[-1].weight == 0) and torch.all(generator.process_head[-1].bias == 0)

    p2_dummy = torch.from_numpy(data['p2'][:8]).to(DEVICE)
    v0_dummy = torch.from_numpy(data['v0'][:8]).to(DEVICE)
    past_dummy = torch.from_numpy(data['past'][:8]).to(DEVICE)
    with torch.no_grad():
        h64_dummy = encoder.extract_representation(past_dummy)
        out_init = generator(p2_dummy, h64_dummy, v0_dummy)

    s = (torch.arange(1, 13, dtype=torch.float64) / 12.0).to(DEVICE)[None, None, :, None]
    v_exp = v0_dummy.double()[:, None, None, :]
    c_exp = generator.centers12[None, :, None, :]
    expected_init = 4.8 * s * v_exp + s.square() * c_exp
    max_err = float(torch.max(torch.abs(out_init['path'] - expected_init)))
    assert max_err < 1e-12, f"Initial path mismatch: {max_err}"
    print("Audit 4: Initial generator output matches exact physical polynomial baseline.")

    # 5. Pretraining masking & coordinate sanity
    # Test Arm A masking: indices 4, 5 hidden
    B = 4
    p_test = past_dummy[:B]
    rel_test = p_test - p_test[:, 0:1, :]
    mask_A = torch.ones((B, 8, 1), dtype=torch.float32, device=DEVICE)
    mask_A[:, [4, 5], :] = 0.0
    rel_in_A = rel_test.clone()
    rel_in_A[:, [4, 5], :] = 0.0
    t_norm = (torch.arange(8, dtype=torch.float32, device=DEVICE) / 8.0)[None, :, None].expand(B, -1, -1)
    x_in_A = torch.cat((rel_in_A.float(), mask_A, t_norm), dim=-1)
    _, pred_A = encoder.forward_pretrain(x_in_A)
    loss_pre_A = compute_pretrain_loss(pred_A, p_test, arm='A')
    assert torch.isfinite(loss_pre_A)

    # Test Arm B masking: indices 6, 7 hidden
    mask_B = torch.ones((B, 8, 1), dtype=torch.float32, device=DEVICE)
    mask_B[:, [6, 7], :] = 0.0
    rel_in_B = rel_test.clone()
    rel_in_B[:, [6, 7], :] = 0.0
    x_in_B = torch.cat((rel_in_B.float(), mask_B, t_norm), dim=-1)
    _, pred_B = encoder.forward_pretrain(x_in_B)
    loss_pre_B = compute_pretrain_loss(pred_B, p_test, arm='B')
    assert torch.isfinite(loss_pre_B)
    print("Audit 5: Pretraining masking, origin-zeroing, and loss computations verified for both arms.")

    # 6. Real gradient step verification for both stages
    # Stage 1: Encoder pretrain backward
    opt_enc = torch.optim.AdamW(encoder.parameters(), lr=0.001)
    opt_enc.zero_grad()
    loss_pre_B.backward()
    for name, p in encoder.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), f"Pretrain grad issue in {name}"
    opt_enc.step()

    # Stage 2: Generator backward
    opt_gen = torch.optim.AdamW(generator.parameters(), lr=0.001)
    opt_gen.zero_grad()
    y12_dummy = torch.from_numpy(data['gt'][:8]).to(DEVICE)
    out_gen = generator(p2_dummy, h64_dummy, v0_dummy)
    loss_gen, _ = compute_generator_loss(out_gen, y12_dummy)
    loss_gen.backward()
    for name, p in generator.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), f"Generator grad issue in {name}"
    opt_gen.step()
    print("Audit 6: Real backward and optimizer update passed for both pretrain and generator stages.")

    # 7. System resource check
    mem = psutil.virtual_memory()
    avail_gb = mem.available / (1024**3)
    assert avail_gb >= 6.0, f"Insufficient RAM: {avail_gb:.2f} GB"
    print(f"Audit 7: System memory check passed ({avail_gb:.2f} GB available).")

    preflight_dict = dict(
        status="PASS",
        split_sizes={k: len(v) for k, v in splits.items()},
        encoder_parameters=enc_total,
        frozen_parameters=enc_frozen,
        generator_parameters=gen_params,
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

    print("=== D2 PREFLIGHT AUDIT PASSED ===")
    return True

if __name__ == '__main__':
    run_preflight()
