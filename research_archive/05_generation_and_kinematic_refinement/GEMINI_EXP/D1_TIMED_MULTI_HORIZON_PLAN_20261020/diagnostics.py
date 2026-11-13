import numpy as np
import torch

def compute_plan_diagnostics(paths, plans, gt, y20, m20, arm):
    """
    Computes plan-specific diagnostics for D1:
    1. Plan point errors at horizons 4, 8, 12, 20:
       - Horizon 4: e_4 = ||plan[:, 0] - 3*y4||
       - Horizon 8: e_8 = ||plan[:, 1] - 1.5*y8||
       - Horizon 12: e_12 = ||plan[:, 2] - y12||
       - Horizon 20: e_20 = ||plan[:, 3] - 0.6*y20||
    2. Path-to-plan gap:
       Distance between generated path at t and plan point:
       - t=4 (idx 3): ||p[:, 3] - plan[:, 0] / 3||
       - t=8 (idx 7): ||p[:, 7] - plan[:, 1] / 1.5||
       - t=12 (idx 11): ||p[:, 11] - plan[:, 2]||
    """
    # paths: (N, 20, 12, 2)
    # plans: (N, 20, 4, 2)
    # gt: (N, 12, 2)
    # y20: (N, 2)
    # m20: (N,) bool
    
    # Evaluate on the winner candidate
    d = np.linalg.norm(paths - gt[:, None], axis=-1)
    q = d.mean(-1) + 0.5 * d[:, :, -1]
    winner = q.argmin(1)
    N = len(paths)
    row = np.arange(N)

    w_plan = plans[row, winner] # (N, 4, 2)
    w_path = paths[row, winner] # (N, 12, 2)

    m20_mask = m20.astype(bool)
    if np.any(m20_mask):
        w_plan_m20 = w_plan[m20_mask]
        y4 = 3.0 * gt[m20_mask, 3, :]
        y8 = 1.5 * gt[m20_mask, 7, :]
        y12 = 1.0 * gt[m20_mask, 11, :]
        y20_scaled = 0.6 * y20[m20_mask]
        target = np.stack((y4, y8, y12, y20_scaled), axis=1) # (M, 4, 2)
        plan_err_m20 = np.linalg.norm(w_plan_m20 - target, axis=-1) # (M, 4)
    else:
        plan_err_m20 = np.zeros((0, 4))

    # Path to plan gaps:
    # plan points are scaled to 12-step meters. To compare at t=4: plan[:, 0] / 3.0
    gap_4 = np.linalg.norm(w_path[:, 3, :] - w_plan[:, 0, :] / 3.0, axis=-1)
    gap_8 = np.linalg.norm(w_path[:, 7, :] - w_plan[:, 1, :] / 1.5, axis=-1)
    gap_12 = np.linalg.norm(w_path[:, 11, :] - w_plan[:, 2, :], axis=-1)

    return dict(
        mean_plan_error_step4=float(plan_err_m20[:, 0].mean()) if len(plan_err_m20) else None,
        mean_plan_error_step8=float(plan_err_m20[:, 1].mean()) if len(plan_err_m20) else None,
        mean_plan_error_step12=float(plan_err_m20[:, 2].mean()) if len(plan_err_m20) else None,
        mean_plan_error_step20=float(plan_err_m20[:, 3].mean()) if len(plan_err_m20) else None,
        mean_path_to_plan_gap_step4=float(gap_4.mean()),
        mean_path_to_plan_gap_step8=float(gap_8.mean()),
        mean_path_to_plan_gap_step12=float(gap_12.mean())
    )

@torch.no_grad()
def evaluate_frozen_plan_swap(model, p2_tensor, v0_tensor):
    """
    Diagnostic: Replace intermediate plan points (r=0, 1, 2) with z4 (r=3),
    decode path, and measure path change (sensitivity to intermediate plan).
    """
    model.eval()
    B = len(p2_tensor)
    hq = model.encode(p2_tensor)
    delta = model.plan_head(hq).reshape(B, 20, 4, 2)
    base = 4.8 * v0_tensor[:, None, None, :] + model.centers12[None, :, None, :]
    orig_plan = base + delta.double()

    # Original decoding
    orig_out = model.decode_with_plan(hq, v0_tensor, orig_plan)

    # Swapped plan: set plan[:, :, :3] = plan[:, :, 3:4]
    swapped_plan = orig_plan.clone()
    swapped_plan[:, :, :3] = orig_plan[:, :, 3:4].expand(-1, -1, 3, -1)

    swapped_out = model.decode_with_plan(hq, v0_tensor, swapped_plan)

    drift = torch.linalg.vector_norm(swapped_out['path'] - orig_out['path'], dim=-1)
    mean_drift = float(drift.mean().item())
    max_drift = float(drift.max().item())

    return dict(
        plan_swap_mean_path_drift=mean_drift,
        plan_swap_max_path_drift=max_drift
    )
