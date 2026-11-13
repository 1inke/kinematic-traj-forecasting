import numpy as np
import torch

@torch.no_grad()
def evaluate_pretrain_loss_split(encoder, past_split, arm, batch_size=256, device='cuda:0'):
    """
    Evaluates self-supervised pretraining loss on a data split.
    arm='A': infilling [4, 5]
    arm='B': causal forecast [6, 7]
    """
    encoder.eval()
    losses = []
    N = len(past_split)
    hidden_indices = [4, 5] if arm == 'A' else [6, 7]

    for i in range(0, N, batch_size):
        p_b = torch.from_numpy(past_split[i:i + batch_size]).to(device)
        B = len(p_b)
        rel_pos = p_b - p_b[:, 0:1, :]
        mask = torch.ones((B, 8, 1), dtype=torch.float32, device=device)
        mask[:, hidden_indices, :] = 0.0
        rel_in = rel_pos.clone()
        rel_in[:, hidden_indices, :] = 0.0
        t_norm = (torch.arange(8, dtype=torch.float32, device=device) / 8.0)[None, :, None].expand(B, -1, -1)
        x_in = torch.cat((rel_in.float(), mask, t_norm), dim=-1)

        _, pred = encoder.forward_pretrain(x_in)
        target = rel_pos[:, hidden_indices, :]
        pred_sub = pred[:, hidden_indices, :]
        err = torch.linalg.vector_norm(pred_sub - target, dim=-1) # (B, 2)
        losses.append(err.mean().item())

    return float(np.mean(losses))

@torch.no_grad()
def evaluate_representation_ablations(generator, p2_tensor, h64_tensor, v0_tensor):
    """
    Evaluates sensitivity to the pre-trained representation:
    1. Zero-ablation: set h64 = 0
    2. Track-mismatch ablation: shuffle h64
    """
    generator.eval()
    orig_out = generator(p2_tensor, h64_tensor, v0_tensor)
    orig_path = orig_out['path']

    # 1. Zero ablation
    h_zero = torch.zeros_like(h64_tensor)
    zero_out = generator(p2_tensor, h_zero, v0_tensor)
    drift_zero = torch.linalg.vector_norm(zero_out['path'] - orig_path, dim=-1)

    # 2. Track mismatch ablation
    perm = torch.randperm(len(h64_tensor))
    h_shuffled = h64_tensor[perm]
    shuffled_out = generator(p2_tensor, h_shuffled, v0_tensor)
    drift_shuffled = torch.linalg.vector_norm(shuffled_out['path'] - orig_path, dim=-1)

    return dict(
        h64_zero_mean_drift=float(drift_zero.mean().item()),
        h64_zero_max_drift=float(drift_zero.max().item()),
        h64_shuffle_mean_drift=float(drift_shuffled.mean().item()),
        h64_shuffle_max_drift=float(drift_shuffled.max().item())
    )
