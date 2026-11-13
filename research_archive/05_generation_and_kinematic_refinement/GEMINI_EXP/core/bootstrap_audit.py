import numpy as np

def paired_bootstrap_test(metric_treatment, metric_control, tracks, n_draws=10000, seed=20261018, alpha=0.05):
    """
    Performs clustered (by target track) paired bootstrap hypothesis test.
    Returns:
      mean_diff: treatment - control
      ci_lower: (1 - alpha) bootstrap percentile CI lower bound
      ci_upper: (1 - alpha) bootstrap percentile CI upper bound
      p_value: two-sided bootstrap p-value
      crosses_zero: whether CI covers 0
    """
    assert len(metric_treatment) == len(metric_control) == len(tracks)
    N = len(tracks)
    unique_tracks = np.unique(tracks)
    n_tracks = len(unique_tracks)

    track_to_indices = {t: np.where(tracks == t)[0] for t in unique_tracks}

    diff = np.asarray(metric_treatment, dtype=np.float64) - np.asarray(metric_control, dtype=np.float64)
    obs_mean = float(diff.mean())

    rng = np.random.Generator(np.random.PCG64(seed))
    boot_means = np.empty(n_draws, dtype=np.float64)

    for b in range(n_draws):
        sampled_tracks = rng.choice(unique_tracks, size=n_tracks, replace=True)
        idx_list = [track_to_indices[t] for t in sampled_tracks]
        all_idx = np.concatenate(idx_list)
        boot_means[b] = diff[all_idx].mean()

    ci_lower = float(np.percentile(boot_means, 100 * (alpha / 2)))
    ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    crosses_zero = bool(ci_lower <= 0 <= ci_upper)

    # Two-sided empirical p-value relative to 0
    p_pos = (boot_means >= 0).mean()
    p_neg = (boot_means <= 0).mean()
    p_value = float(2 * min(p_pos, p_neg))

    return dict(
        observed_mean_diff=obs_mean,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        alpha=alpha,
        crosses_zero=crosses_zero,
        p_value=p_value,
        n_draws=n_draws,
        n_tracks=n_tracks,
        n_samples=N
    )
