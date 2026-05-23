import numpy as np


def get_percentiles_timeout(
    percentile_values,
    min_connection_time,
    max_connection_time,
    speed_tier_definitions,
):
    sample_count = 1_000_000
    rng = np.random.default_rng(42)
    connection_time_samples = rng.uniform(
        min_connection_time, max_connection_time, size=sample_count
    )

    tier_probabilities = np.array(
        [tier[3] for tier in speed_tier_definitions], dtype=float
    )
    tier_probabilities /= tier_probabilities.sum()
    sampled_tier_indices = rng.choice(
        len(speed_tier_definitions), size=sample_count, p=tier_probabilities
    )

    train_time_samples = np.empty(sample_count)
    for tier_index, (_, min_train_time, max_train_time, _) in enumerate(
        speed_tier_definitions
    ):
        tier_sample_mask = sampled_tier_indices == tier_index
        tier_sample_count = int(tier_sample_mask.sum())
        if tier_sample_count:
            train_time_samples[tier_sample_mask] = rng.uniform(
                min_train_time, max_train_time, size=tier_sample_count
            )

    total_duration_samples = connection_time_samples + train_time_samples
    timeout_by_percentile = np.percentile(total_duration_samples, percentile_values)
    print(f"Os timeouts sao {timeout_by_percentile}")
    return timeout_by_percentile
