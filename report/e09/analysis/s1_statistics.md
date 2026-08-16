# s1_statistics — Per-group and pooled nonparametric statistics (Reviewer R1)

- Data: `results/e09-gradual-drift/cifar-10/grid-v1/` per-episode `agent.json`/`baseline.json`, 12 groups × 5 seeds = 60 paired episodes (4 corruptions × 3 ramp durations × 5 seeds, horizon 600 s).
- Generated: 2026-08-16. Python: numpy only (scipy 1.15.3 in the venv fails to import — broken `scipy.sparse` wheel on this machine — so the exact signed-rank test and the normal-approximation p-value were re-implemented directly, replicating scipy's formulas; results are equivalent).

## Methods
- **Paired difference** per episode: `d = baseline_downtime − agent_downtime` (s). Positive → agent wins (less downtime). Ties (d = 0) count as neither win nor loss.
- **Median**: median of the 5 (or 60) differences.
- **Hodges–Lehmann estimator**: median of the n(n+1)/2 Walsh averages (d_i + d_j)/2, i ≤ j.
- **CI (exact, per group)**: two-sided CI for the pseudo-median from the signed-rank statistic, [W_(c+1), W_(N−c)] over sorted Walsh averages, with c the largest integer s.t. P(T+ ≤ c) ≤ 0.025 under the exact null distribution (subset-sum DP). **With n = 5 no exact 95% CI exists** (the smallest achievable tail mass is 1/32 = 0.03125 > 0.025); the closest achievable level is reported (93.75%). This is a data-independent property of n = 5, not a data artifact.
- **CI (pooled, n = 60)**: normal-approximation quantile of T+ (mean m(m+1)/4, var m(m+1)(2m+1)/24, continuity correction), ≈95% coverage.
- **Cliff's delta (paired)**: (P(win) − P(loss)) = (n_pos − n_neg)/n — the matched-pairs analogue of Cliff's dominance measure. Interpretation thresholds (Romano et al. 2006): |d| < 0.147 negligible; [0.147, 0.33) small; [0.33, 0.474) medium; ≥ 0.474 large.
- **prop_win**: fraction of the 5 (or 60) episodes where the agent has strictly lower downtime.
- **Wilcoxon pooled**: one-sided signed-rank test on d = baseline − agent (H1: agent < baseline, i.e., positive d). Zeros dropped (zero_method='wilcox', 1 tie dropped), mid-ranks for ties, normal approximation with continuity correction (scipy's 'auto' choice for n = 60). W+ = sum of ranks of positive differences (scipy-style statistic = min(W+, W−) = 0 here).

## 1. Downtime — per group (n = 5)

| Group | Median Δ | HL Δ | CI (lo, hi) | CI cov. | Cliff's δ | Interpret. | Wins | Ties | prop_win | exact p (2-s) |
|---|---|---|---|---|---|---|---|---|---|---|
| fog_sev4/ramp_5 | 492.0 | 492.0 | (492.0, 492.0) | 94% | 1.00 | large | 5 | 0 | 1.0 | 0.0625 |
| fog_sev4/ramp_10 | 482.0 | 482.0 | (472.0, 482.0) | 94% | 1.00 | large | 5 | 0 | 1.0 | 0.0625 |
| fog_sev4/ramp_20 | 400.0 | 405.0 | (400.0, 430.0) | 94% | 1.00 | large | 5 | 0 | 1.0 | 0.0625 |
| frosted_glass_blur_sev4/ramp_5 | 492.0 | 492.0 | (492.0, 492.0) | 94% | 1.00 | large | 5 | 0 | 1.0 | 0.0625 |
| frosted_glass_blur_sev4/ramp_10 | 482.0 | 482.0 | (482.0, 492.0) | 94% | 1.00 | large | 5 | 0 | 1.0 | 0.0625 |
| frosted_glass_blur_sev4/ramp_20 | 452.0 | 451.0 | (440.0, 460.0) | 94% | 1.00 | large | 5 | 0 | 1.0 | 0.0625 |
| gaussian_noise_sev3/ramp_5 | 472.0 | 447.0 | (292.0, 482.0) | 94% | 1.00 | large | 5 | 0 | 1.0 | 0.0625 |
| gaussian_noise_sev3/ramp_10 | 422.0 | 422.0 | (382.0, 472.0) | 88% | 0.80 | large | 4 | 1 | 0.8 | 0.1250 |
| gaussian_noise_sev3/ramp_20 | 372.0 | 372.0 | (132.0, 412.0) | 94% | 1.00 | large | 5 | 0 | 1.0 | 0.0625 |
| motion_blur_sev1/ramp_5 | 492.0 | 492.0 | (492.0, 502.0) | 94% | 1.00 | large | 5 | 0 | 1.0 | 0.0625 |
| motion_blur_sev1/ramp_10 | 482.0 | 482.0 | (472.0, 482.0) | 94% | 1.00 | large | 5 | 0 | 1.0 | 0.0625 |
| motion_blur_sev1/ramp_20 | 452.0 | 452.0 | (442.0, 462.0) | 94% | 1.00 | large | 5 | 0 | 1.0 | 0.0625 |

*exact p (2-s) = exact two-sided signed-rank p (sign-pattern enumeration, mid-ranks); for gaussian_noise_sev3/ramp_10 the zero difference is dropped (m = 4). These per-group values illustrate why p-values are uninformative at n = 5 (two-sided p = 0.0625 even for 5/5 wins): effect sizes and CIs are the primary evidence.*

### Per-group interpretation

- **fog_sev4/ramp_5**: uniformly large benefit: the agent wins all 5/5 episodes, Cliff's δ = 1.00 (large), HL median downtime reduction 492 s.
- **fog_sev4/ramp_10**: uniformly large benefit: the agent wins all 5/5 episodes, Cliff's δ = 1.00 (large), HL median downtime reduction 482 s.
- **fog_sev4/ramp_20**: uniformly large benefit: the agent wins all 5/5 episodes, Cliff's δ = 1.00 (large), HL median downtime reduction 405 s.
- **frosted_glass_blur_sev4/ramp_5**: uniformly large benefit: the agent wins all 5/5 episodes, Cliff's δ = 1.00 (large), HL median downtime reduction 492 s.
- **frosted_glass_blur_sev4/ramp_10**: uniformly large benefit: the agent wins all 5/5 episodes, Cliff's δ = 1.00 (large), HL median downtime reduction 482 s.
- **frosted_glass_blur_sev4/ramp_20**: uniformly large benefit: the agent wins all 5/5 episodes, Cliff's δ = 1.00 (large), HL median downtime reduction 451 s.
- **gaussian_noise_sev3/ramp_5**: uniformly large benefit: the agent wins all 5/5 episodes, Cliff's δ = 1.00 (large), HL median downtime reduction 447 s.
- **gaussian_noise_sev3/ramp_10**: 4/5 wins, 1 tie(s), Cliff's δ = 0.80 (large), HL reduction 422 s.
- **gaussian_noise_sev3/ramp_20**: uniformly large benefit: the agent wins all 5/5 episodes, Cliff's δ = 1.00 (large), HL median downtime reduction 372 s.
- **motion_blur_sev1/ramp_5**: uniformly large benefit: the agent wins all 5/5 episodes, Cliff's δ = 1.00 (large), HL median downtime reduction 492 s.
- **motion_blur_sev1/ramp_10**: uniformly large benefit: the agent wins all 5/5 episodes, Cliff's δ = 1.00 (large), HL median downtime reduction 482 s.
- **motion_blur_sev1/ramp_20**: uniformly large benefit: the agent wins all 5/5 episodes, Cliff's δ = 1.00 (large), HL median downtime reduction 452 s.

## 2. Downtime — pooled (60 paired episodes)

- Median Δ: 472.0 s; mean Δ: 445.3 s; Hodges–Lehmann: 462.0 s (CI [450.0, 476.0], 95% normal-approx).
- Cliff's delta (paired): 0.98 (large); wins 59/60, ties 1 (1 zero difference in gaussian_noise_sev3/ramp_10 seed 42, dropped for the test, zero_method='wilcox'), prop_win = 0.983.
- Wilcoxon signed-rank (one-sided, agent < baseline): W+ = 1770 on m = 59 nonzero differences (mid-ranks; normal approx. with continuity correction), p = 1.23e-11 — **confirms p < 0.001**. Two-sided p = 2.45e-11.
- Interpretation: **large** effect by Romano et al. thresholds.

## 3. Detection delay (agent) — per group

| Group | Median (s) | Range (min–max, s) | n |
|---|---|---|---|
| fog_sev4/ramp_5 | 100 | 100–100 | 5 |
| fog_sev4/ramp_10 | 110 | 110–120 | 5 |
| fog_sev4/ramp_20 | 140 | 130–140 | 5 |
| frosted_glass_blur_sev4/ramp_5 | 100 | 100–100 | 5 |
| frosted_glass_blur_sev4/ramp_10 | 110 | 100–110 | 5 |
| frosted_glass_blur_sev4/ramp_20 | 140 | 130–140 | 5 |
| gaussian_noise_sev3/ramp_5 | 120 | 110–300 | 5 |
| gaussian_noise_sev3/ramp_10 | 150 | 120–210 | 4 |
| gaussian_noise_sev3/ramp_20 | 220 | 180–460 | 5 |
| motion_blur_sev1/ramp_5 | 100 | 90–100 | 5 |
| motion_blur_sev1/ramp_10 | 110 | 110–120 | 5 |
| motion_blur_sev1/ramp_20 | 140 | 130–150 | 5 |

*Note: detection_delay is null for episodes where the detector never triggered (gaussian_noise_sev3/ramp_10, seed 42 — the same episode counted as a tie in downtime, both sides 540 s). The paper's reported mean for that group (157.5 s) is over n = 4 for the same reason.*

## 3b. Appendix — per-seed paired differences (baseline − agent, s)

| Group | s42 | s43 | s44 | s45 | s46 |
|---|---|---|---|---|---|
| fog_sev4/ramp_5 | 492.0 | 492.0 | 492.0 | 492.0 | 492.0 |
| fog_sev4/ramp_10 | 472.0 | 482.0 | 482.0 | 482.0 | 482.0 |
| fog_sev4/ramp_20 | 410.0 | 400.0 | 400.0 | 400.0 | 430.0 |
| frosted_glass_blur_sev4/ramp_5 | 492.0 | 492.0 | 492.0 | 492.0 | 492.0 |
| frosted_glass_blur_sev4/ramp_10 | 482.0 | 482.0 | 482.0 | 492.0 | 482.0 |
| frosted_glass_blur_sev4/ramp_20 | 460.0 | 452.0 | 440.0 | 452.0 | 450.0 |
| gaussian_noise_sev3/ramp_5 | 292.0 | 472.0 | 482.0 | 472.0 | 422.0 |
| gaussian_noise_sev3/ramp_10 | 0.0 | 382.0 | 472.0 | 462.0 | 422.0 |
| gaussian_noise_sev3/ramp_20 | 132.0 | 372.0 | 402.0 | 412.0 | 362.0 |
| motion_blur_sev1/ramp_5 | 492.0 | 492.0 | 502.0 | 492.0 | 492.0 |
| motion_blur_sev1/ramp_10 | 482.0 | 472.0 | 482.0 | 482.0 | 482.0 |
| motion_blur_sev1/ramp_20 | 462.0 | 442.0 | 450.0 | 452.0 | 452.0 |

## 4. Cases where the CI could not be computed / special handling

- **95% CI unattainable for every n = 5 group** (not data-dependent): the exact two-sided 95% signed-rank CI does not exist at n = 5; the reported intervals are exact at the closest achievable level (93.75%). For gaussian_noise_sev3/ramp_10, the 1 tie leaves m = 4 nonzero differences and the closest achievable coverage is 87.5%.
- **Degenerate CIs**: groups whose 5 differences are identical (fog_sev4/ramp_5 and ramp_10, frosted_glass_blur_sev4/ramp_5, motion_blur_sev1/ramp_5) have all 15 Walsh averages equal → CI collapses to a point at the exact level; coverage statements still hold formally.
- **Ties**: 1 zero difference overall (gaussian_noise_sev3/ramp_10, seed 42). Dropped for the Wilcoxon test (zero_method='wilcox', matching scipy's default) and for the exact CI (m = 4); kept in prop_win and Cliff's delta as a tie (counts 0).
- **scipy note**: `scipy.stats.wilcoxon` could not be imported (broken `scipy.sparse` wheel in the venv); the exact signed-rank p-value (per group) and the normal-approximation p-value (pooled, n = 60 — scipy's own choice for n ≥ 50) were computed directly from the same formulas. Pooled W and p are therefore reported with exact treatment of mid-ranks and the continuity correction scipy applies.

## 5. Ready-to-use sentences (English)

- **fog_sev4/ramp_5**: "Hodges–Lehmann median downtime reduction: 492 s (93.8% exact CI [492, 492]), Cliff's delta = 1.00 (large), agent wins 5/5."
- **fog_sev4/ramp_10**: "Hodges–Lehmann median downtime reduction: 482 s (93.8% exact CI [472, 482]), Cliff's delta = 1.00 (large), agent wins 5/5."
- **fog_sev4/ramp_20**: "Hodges–Lehmann median downtime reduction: 405 s (93.8% exact CI [400, 430]), Cliff's delta = 1.00 (large), agent wins 5/5."
- **frosted_glass_blur_sev4/ramp_5**: "Hodges–Lehmann median downtime reduction: 492 s (93.8% exact CI [492, 492]), Cliff's delta = 1.00 (large), agent wins 5/5."
- **frosted_glass_blur_sev4/ramp_10**: "Hodges–Lehmann median downtime reduction: 482 s (93.8% exact CI [482, 492]), Cliff's delta = 1.00 (large), agent wins 5/5."
- **frosted_glass_blur_sev4/ramp_20**: "Hodges–Lehmann median downtime reduction: 451 s (93.8% exact CI [440, 460]), Cliff's delta = 1.00 (large), agent wins 5/5."
- **gaussian_noise_sev3/ramp_5**: "Hodges–Lehmann median downtime reduction: 447 s (93.8% exact CI [292, 482]), Cliff's delta = 1.00 (large), agent wins 5/5."
- **gaussian_noise_sev3/ramp_10**: "Hodges–Lehmann median downtime reduction: 422 s (87.5% exact CI [382, 472]), Cliff's delta = 0.80 (large), agent wins 4/5."
- **gaussian_noise_sev3/ramp_20**: "Hodges–Lehmann median downtime reduction: 372 s (93.8% exact CI [132, 412]), Cliff's delta = 1.00 (large), agent wins 5/5."
- **motion_blur_sev1/ramp_5**: "Hodges–Lehmann median downtime reduction: 492 s (93.8% exact CI [492, 502]), Cliff's delta = 1.00 (large), agent wins 5/5."
- **motion_blur_sev1/ramp_10**: "Hodges–Lehmann median downtime reduction: 482 s (93.8% exact CI [472, 482]), Cliff's delta = 1.00 (large), agent wins 5/5."
- **motion_blur_sev1/ramp_20**: "Hodges–Lehmann median downtime reduction: 452 s (93.8% exact CI [442, 462]), Cliff's delta = 1.00 (large), agent wins 5/5."
- **Pooled**: "Pooled Hodges–Lehmann median downtime reduction: 462 s (≈95% CI [450, 476]), Cliff's delta = 0.98 (large), agent wins 59/60 episodes; Wilcoxon signed-rank one-sided p = 1.2e-11 (W+ = 1770, m = 59, 1 tie dropped)."
