# QA comparison

## Entradas

- Smoke UDD: `output-mnist/smoke_drift.json#T_drift_800_udd_smoke`
- Temporal Sync: `output-mnist/accuracy_data_iid_qa_temporal_small_T100_sync.json#T_drift_100_p75_sync`

## Parametros

- Alvos de acuracia: 50%
- Horizontes: 500s

## Metricas

| scenario | path | key | n_evals | time_total_min | acc_initial | acc_final | max_acc | avg_last10 | tail_std | delta_peak_final | loss_final | time_to_90pct_peak | area_under_accuracy_time | time_to_acc_50 | acc_at_500s | max_acc_until_500s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Smoke UDD | output-mnist/smoke_drift.json | T_drift_800_udd_smoke | 106 | 26.7398 | 0.4732 | 0.4792 | 0.6689 | 0.4790 | 0.0005 | 0.1897 | 6.5649 | 815.7054 | 783.8773 | 77.4329 | 0.5094 | 0.5099 |
| Temporal Sync | output-mnist/accuracy_data_iid_qa_temporal_small_T100_sync.json | T_drift_100_p75_sync | 16 | 3.4990 | 0.5107 | 0.4829 | 0.5132 | 0.4877 | 0.0000 | 0.0303 | 4.4984 | 15.7054 | 96.4850 | 15.7054 | 0.4829 | 0.5132 |

## Comparacoes Objetivas

- Maior max_acc: Smoke UDD (0.6689)
- Maior acc_final: Temporal Sync (0.4829)
- Menor queda pico-final: Temporal Sync (0.0303)
- Menor tempo ate 50%: Temporal Sync (15.7054)
- Maior max_acc ate 500s: Temporal Sync (0.5132)
