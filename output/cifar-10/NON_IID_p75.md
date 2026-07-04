# Comparacao de Resultados - NON-IID - CIFAR-10 - 75% de Conexão

## Entradas

- Sync NON-IID p75: `output-cifar-10\accuracy_data_non_iid_compare_5000_eval10_p75_sync.json#75`
- Async NON-IID p75: `output-cifar-10\accuracy_data_non_iid_compare_5000_eval10_p75_async.json#75`

## Parametros

- Alvos de acuracia: 50%, 60%
- Horizontes: 5000s

## Metricas

| scenario | path | key | n_evals | time_total_min | acc_initial | acc_final | max_acc | avg_last10 | tail_std | delta_peak_final | loss_final | time_to_90pct_peak | area_under_accuracy_time | time_to_acc_50 | time_to_acc_60 | acc_at_5000s | max_acc_until_5000s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Sync NON-IID p75 | output-cifar-10\accuracy_data_non_iid_compare_5000_eval10_p75_sync.json | 75 | 33 | 83.7619 | 0.1093 | 0.6003 | 0.6171 | 0.5983 | 0.0093 | 0.0168 | 1.6879 | 2669.9116 | 2440.9801 | 2355.8044 | 3612.2334 | 0.6079 | 0.6171 |
| Async NON-IID p75 | output-cifar-10\accuracy_data_non_iid_compare_5000_eval10_p75_async.json | 75 | 1166 | 83.5002 | 0.1000 | 0.5366 | 0.5372 | 0.5364 | 0.0003 | 0.0006 | 1.4860 | 219.5958 | 2611.3472 | 270.1578 | N/A | 0.5366 | 0.5372 |

## Comparacoes Objetivas

- Maior max_acc: Sync NON-IID p75 (0.6171)
- Maior acc_final: Sync NON-IID p75 (0.6003)
- Menor queda pico-final: Async NON-IID p75 (0.0006)
- Menor tempo ate 50%: Async NON-IID p75 (270.1578)
- Menor tempo ate 60%: Sync NON-IID p75 (3612.2334)
- Maior max_acc ate 5000s: Sync NON-IID p75 (0.6171)
