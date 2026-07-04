# Comparacao de Resultados - NON-IID - MNIST - 75% de Conexão

## Entradas

- Sync NON-IID p75: `output-mnist\accuracy_data_non_iid_compare_5000_eval10_p75_sync.json#75`
- Async NON-IID p75: `output-mnist\accuracy_data_non_iid_compare_5000_eval10_p75_async.json#75`

## Parametros

- Alvos de acuracia: 80%, 90%
- Horizontes: 5000s

## Metricas

| scenario | path | key | n_evals | time_total_min | acc_initial | acc_final | max_acc | avg_last10 | tail_std | delta_peak_final | loss_final | time_to_90pct_peak | area_under_accuracy_time | time_to_acc_80 | time_to_acc_90 | acc_at_5000s | max_acc_until_5000s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Sync NON-IID p75 | output-mnist\accuracy_data_non_iid_compare_5000_eval10_p75_sync.json | 75 | 33 | 83.7619 | 0.4655 | 0.9881 | 0.9907 | 0.9899 | 0.0008 | 0.0026 | 0.0394 | 157.0536 | 4903.2338 | 157.0536 | 157.0536 | 0.9902 | 0.9907 |
| Async NON-IID p75 | output-mnist\accuracy_data_non_iid_compare_5000_eval10_p75_async.json | 75 | 1170 | 83.7619 | 0.0892 | 0.9807 | 0.9807 | 0.9807 | 0.0001 | 0.0000 | 0.0602 | 38.7193 | 4893.9863 | 38.7193 | 43.8089 | 0.9807 | 0.9807 |

## Comparacoes Objetivas

- Maior max_acc: Sync NON-IID p75 (0.9907)
- Maior acc_final: Sync NON-IID p75 (0.9881)
- Menor queda pico-final: Async NON-IID p75 (0.0000)
- Menor tempo ate 80%: Async NON-IID p75 (38.7193)
- Menor tempo ate 90%: Async NON-IID p75 (43.8089)
- Maior max_acc ate 5000s: Sync NON-IID p75 (0.9907)
