# Comparacao de Resultados - NON-IID - FASHION MNIST - 75% de Conexão

## Entradas

- Sync NON-IID p75: `output-fashion-mnist\accuracy_data_non_iid_compare_5000_eval10_p75_sync.json#75`
- Async NON-IID p75: `output-fashion-mnist\accuracy_data_non_iid_compare_5000_eval10_p75_async.json#75`

## Parametros

- Alvos de acuracia: 80%, 70%
- Horizontes: 5000s

## Metricas

| scenario | path | key | n_evals | time_total_min | acc_initial | acc_final | max_acc | avg_last10 | tail_std | delta_peak_final | loss_final | time_to_90pct_peak | area_under_accuracy_time | time_to_acc_80 | time_to_acc_70 | acc_at_5000s | max_acc_until_5000s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Sync NON-IID p75 | output-fashion-mnist\accuracy_data_non_iid_compare_5000_eval10_p75_sync.json | 75 | 33 | 83.7619 | 0.3771 | 0.8427 | 0.8857 | 0.8461 | 0.0222 | 0.0430 | 0.7815 | 1256.4290 | 4067.9417 | 1256.4290 | 314.1073 | 0.8121 | 0.8857 |
| Async NON-IID p75 | output-fashion-mnist\accuracy_data_non_iid_compare_5000_eval10_p75_async.json | 75 | 1170 | 83.7619 | 0.1000 | 0.8132 | 0.8411 | 0.8132 | 0.0013 | 0.0279 | 0.5484 | 92.3707 | 4022.1877 | 112.3374 | 43.8089 | 0.8132 | 0.8411 |

## Comparacoes Objetivas

- Maior max_acc: Sync NON-IID p75 (0.8857)
- Maior acc_final: Sync NON-IID p75 (0.8427)
- Menor queda pico-final: Async NON-IID p75 (0.0279)
- Menor tempo ate 80%: Async NON-IID p75 (112.3374)
- Menor tempo ate 70%: Async NON-IID p75 (43.8089)
- Maior max_acc ate 5000s: Sync NON-IID p75 (0.8857)
