# Comparacao de Resultados - IID - CIFAR-10 - 75% de Conexão

## Entradas

- Sync IID p75: `output-cifar-10\accuracy_data_iid_compare_5000_eval10_p75_sync.json#75`
- Async IID p75: `output-cifar-10\accuracy_data_iid_compare_5000_eval10_p75_async.json#75`

## Parametros

- Alvos de acuracia: 70%, 60%
- Horizontes: 5000s

## Metricas

| scenario | path | key | n_evals | time_total_min | acc_initial | acc_final | max_acc | avg_last10 | tail_std | delta_peak_final | loss_final | time_to_90pct_peak | area_under_accuracy_time | time_to_acc_70 | time_to_acc_60 | acc_at_5000s | max_acc_until_5000s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Sync IID p75 | output-cifar-10\accuracy_data_iid_compare_5000_eval10_p75_sync.json | 75 | 33 | 83.7619 | 0.2863 | 0.7786 | 0.7803 | 0.7786 | 0.0012 | 0.0017 | 1.0315 | 471.1609 | 3781.4524 | 471.1609 | 157.0536 | 0.7786 | 0.7803 |
| Async IID p75 | output-cifar-10\accuracy_data_iid_compare_5000_eval10_p75_async.json | 75 | 1166 | 83.5002 | 0.3190 | 0.7481 | 0.7481 | 0.7481 | 0.0001 | 0.0000 | 0.7859 | 116.5270 | 3715.8495 | 154.9727 | 60.3719 | 0.7481 | 0.7481 |

## Comparacoes Objetivas

- Maior max_acc: Sync IID p75 (0.7803)
- Maior acc_final: Sync IID p75 (0.7786)
- Menor queda pico-final: Async IID p75 (0.0000)
- Menor tempo ate 70%: Async IID p75 (154.9727)
- Menor tempo ate 60%: Async IID p75 (60.3719)
- Maior max_acc ate 5000s: Sync IID p75 (0.7803)
