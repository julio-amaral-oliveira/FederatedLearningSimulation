# Comparacao de Resultados - IID - FASHION MNIST - 75% de Conexão

## Entradas

- Sync IID p75: `output-fashion-mnist\accuracy_data_iid_compare_5000_eval10_p75_sync.json#75`
- Async IID p75: `output-fashion-mnist\accuracy_data_iid_compare_5000_eval10_p75_async.json#75`

## Parametros

- Alvos de acuracia: 80%, 90%
- Horizontes: 5000s

## Metricas

| scenario | path | key | n_evals | time_total_min | acc_initial | acc_final | max_acc | avg_last10 | tail_std | delta_peak_final | loss_final | time_to_90pct_peak | area_under_accuracy_time | time_to_acc_80 | time_to_acc_90 | acc_at_5000s | max_acc_until_5000s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Sync IID p75 | output-fashion-mnist\accuracy_data_iid_compare_5000_eval10_p75_sync.json | 75 | 33 | 83.7619 | 0.7396 | 0.9359 | 0.9361 | 0.9350 | 0.0010 | 0.0002 | 0.3224 | 157.0536 | 4651.0018 | 157.0536 | 314.1073 | 0.9350 | 0.9361 |
| Async IID p75 | output-fashion-mnist\accuracy_data_iid_compare_5000_eval10_p75_async.json | 75 | 1170 | 83.7619 | 0.7570 | 0.9287 | 0.9291 | 0.9287 | 0.0000 | 0.0004 | 0.2136 | 16.5679 | 4657.2182 | 7.4082 | 85.1454 | 0.9287 | 0.9291 |

## Comparacoes Objetivas

- Maior max_acc: Sync IID p75 (0.9361)
- Maior acc_final: Sync IID p75 (0.9359)
- Menor queda pico-final: Sync IID p75 (0.0002)
- Menor tempo ate 80%: Async IID p75 (7.4082)
- Menor tempo ate 90%: Async IID p75 (85.1454)
- Maior max_acc ate 5000s: Sync IID p75 (0.9361)
