# Comparacao de Resultados - IID - MNIST - 75% de Conexão

## Entradas

- Sync IID p75: `output-mnist\accuracy_data_iid_compare_5000_eval10_p75_sync.json#75`
- Async IID p75: `output-mnist\accuracy_data_iid_compare_5000_eval10_p75_async.json#75`

## Parametros

- Alvos de acuracia: 95%, 98%
- Horizontes: 5000s

## Metricas

| scenario | path | key | n_evals | time_total_min | acc_initial | acc_final | max_acc | avg_last10 | tail_std | delta_peak_final | loss_final | time_to_90pct_peak | area_under_accuracy_time | time_to_acc_95 | time_to_acc_98 | acc_at_5000s | max_acc_until_5000s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Sync IID p75 | output-mnist\accuracy_data_iid_compare_5000_eval10_p75_sync.json | 75 | 33 | 83.7619 | 0.9293 | 0.9953 | 0.9953 | 0.9950 | 0.0003 | 0.0000 | 0.0241 | 15.7054 | 4975.7667 | 157.0536 | 157.0536 | 0.9953 | 0.9953 |
| Async IID p75 | output-mnist\accuracy_data_iid_compare_5000_eval10_p75_async.json | 75 | 1170 | 83.7619 | 0.9422 | 0.9937 | 0.9941 | 0.9937 | 0.0000 | 0.0004 | 0.0199 | 1.3574 | 4990.4452 | 7.4082 | 30.2146 | 0.9937 | 0.9941 |

## Comparacoes Objetivas

- Maior max_acc: Sync IID p75 (0.9953)
- Maior acc_final: Sync IID p75 (0.9953)
- Menor queda pico-final: Sync IID p75 (0.0000)
- Menor tempo ate 95%: Async IID p75 (7.4082)
- Menor tempo ate 98%: Async IID p75 (30.2146)
- Maior max_acc ate 5000s: Sync IID p75 (0.9953)
