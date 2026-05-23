# Comparacao de Resultados

Use `experiments/compare_results.py` ou `experiments/comparison_ui.py` para
comparar dois ou mais JSONs de resultado de FL por tempo virtual simulado.

A regra principal e: o comparador nao infere significado pelo nome do arquivo.
O usuario informa explicitamente o label, o arquivo e, quando necessario, a
chave interna do JSON.

## CLI

```bash
python experiments/compare_results.py \
  --scenario "Sync IID p75=output-cifar-10/accuracy_data_iid_compare_5000_eval10_p75_sync.json#75" \
  --scenario "Async IID p75=output-cifar-10/accuracy_data_iid_compare_5000_eval10_p75_async.json#75" \
  --target-accuracy 0.50 \
  --target-accuracy 0.60 \
  --horizon-seconds 4000 \
  --horizon-seconds 5000 \
  --title "Sync vs Async - CIFAR-10 IID p75" \
  --output output/comparison_iid_p75_5000
```

Saidas sempre geradas:

```text
output/comparison_iid_p75_5000.md
output/comparison_iid_p75_5000.csv
output/comparison_iid_p75_5000.png
```

## Formato de `--scenario`

```text
--scenario "Label legivel=caminho/arquivo.json#chave"
```

Exemplos:

```bash
--scenario "Sync IID p75=output-cifar-10/sync.json#75"
--scenario "Async Non-IID p50=output-cifar-10/async.json#50"
```

Regras:

- O label e apenas o nome exibido no grafico, no CSV e no Markdown.
- Se o JSON for uma lista direta de avaliacoes, nao use `#chave`.
- Se o JSON tiver uma unica chave, por exemplo `{ "75": [...] }`, a chave pode ser omitida.
- Se o JSON tiver varias chaves, por exemplo `{ "25": [...], "50": [...] }`, informe `#25`, `#50` ou `#75`.
- O comando exige pelo menos dois cenarios.

## UI Streamlit

```bash
python -m streamlit run experiments/comparison_ui.py
```

Campos:

- `Titulo`: aparece no topo do Markdown e no titulo do grafico.
- `Numero de cenarios`: quantidade de curvas comparadas.
- `Label`: nome legivel da curva, por exemplo `Sync IID p75`.
- `Arquivo JSON`: caminho do arquivo de resultado.
- `Chave`: chave interna do JSON, por exemplo `75`; pode ficar vazia quando o JSON tem uma unica chave ou e uma lista direta.
- `Alvos de acuracia`: lista opcional, por exemplo `0.50, 0.60`.
- `Horizontes em segundos`: lista opcional, por exemplo `4000, 5000`.
- `Saida`: caminho base sem extensao para gerar `.md`, `.csv` e `.png`.

A UI chama o mesmo `comparison_core.py` usado pelo CLI. Portanto, os resultados
seguem as mesmas metricas e validacoes.

## Metricas Padrao

| Metrica | Significado |
|---|---|
| `n_evals` | Numero de avaliacoes registradas. |
| `time_total_min` | Tempo virtual final em minutos. |
| `acc_initial` | Primeira acuracia registrada. |
| `acc_final` | Ultima acuracia registrada. |
| `max_acc` | Maior acuracia atingida. |
| `avg_last10` | Media das ultimas 10 avaliacoes. |
| `tail_std` | Desvio padrao do trecho final da curva. |
| `delta_peak_final` | Diferenca entre pico e acuracia final. |
| `loss_final` | Ultima loss registrada. |
| `time_to_90pct_peak` | Tempo ate atingir 90% do proprio pico da curva. |
| `area_under_accuracy_time` | Area sob a curva acuracia x tempo virtual. |

## Metricas Opcionais

Use `--target-accuracy` para medir tempo ate uma acuracia alvo:

```bash
--target-accuracy 0.50 --target-accuracy 0.60
```

Isso cria colunas como:

```text
time_to_acc_50
time_to_acc_60
```

Use `--horizon-seconds` para medir desempenho ate um tempo virtual:

```bash
--horizon-seconds 4000 --horizon-seconds 5000
```

Isso cria colunas como:

```text
acc_at_4000s
max_acc_until_4000s
acc_at_5000s
max_acc_until_5000s
```

`acc_at_Xs` usa a ultima avaliacao observada ate o horizonte. Ja
`max_acc_until_Xs` usa o melhor valor observado ate o horizonte.

## Grafico

O PNG usa tempo virtual no eixo X e acuracia no eixo Y.

Por padrao, cada cenario mostra:

- curva bruta fina e translucida;
- curva suavizada por EMA por cima.

Opcoes:

```bash
--ema-alpha 0.1
--no-ema
```

Alvos de acuracia aparecem como linhas horizontais tracejadas. Horizontes de
tempo aparecem como linhas verticais tracejadas com rotulo.

## Markdown

O `.md` gerado contem:

- titulo configuravel;
- entradas usadas;
- parametros globais;
- tabela de metricas;
- comparacoes objetivas, como melhor `max_acc` e menor tempo ate alvo.

Ele nao tenta explicar causalidade. Interpretacoes sobre staleness,
heterogeneidade, FedAvg ou agregacao assincrona devem ser feitas em analise
separada, com literatura quando necessario.

## Valores Ausentes

`N/A` significa que a metrica nao existe para aquele cenario. Exemplos:

- o cenario nao atingiu uma acuracia alvo;
- nao havia avaliacao antes do horizonte informado.
