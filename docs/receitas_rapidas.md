# Receitas Rapidas

Comandos curtos para executar cenarios comuns sem precisar lembrar todos os
argumentos.

Todos os comandos assumem execucao na raiz do repositorio.

## Preparacao

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Rodar simulacao sincrona

```bash
# CIFAR-10, IID + Non-IID, percentis 25/50/75
python src/synchronous/main.py

# MNIST, apenas IID
python src/synchronous/main.py --dataset mnist --iid

# GTSRB, apenas Non-IID, p50
python src/synchronous/main.py --dataset gtsrb --non-iid --percentile 50

# CIFAR-10 com cenario sem timeout
python src/synchronous/main.py --dataset cifar10 --include-no-timeout

# Teste rapido com menos rounds
python src/synchronous/main.py --dataset cifar10 --iid --percentile 50 --num-rounds 10 --eval-every 1 --output-prefix smoke_sync

# Execucao longa com avaliacao mais esparsa
python src/synchronous/main.py --dataset cifar10 --iid --percentile 75 --num-rounds 5000 --eval-every 10 --output-prefix compare_5000_eval10_p75_sync
```

## Rodar simulacao assincrona

```bash
# Padrao
python src/asynchronous/main.py

# Fashion-MNIST, Non-IID, p50
python src/asynchronous/main.py --dataset fashion_mnist --non-iid --percentile 50

# Ajuste de hiperparametros assincronos
python src/asynchronous/main.py --base-alpha 0.5 --decay-of-base-alpha 0.99 --tardiness-sensivity 0.1

# Menos atualizacoes para teste rapido
python src/asynchronous/main.py --num-updates 20 --percentile 50

# Execucao longa com avaliacao mais esparsa
python src/asynchronous/main.py --dataset cifar10 --iid --percentile 75 --num-updates 5000 --eval-every 10 --output-prefix compare_5000_eval10_p75_async

# Parada por estabilidade
python src/asynchronous/main.py --dataset cifar10 --non-iid --stop-on-stability --stability-patience 60 --stability-eval-every 10 --output-prefix stop_on_stability

# Parada por acuracia alvo
python src/asynchronous/main.py --dataset cifar10 --iid --target-accuracy 0.70 --output-prefix target_70
```

## Temporal Drift

Experimento de concept drift: a cada `T_drift` segundos virtuais, as classes
ativas alternam entre Grupo A (0-4) e Grupo B (5-9). O modelo precisa
reaprender o grupo ativo a cada fase.

```bash
# Fase longa: 200s por fase, 15 fases (3000s total)
python experiments/temporal_drift.py --mode both --T-drift 200 --num-phases 15 --eval-every 10

# Fases curtas variadas: 3 cenarios com T=10,40,70, 10 fases cada
python experiments/temporal_drift.py --mode both --T-drift "10,40,70" --num-phases "10,10,10" --eval-every 10
```

### Gerar graficos do drift

```bash
# Um cenario (single)
python experiments/plot_drift.py --mode single --json output-cifar-10/accuracy_data_iid_T200_sync.json --target-accuracy 0.50,0.60

# Comparacao sync vs async
python experiments/plot_drift.py --mode compare --sync-json output-cifar-10/accuracy_data_iid_T200_sync.json --async-json output-cifar-10/accuracy_data_iid_T200_async.json --target-accuracy 0.50,0.60 --title "Sync vs Async T=200"
```

## Estudo de ablacao

```bash
# Default
python experiments/ablation_study.py

# Mais atualizacoes
python experiments/ablation_study.py --num-updates 80 --percentile 50
```

## Gerar graficos

```bash
# Graficos gerais
python -m utils.plot_accuracy --output-dir output-cifar-10

# Graficos gerais Non-IID para MNIST
python -m utils.plot_accuracy --output-dir output-mnist --non-iid --x-label atualizacoes

# Graficos de ablacao
python experiments/plot_ablation.py --distribution all --vary all --percentile 50
```

## Comparar resultados

O comparador aceita dois ou mais cenarios explicitos. O label e apenas o nome
que aparece no grafico e nos relatorios.

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

Saidas geradas:

```text
output/comparison_iid_p75_5000.md
output/comparison_iid_p75_5000.csv
output/comparison_iid_p75_5000.png
```

## UI de comparacao

```bash
python -m streamlit run experiments/comparison_ui.py
```

Campos principais:

- `Titulo`: aparece no Markdown e no grafico.
- `Numero de cenarios`: quantidade de curvas.
- `Label`: nome legivel da curva.
- `Arquivo JSON`: caminho do resultado.
- `Chave`: chave interna do JSON, como `75`; pode ficar vazia se o JSON tiver uma unica chave.
- `Alvos de acuracia`: lista como `0.50, 0.60`.
- `Horizontes em segundos`: lista como `4000, 5000`.
- `Saida`: caminho base sem extensao.

## Benchmark estimado via Monte Carlo

### Sincrono

```bash
python -c "from src.synchronous.monte_carlo import get_percentiles_timeout;from src.synchronous.constants import MIN_CONNECTION_TIME,MAX_CONNECTION_TIME,SPEED_TIERS,NUM_UPDATES;ps=[25,50,75];ts=get_percentiles_timeout(ps,MIN_CONNECTION_TIME,MAX_CONNECTION_TIME,SPEED_TIERS);print({f'p{p}':{'timeout_por_rodada_s':float(t),'tempo_total_estimado_s':float(t*NUM_UPDATES)} for p,t in zip(ps,ts)})"
```

### Assincrono

```bash
python -c "from src.asynchronous.monte_carlo import get_percentiles_timeout;from src.asynchronous.constants import MIN_CONNECTION_TIME,MAX_CONNECTION_TIME,SPEED_TIERS,NUM_UPDATES;ps=[25,50,75];ts=get_percentiles_timeout(ps,NUM_UPDATES,MIN_CONNECTION_TIME,MAX_CONNECTION_TIME,SPEED_TIERS);print({f'p{p}':{'tempo_total_estimado_s':float(t)} for p,t in zip(ps,ts)})"
```

Observacoes:

- Sincrono: o timeout do modulo e por rodada.
- Assincrono: o timeout do modulo ja retorna total para `NUM_UPDATES`.
- O campo `time` salvo nos JSONs e tempo virtual simulado.
