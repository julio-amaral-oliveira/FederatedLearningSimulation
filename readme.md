# Simulador de Aprendizado Federado

Simulador de aprendizado federado com abordagens sincronas e assincronas,
suporte a multiplos datasets, modelos CNN e ferramentas para comparar
resultados por tempo simulado.

O projeto foi desenvolvido no contexto de iniciacao cientifica FAPESP na
UNICAMP.

## Mapa dos experimentos

O projeto contém sete familias de experimento. Leia o [catalogo de
experimentos](docs/experimentos.md) antes de executar um script em
`experiments/` ou comparar resultados.

O registro operacional em `experiments/registry.py` define os IDs e as raízes
em `results/`. Ele não move resultados por conta própria.

- E01 a E04: simulador federado estatico, ablação, timeout e comparação Sync
  contra Async.
- E05: drift temporal sazonal com alternância de grupos de classes. É um
  estudo histórico.
- E06: primeiro protótipo do drift agent. É legado.
- E07: drift agent endurecido, com calibração, pares auditados, matriz de
  seeds, controles e horizonte fixo. É o fluxo atual de drift.

O [Relatório Final FAPESP](report/Relat%C3%B3rio_Final_FAPESP.pdf) é o documento
de origem do E01. Os relatórios posteriores registram a evolução dos estudos e
não substituem a configuração persistida de cada experimento.

Os comandos de simulação base deste README cobrem E01 a E04. Use as
[receitas rápidas](docs/receitas_rapidas.md) para exemplos adicionais e o
[runbook do E07](docs/literatura/15.%20handoff-experimento-drift-agent.md) para
o experimento de drift atual.

A calibração de origem do E07 está em
`output/cifar-10/severity-calibration/`. Sua cópia publicada está em
`results/e07-drift-agent/cifar-10/calibration/`. A matriz de origem está em
`output/cifar-10/drift-agent-2/final-matrix-rerun/`. Sua cópia publicada está
em `results/e07-drift-agent/cifar-10/uniform/matrix-v1/`.

## O que este repositorio faz

- Simulacao de FL sincrono com FedAvg por rodada.
- Simulacao de FL assincrono com agregacao on-the-fly e penalidade por staleness.
- Particionamento IID e Non-IID.
- Timeout por percentis via Monte Carlo.
- Tempo virtual para comparar cenarios sem depender do wall-clock real.
- Heterogeneidade de clientes por tiers de velocidade.
- Avaliacao esparsa com `--eval-every` para reduzir custo em execucoes longas.
- Geracao automatica de graficos de acuracia.
- Comparacao de cenarios por CLI e por UI Streamlit.
- Estudo de ablacao para parametros do servidor assincrono.
- Drift temporal e drift visual com monitoramento e retreino reativo.

## Datasets suportados

Os scripts principais aceitam `--dataset` com:

- `cifar10`
- `mnist`
- `fashion_mnist`
- `gtsrb`

Os entrypoints organizados usam `output/<experiment-id>/<dataset>/` por
default. Os diretórios abaixo continuam como defaults legados da camada de
dados quando ela é usada diretamente:

- `cifar10` -> `output-cifar-10`
- `mnist` -> `output-mnist`
- `fashion_mnist` -> `output-fashion-mnist`
- `gtsrb` -> `output-gtsrb`

## Estrutura do projeto

```text
.
├── src/
│   ├── synchronous/
│   │   ├── main.py
│   │   ├── server.py
│   │   ├── client.py
│   │   ├── constants.py
│   │   └── monte_carlo.py
│   ├── asynchronous/
│   │   ├── main.py
│   │   ├── server.py
│   │   ├── client.py
│   │   ├── constants.py
│   │   └── monte_carlo.py
│   └── utils/
│       ├── data_loader.py
│       ├── data_split.py
│       ├── models.py
│       └── plot_accuracy.py
├── experiments/
│   ├── ablation_study.py
│   ├── plot_ablation.py
│   ├── comparison_core.py
│   ├── compare_results.py
│   ├── comparison_ui.py
│   ├── temporal_drift.py
│   ├── smoke_drift.py
│   ├── run_smoke_drift.py
│   ├── severity_calibration.py
│   └── plot_smoke_drift.py
├── docs/
├── checkpoints/
└── output-*/
```

## Requisitos

Python 3.10+ recomendado.

Pacotes principais:

- `numpy`
- `scipy`
- `matplotlib`
- `torch`
- `torchvision`
- `pillow`
- `streamlit`

Arquivo de dependencias:

- `requirements.txt`

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

Instalacao rapida com venv:

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Guia de comandos curtos:

- [Mapa dos experimentos](docs/experimentos.md)
- [Índice dos relatórios](report/README.md)
- [Receitas rápidas](docs/receitas_rapidas.md)
- [Runbook do E07](docs/literatura/15.%20handoff-experimento-drift-agent.md)

## Como executar

Todos os comandos abaixo assumem execucao na raiz do repositorio.

### 1) Simulacao sincrona

Comportamento:

- Se nenhum flag de distribuicao for passado, roda IID e Non-IID.
- Por padrao roda percentis 25, 50 e 75.
- `--include-no-timeout` adiciona um cenario sem timeout.
- `--eval-every` controla a frequencia de avaliacao e escrita na curva.
- O tempo salvo em `time` e tempo virtual, nao wall-clock real.

Comandos:

```bash
python src/synchronous/main.py --experiment-id e01-static
python src/synchronous/main.py --experiment-id e01-static --dataset mnist --iid
python src/synchronous/main.py --experiment-id e01-static --dataset fashion_mnist --non-iid
python src/synchronous/main.py --experiment-id e01-static --dataset gtsrb --iid --percentile 50
python src/synchronous/main.py --experiment-id e03-timeout --dataset cifar10 --include-no-timeout
python src/synchronous/main.py --experiment-id e04-static-comparison --dataset cifar10 --iid --percentile 75 --num-rounds 5000 --eval-every 10 --output-prefix compare_5000_p75_sync
```

Argumentos principais:

- `--num-clients INT`
- `--num-rounds INT`
- `--epochs INT`
- `--batch-size INT`
- `--dataset {cifar10,mnist,fashion_mnist,gtsrb}`
- `--iid`
- `--non-iid`
- `--percentile INT`
- `--include-no-timeout`
- `--output-prefix STR`
- `--output-dir PATH` (default: `output/<experiment-id>/<dataset>/`)
- `--experiment-id ID`
- `--eval-every INT`

### 2) Simulacao assincrona

Comportamento:

- Se nenhum flag de distribuicao for passado, roda IID e Non-IID.
- Por padrao roda percentis 25, 50 e 75.
- Permite ajustar hiperparametros da agregacao assincrona.
- `--eval-every` controla a frequencia de avaliacao e escrita na curva.
- O servidor usa simulacao por eventos discretos com heap de eventos.

Comandos:

```bash
python src/asynchronous/main.py --experiment-id e01-static
python src/asynchronous/main.py --experiment-id e01-static --dataset mnist --percentile 50
python src/asynchronous/main.py --experiment-id e01-static --dataset gtsrb --non-iid --num-updates 40
python src/asynchronous/main.py --experiment-id e02-ablation --base-alpha 0.5 --decay-of-base-alpha 0.99 --tardiness-sensivity 0.1
python src/asynchronous/main.py --experiment-id e04-static-comparison --dataset cifar10 --iid --percentile 75 --num-updates 5000 --eval-every 10 --output-prefix compare_5000_p75_async
```

Argumentos principais:

- `--num-clients INT`
- `--num-updates INT`
- `--epochs INT`
- `--batch-size INT`
- `--dataset {cifar10,mnist,fashion_mnist,gtsrb}`
- `--iid`
- `--non-iid`
- `--percentile INT`
- `--base-alpha FLOAT`
- `--decay-of-base-alpha FLOAT`
- `--tardiness-sensivity FLOAT`
- `--output-prefix STR`
- `--output-dir PATH` (default: `output/<experiment-id>/<dataset>/`)
- `--experiment-id ID`
- `--eval-every INT`
- `--stop-on-stability`
- `--target-accuracy FLOAT`

### 3) Estudo de ablacao

Entrada organizada: `experiments/e02_ablation/run.py`

O estudo varia um parametro por vez no cenario assincrono e executa IID e
Non-IID.

Comandos:

```bash
python -m experiments.e02_ablation.run
python -m experiments.e02_ablation.run --num-updates 80 --percentile 75
python -m experiments.e02_ablation.run --num-clients 20 --epochs 1 --batch-size 32
```

### 4) Graficos e comparacao de resultados

Graficos de acuracia gerais:

```bash
python -m utils.plot_accuracy --output-dir output/e01-static/cifar-10
python -m utils.plot_accuracy --output-dir output/e01-static/mnist --non-iid --x-label atualizacoes
```

Graficos do estudo de ablacao:

```bash
python -m experiments.e02_ablation.plot --distribution iid --vary all
python -m experiments.e02_ablation.plot --distribution all --vary all --percentile 50
```

Comparacao entre cenarios:

```bash
python -m experiments.e04_static_comparison.compare \
  --scenario "Sync IID p75=output/e04-static-comparison/cifar-10/accuracy_data_iid_compare_5000_eval10_p75_sync.json#75" \
  --scenario "Async IID p75=output/e04-static-comparison/cifar-10/accuracy_data_iid_compare_5000_eval10_p75_async.json#75" \
  --target-accuracy 0.50 \
  --target-accuracy 0.60 \
  --horizon-seconds 4000 \
  --horizon-seconds 5000 \
  --title "Sync vs Async - CIFAR-10 IID p75" \
  --output output/comparison_iid_p75_5000
```

UI simples em Streamlit:

```bash
python -m streamlit run experiments/comparison_ui.py
```

Mais detalhes:

- `docs/comparacao_resultados.md`

## Benchmark estimado via Monte Carlo

Para estimar tempo sem rodar treino completo, use os mesmos modulos Monte Carlo
do projeto.

### Sincrono

No sincrono, o modulo retorna timeout por rodada. Uma estimativa simples de
tempo total e:

```text
tempo_total_estimado ~= timeout_por_rodada * NUM_UPDATES
```

Comando:

```bash
python -c "from src.synchronous.monte_carlo import get_percentiles_timeout;from src.synchronous.constants import MIN_CONNECTION_TIME,MAX_CONNECTION_TIME,SPEED_TIERS,NUM_UPDATES;ps=[25,50,75];ts=get_percentiles_timeout(ps,MIN_CONNECTION_TIME,MAX_CONNECTION_TIME,SPEED_TIERS);print({f'p{p}':{'timeout_por_rodada_s':float(t),'tempo_total_estimado_s':float(t*NUM_UPDATES)} for p,t in zip(ps,ts)})"
```

### Assincrono

No assincrono, o modulo retorna timeout total para `NUM_UPDATES`.

```bash
python -c "from src.asynchronous.monte_carlo import get_percentiles_timeout;from src.asynchronous.constants import MIN_CONNECTION_TIME,MAX_CONNECTION_TIME,SPEED_TIERS,NUM_UPDATES;ps=[25,50,75];ts=get_percentiles_timeout(ps,NUM_UPDATES,MIN_CONNECTION_TIME,MAX_CONNECTION_TIME,SPEED_TIERS);print({f'p{p}':{'tempo_total_estimado_s':float(t)} for p,t in zip(ps,ts)})"
```

## Formato das saidas

Os resultados de treino sao salvos em JSON:

```json
{
  "75": [
    {"loss": 2.302, "accuracy": 0.098, "time": 125.45}
  ]
}
```

Campos:

- `loss`: cross-entropy no conjunto de teste.
- `accuracy`: acuracia no conjunto de teste.
- `time`: tempo virtual simulado em segundos.

Convencao de nome:

- `accuracy_data_{iid|non_iid}.json`
- `accuracy_data_{iid|non_iid}_{output_prefix}.json`

## Modelos usados

Factory centralizada em `src/utils/models.py`:

- `cnn_cifar10`
- `cnn_mnist`
- `cnn_fashion_mnist`
- `cnn_gtsrb`

O mapeamento dataset -> modelo e feito em `src/utils/data_loader.py`.

## Boas praticas

- Evite versionar resultados massivos novos em `output-*`.
- Prefira `--percentile 50` e `--eval-every 10` em testes rapidos.
- Reduza `--num-rounds` ou `--num-updates` durante debug inicial.
- Use `--output-prefix` para separar experimentos sem sobrescrever arquivos.
- Use `experiments/compare_results.py` para comparacoes reproduziveis.
- Use a UI apenas como atalho; a logica oficial esta em `comparison_core.py`.

## Troubleshooting rapido

- Erro de memoria/tempo: reduza `--num-updates`, `--num-rounds` e/ou `--num-clients`.
- Erro com GTSRB: confira permissao de escrita em `data/gtsrb` para download/extracao.
- Sem graficos: confirme se os JSONs estao no diretorio passado em `--output-dir`.
- Erro de `pandas` na UI: a tela de metricas usa Markdown para evitar depender de `st.dataframe`.
