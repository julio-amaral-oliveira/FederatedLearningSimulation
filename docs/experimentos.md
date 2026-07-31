# Catálogo de Experimentos

Este documento identifica os experimentos do repositório e separa suas perguntas,
protocolos, resultados e códigos.

Leia este documento antes de executar um script em `experiments/`.
O mesmo nome, como `drift` ou `comparison`, aparece em estudos diferentes.

## Regra principal

Não compare resultados de estudos diferentes sem verificar o protocolo.

Os estudos usam perguntas diferentes:

- treinamento estático mede convergência e acurácia.
- timeout mede o efeito de clientes lentos no modo síncrono.
- ablação mede parâmetros da agregação assíncrona.
- drift temporal muda as classes disponíveis durante o treinamento.
- drift agent aplica corrupção visual durante a produção e pode retreinar o modelo.

Um resultado com `accuracy`, `loss` e `time` não pertence automaticamente ao
mesmo protocolo de um resultado com `downtime`, `drift_events` e `retrain_decisions`.

## Linha do tempo

| Período | Marco | Resultado da evolução |
|---|---|---|
| Fevereiro a março de 2026 | Simulador básico | Criação dos clientes, servidores, FedAvg, FedAsync, datasets e gráficos. |
| Março de 2026 | Ablação, generalização e timeout | Estudos separados sobre a fórmula assíncrona e sobre clientes lentos no modo síncrono. |
| Abril a maio de 2026 | Tempo virtual e comparação | Substituição gradual de esperas reais por eventos e tempo virtual. |
| Maio de 2026 | Drift temporal sazonal | Comparação Sync/Async com alternância de grupos de classes. |
| Maio a julho de 2026 | Drift agent | Criação de detecção local, retreino reativo e métrica de downtime. |
| 29 a 31 de julho de 2026 | Drift agent endurecido | Pareamento auditável, matriz de seeds, calibração, controles e schema v3. |

O histórico confirma essa sequência. O commit `7fded57` criou o drift temporal.
O commit `0540e21` criou o detector de drift e o monitoramento por incerteza.
O commit `413a59a` adicionou as corrupções e a calibração.
Os commits de `29/07/2026` endureceram o experimento do drift agent.
Os commits `852090e` e `b059545` atualizaram a corrupção Gaussian noise e os
gráficos do E07. Eles não criaram uma nova família de experimentos.

## Mapa operacional

### Estados dos scripts

| Estado | Significado |
|---|---|
| `atual` | Entrada oficial ou motor usado pelo experimento atual da família. |
| `suporte` | Módulo usado por uma entrada oficial. Ele não define um protocolo sozinho. |
| `histórico` | Código preservado para reproduzir ou entender um estudo anterior. |
| `legado` | Código ou interface preservado, mas que não deve iniciar uma nova execução. |
| `misturado` | Diretório que contém resultados de mais de uma execução ou política. |

O estado é relativo à família do experimento. Um script pode ser `atual` para
E01 e histórico para E07.

### Scripts e módulos

| Arquivo | Experimento | Estado | Papel |
|---|---|---|---|
| `src/synchronous/main.py` | E01, E03, E04 | `atual` | Entrada do simulador síncrono. |
| `src/asynchronous/main.py` | E01, E02, E04 | `atual` | Entrada do simulador assíncrono. |
| `src/orchestrator/orchestrator.py` | E06, E07 | `suporte` | Monitor local e decisão coletiva do drift agent. |
| `src/utils/drift.py` | E05 | `suporte` | Agenda de fases e mudança de classes. |
| `src/utils/drift_detector.py` | E06, E07 | `suporte` | Detector de drift baseado em incerteza. |
| `src/utils/plot_accuracy.py` | E01 a E04 | `suporte` | Gera gráficos de acurácia do simulador base. |
| `src/utils/corruptions.py` | E06, E07 | `suporte` | Implementa o contrato das corrupções visuais. |
| `experiments/ablation_study.py` | E02 | `histórico` | Executa a ablação da agregação assíncrona. |
| `experiments/plot_ablation.py` | E02 | `histórico` | Plota os resultados da ablação. |
| `experiments/compare_results.py` | E01, E03, E04 | `suporte` | Compara JSONs de acurácia do simulador base. |
| `experiments/comparison_core.py` | E01, E03, E04, E07 | `suporte` | Compartilha métricas e leitura de resultados. |
| `experiments/comparison_ui.py` | E04 | `histórico` | UI para comparação estática. |
| `experiments/temporal_drift.py` | E05 | `histórico` | Executa fases de drift temporal. |
| `experiments/plot_drift.py` | E05 | `histórico` | Plota o drift temporal. |
| `experiments/drift_ui.py` | E05 | `histórico` | UI para os resultados do drift temporal. |
| `experiments/smoke_drift.py` | E07 | `atual` | Motor de um episódio e comparação dos braços. A CLI direta é diagnóstica. |
| `experiments/run_smoke_drift.py` | E07 | `atual` | Entrada oficial da matriz de cenários e seeds. |
| `experiments/severity_calibration.py` | E07 | `atual` | Entrada oficial da calibração. |
| `experiments/drift_controls.py` | E07 | `suporte` | Controles `identity` e Oracle. |
| `experiments/drift_results.py` | E07 | `suporte` | Valida pares e resume resultados v2 e v3. |
| `experiments/result_io.py` | E07 | `suporte` | Publica artefatos de forma atômica. |
| `experiments/registry.py` | E01 a E07 | `suporte` | Registro operacional de IDs, estados e raízes em `results/`. |
| `experiments/migrate_outputs.py` | E07 | `suporte` | Planeja e publica cópias validadas de resultados. |
| `experiments/plot_smoke_drift.py` | E07 | `atual` | Entrada oficial da visualização individual ou agregada. |
| `experiments/smoke_drift_ui.py` | E06 | `legado` | UI para JSONs antigos `smoke_drift*.json`. |

Os novos namespaces em `experiments/eXX_*` são os pontos de entrada organizados.
Nesta etapa eles funcionam como adapters de compatibilidade para os módulos
planos existentes. A implementação ainda fica nos módulos antigos até que os
imports e os comandos históricos sejam migrados sem quebrar usuários.

Os módulos de dados, modelos e servidores são infraestrutura compartilhada.
Eles não definem sozinhos uma família de experimento.

### Diretórios de saída observados

| Diretório | Experimento | Estado | Conteúdo observado |
|---|---|---|---|
| `output-cifar-10/` | E04 e E05 | `misturado` | JSONs estáticos `compare_5000` e JSONs de drift com `T20`, `T200`, `T500` e outros. |
| `output-mnist/` | E06 | `histórico` | JSONs `smoke_drift*.json` e uma execução temporal de QA. |
| `output/<dataset>/` | E04 | `histórico` | CSVs, Markdown e PNGs de comparação estática. |
| `output/drift/T_<tempo>/` | E05 | `histórico` | Gráficos por fase e por grupo de classes. |
| `output/corrupted-drift/T_<tempo>/` | E05 | `histórico` | Variante antiga de drift temporal com corrupção visual. |
| `drift-agent/<cenário>/` | E06 | `legado` | Pares `agent.json` e `baseline.json` schema v2. |
| `output/cifar-10/severity-calibration/` | E07 | `atual` | Calibração oficial schema 1. Seleciona `gaussian_noise:3`, `frosted_glass_blur:4`, `motion_blur:1` e `fog:4`. |
| `output/cifar-10/drift-agent/severity-calibration/` | E07 | `histórico` | Destino documentado em uma versão anterior. Não é a fonte da calibração atual. |
| `output/cifar-10/drift-agent/final-matrix/` | E07 | `histórico` | Saída local anterior. Não use como matriz oficial. |
| `output/cifar-10/drift-agent-2/` | E07 | `atual` | Matriz científica importada. `final-matrix-rerun/` contém 25 pares v3. `omp-smoke/` fica separado. |
| `results/e07-drift-agent/cifar-10/calibration/` | E07 | `atual` | Cópia publicada e validada da calibração oficial. |
| `results/e07-drift-agent/cifar-10/uniform/matrix-v1/` | E07 | `atual` | Cópia publicada e validada dos 25 pares, gráficos e resumo. |
| `results/e07-drift-agent/cifar-10/uniform/smoke/` | E07 | `atual` | Cópia publicada e validada do smoke separado. |

Não use `output-cifar-10/` como uma população única. O diretório contém saídas
de E04 e E05. Não use `drift-agent/` junto com
`output/cifar-10/drift-agent-2/`. O primeiro contém schema v2 legado. O
segundo contém a matriz científica atual em schema v3.

A calibração de origem está em
`output/cifar-10/severity-calibration/severity_calibration.json`. A cópia
publicada está em `results/e07-drift-agent/cifar-10/calibration/`. A matriz de
origem está em `output/cifar-10/drift-agent-2/final-matrix-rerun/`. A cópia
publicada está em `results/e07-drift-agent/cifar-10/uniform/matrix-v1/`. Ela
contém quatro cenários visuais e o controle `identity:0`, com cinco seeds por
grupo. Isso produz 25 pares. O diretório `omp-smoke/` não faz parte da matriz.
Sua cópia publicada fica em `results/e07-drift-agent/cifar-10/uniform/smoke/`.

### Comandos oficiais por família

Use os comandos abaixo para iniciar uma execução da família correta.

| Experimento | Comando oficial | Saída principal |
|---|---|---|
| E01 | `python src/synchronous/main.py --experiment-id e01-static` ou `python src/asynchronous/main.py --experiment-id e01-static` | `output/e01-static/<dataset>/` |
| E02 | `python -m experiments.e02_ablation.run` | `output/e02-ablation/cifar-10/` e gráficos de ablação. |
| E03 | `python src/synchronous/main.py --experiment-id e03-timeout --include-no-timeout` | `output/e03-timeout/<dataset>/` |
| E04 | Entradas Sync/Async com `--experiment-id e04-static-comparison`, depois `python -m experiments.e04_static_comparison.compare ...` | `output/e04-static-comparison/<dataset>/` ou o caminho explícito. |
| E05 | `python -m experiments.e05_temporal_drift.run --mode both ...` | `output/e05-temporal-drift/<dataset>/`. |
| E06 | `python -m experiments.e06_drift_agent_prototype.prototype ...` | Episódio legado em `output/e06-drift-agent-prototype/<dataset>/`. |
| E07 calibração | `MPLCONFIGDIR=/private/tmp/fl-mpl .venv/bin/python -m experiments.e07_drift_agent.calibrate ...` | `output/e07-drift-agent/cifar-10/calibration/`. |
| E07 matriz | `MPLCONFIGDIR=/private/tmp/fl-mpl .venv/bin/python -m experiments.e07_drift_agent.run_matrix ...` | `output/e07-drift-agent/cifar-10/uniform/matrix-v1/`. |
| E07 plot | `MPLCONFIGDIR=/private/tmp/fl-mpl .venv/bin/python -m experiments.e07_drift_agent.plot --scenario-dir <dir> --output <png>` | PNG do cenário. |

Para E07, use o módulo com `-m`. O handoff oficial recomenda essa forma para
preservar a resolução do pacote `experiments`.

### Comandos históricos ou fora do fluxo atual

Não use estes comandos para iniciar o E07 atual:

- `python experiments/temporal_drift.py`. Ele executa E05.
- `python experiments/ablation_study.py`. Ele executa E02.
- `python experiments/smoke_drift_ui.py`. Ele lê o formato antigo de E06.
- `python experiments/compare_results.py` para pares agente/baseline v3. Use
  `experiments.drift_results` e `experiments.e07_drift_agent.plot` para E07.
- `experiments/plot_comparison.py`. O arquivo aparece em
  `handoff-7MZns2.md`, mas não existe no checkout atual.

As receitas rápidas identificam os comandos E02 e E05 como históricos. O
comando do E05 usa `--output-prefix compare`, por isso seus artefatos esperados
incluem `accuracy_data_iid_compare_T200_sync.json`. Não confunda esses JSONs
com os pares v3 do E07.

## E01. Treinamento federado estático

### Pergunta

Como o aprendizado federado síncrono e assíncrono converge em um cenário sem
mudança temporal na distribuição dos dados?

### Protocolo

- CIFAR-10, MNIST, Fashion-MNIST e GTSRB em parte dos estudos.
- Cenários IID e Non-IID.
- Clientes treinam durante rounds ou updates.
- O tempo virtual representa o custo de comunicação e treinamento.
- O modelo não recebe corrupção visual durante a produção.

### Métricas

- acurácia máxima e final.
- loss final.
- estabilidade da cauda.
- tempo até uma acurácia alvo.
- área sob a curva de acurácia e tempo.

### Código

- `src/synchronous/`.
- `src/asynchronous/`.
- `src/utils/data_loader.py`.
- `src/utils/data_split.py`.
- `src/utils/models.py`.
- `experiments/comparison_core.py`.
- `experiments/compare_results.py`.

### Estado

Este estudo forma a base do projeto. Os resultados históricos continuam úteis
para entender o simulador, mas não representam os protocolos de drift agent.

### Fontes

- [README principal](../readme.md).
- [Relatório Final FAPESP — origem do E01](../report/Relat%C3%B3rio_Final_FAPESP.pdf).
- [relatório inicial do simulador](../report/relatorio.pdf).
- [Resultados Experimentais do E01](../report/Resultados%20Experimentais.pdf).
- [comparação entre datasets](../report/comparacao_sync_async.pdf).
- [documentação da comparação](comparacao_resultados.md).

## E02. Ablação da agregação assíncrona

### Pergunta

Como os parâmetros da agregação assíncrona alteram a acurácia e a estabilidade?

### Protocolo

O estudo varia um parâmetro por vez na fórmula de agregação:

```text
agg_factor = alpha * beta^version * 1 / (1 + gamma * staleness)
```

Os parâmetros principais são:

- `alpha`, taxa de incorporação.
- `beta`, decaimento temporal.
- `gamma`, penalização por staleness.

O estudo usa CIFAR-10, 40 clientes, 40 updates por configuração e cenários IID
e Non-IID.

### Métricas

- acurácia máxima.
- média das últimas avaliações.
- desvio-padrão da cauda.
- comparação IID contra Non-IID.

### Código

- `experiments/ablation_study.py`.
- `experiments/plot_ablation.py`.
- `src/asynchronous/server.py`.
- `src/asynchronous/client.py`.
- `src/asynchronous/constants.py`.

### Estado

Este estudo é histórico. Ele explica a escolha de valores como `alpha=0.8`,
`beta=0.999` e `gamma=0.075` no simulador assíncrono.

Não use esses valores como parâmetros do drift agent. O drift agent tem sua
própria política de detecção, quorum, rounds e horizonte.

### Fontes

- [análise da ablação](ablation_analysis.md).
- [Resultados Experimentais](../report/resultados.pdf), seção de ablação.

## E03. Impacto do timeout síncrono

### Pergunta

Como a fração de clientes esperada pelo servidor síncrono altera o trade-off
entre tempo virtual e acurácia?

### Protocolo

- CIFAR-10.
- 40 clientes.
- 80 rounds.
- clientes com atraso de conexão e treinamento.
- percentis P25, P50, P75 e cenário sem timeout.
- cenários IID e Non-IID.

Na versão descrita no relatório inicial, P25, P50 e P75 representam frações de
participação no modo síncrono. Versões posteriores usam percentis para estimar
timeouts por Monte Carlo. Eles não representam o quorum do drift agent.

### Métricas

- acurácia por round.
- acurácia final e máxima.
- tempo por round.
- tempo total.
- acurácia por hora.
- estabilidade no cenário Non-IID.

### Código

- `src/synchronous/server.py`.
- `src/synchronous/monte_carlo.py`.
- `src/synchronous/constants.py`.
- `experiments/comparison_core.py`.

### Estado

Este estudo é histórico. Ele analisa o custo de esperar clientes lentos durante
o treinamento federado. Ele não mede detecção de drift nem recuperação.

### Fontes

- [análise do impacto do timeout](timeout_impact_analysis.md).
- [Resultados Experimentais](../report/resultados.pdf), seção de timeout.

## E04. Comparação Sync contra Async em cenário estático

### Pergunta

Qual modo oferece melhor acurácia e qual modo converge mais rápido quando o
dataset não muda durante a execução?

### Protocolo

- comparação entre FedAvg síncrono e agregação assíncrona.
- datasets MNIST, Fashion-MNIST e CIFAR-10.
- cenários IID e Non-IID.
- P75 no modo síncrono.
- `alpha=0.8`, `beta=0.999` e `gamma=0.075` no modo assíncrono.
- 5000 rounds ou updates.
- avaliação a cada 10 steps.

### Resultado de referência

No CIFAR-10 IID, o relatório registra que o síncrono alcança acurácia máxima
maior, enquanto o assíncrono atinge o pico mais rápido.

No CIFAR-10 Non-IID, a diferença de acurácia aumenta, mas o assíncrono mantém
vantagem de velocidade e estabilidade da cauda.

### Código

- `experiments/comparison_core.py`.
- `experiments/compare_results.py`.
- `experiments/comparison_ui.py`.
- `src/utils/plot_accuracy.py`.

### Estado

Este estudo é a comparação estática de referência. Não o confunda com o estudo
E05, que alterna grupos de classes, nem com E06 e E07, que aplicam corrupção
visual durante a produção.

### Fontes

- [relatório de comparação Sync/Async](../report/comparacao_sync_async.pdf).
- [documentação de comparação](comparacao_resultados.md).
- [diário de 20/05/2026](diario_2026-05-20.md).

## E05. Drift temporal sazonal Sync contra Async

### Pergunta

O modo assíncrono mantém vantagem quando a distribuição de classes muda durante
o treinamento?

### Protocolo

- CIFAR-10.
- grupo A com classes 0 a 4.
- grupo B com classes 5 a 9.
- fases alternadas no formato 100/0.
- teste completo com as 10 classes.
- valores de `T_drift` entre 20 e 1000 segundos.
- comparação Sync contra Async.

Este estudo muda os dados disponíveis em cada fase. Ele não injeta uma
corrupção visual em imagens de produção.

### Métricas

- plasticidade por grupo.
- forgetting.
- sustained accuracy.
- área sob a curva.
- tempo de recuperação.
- acurácia do grupo ativo e do grupo inativo.

### Código

- `experiments/temporal_drift.py`.
- `experiments/plot_drift.py`.
- `experiments/drift_ui.py`.
- `src/utils/drift.py`.

### Estado

Este estudo é histórico e independente do drift agent. Ele continua útil para
comparar Sync contra Async sob mudança temporal de classes.

### Fontes

- [desenho do drift temporal](literatura/4.%20desenho-experimento-drift-temporal.md).
- [resultados do drift temporal](literatura/5.%20resultados-drift-temporal.md).
- [relatório do drift temporal](../report/drift_temporal_sync_async.pdf).

## E06. Drift agent inicial

### Pergunta

Um detector local sem rótulos consegue identificar degradação visual e disparar
um retreino federado síncrono?

### Protocolo

1. Treinar o modelo com dados limpos.
2. Executar um warm-up limpo para o detector.
3. Aplicar corrupção visual durante a produção.
4. Usar MC Dropout, entropia e ADWIN para detectar mudança local.
5. Agregar os sinais dos clientes com uma decisão coletiva.
6. Executar um bloco de retreino no agente.
7. Comparar o agente com uma baseline sem retreino.

O detector recebe somente `x`. Os rótulos permanecem no treinamento e na
avaliação offline.

### Código

- `experiments/smoke_drift.py`.
- `experiments/smoke_drift_ui.py`.
- `src/orchestrator/orchestrator.py`.
- `src/utils/drift_detector.py`.
- `src/utils/corruptions.py`.
- `experiments/comparison_core.py`.

### Estado

Este estudo é um protótipo histórico. Ele provou o encadeamento técnico, mas
não ofereceu evidência estatística suficiente para uma conclusão científica.

O nome `smoke_drift.py` ficou inadequado. O arquivo ainda contém parte do fluxo
principal usado pelo experimento endurecido E07.

### Fontes

- [resumo do drift agent](literatura/6.%20Drift%20agent.md).
- [métodos de detecção](literatura/7.%20Metodos%20de%20deteccao%20de%20drift.md).
- [especificação do CIFAR corrompido](literatura/8.%20especificacao-drift-agent-cifar-corrompido.md).
- [especificação do fluxo síncrono](literatura/9.%20especificacao-fluxo-drift-agent-sincrono.md).

## E07. Drift agent endurecido e auditável

### Pergunta

O retreino reativo disparado por detecção local reduz o downtime em comparação
com uma baseline sem retreino, sob o mesmo cenário e horizonte?

### Protocolo oficial atual

- CIFAR-10 com todas as classes.
- treinamento federado síncrono.
- corrupção visual global e simultânea.
- detector local sem rótulos com MC Dropout, entropia e ADWIN.
- decisão coletiva por quorum.
- no máximo um bloco de retreino por episódio.
- baseline pareada sem retreino.
- horizonte de produção fixo.
- matriz explícita de seeds, cenários e sensibilidades.
- schema v3 com metadados e trace.
- validação do par agente/baseline.
- persistência atômica e manifest-last.

As corrupções atuais são `gaussian_noise`, `frosted_glass_blur`, `motion_blur`
e `fog`. Elas são inspiradas em CIFAR-10-C, mas não reproduzem o benchmark
oficial.

### Métricas

- `detection_delay_seconds`.
- `retraining_duration_seconds`.
- `time_to_recovery_seconds`.
- `downtime_seconds`.
- `clean_retention_delta`.
- `corrupted_accuracy_gain`.

### Código oficial

- `experiments/smoke_drift.py`: execução de um episódio e comparação dos braços.
- `experiments/run_smoke_drift.py`: matriz de seeds, cenários, controles e sensibilidades.
- `experiments/severity_calibration.py`: calibração das severidades.
- `experiments/drift_results.py`: leitura e validação de resultados.
- `experiments/result_io.py`: publicação atômica dos artefatos.
- `experiments/plot_smoke_drift.py`: visualização de um par ou de várias seeds.
- `src/orchestrator/orchestrator.py`: monitor local e decisão coletiva.
- `src/utils/corruptions.py`: contrato das corrupções.

### Estado

Este é o experimento de drift atual. Use o handoff e o checkpoint como fontes
operacionais. Não misture seus resultados com os resultados históricos E05 ou
com o piloto inicial E06.

O perfil `uniform` é o padrão do experimento atual. O perfil `heterogeneous` é
uma análise de sensibilidade. Resultados de perfis diferentes não formam uma
mesma população estatística.

### Fontes

- [endurecimento do experimento](literatura/13.%20endurecimento-experimento-drift-agent.md).
- [plano de implementação](literatura/14.%20plano-endurecimento-experimento-drift-agent.md).
- [handoff operacional](literatura/15.%20handoff-experimento-drift-agent.md).
- [estado atual](../checkpoints/drift-agent-hardening/STATE.md).
- [diário cronológico](../checkpoints/drift-agent-hardening/JOURNAL.md).

## Como identificar um resultado

Use esta sequência antes de abrir ou comparar um JSON.

1. Verifique se o resultado contém `downtime_seconds`, `drift_events` ou
   `retrain_decisions`. Se contiver, ele pertence a E06 ou E07.
2. Verifique se o resultado contém `phase`, `accuracy_A` ou `accuracy_B`. Se
   contiver, ele pertence a E05.
3. Verifique se o nome contém `alpha`, `beta`, `gamma` ou `staleness`. Se
   contiver, ele pertence a E02.
4. Verifique se o nome contém `timeout`, `p25`, `p50`, `p75` ou `no-timeout`.
   Se contiver, ele pertence a E03 ou a uma comparação estática relacionada.
5. Verifique se o resultado compara apenas curvas `{loss, accuracy, time}`.
   Nesse caso, ele pertence a E01 ou E04.

Quando houver dúvida, leia a configuração persistida no JSON. O nome do
arquivo não é suficiente para identificar o protocolo.

## O que não deve ser misturado

| Não misture | Motivo |
|---|---|
| E05 com E06/E07 | E05 muda classes por fase. E06/E07 aplicam corrupção visual em produção. |
| E06 com E07 | E06 é prova de conceito. E07 adiciona horizonte, calibração, pares, schema e validação. |
| E02 com E07 | E02 estuda a fórmula FedAsync. E07 usa o fluxo síncrono do drift agent. |
| `uniform` com `heterogeneous` | O perfil muda timeout, participação e tempo virtual. |
| schema v2 com schema v3 auditado | O schema v2 continua legível, mas não oferece as mesmas garantias de auditoria. |
| resultados de seeds diferentes sem configuração | A seed só pode ser agregada quando cenário, checkpoint, política e horizonte coincidem. |

## Plano de reorganização física

O catálogo já separa as perguntas e os protocolos. A árvore do repositório ainda
mistura códigos e resultados de famílias diferentes. Esta seção define o alvo
da reorganização. Ela não autoriza mover ou apagar artefatos nesta etapa.

### Problema observado

- `experiments/` mantém entradas de E02, E05, E06 e E07 no mesmo nível.
- `src/synchronous/` e `src/asynchronous/` servem E01, E03 e E04 sem uma
  entrada que mostre a família executada.
- `output-cifar-10/` mistura JSONs de E04 e E05.
- `output/` mistura gráficos de E04, E05 e saídas de E07.
- `drift-agent/` contém o formato v2 legado do E06.
- `output/cifar-10/` mistura E04 com o E07 atual.
- O `.gitignore` ignora as saídas. Uma mudança de caminho não aparece no
  histórico do Git sem um registro documentado.

### Layout alvo dos resultados

Use `output/` como área temporária e ignorada. Publique os resultados oficiais
em `results/`, que será versionado pelo Git. Use o ID do experimento como a
primeira pasta dentro de `results/`. Use o dataset dentro do experimento. Use
nomes de protocolo para as variações.

```text
results/
├── e01-static/
│   └── <dataset>/
├── e02-ablation/
│   └── cifar-10/
├── e03-timeout/
│   └── cifar-10/
├── e04-static-comparison/
│   └── <dataset>/
├── e05-temporal-drift/
│   └── cifar-10/
│       ├── raw/
│       └── plots/
├── e06-drift-agent-prototype/
│   └── <dataset>/
└── e07-drift-agent/
    └── cifar-10/
        ├── calibration/
        ├── uniform/
        │   ├── matrix/
        │   └── smoke/
        └── heterogeneous/
            └── sensitivity/
```

A cópia validada da matriz oficial do E07 está em
`results/e07-drift-agent/cifar-10/uniform/matrix-v1/`. A cópia validada da
calibração está em `results/e07-drift-agent/cifar-10/calibration/`. A fonte
original permanece em `output/cifar-10/drift-agent-2/final-matrix-rerun/` e
`output/cifar-10/severity-calibration/`. A cópia não altera a origem.

### Registro operacional

`experiments/registry.py` define o ID, o estado e a raiz de resultados de cada
experimento. `experiments/__init__.py` exporta `get_experiment()` e
`output_path()` para os novos entrypoints.

| ID | Estado | Raiz de resultados |
|---|---|---|
| `e01-static` | `historical` | `results/e01-static/` |
| `e02-ablation` | `historical` | `results/e02-ablation/` |
| `e03-timeout` | `historical` | `results/e03-timeout/` |
| `e04-static-comparison` | `historical` | `results/e04-static-comparison/` |
| `e05-temporal-drift` | `historical` | `results/e05-temporal-drift/` |
| `e06-drift-agent-prototype` | `legacy` | `results/e06-drift-agent-prototype/` |
| `e07-drift-agent` | `current` | `results/e07-drift-agent/` |

O registro define caminhos. Ele não substitui a configuração persistida, os
manifests ou a validação científica dos resultados.

Use `experiments/migrate_outputs.py` com `--dry-run` para revisar cada origem
e destino. Use `--copy` somente depois de revisar o plano. O modo de cópia
recusa destinos existentes, valida os digests e valida os pares publicados.

### Layout alvo do código de experimento

As entradas devem mostrar a família no caminho. Os módulos compartilhados
continuam fora dessas pastas até a migração de imports passar nos testes.

```text
experiments/
├── e01_static/
├── e02_ablation/
├── e03_timeout/
├── e04_static_comparison/
├── e05_temporal_drift/
├── e06_drift_agent_prototype/
├── e07_drift_agent/
└── shared/
```

Na primeira fase, cada pasta recebe um README curto e uma entrada explícita.
As entradas antigas podem permanecer como wrappers temporários. O wrapper deve
apontar para a entrada nova e deve mostrar uma mensagem de migração.

As entradas criadas nesta fase são adapters sem mensagem obrigatória: elas
preservam o comportamento dos comandos antigos e tornam a família visível no
comando. A mensagem e a remoção dos adapters ficam para uma etapa posterior,
depois da migração dos imports.

### Mapa de migração

| Origem atual | Destino planejado | Regra |
|---|---|---|
| `output-cifar-10/accuracy_data_iid_compare*` e `*_non_iid_compare*` | `results/e04-static-comparison/cifar-10/raw/` | Copiar e validar os JSONs como E04. |
| `output-cifar-10/accuracy_data_iid_T*` e `*_corrupted_*` | `results/e05-temporal-drift/cifar-10/raw/` | Copiar sem misturar com E04. |
| `output/drift/` e `output/corrupted-drift/` | `results/e05-temporal-drift/cifar-10/plots/` | Preservar a variante visual no nome ou em um índice. |
| `output/cifar-10/IID_p75.*` e `NON_IID_p75.*` | `results/e04-static-comparison/cifar-10/plots/` | Associar ao comando e aos JSONs de origem. |
| `drift-agent/` | `results/e06-drift-agent-prototype/` | Marcar como schema v2 legado. |
| `output-mnist/smoke_drift*.json` | `results/e06-drift-agent-prototype/mnist/` | Separar o QA temporal antes de copiar. |
| `output/cifar-10/severity-calibration/` | `results/e07-drift-agent/cifar-10/calibration/` | Cópia validada. Manter a origem e seu schema. |
| `output/cifar-10/drift-agent-2/final-matrix-rerun/` | `results/e07-drift-agent/cifar-10/uniform/matrix-v1/` | Cópia validada. Manter os 25 pares e os manifests. |
| `output/cifar-10/drift-agent-2/omp-smoke/` | `results/e07-drift-agent/cifar-10/uniform/smoke/` | Cópia validada e separada da matriz científica. |

### Ordem de execução

1. Criar as entradas, o registro operacional e os contratos de saída sem mover
   resultados.
2. Adicionar `--output-dir` explícito às entradas que ainda usam caminhos
   fixos. Novas execuções usam `output/<experiment-id>/<dataset>/`.
3. A cópia validada do E07 está publicada em `results/`. Mantenha a origem.
4. Atualizar comandos, README, receitas e links para os novos destinos.
5. Comparar contagens, schemas, manifests e digests entre origem e destino.
6. Marcar os caminhos antigos como históricos antes de qualquer remoção.

O E07 deve migrar primeiro porque possui o protocolo atual e artefatos oficiais.
E01 a E05 podem migrar depois, em ordem de dependência. O E06 deve permanecer
separado como protótipo histórico.

## Tratamento do código e dos relatórios

O código, os resultados e os relatórios têm funções diferentes. A reorganização
mantém essas funções separadas.

### Código em `src/`

`src/` contém o motor do simulador e módulos reutilizáveis. Ele permanece
versionado pelo Git e não deve ser copiado para cada experimento.

| Caminho | Papel | Famílias | Tratamento |
|---|---|---|---|
| `src/synchronous/` | Motor federado síncrono | E01, E03, E04, E05 e E07 | Manter como infraestrutura compartilhada. |
| `src/asynchronous/` | Motor federado assíncrono | E01, E02, E04 e E05 | Manter como infraestrutura compartilhada. |
| `src/utils/data_loader.py`, `data_split.py` e `models.py` | Dados, divisão e modelos | E01 a E07 | Manter como infraestrutura compartilhada. |
| `src/utils/experiment_runner.py`, `ema.py` e `plot_accuracy.py` | Auxiliares de execução e gráficos | E01 a E04 | Manter como suporte compartilhado. |
| `src/utils/drift.py` | Agenda de mudança de classes | E05 | Associar ao namespace do E05 em uma migração futura. |
| `src/orchestrator/orchestrator.py` | Monitor local e decisão coletiva | E06 e E07 | Associar ao suporte do drift agent sem duplicar o motor síncrono. |
| `src/utils/drift_detector.py` | Detector baseado em incerteza | E06 e E07 | Associar ao suporte do drift agent. |
| `src/utils/corruptions.py` | Corrupções visuais | E06 e E07 | Associar ao suporte do drift agent. |
| `src/federated_learning_sim.egg-info/` | Metadados gerados do pacote | Infraestrutura | Não classificar como código de experimento. |

As entradas e configurações específicas ficam em `experiments/eXX_*`. Os
imports antigos devem continuar funcionando durante a migração. Não mova os
motores compartilhados antes de validar os testes e os comandos oficiais.

### Relatórios em `report/`

`report/` contém documentos de origem, relatórios históricos e fontes LaTeX.
Ele não é uma pasta de saída de execução. Os relatórios devem permanecer
versionados pelo Git, com o PDF e seu `.tex` tratados como um par quando ambos
existirem. Consulte também o [índice dos relatórios](../report/README.md).

| Arquivo | Papel no mapa | Experimento | Estado |
|---|---|---|---|
| `report/Relatório_Final_FAPESP.pdf` | Documento que origina e motiva o estudo inicial | E01 | Origem do E01 |
| `report/relatorio.pdf` e `relatorio.tex` | Relatório inicial do simulador | E01 | Histórico do E01 |
| `report/Resultados Experimentais.pdf` | Primeira consolidação dos resultados do simulador | E01 e estudos iniciais | Histórico |
| `report/resultados.pdf` e `resultados.tex` | Resultados dos experimentos posteriores, incluindo ablação e timeout | E02 e E03 | Histórico |
| `report/comparacao_sync_async.pdf` e `comparacao_sync_async.tex` | Comparação estática entre Sync e Async | E04 | Histórico |
| `report/drift_temporal_sync_async.pdf` e `drift_temporal_sync_async.tex` | Comparação sob drift temporal de classes | E05 | Histórico |

O Relatório Final FAPESP é a fonte de origem do E01. Ele não é apenas um
contexto geral do projeto. Os relatórios posteriores mostram a evolução do
simulador e dos estudos. Eles não substituem a configuração persistida dos
experimentos nem a matriz oficial do E07.

A organização lógica futura de `report/` é:

```text
report/
├── e01-origin/
│   └── Relatório_Final_FAPESP.pdf
├── historical/
│   ├── e01/
│   ├── e02/
│   ├── e03/
│   ├── e04/
│   └── e05/
└── sources/
```

Não mova os PDFs ou os arquivos `.tex` antes de verificar os caminhos de
imagens e a compilação. Primeiro crie um índice de relatórios e associe cada
documento ao ID do experimento.

## Fonte operacional atual

Para executar o drift agent atual, siga [o handoff do E07](literatura/15.%20handoff-experimento-drift-agent.md).

Para entender a origem do E01, comece pelo
[Relatório Final FAPESP](../report/Relat%C3%B3rio_Final_FAPESP.pdf). Depois leia o
[relatório inicial](../report/relatorio.pdf) e os
[Resultados Experimentais](../report/Resultados%20Experimentais.pdf).

Para entender a evolução dos estudos, consulte:

1. [Novos Experimentos](../report/resultados.pdf).
2. [Comparação Sync/Async](../report/comparacao_sync_async.pdf).
3. [Drift Temporal Sync/Async](../report/drift_temporal_sync_async.pdf).
4. [Handoff do Drift Agent](literatura/15.%20handoff-experimento-drift-agent.md).

Os relatórios explicam a evolução das perguntas. A configuração persistida de
cada resultado continua sendo a fonte operacional do experimento correspondente.
