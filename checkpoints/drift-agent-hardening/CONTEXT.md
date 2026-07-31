# Contexto de domínio — drift-agent-hardening

> Glossário e fronteiras conceituais. Estado e decisões vigentes estão em
> [STATE.md](STATE.md).

## Termos

### Drift agent

- **Definição:** braço que monitora o stream, aplica a política coletiva e
  pode executar um bloco de retreino.
- **Não confundir com:** um detector local isolado.
- **Aliases ou nomes usados no código:** `agent`, `baseline=False`.

### Baseline sem retreino

- **Definição:** braço pareado que observa o mesmo cenário, registra o primeiro
  gatilho contrafactual e não executa retreino.
- **Não confundir com:** execução sem detector ou sem corrupção.
- **Aliases ou nomes usados no código:** `baseline`,
  `counterfactual_triggers`, `record_action=False`.

### Par auditado

- **Definição:** `agent.json` e `baseline.json` schema v3 que passaram pela
  validação interna e cruzada e pertencem ao mesmo manifesto.
- **Não confundir com:** dois arquivos que apenas possuem o mesmo nome de
  cenário.
- **Aliases ou nomes usados no código:** `validate_pair`,
  `audited_pair=True`, `pair-manifest.json`.

### Registro operacional

- **Definição:** mapa versionado que associa cada ID de experimento ao estado e
  à raiz publicada de resultados.
- **Não confundir com:** catálogo humano, configuração científica ou manifest
  de um par.
- **Alias ou nome usado no código:** `experiments.registry`.

### Dry-run de migração

- **Definição:** listagem de todos os arquivos e destinos planejados sem criar,
  copiar, mover ou apagar arquivos.
- **Não confundir com:** uma cópia validada ou uma publicação em `results/`.
- **Alias ou nome usado no código:** `experiments.migrate_outputs`.

### Namespace de experimento

- **Definição:** caminho Python que torna a família do experimento visível na
  entrada, como `experiments.e07_drift_agent.run_matrix`.
- **Não confundir com:** o módulo compartilhado que implementa o motor ou com
  a raiz onde os resultados são publicados.
- **Papel:** oferecer um seam estável para migrar imports sem quebrar os
  comandos planos antigos.

### Raiz temporária de execução

- **Definição:** diretório em `output/` usado por uma nova execução antes de
  sua validação e publicação.
- **Não confundir com:** a raiz oficial em `results/` ou com uma origem
  histórica importada.
- **Alias ou nome usado no código:** `temporary_output_path`.

### Raiz legada de saída

- **Definição:** diretório criado pela convenção antiga antes da identificação
  dos experimentos por namespace.
- **Exemplos:** `output-cifar-10/` e `output-mnist/`.
- **Estado atual:** as raízes foram migradas e não são válidas para novas
  execuções.

### Resultado histórico bruto

- **Definição:** arquivo produzido por uma execução antiga e preservado para
  rastreabilidade.
- **Não confundir com:** uma publicação oficial atual ou um par E07 auditado.
- **Local:** subdiretório `raw/`, `legacy/` ou `legacy-v2/` em `results/`.

### QA cruzado

- **Definição:** comparação ou verificação que usa dados de duas famílias de
  experimentos.
- **Não confundir com:** resultado de uma única família.
- **Local:** `results/qa/cross-experiment/`.

### UDD

- **Definição:** detector de drift baseado em incerteza: múltiplos forwards
  com Dropout, entropia e ADWIN.
- **Não confundir com:** threshold fixo de entropia ou avaliação por
  acurácia/rótulo.
- **Aliases ou nomes usados no código:** `detector_kind="udd"`,
  `UDDDetector`.

### MC Dropout

- **Definição:** múltiplas inferências com Dropout ativo usadas para estimar
  incerteza preditiva.
- **Não confundir com:** treinamento adicional.
- **Aliases ou nomes usados no código:** `detector_T`, `T`.

### ADWIN

- **Definição:** detector adaptativo que compara segmentos de uma janela de
  scores e emite mudança.
- **Não confundir com:** quorum do servidor.
- **Aliases ou nomes usados no código:** `detector_alpha`, `alpha`.

### Detecção local

- **Definição:** cada cliente avalia seu batch sem rótulos e produz score/flag.
- **Não confundir com:** decisão global de retreino.
- **Aliases ou nomes usados no código:** `drift_events`, flags por cliente.

### Ação coletiva

- **Definição:** o servidor observa a fração recente de clientes com flag e
  decide uma única ação quando o quorum é atingido.
- **Não confundir com:** o evento ADWIN de um cliente.
- **Aliases ou nomes usados no código:** `trigger_threshold`,
  `trigger_window_ticks`, `retrain_decisions`.

### Warm-up

- **Definição:** observações limpas anteriores à produção usadas para formar a
  referência do detector.
- **Não confundir com:** os rounds de treinamento inicial.
- **Aliases ou nomes usados no código:** `warmup_ticks`,
  `warmup_completed_time`.

### Produção

- **Definição:** período após o warm-up no qual os batches de monitoramento e
  a avaliação usam o cenário configurado.
- **Aliases ou nomes usados no código:** `production_start_time`,
  `drift_onset`.

### Horizonte de produção

- **Definição:** duração virtual fixa observada após o início da produção.
- **Não confundir com:** wall-clock do programa ou duração do retreino.
- **Aliases ou nomes usados no código:** `production_horizon_seconds`,
  `end_time_seconds`.

### Tempo virtual

- **Definição:** relógio do simulador derivado das durações virtuais de
  conexão/treino e dos ticks.
- **Não confundir com:** minutos reais transcorridos no terminal.
- **Aliases ou nomes usados no código:** `server.virtual_time`.

### Wall-clock

- **Definição:** tempo real consumido por Python, CPU, GPU, I/O e
  sincronizações.
- **Não confundir com:** downtime ou duração virtual de round.

### Round federado

- **Definição:** distribuição dos pesos globais, fits locais dos clientes
  participantes e agregação.
- **Não confundir com:** um único batch ou um único optimizer step. Um round
  contém muitos batches e vários fits locais.
- **Aliases ou nomes usados no código:** `run_one_round`, `run_rounds`.

### Perfil de velocidade dos clientes

- **Definição:** distribuição das faixas de duração virtual de treino.
  O simulador usa essas faixas para calcular o timeout e a duração de cada
  cliente.
- **Não confundir com:** wall-clock real, seed do experimento ou quantidade
  de rounds.
- **Aliases ou nomes usados no código:** `client_speed_profile`,
  `client_speed_tiers`, `SPEED_PROFILES`.

### Perfil uniforme

- **Definição:** todos os clientes usam o tier virtual
  `("uniform", 0, 10, 1.0)`.
- **Limite:** o simulador ainda gera durações diferentes para os clientes.
  O timeout p75 não garante a participação de todos.
- **Aliases ou nomes usados no código:** `client_speed_profile="uniform"`.

### Perfil heterogêneo

- **Definição:** os clientes usam os tiers `fast`, `medium`, `slow` e
  `very_slow`. Cada tier tem uma faixa de duração virtual.
- **Não confundir com:** distribuição não-IID dos dados. A heterogeneidade
  temporal e a heterogeneidade estatística são eixos diferentes.
- **Aliases ou nomes usados no código:**
  `client_speed_profile="heterogeneous"`.

### Covariate drift visual

- **Definição:** mudança em `P(X)` causada por transformação das imagens, com
  semântica de `Y` preservada.
- **Não confundir com:** pure concept drift, no qual muda somente `P(Y|X)`.

### Cenário inspirado em CIFAR-10-C

- **Definição:** corrupção própria com nomes/parâmetros do projeto.
- **Não confundir com:** reprodução do dataset ou implementação oficial
  CIFAR-10-C.
- **Aliases ou nomes usados no código:** `gaussian_noise`,
  `frosted_glass_blur`, `motion_blur`, `fog`.

### Severidade

- **Definição:** nível interno inteiro `1..5` de uma corrupção visual.
- **Não confundir com:** escala comparável entre corrupções diferentes.

### Calibração

- **Definição:** treino de um checkpoint limpo e avaliação da tabela completa
  corrupção × severidade.
- **Não confundir com:** escolher a severidade que produz o melhor resultado
  do agente.
- **Aliases ou nomes usados no código:** `severity_calibration.json`,
  `calibrate_clean_checkpoint`.

### Severidade selecionada

- **Definição:** menor severidade cuja acurácia corrompida fica estritamente no
  intervalo predefinido `(0.25, 0.50)`.
- **Não confundir com:** severidade escolhida depois de observar downtime.
- **Aliases ou nomes usados no código:** `selected`.

### Tau

- **Definição:** limiar operacional de acurácia abaixo do qual o serviço é
  considerado indisponível.
- **Não confundir com:** threshold do detector ou quorum.
- **Aliases ou nomes usados no código:** `tau`, atualmente 0.50.

### Downtime

- **Definição:** integral stepwise do tempo virtual em que a acurácia
  corrompida está estritamente abaixo de `tau`.
- **Não confundir com:** wall-clock de treinamento.
- **Aliases ou nomes usados no código:** `downtime_seconds`.

### Atraso de detecção

- **Definição:** tempo virtual entre onset do drift e decisão coletiva.
- **Aliases ou nomes usados no código:** `detection_delay_seconds`.

### Duração do retreino

- **Definição:** tempo virtual entre início do primeiro e término do último
  round do bloco.
- **Não confundir com:** tempo até recuperação.
- **Aliases ou nomes usados no código:** `retraining_duration_seconds`.

### Sensibilidade de rounds

- **Definição:** comparação de valores de `retrain_rounds`. A comparação
  mantém cenário, seed, quorum, horizonte e perfil de velocidade.
- **Não confundir com:** replicação entre seeds. Mais rounds não garantem
  acurácia maior.

### Tempo até recuperação

- **Definição:** tempo entre a decisão e a primeira avaliação posterior com
  acurácia maior ou igual a `tau`.
- **Limite:** quando nunca houve acurácia abaixo de `tau`, o valor não
  representa recuperação de indisponibilidade.
- **Aliases ou nomes usados no código:** `time_to_recovery_seconds`.

### Ganho corrompido

- **Definição:** acurácia corrompida final menos acurácia no onset, dentro do
  mesmo braço.
- **Não confundir com:** diferença de downtime entre braços.
- **Aliases ou nomes usados no código:** `corrupted_accuracy_gain`.

### Retenção limpa

- **Definição:** acurácia limpa final menos acurácia limpa pré-drift.
- **Aliases ou nomes usados no código:** `clean_retention_delta`.

### Checkpoint limpo

- **Definição:** estado produzido depois dos rounds iniciais e antes da
  produção corrompida.
- **Aliases ou nomes usados no código:** `clean_checkpoint_digest`.

### Schema v2

- **Definição:** formato legado ainda legível.
- **Limite:** não contém evidência suficiente para pareamento auditado.
- **Aliases ou nomes usados no código:** `legacy_unverified_pair=True`.

### Schema v3

- **Definição:** formato atual com configuração, detector, runtime, trace,
  ações e invariantes temporais.
- **Aliases ou nomes usados no código:** `schema_version=3`.

### Manifest-last

- **Definição:** protocolo no qual os dois payloads são preparados/publicados
  e o manifesto com digests é publicado por último.
- **Não confundir com:** atomicidade transacional de um banco de dados.
- **Aliases ou nomes usados no código:** `pair-manifest.json`,
  `save_validated_pair`, `load_persisted_pair`.

### Controle Oracle

- **Definição:** controle que dispara no primeiro tick de produção e mede o
  caminho de adaptação quando o evento é conhecido.
- **Não confundir com:** desempenho do detector real ou falso positivo.
- **Aliases ou nomes usados no código:** `OracleMonitor`,
  `population="oracle_control"`, `trigger_policy="oracle"`.

### Identity

- **Definição:** transformação que preserva a entrada; no schema v3 usa
  severidade 0.
- **Aliases ou nomes usados no código:** `identity_corruption`,
  `identity_sev0`.

### Controle identity com detector real

- **Definição:** cenário limpo que mantém o detector real para medir falso
  positivo.
- **Limite atual:** é uma distinção do desenho; `--include-controls` gera o
  controle Oracle, não este controle.
- **Aliases ou nomes usados no código:** `population="identity_control"` quando
  construído explicitamente.

### Seed

- **Definição:** identificador da realização estocástica usada no cenário e no
  pareamento.
- **Não confundir com:** replicação estatística suficiente; uma seed é uma
  observação.

### Quorum

- **Definição:** fração de clientes sinalizados necessária para ação coletiva.
- **Não confundir com:** threshold ADWIN ou `tau`.
- **Aliases ou nomes usados no código:** `trigger_threshold`.

### Resumo particionado

- **Definição:** agregação apenas entre runs com dimensões experimentais
  idênticas, normalmente variando seeds.
- **Não confundir com:** média global de todos os controles e corrupções.
- **Aliases ou nomes usados no código:** `summary.json`, `groups`,
  `dimensions`.

### Catálogo de experimentos

- **Definição:** mapa que identifica perguntas, protocolos, métricas, código e
  fontes de cada estudo do repositório.
- **Fonte:** `docs/experimentos.md`.
- **Não confundir com:** o estado atual do drift agent ou o diário de decisões.

### ID de experimento

- **Definição:** rótulo humano estável de um estudo, como `E01` ou `E07`.
- **Limite:** os IDs ainda não fazem parte dos nomes dos artefatos persistidos.
- **Fonte:** `docs/experimentos.md`.

### Treinamento federado estático

- **Definição:** execução sem mudança temporal de classes ou corrupção visual
  durante a produção.
- **Não confundir com:** drift temporal ou drift agent visual.
- **ID do catálogo:** `E01` e `E04`.

### Ablação assíncrona

- **Definição:** estudo que varia `alpha`, `beta` e `gamma` da agregação
  assíncrona.
- **Não confundir com:** sensibilidade de quorum ou rounds do drift agent.
- **ID do catálogo:** `E02`.

### Impacto do timeout

- **Definição:** estudo do custo de esperar ou descartar clientes durante
  rounds síncronos.
- **Não confundir com:** quorum de detecção coletiva.
- **ID do catálogo:** `E03`.

### Drift temporal sazonal

- **Definição:** alternância de grupos de classes durante o treinamento, com
  fases definidas por `T_drift`.
- **Não confundir com:** corrupção visual de imagens em produção.
- **ID do catálogo:** `E05`.

### Drift agent inicial

- **Definição:** primeira implementação do fluxo de detecção local, ação
  coletiva, retreino e baseline.
- **Limite:** prova de conceito sem as garantias completas de pareamento,
  calibração e auditoria do fluxo atual.
- **ID do catálogo:** `E06`.

### Drift agent endurecido

- **Definição:** fluxo síncrono atual com horizonte fixo, matriz de seeds,
  calibração, controles, schema v3 e validação de pares.
- **Não confundir com:** o protótipo inicial do drift agent.
- **ID do catálogo:** `E07`.

### Estado de script

- **Definição:** classificação operacional de um arquivo executável ou módulo.
- **Valores:** `atual`, `suporte`, `histórico` e `legado`.
- **Limite:** o estado é relativo à família do experimento. Um arquivo pode ser
  atual para uma família e histórico para outra.
- **Fonte:** `docs/experimentos.md`.

### Comando oficial

- **Definição:** comando documentado para iniciar ou visualizar um experimento
  dentro do protocolo aceito.
- **Não confundir com:** qualquer CLI que ainda exista no checkout.
- **Fonte atual do E07:** `python -m experiments.e07_drift_agent.calibrate`,
  `python -m experiments.e07_drift_agent.run_matrix` e
  `python -m experiments.e07_drift_agent.plot`. Os caminhos planos continuam
  disponíveis durante a migração.

### Documento histórico

- **Definição:** documentação preservada para explicar ou reproduzir um estudo
  anterior, sem autorizar uma nova execução no protocolo atual.
- **Exemplos:** `handoff-7MZns2.md` e as seções E02 e E05 das receitas rápidas.
- **Não confundir com:** o runbook operacional do E07.
- **Fonte:** `docs/experimentos.md`.

### Diretório de saída misturado

- **Definição:** diretório que contém artefatos de mais de uma família ou
  protocolo.
- **Exemplo:** `output-cifar-10/` contém resultados de E04 e E05.
- **Limite:** não agregue os arquivos apenas porque eles compartilham a pasta.

### Saída v2 legada

- **Definição:** par de resultados do drift agent anterior ao contrato v3.
- **Exemplo:** pares em `drift-agent/<cenário>/`.
- **Não confundir com:** a matriz v3 oficial em
  `output/cifar-10/drift-agent-2/final-matrix-rerun/`.

### Importação externa de artefatos

- **Definição:** resultados gerados em outra máquina e copiados para o
  checkout local para auditoria ou análise.
- **Exemplo atual:** `output/cifar-10/drift-agent-2/`.
- **Papel atual:** contém a matriz científica oficial em
  `final-matrix-rerun/`. A calibração oficial fica em
  `output/cifar-10/severity-calibration/`.
- **Regra:** manter `omp-smoke/` separado e registrar o ambiente e a
  proveniência antes de agregar os resultados.

### ID do experimento

- **Definição:** identificador estável de uma família experimental, como E01
  ou `e07-drift-agent`.
- **Não confundir com:** seed, severidade, cenário ou nome de uma execução.
- **Papel:** separar protocolo, código de entrada e raiz de resultados.

### Layout de execução

- **Definição:** estrutura temporária que coloca o ID do experimento antes do
  dataset e das variações de protocolo.
- **Formato:** `output/<experiment-id>/<dataset>/`.
- **Não confundir com:** o layout canônico de publicação em `results/`.

### Layout canônico de publicação

- **Definição:** estrutura versionada que coloca o ID do experimento antes do
  dataset e recebe somente artefatos validados.
- **Formato:** `results/<experiment-id>/<dataset>/`.
- **Não confundir com:** uma raiz temporária ou histórica que apenas contém a
  execução mais recente.

### Raiz de resultados

- **Definição:** diretório base no qual uma entrada publica seus artefatos
  depois da validação.
- **Exemplos atuais:** `results/e07-drift-agent/cifar-10/` e suas subpastas
  `calibration/`, `uniform/matrix-v1/` e `uniform/smoke/`.
- **Regra:** cada raiz deve identificar uma única família experimental.

### Migração não destrutiva

- **Definição:** cópia validada de uma raiz antiga para uma raiz canônica sem
  apagar a origem.
- **Requisitos:** comparar contagens, schemas, manifests e digests antes de
  atualizar o caminho oficial.

### Saída histórica

- **Definição:** artefato preservado para reprodução ou contexto, mas que não
  deve ser usado como padrão para uma nova execução.
- **Não confundir com:** uma saída inválida. Uma saída histórica pode ser
  tecnicamente válida dentro do protocolo que a gerou.

### Documento de origem do experimento

- **Definição:** documento que apresenta a motivação, a pergunta inicial e o
  ponto de partida de uma família experimental.
- **Exemplo atual:** `report/Relatório_Final_FAPESP.pdf` é o documento de origem
  do E01.
- **Não confundir com:** um relatório posterior de resultados ou a configuração
  persistida de uma execução.

### Relatório experimental

- **Definição:** documento que consolida resultados, decisões ou evolução de um
  ou mais estudos.
- **Limite:** um relatório pode explicar o protocolo, mas a configuração e os
  artefatos persistidos continuam sendo a fonte operacional do resultado.

## Atores e sistemas

### Cliente federado

Mantém dataset e modelo local, executa fit local e inferência MC Dropout. No
fluxo atual, clientes participantes são treinados sequencialmente pelo
servidor.

### Servidor síncrono

Distribui pesos, amostra durações virtuais, executa os clientes participantes,
agrega pesos e avança o relógio virtual.

### DriftMonitor

Coordena detectores locais, registra o trace, calcula a fração sinalizada e
mantém o latch de uma única ação.

### Controlador do episódio

`experiments/smoke_drift.py`: constrói/restaura checkpoints, executa warm-up e
produção, aplica corrupção, retreina e produz payloads.

### Runner da matriz

`experiments/run_smoke_drift.py`: materializa configurações, executa pares,
controles e resumo, sem depender de subprocessos.

### Validador e persistência

`experiments/drift_results.py` valida semântica e pareamento;
`experiments/result_io.py` publica e recarrega artefatos.

### Calibrador

`experiments/severity_calibration.py`: treina um checkpoint e avalia todas as
severidades solicitadas.

### Plot

`experiments/plot_smoke_drift.py`: aceita um par validado ou um diretório de
cenário e produz uma visualização temporal individual ou agregada.

## Fronteiras importantes

- Detecção usa somente entradas; treinamento e avaliação podem usar rótulos.
- Evento local e decisão global são níveis diferentes.
- Acurácia e downtime são métricas offline, não entradas do detector.
- O experimento atual observa covariate drift global, não drift heterogêneo ou
  concept drift puro.
- Tempo virtual mede a simulação; wall-clock mede custo computacional.
- Smoke técnico demonstra integração; piloto real demonstra uma realização;
  matriz com seeds e dispersão sustenta análise experimental.
- Um par auditado prova consistência interna, não validade externa da hipótese.
- Controles Oracle, identity com detector e cenários visuais respondem
  perguntas diferentes e não devem compartilhar o mesmo agregado.
- Valores de política devem ser reportados como escolhas e submetidos a
  sensibilidade, não atribuídos à literatura.
- Melhorar performance não autoriza alterar silenciosamente configuração,
  seeds, horizonte, detector ou métrica.
