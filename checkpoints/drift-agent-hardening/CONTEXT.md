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

`experiments/plot_smoke_drift.py`: aceita um par validado e produz a
visualização temporal.

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
