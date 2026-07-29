# Estado atual — drift-agent-hardening

> Verdade vigente do projeto em 2026-07-29. Para a evolução e as evidências,
> consulte [JOURNAL.md](JOURNAL.md). Para terminologia e fronteiras, consulte
> [CONTEXT.md](CONTEXT.md).

## Objetivo

Executar um experimento federado síncrono, reproduzível e auditável que teste
se o retreino disparado por detecção local de covariate drift visual reduz o
downtime do modelo em relação a uma baseline sem retreino, sob o mesmo
checkpoint, stream e horizonte virtual.

## Escopo atual

### Incluído

- CIFAR-10 com todas as classes e corrupção visual global/simultânea.
- Treinamento federado síncrono.
- Detecção local sem rótulos com UDD: MC Dropout, entropia de Shannon e ADWIN.
- Decisão coletiva central por quorum e no máximo um bloco de retreino.
- Baseline pareada sem retreino.
- Horizonte de produção fixo, métricas temporais e retenção limpa.
- Corrupções próprias `gaussian_noise`, `frosted_glass_blur`, `motion_blur` e
  `fog`, apenas inspiradas em CIFAR-10-C.
- Calibração completa das severidades `1..5`.
- Matriz explícita de seeds, cenários e sensibilidades.
- Resultados schema v3, validação do par, persistência atômica, resumo
  particionado e plot.
- Controle Oracle descrito como `identity:0`.

### Adiado ou fora de escopo

- Servidor assíncrono no experimento atual.
- Pure concept drift, no qual muda somente `P(Y|X)`.
- Drift local/escalonado, clustering, múltiplos modelos e retreino por grupo.
- Múltiplos drifts, cooldown e retreinos recorrentes.
- Seleção de clientes por custo, impacto ou staleness.
- Champion/Challenger, Validation Gate, RL e política aprendida.
- Reprodução literal do benchmark CIFAR-10-C.
- Otimizações de wall-clock ainda não medidas e aprovadas.

## Estado do código

- Branch: `feature/drift-agent-hardening`.
- Head observado: `a573478` (`fix: validate identity control severity`).
- Base da implementação endurecida: `c9e9355`.
- A revisão final e a re-revisão concluíram **Ready**.
- Última verificação completa observada: 116 testes passando e 2 skips
  condicionais de CUDA/MPS; `compileall` e `git diff --check` passaram.
- Não houve push nem trailers `Co-authored-by`.
- Alterações locais preexistentes do usuário foram preservadas. Em especial,
  os documentos 11 e 12 estavam não rastreados e são contexto, não prova do
  estado commitado.

Referências principais:

- [Desenho do endurecimento](../../docs/literatura/13.%20endurecimento-experimento-drift-agent.md)
- [Plano de implementação](../../docs/literatura/14.%20plano-endurecimento-experimento-drift-agent.md)
- [Handoff e comandos](../../docs/literatura/15.%20handoff-experimento-drift-agent.md)

## Decisões vigentes

1. **Pareamento por estado completo e validação explícita.** Agente e baseline
   permanecem braços independentes. O checkpoint captura pesos, estado do
   servidor e RNGs de Python, NumPy, Torch CPU, CUDA e MPS quando disponíveis.
   O trace por tick permite rejeitar divergências em vez de escondê-las.
2. **Horizonte fixo.** O fluxo principal usa
   `production_horizon_seconds=400`. O fim deve ser exatamente
   `production_start_time + horizon`. Um retreino cujo limite conservador não
   cabe no restante do horizonte falha antes do primeiro round.
3. **Métricas separadas.** Atraso de detecção, duração do retreino, primeira
   avaliação pós-decisão acima de `tau`, downtime, ganho corrompido e retenção
   limpa são campos distintos.
4. **Schema v3 para novas evidências.** Schema v2 continua legível apenas como
   legado não auditado. Somente um par v3 semanticamente válido pode ser
   marcado como auditado.
5. **Persistência manifest-last.** `agent.json` e `baseline.json` são
   publicados atomicamente e só formam um par consumível quando
   `pair-manifest.json` válido está presente. Calibração e resumo também usam
   substituição atômica.
6. **Calibração antes da matriz principal.** Treinar um checkpoint limpo uma
   vez, avaliar toda a tabela corrupção × severidade e selecionar a menor
   severidade com acurácia estritamente em `(0.25, 0.50)`. Nenhuma combinação
   avaliada é omitida do JSON.
7. **Sem escolha pós-hoc silenciosa.** A severidade usada na matriz deve vir da
   calibração ou ser declarada como cenário alternativo. `tau=0.50` não deve
   ser elevado apenas porque um piloto não produziu downtime.
8. **Sensibilidades somente quando explícitas.** Quorums e quantidades de
   rounds adicionais não multiplicam a matriz por padrão.
9. **Controles e cenários não são agregados juntos.** O resumo é particionado
   por população, corrupção, severidade, política de gatilho, quorum, rounds e
   horizonte. Não existe uma média global misturando controles e cenários.
10. **Convenção de controle v3.** `identity:0` é a única identidade válida;
    corrupções visuais usam severidades inteiras `1..5`.
11. **Valores de política são hiperparâmetros do projeto.** Warm-up 20,
    quorum 0.30, janela 2, cinco rounds e `tau=0.50` não são prescrições dos
    papers e exigem análise de sensibilidade.
12. **Execuções reais só depois dos testes estruturais.** O smoke injetado
    prova o encadeamento técnico, não a qualidade do modelo nem a hipótese
    científica.

## Invariantes e requisitos

- O detector recebe somente `x`; rótulos são usados apenas no treinamento e
  na avaliação offline.
- Agente e baseline de um par usam o mesmo checkpoint, configuração, seed,
  início, horizonte e trace até a primeira decisão.
- A baseline registra o primeiro gatilho como contrafactual e nunca retreina.
- O agente executa no máximo um bloco com exatamente a quantidade configurada
  de rounds.
- Downtime é a integral stepwise dos intervalos com
  `accuracy_corrompida < tau`.
- Corrupções preservam shape, dtype, device, faixa de pixels e labels; não
  alteram a entrada in-place.
- Um skip de teste CUDA/MPS significa ausência de evidência naquele hardware,
  não validação do backend.
- Tempo virtual do simulador e wall-clock real são grandezas diferentes.

## Evidência experimental vigente

O piloto observado foi:

```text
dataset=cifar10
seed=42
scenario=gaussian_noise:5
horizon=400 s
include_controls=true
device=mps:0
```

O par visual foi aceito como v3 auditado e apresentou:

- acurácia corrompida no onset: 0.5353;
- decisão após 120 s, exatamente com quorum 0.30;
- cinco rounds em 78.5268 s virtuais;
- acurácia corrompida final do agente: 0.7409;
- acurácia corrompida final da baseline: 0.5353;
- ganho corrompido do agente: +0.2056;
- retenção limpa do agente: -0.0405;
- downtime do agente e da baseline: 0 s.

Conclusão vigente: o piloto prova que, nesta seed, o detector disparou e o
retreino melhorou a acurácia sob ruído. Ele **não testa a hipótese primária de
redução de downtime**, porque a acurácia no onset permaneceu acima de
`tau=0.50`. O `time_to_recovery` desse piloto não representa recuperação de
uma indisponibilidade real, pois não houve intervalo abaixo de `tau`.

O controle Oracle `identity:0` disparou no primeiro tick, não produziu
downtime e alterou a acurácia limpa em apenas +0.0026. Ele testa o caminho de
ação quando o evento é conhecido; não mede falso positivo do detector real em
dados limpos.

## Bloqueios

1. **Calibração ainda não confirmada.** Na última inspeção não existia
   `output/cifar-10/severity-calibration/severity_calibration.json`. O usuário
   iniciou a calibração, mas sua conclusão e os valores de `selected` ainda
   não foram verificados.
2. **Piloto sem downtime.** `gaussian_noise:5` resultou em 0.5353 no onset,
   acima do limiar de 0.50.
3. **Apenas uma seed.** `std=0` no resumo é consequência de `run_count=1`, não
   evidência de estabilidade.
4. **Wall-clock inviável para a matriz.** O piloto com controle levou cerca de
   uma hora; durante a calibração foram relatados cerca de 30 minutos para 15
   rounds.
5. **Reprodutibilidade bit a bit no MPS não demonstrada.** O piloto registrou
   `effective_device=mps:0`, mas algoritmos determinísticos estavam
   desativados.

## Diagnóstico de desempenho vigente

O MPS foi efetivamente usado no piloto, conforme o metadata do artefato.
Entretanto:

- cada cliente é treinado sequencialmente dentro de `Server.train_clients`;
- cada fit recria `TensorDataset` e `DataLoader`;
- batches são transferidos CPU → device durante o fit;
- pesos são copiados device → CPU depois de cada cliente, agregados e
  redistribuídos;
- batch 32 e uma CNN pequena podem não saturar o MPS;
- `--include-controls` executa outro `run_drift_comparison`, incluindo outro
  treino limpo de 20 rounds.

Os digests dos checkpoints limpos de `gaussian_noise:5` e `identity:0` são
idênticos, evidenciando que o mesmo estado final foi recomputado. Isso sustenta
a hipótese de trabalho duplicado. Ainda não existe um perfil end-to-end no
ambiente Conda do usuário que atribua percentuais do wall-clock a cada causa.

## Questões em aberto

- Quais severidades serão selecionadas pela calibração?
- Alguma corrupção não terá severidade dentro do intervalo estrito?
- O checkpoint limpo deve ser reutilizado por seed entre todos os cenários e
  controles antes da matriz completa?
- Qual modo piloto reduz custo sem ser confundido com evidência científica?
- Quais cópias CPU↔MPS e reconstruções de DataLoader dominam o wall-clock?
- Deve existir também um controle `identity` com o detector real para medir
  falsos positivos, além do controle Oracle atual?
- As flags de determinismo do MPS devem ser endurecidas ou apenas registradas
  e tratadas como limitação?

## Trabalho adiado

- Implementar otimizações antes de medi-las.
- Paralelizar clientes numa única GPU sem benchmark que demonstre benefício.
- Alterar batch size, quantidade de rounds ou frequência de avaliação na
  matriz científica sem registrar a mudança como configuração experimental.
- Executar a matriz de cinco seeds enquanto calibração e wall-clock permanecem
  bloqueadores.

## Próximos passos

1. Aguardar a calibração e inspecionar o JSON completo e `selected`.
2. Confirmar que a severidade escolhida coloca o onset estritamente abaixo de
   0.50 e acima de 0.25 em um piloto.
3. Medir no ambiente Conda `federatedLearning` o wall-clock por:
   construção, cliente, round, avaliação, monitoramento e cópia de pesos.
4. Decidir uma otimização preservando a semântica. A principal candidata é
   treinar e reutilizar um checkpoint limpo por seed entre cenários e
   controles.
5. Criar, se aprovado, um modo piloto explicitamente não científico com
   amostras/rounds reduzidos.
6. Repetir o piloto calibrado, validar o par e gerar o gráfico.
7. Somente então executar múltiplas seeds, analisar grupos separados e
   reportar dispersão.
