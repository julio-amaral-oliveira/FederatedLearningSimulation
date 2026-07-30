# Estado atual — drift-agent-hardening

> Verdade vigente do projeto em 2026-07-30. Para a evolução e as evidências,
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
- Último commit de código observado: `90b230b`
  (`feat: compare drift results across seeds`).
- Base da implementação endurecida: `c9e9355`.
- A revisão final e a re-revisão concluíram **Ready**.
- A última verificação completa executou 127 testes.
- A verificação passou em 125 testes.
- A verificação ignorou 2 testes condicionais de CUDA/MPS.
- O experimento usa `uniform` como default.
- Os módulos legados síncrono e assíncrono usam `heterogeneous` como default.
- O schema v3 registra `client_speed_profile` e `client_speed_tiers`.
- Os testes cobrem o default, a seleção explícita e a divisão do resumo.
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
13. **Seed de teste compartilhada.** A calibração e o episódio usam
    `base_seed + 9999`. O calibrador registra `corrupted_test_seed`.
14. **Perfil de velocidade é uma dimensão experimental.** Os perfis alteram o
    timeout, a participação e o tempo virtual.
    Não agregue resultados de perfis diferentes.
    O perfil uniforme é o cenário principal.
    O perfil heterogêneo é uma análise de sensibilidade.
15. **Checkpoint contínuo.** Registre cada nova evidência, decisão, reversão,
    bloqueio ou termo durável nos três artefatos.
16. **Política da matriz principal.** A matriz usa o perfil uniforme, quorum
    0.30, um round e as seeds 42–46.
    A matriz usa `frosted_glass_blur:4`, `motion_blur:1` e `fog:4`.
    `gaussian_noise` fica fora porque a calibração não selecionou uma
    severidade.

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

### Calibração uniforme corrigida

O artefato de 2026-07-29 usa a seed base 42 e a seed de teste 10041.
O checkpoint é `9432a30f...`.

- `gaussian_noise`: nenhuma severidade no intervalo estrito
- `frosted_glass_blur:4`: 0.4237
- `motion_blur:1`: 0.3744
- `fog:4`: 0.4925.

### Matriz principal anterior, heterogênea

A matriz usou cinco seeds de `motion_blur:1`, cinco rounds e o perfil
heterogêneo. O ganho corrompido médio foi +0.3006.
O agente teve downtime médio de 111.99 s. A baseline teve downtime de 400 s.
A retenção limpa média foi -0.2057.
Esses resultados sustentam a redução de downtime no perfil heterogêneo.
Não combine esses resultados com execuções uniformes.

### Exploração de rounds, uniforme, seed 42

Os quatro pares `1/3/5/10` usam schema v3 e têm manifests válidos.
Eles usam o checkpoint `9432a30f...`.
O onset foi 0.3744. A decisão ocorreu após 90 s.
A primeira recuperação ocorreu após 8.0037 s.

| Rounds | Final corrompida | Ganho | Downtime | Retenção limpa | Duração |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.5895 | +0.2151 | 98.0037 s | -0.2123 | 8.0037 s |
| 3 | 0.5967 | +0.2223 | 106.0073 s | -0.2243 | 24.0110 s |
| 5 | 0.5025 | +0.1281 | 106.0073 s | -0.2409 | 40.0183 s |
| 10 | 0.5758 | +0.2014 | 106.0073 s | -0.2318 | 80.0365 s |

Um round teve o melhor resultado conjunto de downtime, custo e retenção.
Três rounds adicionaram 0.0072 à acurácia corrompida final.
Esse ganho exigiu mais tempo e causou maior perda limpa.
O segundo round reduziu a acurácia para 0.4783.
Mais rounds não garantiram uma acurácia maior.
Um round ainda causou uma perda limpa de 0.2123.

### Matriz principal uniforme concluída

O runner gerou 20 pares em `output/cifar-10/drift-agent/final-matrix`.
Cada par tem `agent.json`, `baseline.json` e `pair-manifest.json`.
A matriz contém cinco seeds para cada cenário selecionado.
Ela contém `frosted_glass_blur:4`, `motion_blur:1` e `fog:4`.
Ela também contém cinco pares do controle Oracle `identity:0`.
O plotter gerou um arquivo `comparison.png` para cada um dos 20 pares.
Todos os 20 arquivos existem e têm conteúdo.
A inspeção visual de `motion_blur:1`, seed 42, confirmou o plot esperado.
Os artefatos estão completos, mas a análise agregada ainda está pendente.

### Visualização por corrupção implementada

Um subagente sem contexto anterior recriou o plotter do zero.
Ele usou apenas `smoke_drift.py` e exemplos JSON como contexto.
O painel superior mostra as cinco seeds como curvas em degraus.
Ele também mostra a média e a faixa mínima–máxima observada.
Três painéis inferiores mostram métricas pareadas por seed.
As métricas são acurácia corrompida final, downtime e retenção limpa.
O commit `90b230b` contém o código e os testes.
O plotter gerou quatro figuras agregadas, uma para cada cenário.

## Bloqueios

1. **A documentação de handoff não contém o perfil de velocidade.**
   O código e o checkpoint registram o perfil.
   O handoff ainda não mostra a nova opção da CLI.
2. **O resultado não registra a participação por round.**
   O payload registra o perfil e os tiers.
   O payload não mostra quais clientes excederam o timeout.
   O JSON não prova a redução de retardatários.
3. **A exploração de rounds usa uma seed.**
   Ela ajuda a escolher o próximo teste.
   Ela não sustenta a escolha final da política.
4. **A perda limpa permanece alta.**
   Um round reduziu a acurácia limpa em 21.23 pontos percentuais.
5. **O MPS não tem reprodução bit a bit confirmada.**
   Os artefatos registram `effective_device=mps:0`.
   Os algoritmos determinísticos estavam desativados.

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

- O perfil uniforme deve manter o timeout p75?
- O perfil uniforme deve incluir todos os clientes?
- Quais campos devem registrar a participação e as durações por round?
- A matriz principal confirma o resultado de um round nas seeds 42–46?
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
- Alterar batch size ou frequência de avaliação na matriz científica sem
  registrar a mudança como configuração experimental.
- Introduzir replay limpo, regularização ou outra estratégia de retenção antes
  de concluir a sensibilidade de rounds dentro do escopo atual.

## Próximos passos

1. Documente a opção de perfil no handoff.
2. Não misture resultados uniformes e heterogêneos.
3. Registre a participação efetiva por round.
4. Carregue os 20 manifests e gere o resumo particionado.
5. Analise downtime, atraso, recuperação e retenção limpa por cenário.
6. Meça no ambiente Conda `federatedLearning` o wall-clock por:
   construção, cliente, round, avaliação, monitoramento e cópia de pesos.
7. Escolha uma otimização que preserve a semântica. A principal opção é
   treinar e reutilizar um checkpoint limpo por seed entre cenários e
   controles.
8. Se um round ainda causar perda limpa alta, avalie replay ou regularização.
9. Trate essa avaliação como uma nova pergunta experimental.
