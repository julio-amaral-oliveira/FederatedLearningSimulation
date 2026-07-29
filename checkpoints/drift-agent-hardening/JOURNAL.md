# Diário cronológico — drift-agent-hardening

> Registro append-only da evolução até 2026-07-29. A verdade vigente está em
> [STATE.md](STATE.md).

## 2026-07-29 — avaliação inicial

### Contexto

O trabalho começou pela leitura dos commits recentes da `main`, dos documentos
de literatura a partir do 8 e do código/artefatos em `drift-agent`. A
especificação original pedia um ciclo síncrono de detecção local sem rótulos,
gatilho coletivo, cinco rounds de retreino e comparação por downtime.

Os documentos 8–10 descreviam `gaussian_noise`, `gaussian_blur` e
`brightness_contrast`. O código observado já havia adotado
`gaussian_noise`, `frosted_glass_blur`, `motion_blur` e `fog`. A decisão foi
não fingir reprodução literal da especificação nem do CIFAR-10-C: os quatro
cenários existentes seriam tratados como corrupções próprias, inspiradas no
benchmark, com contrato tensorial e calibração explícitos.

### Evidências e hipóteses

- O fluxo funcional já executava treino limpo, monitoramento, detecção,
  gatilho e retreino.
- Os resultados piloto não bastavam para conclusão estatística.
- A comparação agente/baseline podia divergir por RNG.
- O endpoint temporal dependia do momento do retreino.
- Faltavam metadados, validação semântica, múltiplas seeds, controles e
  calibração auditável.
- Os valores `20/0.30/5/0.50` eram políticas do projeto, não fatos prescritos
  pela literatura.

### Decisão

Classificar o estado anterior como prova de conceito funcional, não como
experimento conclusivo, e preparar um endurecimento estrutural antes de novas
execuções CIFAR completas.

## 2026-07-29 — desenho, plano e processo de implementação

### Decisões

- Criados os documentos 13 e 14 nos commits `9a2e06d` e `c9e9355`.
- Branch escolhida pelo usuário: `feature/drift-agent-hardening`.
- O usuário rejeitou worktree; o trabalho permaneceu no checkout atual.
- Desenvolvimento escolhido: subagent-driven development.
- Cada ponto funcional deveria formar um commit independente.
- Não fazer push e não adicionar `Co-authored-by`.
- Não realizar revisão entre commits; executar uma única revisão sistemática
  ao final e corrigir cada achado em commit isolado.
- O modelo pedido para revisores foi “5.6 Luna”, indisponível. A alternativa
  aceita foi `gpt-5.6-terra`, effort médio, para a revisão final.
- Alterações dirty/untracked preexistentes deveriam permanecer intactas.

### Alternativa arquitetural escolhida

Entre completar apenas RNG, executar braços em lockstep ou preservar braços
independentes com estado completo/trace/validação, foi escolhida a terceira
opção. Ela preserva a arquitetura e torna divergências observáveis.

## 2026-07-29 — commits estruturais

### `7a6468c` — pareamento estocástico

- Captura/restauração dos RNGs relevantes.
- Trace completo por tick.
- Deep copy de históricos.
- Teste com Dropout real para replay.

### `e27fae8` — horizonte e métricas

- Introdução do horizonte de produção.
- Separação entre duração do retreino e tempo até primeira avaliação acima de
  `tau`.

### `bcf4151` — metadata auditável

- Schema v3 com configuração, detector, runtime, digest e métricas derivadas.
- V2 mantido como legado.

### `f770e10` — contrato das corrupções

- Shape, dtype, device, ausência de mutação, faixa, seed, severidade e
  monotonicidade cobertos por testes.
- Mantidos os quatro nomes existentes, sem alegar CIFAR-10-C oficial.

### `c156c59` — calibração

- Uma construção/treino limpo por calibração.
- Avaliação de todo o produto cartesiano antes da seleção.
- Menor severidade estritamente dentro do intervalo.

### `08a508b` — matriz e controles

- Runner direto, sem subprocess/cwd implícito.
- Seeds/cenários explícitos.
- Sensibilidades apenas sob flags.
- Adapters `identity_corruption` e `OracleMonitor`.

### `9cbf2c7` — leitura, validação e plot

- `DriftResult`, `validate_pair` e `summarize_pair`.
- V2 legível e não auditado; V3 destinado a auditoria.
- Plot baseado no par validado.

### `4ee0406` — handoff

- Documento 15 com comandos, schemas e limites de alegação.

## 2026-07-29 — revisão final única

### Evidência

O revisor `gpt-5.6-terra`, effort médio, examinou cada commit original e
registrou seis achados:

1. Critical: retreino podia ultrapassar o horizonte.
2. Important: V3 validava concordância, não toda a semântica.
3. Important: persistência não era atômica.
4. Important: resumo misturava cenários e controles.
5. Minor: caminhos CUDA/MPS sem cobertura red-capable.
6. Minor: smoke E2E descrito, mas não executável.

### Correções isoladas

- `e09bb90`: horizonte padrão 400 s e recusa antecipada quando o bloco não
  cabe.
- `3ae4728`: contrato semântico V3 completo.
- `ea8679d`: publicação atômica e manifest-last.
- `017288b`: resumo particionado por dimensões.
- `a58cb63`: mocks e testes condicionais de acelerador.
- `1d589be`: smoke E2E executável.

### Re-revisão e reversão parcial

A re-revisão encontrou um conflito novo: o runner criava `identity:0`, mas o
validador V3 aceitava apenas severidades `1..5`. O fluxo documentado
`--include-controls` falhava.

O commit `a573478` definiu a convenção vigente:

- V3 aceita somente `identity:0`;
- corrupções não identidade usam `1..5`;
- V2 mantém comportamento legado;
- o smoke público cobre `include_controls=True`.

Após esse commit, o mesmo revisor concluiu **Ready**, com 116 testes passando,
2 skips de acelerador e nenhum novo achado.

## 2026-07-29 — orientação de execução

### Fluxo documentado

1. Smoke técnico injetado, sem CIFAR.
2. Piloto real com uma seed/cenário.
3. Calibração das severidades.
4. Matriz principal com múltiplas seeds.
5. Plot e análise de pares validados.

Foi esclarecido que o percentual inicial exibido no terminal era o download
do CIFAR-10. Também foi esclarecido que `--include-controls` produz dois
pares: visual agente/baseline e `identity:0` agente/baseline.

## 2026-07-29 — piloto real `gaussian_noise:5`

### Configuração observada

- CIFAR-10, seed 42.
- Gaussian noise severidade 5.
- Horizonte 400 s.
- Quorum 0.30.
- Cinco rounds de retreino.
- Controle incluído.
- Python 3.13.14 conda-forge, Torch 2.13.0, `mps:0`.
- Algoritmos determinísticos desativados.

### Evidência do cenário visual

- Mesmo checkpoint e estado preprodução nos dois braços.
- Acurácia limpa pré-drift: 0.7653.
- Acurácia corrompida no onset: 0.5353.
- O detector local produziu flags e o quorum foi atingido após 120 s.
- O agente executou exatamente cinco rounds em 78.5268 s virtuais.
- Acurácia corrompida final do agente: 0.7409.
- Baseline permaneceu em 0.5353.
- Ganho corrompido do agente: +0.2056.
- Acurácia limpa final do agente: 0.7248, delta -0.0405.
- Downtime de ambos: 0 s.

### Interpretação

O mecanismo detectou e adaptou nesta seed, mas a hipótese de downtime não foi
exercitada porque 0.5353 permaneceu acima de `tau=0.50`. Não se decidiu elevar
`tau` pós-hoc. A severidade deve ser obtida pela calibração predefinida.

`time_to_recovery=15.7054` não foi tratado como recuperação operacional nesse
piloto: não existiu estado indisponível.

### Evidência do controle Oracle

- `identity:0`.
- Disparo no primeiro tick, 10 s.
- Cinco rounds.
- Delta limpo +0.0026.
- Downtime 0.

O controle demonstrou o caminho de ação com evento conhecido, mas não mediu
falsos positivos do detector real em dados limpos.

## 2026-07-29 — calibração e desempenho

### Calibração

Foi fornecido e documentado o comando que treina um checkpoint limpo e avalia
as 20 combinações de quatro corrupções × cinco severidades. Na última
inspeção, o JSON de calibração ainda não existia. O usuário relatou execução
em andamento, com aproximadamente 30 minutos para chegar ao round 15.

### Sintoma de wall-clock

O piloto com controle levou aproximadamente uma hora real, apesar do uso
suposto de GPU.

### Evidências

- O próprio resultado registra `effective_device=mps:0`; a hipótese de que o
  piloto executou somente em CPU foi descartada.
- O ambiente de inspeção do agente não era o Conda `federatedLearning` do
  usuário e não foi usado como prova do device do piloto.
- O servidor percorre clientes em loop sequencial.
- Cada fit recria dataset/loader, move batches ao device, devolve pesos à CPU,
  agrega e redistribui.
- `--include-controls` chama uma segunda comparação completa.
- Os digests limpos do visual e do controle foram idênticos:
  `3b7f02ab62e4b95e668f74a9ca2f3ce288dd7e92704d97cf337423568265de6a`.
  Portanto, o mesmo checkpoint final foi recomputado.

### Hipóteses vigentes, ainda não perfiladas

1. Treino limpo duplicado entre cenário e controle.
2. Clientes locais sequenciais.
3. Cópias/sincronizações CPU↔MPS por cliente.
4. DataLoaders reconstruídos e batches pequenos.
5. CNN pequena e batch 32 com baixa saturação do MPS.

Não foi decidido paralelizar nem alterar batch/rounds. Primeiro deve existir
perfil de wall-clock no Conda real. A candidata principal para otimização é
reutilizar um checkpoint limpo por seed sem mudar a semântica dos pares.

## 2026-07-29 — checkpoint desta sessão

Criados os três artefatos canônicos em
`checkpoints/drift-agent-hardening/`. O estado vigente foi separado da
cronologia e do glossário. Documentos de literatura e relatórios de revisão
permanecem fontes de evidência referenciadas, não fontes concorrentes de
estado atual.
