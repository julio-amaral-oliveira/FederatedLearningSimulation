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

## 2026-07-30 — seed corrigida, uniformidade e exploração de rounds

### Correção da seed de teste

O episódio usava `base_seed + 9999` para a corrupção do teste.
A calibração antiga usava a seed base.
O primeiro teste falhou com `17 != 10016`.
O código passou a usar uma função para calcular a seed de teste.
O calibrador passou a registrar `corrupted_test_seed`.
O commit `982c19d` contém a correção.
A suíte completa passou em 116 testes e ignorou 2 testes de acelerador.

### Matriz heterogênea de cinco seeds

O validador aceitou os pares de `motion_blur:1` para as seeds 42–46.
Cada execução usou cinco rounds e o perfil heterogêneo.
O onset ficou entre 0.3136 e 0.3891.
A decisão ocorreu após 90 s em todas as seeds.
O ganho corrompido médio foi +0.3006.
O agente teve downtime médio de 111.99 s.
A baseline teve downtime de 400 s.
A retenção limpa média foi -0.2057.
A seed 45 ficou abaixo de `tau` durante rounds intermediários.

### Alteração local de perfil de velocidade

O usuário adicionou os perfis `uniform` e `heterogeneous`.
O runner aceita `--client-speed-profile` e `--speed-profile`.
O drift experiment usa `uniform` como default local.
O perfil uniforme usa um tier com faixa de 0 a 10.
Os módulos legados mantêm o perfil heterogêneo.

O perfil uniforme remove as diferenças entre tiers.
Ele não produz durações iguais para todos os clientes.
O simulador ainda gera uma duração para cada cliente.
O timeout p75 ainda pode excluir os clientes mais lentos.
O perfil também altera o checkpoint, o timeout e o tempo virtual.
Por isso, o perfil é uma dimensão experimental.

A mudança está local e não tem commit.
A suíte focada passou em 33 testes e ignorou 1 teste.
Um teste de metadata v3 falhou.
A expectativa não contém `client_speed_profile` e `client_speed_tiers`.
Não há testes novos para a seleção do perfil ou para o default.
Não há teste novo para a divisão do resumo por perfil.

Uma validação anterior usou o ambiente Conda `federatedLearning`.
Essa validação passou em 45 testes de runner, resultados e smoke E2E.
Ela não executou `tests.test_drift_episode`.
Por isso, ela não encontrou a expectativa incorreta.

### Nova calibração uniforme

O novo JSON usa a seed base 42 e `corrupted_test_seed=10041`.
O checkpoint é `9432a30f...`.
A exploração de rounds usa o mesmo checkpoint.

- `gaussian_noise`: nenhuma
- `frosted_glass_blur:4`: 0.4237
- `motion_blur:1`: 0.3744
- `fog:4`: 0.4925.

### Exploração `motion_blur:1`, seed 42

`load_persisted_pair` aceitou os pares com 1, 3, 5 e 10 rounds.
Todos os pares usam o mesmo checkpoint e o perfil uniforme.
O onset foi 0.3744.
A decisão ocorreu após 90 s.

- 1 round: final corrompida 0.5895, downtime 98.0037 s, retenção -0.2123,
  duração 8.0037 s.
- 3 rounds: final 0.5967, downtime 106.0073 s, retenção -0.2243, duração
  24.0110 s.
- 5 rounds: final 0.5025, downtime 106.0073 s, retenção -0.2409, duração
  40.0183 s.
- 10 rounds: final 0.5758, downtime 106.0073 s, retenção -0.2318, duração
  80.0365 s.

O primeiro round atingiu 0.5895.
O segundo round reduziu a acurácia para 0.4783.
O terceiro round aumentou a acurácia para 0.5967.
Mais rounds não produziram uma melhora contínua.
Um round teve o melhor resultado conjunto de downtime, custo e retenção.
Três rounds adicionaram 0.0072 à acurácia final.
Um round ainda causou uma perda limpa de 0.2123.

### Decisão provisória

Não adicione replay limpo ou regularização agora.
Primeiro, consolide e documente o perfil de velocidade.
Depois, corrija os testes e valide um round em outras seeds.
Se a perda limpa continuar alta, defina uma nova pergunta experimental.
Não adicione uma estratégia de retenção como um ajuste sem registro.

O usuário pediu o uso contínuo da skill `maintaining-project-checkpoints`.
As próximas mudanças devem atualizar os três artefatos canônicos.

### Perfis de velocidade consolidados

O commit `73839fd` adicionou os perfis configuráveis.
O drift experiment usa o perfil uniforme como default.
Os executáveis legados usam o perfil heterogêneo como default.

O teste de metadata v3 agora exige o perfil e os tiers resolvidos.
Os testes também verificam o default uniforme.
Outro teste verifica a seleção explícita do perfil heterogêneo.
Um teste verifica a propagação do perfil para a matriz.
Um teste verifica os perfis centrais dos módulos síncrono e assíncrono.
Um teste verifica a divisão do resumo por perfil.

A suíte focada executou 34 testes.
Ela passou em 33 testes e ignorou 1 teste de acelerador.
A suíte completa executou 121 testes.
Ela passou em 119 testes e ignorou 2 testes de acelerador.
`py_compile` e `git diff --check` passaram.

### Papel dos perfis de velocidade

O usuário confirmou o perfil uniforme como cenário principal.
O perfil heterogêneo fica como análise de sensibilidade.
Esta decisão remove a questão pendente sobre o default metodológico.

### Matriz principal definida

O usuário decidiu encerrar os testes ponto a ponto.
A matriz principal usará as seeds 42–46.
Ela usará o perfil uniforme, quorum 0.30 e um round.
Ela incluirá `frosted_glass_blur:4`, `motion_blur:1` e `fog:4`.
Ela também incluirá os controles Oracle.
`gaussian_noise` fica fora porque a calibração não selecionou uma severidade.

### Matriz principal concluída

O runner concluiu a matriz em
`output/cifar-10/drift-agent/final-matrix`.
A inspeção encontrou 20 arquivos `agent.json`.
Ela também encontrou 20 arquivos `baseline.json`.
Cada um dos 20 pares tem um `pair-manifest.json`.

Cada cenário contém as seeds 42–46.
Os cenários são `frosted_glass_blur:4`, `motion_blur:1` e `fog:4`.
O controle Oracle `identity:0` também contém as cinco seeds.
A análise agregada dos resultados ainda está pendente.

### Gráficos individuais gerados

O plotter usou o Python do ambiente Conda `federatedLearning`.
Ele gerou um arquivo `comparison.png` para cada par.
A verificação encontrou 20 arquivos com conteúdo.
O plotter validou cada par antes de salvar o gráfico.

A inspeção visual usou `motion_blur:1`, seed 42.
O gráfico mostrou agente, baseline, tau, onset e horizonte.
Ele também mostrou a decisão, o round e a primeira recuperação.
O aviso de cache do Fontconfig não impediu a geração.

### Proposta de gráfico agregado por corrupção

Um subagente sem contexto anterior analisou os JSONs e o plotter atual.
Ele não editou arquivos.
Ele recomendou uma figura composta para cada corrupção.

O painel superior usaria tempo desde o onset.
Ele mostraria as cinco seeds como curvas em degraus.
Linhas sólidas representariam o agente.
Linhas tracejadas representariam a baseline.
Uma linha espessa mostraria a média de cada braço.
Uma faixa translúcida mostraria o mínimo e o máximo observados.
Essa faixa não seria apresentada como intervalo de confiança.

Três painéis inferiores usariam dumbbells pareados por seed.
Eles mostrariam acurácia corrompida final, downtime e retenção limpa.
A figura manteria o controle `identity:0` separado e identificado como Oracle.

O subagente rejeitou barras, violinos e médias isoladas.
Essas opções esconderiam o pareamento ou a variabilidade com cinco seeds.
A proposta ainda depende de aprovação e implementação.

### Plotter agregado implementado do zero

O usuário pediu a exclusão do plotter anterior.
O arquivo `experiments/plot_smoke_drift.py` foi excluído.
Um subagente sem contexto anterior recriou o arquivo do zero.
Ele recebeu somente `experiments/smoke_drift.py` e exemplos JSON.
Ele não consultou a implementação excluída.

A nova CLI recebe `--scenario-dir` e `--output`.
Ela descobre os pares `agent.json` e `baseline.json` por seed.
Ela rejeita grupos vazios, braços ausentes e dimensões incompatíveis.
Ela normaliza o tempo desde o onset.
Ela usa curvas em degraus e alinhamento por forward-fill.
Ela mostra as trajetórias, as médias e a faixa mínima–máxima.
Ela também mostra os três dumbbells pareados.

A inspeção visual encontrou dois problemas no controle Oracle.
O eixo de downtime ampliava ruído numérico próximo de zero.
Os marcadores coincidentes também escondiam a baseline.
O implementador fixou o eixo entre zero e o horizonte.
Ele também usou um círculo aberto maior para a baseline.

Os testes antigos passaram a usar o contrato por cenário.
A suíte completa executou 127 testes.
Ela passou em 125 testes e ignorou 2 testes de acelerador.
O commit `90b230b` contém o plotter e os testes.

O plotter gerou quatro figuras `seeds_42-46_comparison.png`.
Cada cenário da matriz principal contém uma figura.
A inspeção visual confirmou `motion_blur:1` e o controle Oracle `identity:0`.
