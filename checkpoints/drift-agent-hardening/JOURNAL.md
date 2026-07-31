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

### Gráficos antigos substituídos

Os 20 diretórios de seed ainda continham `comparison.png`.
Esses arquivos vieram do plotter individual excluído.
O usuário pediu a substituição pelos gráficos agregados.
Os 20 arquivos individuais foram removidos.
Os quatro arquivos `seeds_42-46_comparison.png` foram mantidos.
Cada cenário agora contém somente o gráfico do plotter novo.

### Artefatos da matriz apagados

O usuário apagou os arquivos antes de pedir uma nova geração.
O diretório `output/cifar-10/drift-agent` não existe mais.
Por isso, o plotter não encontrou os 20 pares JSON.

O Git não rastreava os artefatos.
As pastas temporárias não contêm uma cópia.
O macOS negou acesso à Lixeira, mesmo fora do sandbox.
A recuperação manual ou uma nova execução da matriz é necessária.

### Regeneração do gráfico de `motion_blur:1`

Em 2026-07-30, a inspeção encontrou os 20 pares v3 em
`output/cifar-10/drift-agent/final-matrix`.
Não havia `comparison.png` antigo nos diretórios de seed.
O gráfico agregado anterior foi removido antes da nova execução.
O comando solicitado falhou no Python base por falta de `matplotlib`.
O mesmo comando passou no ambiente Conda `federatedLearning`.
O plotter criou `motion_blur_sev1/seeds_42-46_comparison.png`.
O arquivo tem 2720 × 1815 pixels e passou na validação PNG.
A inspeção visual confirmou cinco seeds, as curvas pareadas e os três painéis
de métricas.

### Investigação da variação do `gaussian_noise:3`

Em 2026-07-31, a investigação comparou os cinco JSONs com o plotter atual.
Cada seed tem uma decisão de retreino e um período de retreino.
Os tempos de decisão após o onset são 190, 110, 100, 280 e 170 segundos
para as seeds 42, 43, 44, 45 e 46.
O baseline registra o mesmo gatilho contrafactual em cada seed.
O `Agent mean` muda cinco vezes porque o plotter alinha as trajetórias e calcula
a média em cada tempo de recuperação.
O ruído Gaussian usa uma seed derivada de cliente e tick.
O resultado muda entre seeds sem indicar um bug no plotter.
O teste de reexecução de uma seed foi interrompido durante o treinamento inicial
por custo de wall-clock.

## 2026-07-31 — catálogo das famílias de experimentos

### Evidência

Foi feita uma leitura conjunta de `docs/`, `report/`, do código atual e do
histórico de commits. A evolução do repositório contém sete estudos distintos:

- E01, treinamento federado estático.
- E02, ablação da agregação assíncrona.
- E03, impacto do timeout síncrono.
- E04, comparação Sync contra Async em cenário estático.
- E05, drift temporal sazonal Sync contra Async.
- E06, drift agent inicial.
- E07, drift agent endurecido e auditável.

Os relatórios confirmam protocolos diferentes. E05 alterna grupos de classes
por fases. E06 e E07 aplicam corrupção visual durante a produção. E02 estuda a
fórmula FedAsync. E03 estuda clientes lentos no modo síncrono.

O histórico também mostra que os commits `852090e` e `b059545` alteraram a
corrupção Gaussian noise e os gráficos do E07. Eles não criaram um novo estudo.

### Decisão

Foi criado `docs/experimentos.md` como catálogo humano das famílias, perguntas,
protocolos, métricas, código e fontes. O catálogo usa os IDs E01 a E07.

`STATE.md` continua sendo a fonte da verdade atual. `JOURNAL.md` continua sendo
o histórico append-only. `CONTEXT.md` contém apenas os termos e as fronteiras.
O catálogo não substitui nenhum dos três artefatos canônicos.

E07 é o único fluxo de drift considerado atual. E01 a E06 permanecem como
referências históricas ou estudos anteriores. Os IDs ainda não foram incluídos
nos nomes dos diretórios ou JSONs.

### Próximo ponto de revisão

Se a confusão continuar nos artefatos persistidos, avaliar uma mudança futura
de nomenclatura que inclua o ID do experimento nos diretórios de saída. Essa
mudança deve preservar a leitura dos resultados existentes.

## 2026-07-31 — mapa operacional de scripts, comandos e saídas

### Evidência

O inventário encontrou 16 entrypoints executáveis entre `experiments/`, os
módulos `main.py` e as CLIs de `src/utils/`. Também encontrou módulos de suporte
sem CLI própria.

Os artefatos confirmam três problemas de identificação:

- `output-cifar-10/` mistura JSONs estáticos de E04 com JSONs temporais de E05.
- `drift-agent/` contém pares schema v2 do protótipo E06.
- `output/cifar-10/drift-agent/` contém calibração e matriz schema v3 do E07.

O inventário também encontrou comandos históricos. `handoff-7MZns2.md` aponta
para `experiments/plot_comparison.py`, que não existe no checkout. Algumas
receitas apontam para nomes de JSON antigos, como
`accuracy_data_iid_T200_sync.json`.

### Decisão

`docs/experimentos.md` recebeu um mapa operacional com quatro classificações:
`atual`, `suporte`, `histórico` e `legado`.

O catálogo também separa os comandos oficiais por família. Para E07, a forma
oficial usa `python -m` para calibração, matriz e plot. A CLI direta de
`experiments/smoke_drift.py` fica restrita a diagnóstico de um episódio.

Nenhum diretório de saída foi movido ou renomeado. Nenhum script legado foi
apagado. A revisão das receitas e dos handoffs ficou como próximo passo.

## 2026-07-31 — documentação operacional alinhada ao catálogo

### Evidência

O runbook `docs/literatura/15. handoff-experimento-drift-agent.md` ainda usava
o diretório `drift-agent/main`, cenários de exemplo diferentes da matriz
vigente e uma interface antiga do plotter. As receitas rápidas usavam nomes de
JSON do E05 que não correspondiam ao artefato gerado com um prefixo explícito.
O handoff `handoff-7MZns2.md` descrevia um checkout antigo e citava
`experiments/plot_comparison.py`, que não existe no checkout atual.

### Decisão

O runbook do E07 agora declara o perfil `uniform`, os cenários calibrados
`frosted_glass_blur:4`, `motion_blur:1` e `fog:4`, um round, as seeds 42–46 e
os diretórios canônicos de calibração e matriz. Ele também usa a interface
atual de `plot_smoke_drift` e separa `heterogeneous` como sensibilidade.

As receitas rápidas agora identificam a família em cada seção, rotulam E02 e
E05 como históricos, corrigem os nomes dos JSONs do E05 e incluem os pontos de
entrada do E07. O handoff antigo foi preservado e marcado como histórico. O
README passou a apontar para o catálogo, as receitas e o runbook.

### Estado

Nenhum código, resultado ou diretório foi movido ou apagado. O catálogo,
`STATE.md`, `CONTEXT.md` e `JOURNAL.md` continuam sendo as fontes de
identificação, verdade vigente, terminologia e histórico, respectivamente.
O próximo trabalho pode analisar os 20 pares da matriz ou propor uma
nomenclatura física para os artefatos, sem misturar protocolos.

## 2026-07-31 — verificação de resíduos na árvore de saída

### Evidência

Uma checagem posterior à revisão documental mostrou que
`output/cifar-10/drift-agent/severity-calibration/` não existe no checkout.
O JSON local de calibração está em `output/cifar-10/severity-calibration/` e
seleciona `gaussian_noise:3`, além de `frosted_glass_blur:4`,
`motion_blur:1` e `fog:4`.

`output/cifar-10/drift-agent/final-matrix/` contém 25 pares e cinco grupos:
os três cenários da política atual, `identity:0` e `gaussian_noise:3`. O
`summary.json` também inclui o grupo Gaussian. A política vigente no
checkpoint define 20 pares e exclui Gaussian.

### Decisão

Tratar a árvore local como saída misturada, não como uma matriz atual limpa.
O caminho de calibração dentro de `output/cifar-10/drift-agent/` continua
sendo o destino oficial para uma nova execução. Nenhum arquivo será movido,
apagado ou agregado nesta etapa.

O catálogo e o runbook passaram a registrar o resíduo explicitamente. A
próxima execução deve usar um diretório vazio ou isolado, publicar somente os
20 pares definidos pela calibração corrigida e gerar um novo resumo antes da
análise.

## 2026-07-31 — importação externa `drift-agent-2`

### Evidência

O usuário informou que refez a execução em outra máquina e importou os
artefatos em `output/cifar-10/drift-agent-2/`.

A auditoria encontrou 26 diretórios com `pair-manifest.json`:

- 25 pares em `final-matrix-rerun/`, distribuídos entre cinco grupos e as
  seeds 42–46.
- Um par em `omp-smoke/gaussian_noise_sev3/seed_42/`.

Todos os 26 pares passaram por `load_persisted_pair`. Os arquivos são schema v3,
os manifests conferem os digests dos JSONs e `validate_pair` aceitou cada braço
agente/baseline. A configuração comum da matriz é perfil `uniform`, um round,
horizonte de 400 s, timeout p75, `tau=0.50` e CIFAR-10.

O runtime registrado nos JSONs é Python 3.11.14 do Anaconda, Torch 2.5.1+cu121
em `cuda:0`, com algoritmos determinísticos desativados. A importação não
contém a calibração. A matriz inclui `gaussian_noise:3` junto com os três
cenários atuais e o controle Oracle.

Os digests de checkpoint limpo não são iguais entre todos os cenários. Para as
seeds 42, 43 e 46, o digest varia entre grupos. Para as seeds 44 e 45, ele é
igual entre os cinco grupos. Isso não invalida os pares individuais, mas
impede tratar automaticamente a matriz como uma comparação com checkpoint
compartilhado por seed.

### Decisão

Registrar `drift-agent-2` como uma importação externa tecnicamente válida e
preservá-la sem mover ou apagar arquivos. Não promover seus 25 pares a matriz
oficial até resolver a proveniência da calibração, a inclusão de
`gaussian_noise:3` e a política de reutilização do checkpoint limpo.

### Próximo ponto de revisão

Primeiro, confirmar com o usuário se `gaussian_noise:3` foi intencional na
execução da outra máquina. Depois, gerar ou importar a calibração correspondente
e decidir se a análise deve usar os 20 pares da política atual ou os 25 pares
da matriz importada como um cenário alternativo explicitamente nomeado.

## 2026-07-31 — correção da proveniência e oficialização da matriz

### Evidência

O usuário esclareceu que a calibração oficial está no repositório em
`output/cifar-10/severity-calibration/severity_calibration.json`.
O arquivo existe, tem schema 1 e seleciona:

- `gaussian_noise:3`, com acurácia 0.3783.
- `frosted_glass_blur:4`, com acurácia 0.4022.
- `motion_blur:1`, com acurácia 0.3493.
- `fog:4`, com acurácia 0.4875.

Essas quatro seleções aparecem em
`output/cifar-10/drift-agent-2/final-matrix-rerun/`. A matriz contém 25 pares
v3, cinco seeds por cenário visual e cinco pares do controle `identity:0`.
Os 25 pares foram validados localmente. O diretório `omp-smoke/` permanece
separado.

### Decisão

Substituir a decisão anterior. `drift-agent-2/final-matrix-rerun/` é a matriz
científica oficial do E07. `output/cifar-10/severity-calibration/` é a fonte
oficial de calibração. A saída antiga em `output/cifar-10/drift-agent/` não é a
matriz oficial.

A variação de digest de checkpoint entre cenários continua registrada como
limitação metodológica. Ela não invalida a matriz oficial nem os pares que
passaram pela validação interna.

### Próximo ponto de revisão

Analisar os 25 pares por cenário. Separar o smoke, registrar o ambiente CUDA e
relatar a calibração, o perfil `uniform`, o horizonte, o quorum e a limitação
de determinismo.

## 2026-07-31 — retomada da reorganização dos experimentos

### Motivo

A confirmação da calibração e da matriz oficial corrigiu a proveniência do E07.
Ela não deveria substituir a tarefa principal de reorganizar as famílias de
experimentos. A análise dos 25 pares fica depois da reorganização operacional.

### Evidência do inventário

- `experiments/` mantém entradas de E02, E05, E06 e E07 no mesmo nível.
- `src/synchronous/` e `src/asynchronous/` servem E01, E03 e E04 sem uma
  identificação de família no caminho.
- `output-cifar-10/` mistura saídas de E04 e E05.
- `output/` mistura saídas de E04, E05 e E07.
- `drift-agent/` contém pares v2 legados do E06.
- `output/cifar-10/` mistura comparação estática com calibração e E07.
- `.gitignore` ignora as saídas. A reorganização física precisa de registro
  próprio e validação por cópia.

### Decisão

Adotar `output/<experiment-id>/<dataset>/` como layout futuro. Usar namespaces
`experiments/eXX_*` para as entradas. Começar pelo E07, pois ele tem o protocolo
atual e a matriz oficial. Manter a matriz em
`output/cifar-10/drift-agent-2/final-matrix-rerun/` e a calibração em
`output/cifar-10/severity-calibration/` até a cópia validada.

Não mover, apagar ou reclassificar resultados nesta etapa. O catálogo registra o
mapa alvo. O plano local em
`docs/superpowers/plans/2026-07-31-reorganizar-experimentos.md` detalha a
execução futura.

### Próximo ponto de revisão

Criar o registro operacional E01–E07, adicionar caminhos de saída explícitos e
executar uma migração em modo dry-run antes de copiar os artefatos do E07.

## 2026-07-31 — correção do papel dos relatórios e do código compartilhado

### Evidência

O usuário esclareceu que `report/Relatório_Final_FAPESP.pdf` é o ponto de
partida original que dá origem ao E01. A classificação anterior como contexto
geral do projeto estava incompleta.

O mapa também precisava tratar duas camadas que não são resultados gerados:

- `src/` contém os motores síncrono e assíncrono, os utilitários compartilhados
  e o suporte específico do drift agent.
- `report/` contém o documento de origem do E01, relatórios históricos e fontes
  LaTeX.

### Decisão

Classificar o Relatório Final FAPESP como documento de origem do E01. A ordem
de leitura começa por esse documento, passa pelo relatório inicial e pelos
Resultados Experimentais, e só depois segue para E02–E05 e E07.

Manter `src/` versionado como código. Não duplicar os motores compartilhados
nos diretórios dos experimentos. Manter `report/` versionado como documentação,
separado de `results/`, que conterá apenas artefatos experimentais validados.

### Atualização do mapa

`docs/experimentos.md` agora registra o tratamento de `src/`, `report/`,
`results/` e `output/`. O layout futuro usa `output/` para saídas temporárias e
`results/` para resultados oficiais publicados.

### Próximo ponto de revisão

Criar o índice de relatórios e associar cada documento a E01–E07. Verificar os
caminhos e a compilação dos arquivos `.tex` antes de mover PDFs ou fontes.

## 2026-07-31 — índice e verificação das fontes dos relatórios

### Evidência

Foi criado `report/README.md` com a ordem de leitura, o papel de cada PDF, a
relação com E01–E05 e a situação das fontes LaTeX.

A verificação de referências encontrou:

- `comparacao_sync_async.tex`: seis gráficos esperados em `output/`.
- `drift_temporal_sync_async.tex`: nove gráficos esperados em `output/drift/`.
- `relatorio.tex`: referências antigas a
  `synchronous/output-cifar-10/` e `asynchronous/output-cifar-10/`, que não
  existem no checkout atual.
- `resultados.tex`: referências a gráficos de ablação e acurácia ausentes em
  `output-cifar-10/`.

O `latex-doctor` classificou o ambiente como `missing` para TeX Live ou MacTeX.
O Tectonic bundled existe, mas falhou antes de processar qualquer documento
com um panic do runtime de rede do macOS. A compilação dos quatro `.tex` não
foi confirmada.

### Decisão

Manter todos os PDFs e fontes nos caminhos atuais. Tratar `relatorio.tex` e
`resultados.tex` como fontes históricas não reproduzíveis no checkout atual até
recuperar os gráficos ou atualizar seus caminhos. Não instalar uma distribuição
LaTeX e não alterar fontes ou PDFs nesta etapa.

Adicionar o índice de relatórios ao README e ao catálogo. A compilação deve ser
reavaliada antes de uma movimentação física.

### Próximo ponto de revisão

Decidir se os gráficos ausentes serão recuperados ou se os dois relatórios
continuarão apenas como PDFs históricos. Depois disso, criar o registro
operacional E01–E07 e o contrato de publicação em `results/`.

## 2026-07-31 — Tectonic: causa isolada e compilação validada

### Pergunta

O Tectonic está instalado. O panic observado ao compilar os relatórios é um
problema da instalação ou uma limitação do ambiente usado pelo Codex?

### Investigação

- O Tectonic bundled e o Tectonic do Homebrew são versões 0.16.9.
- Ambos reproduzem, dentro do sandbox, o panic em
  `system-configuration-0.6.1/src/dynamic_store.rs` com a mensagem
  `Attempted to create a NULL object`.
- O cache local existe em `~/Library/Caches/Tectonic/bundles`.
- `--only-cached` não evita o panic. Variáveis de proxy vazias também não
  mudam o resultado.
- O diagnóstico oficial do plugin (`latex_doctor.py`) passou com `ready: true`
  quando executado fora do sandbox. Não foi necessário instalar TeX Live.

### Resultado

A instalação do Tectonic está funcional. O bloqueio ocorre na política
sandboxed do processo, antes do processamento do documento, quando o runtime
tenta consultar a configuração de rede do macOS. A execução fora do sandbox
foi usada apenas para gerar artefatos temporários e validar o diagnóstico.

### Compilações

- `comparacao_sync_async.tex`: sucesso, PDF temporário de 9 páginas.
- `drift_temporal_sync_async.tex`: sucesso, PDF temporário de 9 páginas, com
  avisos de referências e caixas largas.
- `relatorio.tex`: processou o LaTeX e parou porque
  `../synchronous/output-cifar-10/accuracy_iid.png` está ausente.
- `resultados.tex`: processou o LaTeX e parou porque
  `ablation_base_alpha_iid_p50.png` está ausente.

Os PDFs versionados não foram alterados. O índice em `report/README.md` agora
separa o bloqueio do sandbox dos problemas de proveniência dos relatórios
históricos.

### Decisão

Não alterar o Tectonic, não instalar outra distribuição e não modificar as
fontes LaTeX nesta etapa. Para futuras compilações neste ambiente, usar uma
execução fora do sandbox ou aprovada com acesso equivalente. O próximo passo
continua sendo o registro operacional E01–E07 e a organização dos resultados
em `results/`, depois de decidir o tratamento das imagens históricas ausentes.

## 2026-07-31 — registro operacional e dry-run de publicação

### Objetivo

Criar o registro operacional E01–E07 e revisar a publicação da matriz oficial
E07 sem mover ou copiar arquivos.

### Implementação

- `experiments/registry.py` define os sete IDs, os estados e as raízes de
  resultados em `results/`.
- `experiments/__init__.py` exporta `ExperimentSpec`, `get_experiment()` e
  `output_path()`.
- `experiments/migrate_outputs.py` cria um plano somente leitura. A CLI exige
  `--dry-run` e recusa executar sem essa opção.
- `tests/test_experiment_registry.py` cobre os IDs, o destino do E07, IDs
  desconhecidos e nomes de dataset inválidos.
- `tests/test_migrate_outputs.py` cobre o mapeamento, a ausência de escrita,
  a origem ausente e o destino dentro da origem.

### Evidência do dry-run

O destino `results/e07-drift-agent/` não existia antes da execução e continua
ausente depois dela.

| Origem | Destino planejado | Arquivos | Estado |
|---|---|---:|---|
| `output/cifar-10/drift-agent-2/final-matrix-rerun/` | `results/e07-drift-agent/cifar-10/uniform/matrix-v1/` | 81 | Listado, não copiado |
| `output/cifar-10/severity-calibration/` | `results/e07-drift-agent/cifar-10/calibration/` | 1 | Listado, não copiado |
| `output/cifar-10/drift-agent-2/omp-smoke/` | `results/e07-drift-agent/cifar-10/uniform/smoke/` | 3 | Listado, não copiado |

A matriz contém 25 pares, cinco gráficos e um resumo. O smoke continua
separado da matriz científica. A calibração continua em sua origem oficial.

### Testes

`python3 -m unittest tests.test_experiment_registry tests.test_migrate_outputs -v`
passou em 7 testes. O `.venv` do projeto existe, mas não fornece `pytest`.
O teste usou o Python global e não alterou o repositório.

A regressão dos módulos E07 passou em 53 testes com o `.venv` e caches
temporários graváveis para Matplotlib e fontconfig. O Python global não tem as
dependências científicas do projeto. O primeiro teste no `.venv` abortou ao
tentar usar caches não graváveis do sandbox.

### Decisão

O registro e o contrato de dry-run estão prontos. Nenhum arquivo foi movido,
copiado, apagado ou criado em `results/`. A próxima ação é implementar e
executar a cópia validada somente após autorização explícita.

## 2026-07-31 — publicação autorizada e validação da cópia E07

O usuário autorizou a execução da cópia depois da revisão do dry-run. A
implementação foi ampliada para aceitar `--copy` com as seguintes proteções:

- o destino precisa estar ausente;
- o destino não pode estar dentro da origem;
- a cópia ocorre primeiro em uma pasta temporária no mesmo diretório-pai;
- os `pair-manifest.json` são copiados por último;
- os arquivos publicados são comparados por SHA-256 com a origem;
- os pares são carregados com `load_persisted_pair` antes da publicação;
- a origem permanece intacta.

### Comandos executados

```text
.venv/bin/python -m experiments.migrate_outputs --source output/cifar-10/drift-agent-2/final-matrix-rerun --destination results/e07-drift-agent/cifar-10/uniform/matrix-v1 --experiment e07-drift-agent --copy
.venv/bin/python -m experiments.migrate_outputs --source output/cifar-10/severity-calibration --destination results/e07-drift-agent/cifar-10/calibration --experiment e07-drift-agent --copy
.venv/bin/python -m experiments.migrate_outputs --source output/cifar-10/drift-agent-2/omp-smoke --destination results/e07-drift-agent/cifar-10/uniform/smoke --experiment e07-drift-agent --copy
```

### Resultado

| Origem | Destino | Arquivos | Resultado da validação |
|---|---|---:|---|
| `output/cifar-10/drift-agent-2/final-matrix-rerun/` | `results/e07-drift-agent/cifar-10/uniform/matrix-v1/` | 81 | 25 pares válidos; arquivos e SHA-256 coincidem |
| `output/cifar-10/severity-calibration/` | `results/e07-drift-agent/cifar-10/calibration/` | 1 | arquivo e SHA-256 coincidem |
| `output/cifar-10/drift-agent-2/omp-smoke/` | `results/e07-drift-agent/cifar-10/uniform/smoke/` | 3 | 1 par válido; arquivos e SHA-256 coincidem |

A validação independente confirmou os conjuntos relativos de arquivos, os
digests SHA-256 e a leitura de todos os pares publicados. A matriz oficial
continua separada do smoke. Nenhum arquivo foi movido, apagado ou sobrescrito
na origem. Os artefatos em `results/` estão não rastreados pelo Git; não houve
staging nem commit.

### Verificações

- Os testes do registro e da migração passaram em 9 testes.
- A regressão dos módulos E07 passou em 53 testes com caches temporários
  graváveis para Matplotlib e fontconfig.
- A cópia foi validada novamente fora do teste unitário por contagem, pares e
  SHA-256.

### Próximo ponto de revisão

Revisar o diff dos artefatos publicados e decidir se `results/e07-drift-agent/`
deve ser staged e commitado. Depois, analisar os 25 pares sem incluir o smoke
ou continuar a explicitar namespaces e diretórios de saída nos entrypoints.

## 2026-07-31 — namespaces e raízes de saída explícitas

A reorganização estrutural continuou sem alterar o protocolo ou os artefatos
oficiais do E07. O objetivo desta etapa foi tornar a família visível no comando
e evitar que uma nova execução escreva diretamente em `results/`.

### Código

- Foram criados namespaces para E01–E07 em `experiments/eXX_*`.
- Os entrypoints novos são adapters dos módulos planos existentes. Isso reduz
  o risco de uma migração de imports alterar a execução histórica.
- O E07 agora pode ser executado por
  `experiments.e07_drift_agent.calibrate`,
  `experiments.e07_drift_agent.run_matrix`,
  `experiments.e07_drift_agent.plot` e
  `experiments.e07_drift_agent.episode`.
- E02, E04, E05 e E06 também têm entradas namespaced de compatibilidade.
- `temporary_output_path()` define raízes temporárias em
  `output/<experiment-id>/<dataset>/`. `output_path()` continua reservado a
  `results/<experiment-id>/<dataset>/`.
- Os runners síncrono e assíncrono, a ablação, o drift temporal e as entradas
  E07 aceitam `--output-dir` ou já possuíam `--output`. Uma raiz explícita não
  altera o default global do dataset.

### Documentação e retenção

- O catálogo, README, receitas rápidas, comparador e handoff do E07 apontam
  para os namespaces organizados.
- `.gitignore` mantém `output/` temporário e deixa `results/` visível para o
  Git.
- `CONTEXT.md` agora distingue namespace, raiz temporária de execução e layout
  canônico de publicação.
- Os adapters e as implementações planas continuam preservados. Nenhum
  resultado histórico foi removido.

### Verificações

- 67 testes passaram: registro, namespaces, raízes de saída, migração e
  regressão do E07.
- Todos os namespaces planejados passaram pelo teste de importação.
- Os comandos `--help` dos runners namespaced principais passaram.
- `py_compile` passou para os módulos de `experiments`, `src` e `tests`.

### Decisão

Esta etapa conclui a primeira camada de organização: identidade do experimento,
namespace de entrada e separação entre execução temporária e publicação. A
próxima camada é migrar imports para os namespaces sem duplicar o motor. A
análise científica permanece fora do escopo.

## 2026-07-31 — validação final da camada de organização

Depois do ajuste do default programático de `DriftEpisodeConfig`, a suíte
completa foi executada novamente:

- `MPLCONFIGDIR=/private/tmp/fl-mpl-cache-20260731-final2 XDG_CACHE_HOME=/private/tmp/fl-xdg-cache-20260731-final2 MPLBACKEND=Agg .venv/bin/python -m unittest discover -s tests`
- 142 testes executados.
- 140 testes passaram.
- 2 testes foram ignorados porque CUDA/MPS não estavam disponíveis.
- `git diff --check` passou.
- `py_compile` passou para `experiments`, `src` e `tests`.
- A verificação de links encontrou 52 links locais e 0 ausências.
- Os três destinos E07 continuam com os mesmos arquivos e SHA-256 das
  origens: 81/25 pares na matriz, 1 arquivo na calibração e 3/1 par no smoke.
- Os checkpoints continuam exatamente em `STATE.md`, `JOURNAL.md` e
  `CONTEXT.md`.

Esta validação encerra as tarefas restantes do plano de reorganização. O
próximo trabalho possível é migrar os imports internos para os namespaces e
depois revisar o diff para staging/commit. A análise científica não faz parte
desse trabalho.

## 2026-07-31 — migração das raízes de saída legadas

O inventário confirmou que `output-cifar-10/` continha 38 JSONs rastreados
desde o commit `8462356` e misturava E04 com E05. `output-mnist/` continha 14
arquivos ignorados. Dez eram smoke tests legados do E06. Um era QA do E05. Os
três arquivos restantes eram uma comparação entre E05 e E06.

As cópias foram criadas nestes destinos:

| Origem | Destino | Arquivos |
|---|---|---:|
| `output-cifar-10/` E04 | `results/e04-static-comparison/cifar-10/raw/` | 4 |
| `output-cifar-10/` E05 | `results/e05-temporal-drift/cifar-10/raw/` | 34 |
| `output-mnist/` E05 QA | `results/e05-temporal-drift/mnist/qa/` | 1 |
| `output-mnist/` E06 | `results/e06-drift-agent-prototype/mnist/legacy/` | 10 |
| `output-mnist/` QA cruzado | `results/qa/cross-experiment/mnist/` | 3 |

As 52 cópias passaram por comparação byte a byte. Os arquivos de origem foram
removidos somente depois dessa validação. As raízes `output-cifar-10/` e
`output-mnist/` não existem mais.

O diretório `drift-agent/` não existia no checkout no momento da migração. O
destino `results/e06-drift-agent-prototype/cifar-10/legacy-v2/` contém somente
seu README. Não foram criados dados ausentes.

Os defaults das UIs de E04, E05 e E06 agora apontam para os caminhos nomeados.
O carregador de datasets usa `output/e01-static/<dataset>/` quando não recebe
uma raiz explícita. O catálogo e o README de `results/` documentam a nova
fronteira entre saída temporária, histórico e publicação oficial.
