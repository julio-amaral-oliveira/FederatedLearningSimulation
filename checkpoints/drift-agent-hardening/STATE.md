# Estado atual — drift-agent-hardening

> Verdade vigente do projeto em 2026-07-31. Para a evolução e as evidências,
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
- Último commit de código observado: `b059545`
  (`feat: enhance plotting functions with improved legends and bar representation`).
- Base da implementação endurecida: `c9e9355`.
- A revisão final e a re-revisão concluíram **Ready**.
- A última verificação completa executou 142 testes.
- A verificação passou em 140 testes.
- A verificação ignorou 2 testes condicionais de CUDA/MPS.
- O experimento usa `uniform` como default.
- Os módulos legados síncrono e assíncrono usam `heterogeneous` como default.
- O schema v3 registra `client_speed_profile` e `client_speed_tiers`.
- Os testes cobrem o default, a seleção explícita e a divisão do resumo.
- Não houve push nem trailers `Co-authored-by`.
- Alterações locais preexistentes do usuário foram preservadas. Em especial,
  os documentos 11 e 12 estavam não rastreados e são contexto, não prova do
  estado commitado.
- `docs/experimentos.md` é o catálogo humano das famílias de experimentos.
  Ele organiza a linha do tempo e identifica E01 a E07.
- `report/Relatório_Final_FAPESP.pdf` é o documento de origem do E01. Ele
  registra a motivação e o ponto de partida do simulador.
- E07 é o experimento atual de drift agent. E01 a E06 são referências históricas
  ou estudos anteriores e não devem ser agregados com E07.
- Os IDs E01 a E07 ainda não fazem parte dos nomes dos diretórios ou JSONs.
  O catálogo é a fonte de identificação até uma futura mudança de nomenclatura.

Referências principais:

- [Desenho do endurecimento](../../docs/literatura/13.%20endurecimento-experimento-drift-agent.md)
- [Plano de implementação](../../docs/literatura/14.%20plano-endurecimento-experimento-drift-agent.md)
- [Handoff e comandos](../../docs/literatura/15.%20handoff-experimento-drift-agent.md)
- [Catálogo de experimentos](../../docs/experimentos.md)

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
    A calibração oficial seleciona `gaussian_noise:3`,
    `frosted_glass_blur:4`, `motion_blur:1` e `fog:4`.
    O controle Oracle `identity:0` completa 25 pares.
17. **Catálogo de experimentos.** `docs/experimentos.md` define as fronteiras
    e os nomes E01 a E07 para leitura humana. Ele não substitui este estado,
    o diário ou o glossário.
18. **Reorganização por família.** `output/` é a área temporária e ignorada.
    A raiz futura de resultados oficiais usa
    `results/<experiment-id>/<dataset>/`. O código de entrada usa namespaces
    `experiments/eXX_*`. A migração começa pelo E07 e copia os resultados antes
    de alterar ou remover qualquer origem.
19. **Origem documental do E01.** `report/Relatório_Final_FAPESP.pdf` é a
    origem do E01. `report/` mantém documentos de origem, relatórios históricos
    e fontes LaTeX. Esses documentos não substituem a configuração persistida
    dos resultados.
20. **Separação do código compartilhado.** `src/` mantém os motores síncrono e
    assíncrono, os utilitários de dados e os módulos reutilizáveis. O código
    específico de cada estudo fica nas entradas de `experiments/eXX_*` sem
    duplicar o motor compartilhado.
21. **Índice e compilabilidade dos relatórios.** `report/README.md` classifica
    o FAPESP como origem do E01, os relatórios históricos e as fontes LaTeX.
    A existência de um PDF não prova que sua fonte ainda compile no checkout
    atual.
22. **Tectonic instalado e limitação do sandbox.** O Tectonic 0.16.9 bundled e
    o executável do Homebrew estão instalados. Ambos compilam fora do sandbox
    do Codex e ambos falham dentro dele com o mesmo panic de
    `system-configuration` ao inicializar a configuração de rede do macOS.
    `--only-cached` e variáveis de proxy vazias não evitam o panic. A
    compilação deve usar um processo fora do sandbox ou uma execução aprovada
    com acesso equivalente. Não instalar TeX Live por causa desse problema.
23. **Registro e publicação não destrutiva.** `experiments/registry.py` é o
    registro operacional de E01–E07. `output_path()` aponta para a publicação
    em `results/` e `temporary_output_path()` aponta para a execução temporária
    em `output/`. `experiments/migrate_outputs.py` exige `--dry-run` ou
    `--copy`, recusa destinos existentes e valida a cópia antes da publicação
    atômica. A calibração, a matriz de 25 pares e o smoke do E07 foram
    publicados em destinos separados dentro de `results/`.

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

### Calibração oficial uniforme

O arquivo oficial é
`output/cifar-10/severity-calibration/severity_calibration.json`.
Ele usa a seed base 42, a seed de teste 10041 e schema 1.
O checkpoint da calibração é `de2fe6b5...`.

- `gaussian_noise:3`: 0.3783
- `frosted_glass_blur:4`: 0.4022
- `motion_blur:1`: 0.3493
- `fog:4`: 0.4875.

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

### Saída uniforme anterior

A árvore `output/cifar-10/drift-agent/final-matrix` existe, mas não está limpa.
Ela contém 25 pares e cinco grupos: `frosted_glass_blur:4`, `motion_blur:1`,
`fog:4`, o controle `identity:0` e o resíduo `gaussian_noise:3`.
Ela é uma saída anterior e não é a matriz oficial atual.

O cenário `motion_blur:1` contém o gráfico agregado regenerado em
`seeds_42-46_comparison.png`. O gráfico tem 2720 × 1815 pixels e passou na
validação PNG. Não existem arquivos `comparison.png` antigos nos diretórios
de seed.

### Matriz científica oficial importada: `drift-agent-2`

Em 2026-07-31, o usuário informou que refez a execução em outra máquina e
importou os artefatos em `output/cifar-10/drift-agent-2/`.

- `final-matrix-rerun/` contém 25 pares: cinco seeds para cada grupo
  `fog_sev4`, `frosted_glass_blur_sev4`, `gaussian_noise_sev3`, `identity_sev0`
  e `motion_blur_sev1`.
- `omp-smoke/` contém um par adicional de `gaussian_noise_sev3` para smoke.
- Os 26 pares passaram por `load_persisted_pair`, incluindo schema v3,
  validação agente/baseline, manifest e digest dos JSONs.
- A configuração comum observada é CIFAR-10, perfil `uniform`, um round,
  horizonte de 400 s, timeout p75, `tau=0.50` e seeds 42–46.
- O runtime registrado é Python 3.11.14 do Anaconda, Torch 2.5.1+cu121 e
  `cuda:0`. Algoritmos determinísticos e `cudnn_deterministic` estão
  desativados.
- A calibração correspondente está no repositório em
  `output/cifar-10/severity-calibration/severity_calibration.json`.
- O digest do checkpoint limpo varia entre cenários para as seeds 42, 43 e 46.
  As seeds 44 e 45 compartilham o digest entre os cinco grupos. A validação
  de um par não prova a reutilização do mesmo checkpoint entre cenários.

Essa importação é a matriz científica oficial do E07. A seleção de
`gaussian_noise:3` vem da calibração oficial. O diretório `omp-smoke/` é uma
verificação técnica separada e não entra nos 25 pares.

### Visualização por corrupção implementada

Um subagente sem contexto anterior recriou o plotter do zero.
Ele usou apenas `smoke_drift.py` e exemplos JSON como contexto.
O painel superior mostra as cinco seeds como curvas em degraus.
Ele também mostra a média e a faixa mínima–máxima observada.
Três painéis inferiores mostram métricas pareadas por seed.
As métricas são acurácia corrompida final, downtime e retenção limpa.
O commit `90b230b` contém o código e os testes.
O plotter gerou cinco figuras agregadas para os grupos da matriz oficial. A
figura de `gaussian_noise:3` pertence à matriz científica atual.

### Investigação do gráfico `gaussian_noise:3`

O gráfico com cinco degraus na média não indica cinco retreinos por seed.
Cada seed tem exatamente uma decisão e um período de retreino.
As decisões ocorreram em 100, 110, 170, 190 e 280 segundos após o onset.
O plotter calcula a média com alinhamento temporal e forward-fill.
Por isso, a média muda quando cada seed conclui seu retreino.
O cenário Gaussian usa ruído aleatório por pixel, cliente e tick.
Essa variação explica os tempos de detecção diferentes.
O par Agent/Baseline conserva o mesmo checkpoint e o mesmo tempo de decisão
por seed.
Uma repetição independente foi interrompida durante o treinamento inicial.
Ainda não existe uma segunda execução completa para confirmar a repetibilidade
do tempo de decisão no ambiente atual.

## Bloqueios

1. **O checkpoint não é comum entre todos os cenários.**
   As seeds 42, 43 e 46 têm digests diferentes entre grupos.
   Registre essa variação como limitação da matriz final antes de interpretar
   diferenças entre cenários.
2. **O resultado não registra a participação por round.**
   O payload registra o perfil e os tiers.
   O payload não mostra quais clientes excederam o timeout.
   O JSON não prova a redução de retardatários.
3. **A exploração de rounds usa uma seed.**
   Ela ajuda a escolher o próximo teste.
   Ela não sustenta a escolha final da política.
4. **A perda limpa permanece alta.**
   Um round reduziu a acurácia limpa em 21.23 pontos percentuais.
5. **A reprodução bit a bit não está confirmada.**
   A matriz importada usa `cuda:0` com algoritmos determinísticos desativados.
   O piloto MPS continua como evidência histórica separada.
6. **A compilação dentro do sandbox não está disponível.** O Tectonic 0.16.9
   falha dentro do sandbox por um panic do runtime de rede do macOS, mas passa
   no smoke test e compila `comparacao_sync_async.tex` e
   `drift_temporal_sync_async.tex` fora do sandbox. `relatorio.tex` e
   `resultados.tex` ainda falham por imagens históricas ausentes. Não há TeX
   Live, MacTeX, `latexmk` ou `pdflatex` no PATH.

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
- Como a variação de digest entre cenários afeta a interpretação dos efeitos?
- O checkpoint limpo deve ser reutilizado por seed em uma matriz futura?
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

1. Migrar gradualmente os imports dos módulos planos para os namespaces,
   começando pelo E07, sem duplicar os motores compartilhados.
2. Manter os módulos planos como adapters de compatibilidade até os comandos e
   testes usarem os namespaces novos.
3. Revisar o diff, separar alterações preexistentes e decidir o staging/commit
   da reorganização e da publicação E07.
4. Não iniciar análise científica como parte da reorganização do código.

## Organização experimental vigente

O repositório contém sete estudos identificados no catálogo:

- E01, treinamento federado estático.
- E02, ablação da agregação assíncrona.
- E03, impacto do timeout síncrono.
- E04, comparação Sync contra Async em cenário estático.
- E05, drift temporal sazonal Sync contra Async.
- E06, drift agent inicial.
- E07, drift agent endurecido e auditável.

E07 é o único fluxo de drift considerado atual. E05 usa mudança de classes por
fases. E06 e E07 usam corrupção visual em produção. Esses protocolos não são
intercambiáveis.

## Mapa operacional vigente

O catálogo [docs/experimentos.md](../../docs/experimentos.md) agora classifica
os scripts, módulos, diretórios de saída e comandos por E01 a E07.

- Existem entrypoints planos e adapters namespaced entre `experiments/` e os
  módulos CLI de `src/` e `src/utils/`, além de módulos de suporte.
- Os entrypoints organizados do E07 são
  `experiments.e07_drift_agent.calibrate`,
  `experiments.e07_drift_agent.run_matrix` e
  `experiments.e07_drift_agent.plot`, executados com `python -m`. Os módulos
  planos antigos permanecem como implementação e compatibilidade durante a
  migração.
- `experiments/smoke_drift.py` continua sendo o motor de um episódio. Sua CLI
  direta é diagnóstica e não substitui a matriz oficial.
- `output-cifar-10/` mistura saídas históricas de E04 e E05.
- `drift-agent/` contém pares schema v2 legados.
- `output/cifar-10/severity-calibration/` contém a calibração oficial schema 1.
- `output/cifar-10/drift-agent/` contém uma saída anterior e não oficial.
- `output/cifar-10/drift-agent-2/final-matrix-rerun/` contém a matriz
  científica oficial schema v3 com 25 pares.
- `handoff-7MZns2.md` está marcado como documento histórico e aponta para as
  fontes atuais.
- `docs/receitas_rapidas.md` identifica as famílias históricas E02 e E05,
  corrige os nomes dos JSONs do E05 e aponta para o runbook do E07.
- O runbook e as receitas do E07 usam `output/` como origem de novas execuções
  e `results/e07-drift-agent/` como raiz publicada para auditoria.
- A saída antiga em `output/cifar-10/drift-agent/` permanece separada da
  matriz científica final.
- `report/README.md` foi criado como índice dos relatórios. Ele registra que
  `comparacao_sync_async.tex` e `drift_temporal_sync_async.tex` compilam fora
  do sandbox e que `relatorio.tex` e `resultados.tex` apontam para imagens
  ausentes no checkout atual.
- `experiments/registry.py` registra os sete IDs, seus estados e suas raízes
  em `results/`.
- `experiments/migrate_outputs.py` lista e publica mapeamentos de origem e
  destino. A cópia validada publicou 81 arquivos da matriz, 1 arquivo de
  calibração e 3 arquivos do smoke. Os 25 pares da matriz e o par do smoke
  foram carregados novamente com `load_persisted_pair`, e os digests SHA-256
  de origem e destino coincidiram.
- A primeira etapa física da reorganização foi concluída para E07. O layout
  está documentado em `docs/experimentos.md`; `output/` foi preservado como
  origem e `results/e07-drift-agent/` contém as cópias publicadas. Os arquivos
  em `results/` ainda estão não rastreados pelo Git até uma decisão explícita
  de staging e commit.
- Os namespaces `experiments/e01_static` até `experiments/e07_drift_agent`
  foram criados. Os adapters preservam os módulos planos enquanto os imports
  são migrados.
- Os runners síncrono, assíncrono, E02, E05 e E07 aceitam raízes explícitas.
  Novos defaults temporários usam `output/<experiment-id>/<dataset>/`; a raiz
  publicada em `results/` não é usada automaticamente por uma nova execução.
- O catálogo agora trata `src/` como código versionado compartilhado e `report/`
  como documentos versionados. O Relatório Final FAPESP está associado à
  origem do E01.
- A próxima mudança estrutural é migrar os imports dos módulos planos para os
  namespaces e completar a documentação de E01–E06. Staging e commit da
  publicação devem ocorrer somente após revisão do diff.

## 2026-07-31 — diagnóstico operacional do Tectonic

O diagnóstico do ambiente LaTeX foi concluído antes da reorganização dos
resultados:

- O Tectonic bundled está em
  `/Users/julio-vinicius/.codex/plugins/cache/openai-bundled/latex/0.2.4/bin/tectonic`.
- O Tectonic do Homebrew está em `/opt/homebrew/bin/tectonic`.
- Ambos informam `Tectonic 0.16.9` e reproduzem o panic dentro do sandbox.
- O cache do Tectonic existe em
  `~/Library/Caches/Tectonic/bundles` e contém os recursos usados pelos
  relatórios.
- O `latex_doctor.py` passou com `ready: true` fora do sandbox. Não há TeX
  Live ou MacTeX instalados.
- `comparacao_sync_async.tex` gerou um PDF de 9 páginas e
  `drift_temporal_sync_async.tex` gerou um PDF de 9 páginas em diretórios
  temporários. Os PDFs versionados não foram substituídos.
- `relatorio.tex` e `resultados.tex` chegaram ao processamento LaTeX, mas
  pararam por imagens históricas ausentes. Isso é um problema de proveniência
  dos artefatos, não do Tectonic.

O índice de relatórios foi atualizado para refletir essa evidência. Nenhum
arquivo do repositório foi criado pelo compilador.

### Próximo ponto de revisão

Escolher entre recuperar as imagens históricas ou marcar as duas fontes como
não recompiláveis. O registro operacional e a publicação validada em
`results/` foram retomados e estão descritos na seção abaixo; a decisão sobre
as imagens históricas continua separada dessa reorganização.

## 2026-07-31 — publicação validada dos resultados E07

A cópia autorizada do E07 foi executada sem mover, apagar ou modificar as
origens em `output/`:

| Origem | Destino publicado | Arquivos | Validação |
|---|---|---:|---|
| `output/cifar-10/drift-agent-2/final-matrix-rerun/` | `results/e07-drift-agent/cifar-10/uniform/matrix-v1/` | 81 | 25 pares válidos; SHA-256 coincidente |
| `output/cifar-10/severity-calibration/` | `results/e07-drift-agent/cifar-10/calibration/` | 1 | SHA-256 coincidente |
| `output/cifar-10/drift-agent-2/omp-smoke/` | `results/e07-drift-agent/cifar-10/uniform/smoke/` | 3 | 1 par válido; SHA-256 coincidente |

O comando de cópia usa diretório temporário, publica o manifest por último,
valida os pares e recusa sobrescrever um destino existente. A validação
independente comparou os conjuntos relativos de arquivos, recalculou os
digests SHA-256 e carregou os 25 pares da matriz e o par do smoke com
`load_persisted_pair`. A raiz `results/` agora contém a primeira publicação
física da reorganização e os arquivos aparecem como não rastreados pelo Git;
nenhum arquivo foi staged ou commitado.

O próximo ponto de revisão é decidir se a publicação E07 será staged/commitada
como um conjunto e, depois, continuar a análise dos 25 pares ou implementar
namespaces e argumentos de saída explícitos nos entrypoints.
