# Índice dos relatórios

Este diretório contém o documento de origem do E01, relatórios históricos e
fontes LaTeX. Ele não contém a saída temporária das execuções.

Leia este índice antes de usar um relatório como evidência. O PDF mostra o que
foi documentado. A configuração persistida e os resultados publicados mostram
como uma execução específica foi feita.

## Ordem de leitura

1. Leia o [Relatório Final FAPESP](Relatório_Final_FAPESP.pdf) para entender a
   origem e a motivação do E01.
2. Leia o [relatório inicial](relatorio.pdf) e os
   [Resultados Experimentais](Resultados%20Experimentais.pdf) para entender o
   primeiro simulador e seus resultados.
3. Leia [Novos Experimentos](resultados.pdf) para E02 e E03.
4. Leia [Comparação Sync/Async](comparacao_sync_async.pdf) para E04.
5. Leia [Drift Temporal Sync/Async](drift_temporal_sync_async.pdf) para E05.
6. Leia o [handoff do E07](../docs/literatura/15.%20handoff-experimento-drift-agent.md)
   e os resultados publicados do E07 para o fluxo atual.

## Mapa dos documentos

| Documento | Experimento | Papel | Fonte LaTeX | Verificação atual |
|---|---|---|---|---|
| `Relatório_Final_FAPESP.pdf` | E01 | Origem e motivação do estudo inicial | Não presente neste diretório | PDF preservado. Não é um PDF compilável aqui. |
| `relatorio.pdf` | E01 | Relatório inicial do simulador | `relatorio.tex` | PDF preservado. A fonte aponta para imagens antigas fora do checkout atual. |
| `Resultados Experimentais.pdf` | E01 e estudos iniciais | Primeira consolidação dos resultados | Não presente neste diretório | PDF preservado. A fonte original não está presente aqui. |
| `resultados.pdf` | E02 e E03 | Novos experimentos, ablação e timeout | `resultados.tex` | PDF preservado. A fonte referencia gráficos que não estão em `output-cifar-10/`. |
| `comparacao_sync_async.pdf` | E04 | Comparação estática entre Sync e Async | `comparacao_sync_async.tex` | Todos os gráficos foram encontrados e a fonte compilou com Tectonic 0.16.9 fora do sandbox do Codex. |
| `drift_temporal_sync_async.pdf` | E05 | Comparação sob drift temporal de classes | `drift_temporal_sync_async.tex` | Todos os gráficos foram encontrados e a fonte compilou com Tectonic 0.16.9 fora do sandbox do Codex, com avisos de referências e caixas largas. |

## Verificação das fontes

Os arquivos `.tex` são fontes históricas. Eles não devem ser movidos até que
seus caminhos de imagens e dependências estejam verificados.

Nesta verificação:

- `comparacao_sync_async.tex` encontrou os seis gráficos esperados em
  `results/e04-static-comparison/<dataset>/plots/`.
- `drift_temporal_sync_async.tex` encontrou os nove gráficos esperados em
  `results/e05-temporal-drift/cifar-10/plots/temporal/`.
- `relatorio.tex` aponta para diretórios antigos em
  `synchronous/output-cifar-10/` e `asynchronous/output-cifar-10/`. Esses
  diretórios não existem no checkout atual.
- `resultados.tex` referencia gráficos de ablação e acurácia que não existem
  em `output-cifar-10/`.
- O Tectonic 0.16.9 está instalado e passa no smoke test quando executado fora
  do sandbox do Codex. Dentro do sandbox, ele falha antes de processar o
  documento com um panic de `system-configuration` ao inicializar a camada de
  rede do macOS. `--only-cached` e proxies vazios não evitam o panic.
- `relatorio.tex` foi executado fora do sandbox e parou em
  `../synchronous/output-cifar-10/accuracy_iid.png`, que não existe no
  checkout atual.
- `resultados.tex` foi executado fora do sandbox e parou em
  `ablation_base_alpha_iid_p50.png`, que não existe no checkout atual.
- Não existe TeX Live ou MacTeX disponível neste ambiente. O Tectonic é
  suficiente para os dois relatórios cujas imagens ainda estão presentes.

Por isso, esta verificação confirma a compilação de `comparacao_sync_async.tex`
e `drift_temporal_sync_async.tex`, mas não confirma a recompilação dos dois
relatórios históricos com imagens ausentes. Os PDFs preservados continuam
válidos como registros históricos. Não devemos prometer a recompilação de
`relatorio.tex` ou `resultados.tex` sem recuperar os gráficos ou atualizar as
fontes com uma decisão documental.

## Regra de organização

`report/` deve permanecer separado de `output/` e `results/`:

- `report/` guarda documentos de origem, relatórios e fontes.
- `output/` guarda artefatos temporários.
- `results/` guarda artefatos experimentais publicados, históricos ou de QA.

Os caminhos `synchronous/output-cifar-10/`, `asynchronous/output-cifar-10/`
e `output-cifar-10/` citados acima pertencem ao layout histórico dos relatórios.
Eles não são raízes válidas para novas execuções.

A migração dos gráficos históricos foi concluída em 2026-07-31. Os pares PDF e
`.tex` continuam em `report/`. As fontes usam os artefatos publicados em
`results/`.
