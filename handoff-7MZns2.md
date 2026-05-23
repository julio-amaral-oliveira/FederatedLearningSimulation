# Handoff - FederatedLearningSimulation

## Contexto
Projeto em C:\Users\julio\Documents\Códigos\FederatedLearningSimulation. O usuário está comparando FL síncrono vs assíncrono no CIFAR-10, com foco em tempo virtual, stragglers e métricas por JSON {loss, accuracy, time}.

Instruções relevantes do repo estão em AGENTS.md e foram seguidas: preservar formato JSON, usar --output-prefix, reutilizar utils/, e para análise de resultados calcular métricas padronizadas. Há worktree suja; não reverter mudanças que não forem suas.

## Mudanças principais feitas nesta sessão
- src/synchronous foi convertido para tempo virtual e abordagem de stragglers compatível com src/asynchronous/constants.py:
  - src/synchronous/constants.py: MAX_CONNECTION_TIME = 1, SPEED_TIERS, SPEED_TIER_SEED, SIMULATION_SEED.
  - src/synchronous/main.py: atribui tiers por cliente, aceita --num-rounds, --output-prefix, --eval-every.
  - src/synchronous/server.py: remove threads/sleeps, amostra duração virtual por cliente/tier, registra tempo virtual, avalia de forma periódica.
  - src/synchronous/client.py: usa perform_fit(round_start_weights, ...).
  - src/synchronous/monte_carlo.py: estima timeout por mistura de tiers, ainda por round.
- src/asynchronous/main.py e src/asynchronous/server.py: adicionado --eval-every/evaluation_frequency para reduzir custo de avaliação sem mudar o fluxo de treino; JSON salva pontos avaliados e ponto final.
- experiments/plot_comparison.py: refeito para aceitar prefixos, calcular as três perguntas, e plotar alvo/horizonte.
- Testes adicionados:
  - tests/test_synchronous_virtual_time.py
  - tests/test_comparison_metrics.py

## Experimentos executados
Bateria atual concluída em CIFAR-10 com p50, horizonte ~4000s:
- Síncrono: 706 rounds, --eval-every 10
- Assíncrono: 706 updates por cliente, --eval-every 10
- Prefixos:
  - Sync: compare_4000_eval10_sync
  - Async: compare_4000_eval10_async

JSONs gerados:
- output-cifar-10/accuracy_data_iid_compare_4000_eval10_sync.json
- output-cifar-10/accuracy_data_non_iid_compare_4000_eval10_sync.json
- output-cifar-10/accuracy_data_iid_compare_4000_eval10_async.json
- output-cifar-10/accuracy_data_non_iid_compare_4000_eval10_async.json

Resumo e gráficos:
- experiments/comparison_4000_summary.md
- experiments/comparison_iid.png
- experiments/comparison_non_iid.png

Não repetir o resumo todo se não for necessário; use experiments/comparison_4000_summary.md como fonte. Resultado de alto nível: async superou sync em IID e Non-IID até 4000s, mas levou muito mais wall-clock porque processou muito mais agregações/eventos.

## Observação importante sobre custo do assíncrono
O usuário perguntou por que o assíncrono demorou muito mais. Resposta dada: para o mesmo tempo virtual, o síncrono faz uma agregação global por round (~706 agregações), enquanto o assíncrono processa cada update que chega antes do timeout total. Nos experimentos de 4000s, os logs finais mostraram Total de agregacoes: 19836 para async IID e Non-IID, cerca de 28x mais agregações que o síncrono.

A tentativa inicial de 10000s foi interrompida por inviabilidade interativa; ela gerou apenas logs .err vazios e nenhum JSON final. Depois a bateria de 4000s foi concluída.

## Comandos/verificação usados
Ambiente Python usado explicitamente:
C:\Users\julio\anaconda3\envs\federated-learning\python.exe

Variável necessária no Windows para evitar erro OpenMP:
$env:KMP_DUPLICATE_LIB_OK='TRUE'

Verificação final passou:
C:\Users\julio\anaconda3\envs\federated-learning\python.exe -m unittest tests.test_comparison_metrics tests.test_synchronous_virtual_time -v

Saída final: 5 testes OK.

Script de análise final:
C:\Users\julio\anaconda3\envs\federated-learning\python.exe experiments\plot_comparison.py --dataset cifar10 --sync-dir output-cifar-10 --async-dir output-cifar-10 --sync-prefix compare_4000_eval10_sync --async-prefix compare_4000_eval10_async --percentile 50 --horizon-seconds 4000 --eval-every 10

## Estado do worktree
git status --short --untracked-files=all mostrou modificados:
- experiments/plot_comparison.py
- src/asynchronous/main.py
- src/asynchronous/server.py
- src/synchronous/client.py
- src/synchronous/constants.py
- src/synchronous/main.py
- src/synchronous/monte_carlo.py
- src/synchronous/server.py

Untracked relevantes da sessão:
- tests/test_comparison_metrics.py
- tests/test_synchronous_virtual_time.py
- experiments/comparison_4000_summary.md
- experiments/comparison_iid.png
- experiments/comparison_non_iid.png
- logs .err de tentativas em experiments/*compare*log.err (vazios ou irrelevantes; podem ser limpados se o usuário pedir).

Untracked/preexistentes que não devem ser assumidos como nossos sem checar:
- AGENTS.md
- report/comparacao_sync_async.pdf
- report/comparacao_sync_async.tex

## Próximos passos sugeridos
1. Se a próxima sessão for de análise científica: usar academic-deep-research antes de interpretar mecanismos como heterogeneidade Non-IID, staleness, ou comparar FedAvg/FedAsync com literatura.
2. Se for preparar relatório: usar experiments/comparison_4000_summary.md e os PNGs, mas checar visualmente os gráficos.
3. Se for engenharia: revisar se --eval-every deve virar padrão documentado e se logs antigos .err devem ser removidos.
4. Se for controle de versão: revisar diff cuidadosamente antes de commit; não adicionar Co-Authored-By:.

## Skills recomendadas para a próxima sessão
- superpowers:verification-before-completion antes de qualquer conclusão.
- academic-deep-research se for interpretar os resultados com literatura.
- superpowers:receiving-code-review se o usuário revisar os resultados/código e pedir ajustes.
- pdf ou documents se for transformar os resultados em relatório formal.
