# E08 — Drift parcial escalonado e política de gatilho coletivo

> Especificação do desenho aprovado em 2026-08-05 (revisão 2).
> O estado vigente e o histórico estão em `checkpoints/e08-drift-parcial/`.

## Pergunta do experimento

Qual é a fronteira de disparo da política de gatilho coletivo (quorum,
janela) quando o drift é parcial e escalonado, e qual é o custo da mistura
para os clientes limpos?

O E08 mede a fronteira. Ele não tenta fazer a política funcionar sob drift
parcial.

## Fundamentação na literatura publicada

### Quorum — método, não valor

Nenhum trabalho publicado usa fração de clientes flagados como gatilho.
A prática publicada é a varredura de limiar com curva de trade-off:

- DriftGuard (arXiv:2603.18872, 2026): varre o limiar de gatilho global
  τ ∈ [0.48, 0.59] e mede acurácia contra custo de retreino. O impacto é
  limitado até τ = 0.54; o custo dispara acima de τ = 0.57.
- FedPLC (Sensors 26(1):283, 2026): ablação do limiar de comunidade
  ζ ∈ [0.6, 0.95], melhor em [0.8, 0.9].
- FLARE (IWCMC 2023): parâmetros escolhidos empiricamente com validation
  set, valores estáticos declarados.

Conclusão: o quorum continua parâmetro próprio do projeto. A varredura com
curva de custo segue a metodologia publicada.

### Janela — teoria liga janela à velocidade do drift

- "The Window Dilemma" (Gower-Winter, Groen & Krempl, arXiv:2602.06456,
  2026): janela pequena dá reatividade; janela grande dá estabilidade.
  Drift rápido exige janela pequena; drift lento exige janela grande.
  Quase todos os detectores tratam o tamanho da janela como parâmetro
  ajustável.
- Tschumitschew & Klawonn (2016): análise teórica do tamanho ótimo de
  janela sob drift e ruído. Drift forte pede janela pequena (até 1).
  Ruído alto pede janela maior. Janela errada pode dobrar o erro.
- Kuncheva & Žliobaitė (2009): relação teórica entre tamanho de janela e
  erro de classificação na transição de drift.
- Valores publicados: janela 1 e 2 (FedDrift, win-1/win-2), janela de 10
  rounds (FL-MalDrift), 1000 amostras (Casado), janela adaptativa (ADWIN).

Conclusão: a grade janela × ritmo testa diretamente a relação teórica
publicada entre velocidade do drift e tamanho da janela.

### Tempo entre drifts — cenário, com teoria de detectabilidade

- A maioria injeta um drift por execução. FedDrift usa 10 time steps com
  padrões fixos declarados. FedPLC usa duas fases: abrupta no round t0 e
  assíncrona depois.
- Webb et al. definem Drift Rate (frequência do drift) e Drift Duration.
- O Window Dilemma formaliza o caso "blip drift": conceito com pouca
  persistência pode ser fundamentalmente indetectável, porque a amostragem
  não tem poder estatístico.

Conclusão: o ritmo do stagger é parâmetro de cenário, declarado no schema.
A teoria prevê o regime de falha que a grade deve confirmar ou refutar.

## Desenho final (revisão 2)

| Eixo | Valor | Justificativa |
|---|---|---|
| Corrupção | motion_blur:1 | Detecção mais confiável do E07. Isola a política do ruído do detector |
| Driftados | k = 5 de 10, ids 0..4 | Quorum 0.5 exige 5 flags. Com k < 5, a coluna q=0.5 morre por construção |
| Quorum | {0.2, 0.3, 0.5} | Varredura com curva de custo, como DriftGuard e FedPLC |
| Janela | {1, 2, 5} | Intervalo do FedDrift (1, 2) ao padrão do E07 (2), mais 5 |
| Ritmo do stagger | {1, 3, 5} ticks entre onsets | Cenário declarado. 1 = rápido, 3 = médio, 5 = lento |
| Seeds | 42 | Leitura inicial. Expande só se justificar |
| Fixos | rounds 1, T 5, horizonte 400 s, janela de monitor 10 s | Matriz oficial do E07 |
| Dados do retreino | Driftados: partição corrompida. Limpos: partição limpa | Realismo. Sem isso o cenário é mentiroso |
| Onset | Escalonado por cliente, deterministico | Padrão declarado no schema, no formato do FedDrift |

Grade completa: 3 quoruns × 3 janelas × 3 ritmos = 27 configurações.
Com 1 seed: 27 pares, cerca de 1 hora.

Controle: identidade (sem drift) para medir falso positivo da política.
Opcional na primeira rodada.

## Linha de disparo esperada

O atraso de detecção do motion_blur no E07 é de cerca de 88 s (tick 9).
No E08, a flag de cada cliente chega cerca de 9 ticks depois do onset dele,
com os onsets no frame de produção (0 = primeiro tick corrompido da
produção; o warm-up não conta).

| Ritmo (ticks entre onsets) | Flags chegam nos ticks | q=0.2 (2 flags) | q=0.3 (3 flags) | q=0.5 (5 flags) |
|---|---|---|---|---|
| 1 | 10, 11, 12, 13, 14 | dispara com W >= 2 | dispara com W >= 3 | dispara com W = 5 |
| 3 | 10, 13, 16, 19, 22 | dispara com W >= 4 | nunca com W <= 5 | nunca |
| 5 | 10, 15, 20, 25, 30 | nunca com W <= 5 | nunca | nunca |

Leitura esperada: a relação teórica publicada se confirma. Drift rápido
(ritmo 1) exige janela pequena e dispara até para q=0.5 com W=5. No ritmo 3
apenas q=0.2 com W=5 dispara (2 flags a 3 ticks de distância exigem W >= 4).
O ritmo 5 entra no regime de falha: flags a 5 ticks de distância, então 2
flags na janela exigem W >= 6 e nenhum quorum dispara com W <= 5, apesar de
5 de 10 clientes driftarem.

Os valores exatos dependem do atraso de detecção real por cliente. A
tabela é previsão, não promessa.

## Função de custo

Pesos iguais para os três termos, com normalização por unidade:

```text
custo = downtime / 400 + retenção / 0.21 + retreinos / 1
```

- downtime em segundos, normalizado pelo horizonte.
- retenção limpa em pontos de acurácia, normalizado pela perda máxima
  observada no E07 (0.21).
- retreinos em contagem (0 ou 1).

A normalização é premissa declarada. A sensibilidade aos pesos é análise
futura.

## Métricas novas

- Linha de disparo: menor (quorum, janela) que dispara por ritmo.
- Custo da mistura: acurácia do grupo limpo no teste limpo e do grupo
  driftado no teste corrompido, após o retreino. Métricas globais
  existentes (clean_retention_delta, corrupted_accuracy_gain) continuam.
- Falso positivo: flags sem drift (controle identidade).

## Status da implementação — CONCLUÍDA

As mudanças de código listadas abaixo foram implementadas e registradas no
registry. A suíte completa passa, com a única exceção pré-existente de
hardware (MPS device, presente no commit base dd42c1c). O diff no E07 toca
apenas `episode.py`, `run_matrix.py`, `drift_schedule.py` e testes; nada em
`experiments/shared/drift_results.py` nem em `result_io.py`.

1. Config: campos `drifted_client_ids` e `drift_onset_ticks` — `episode.py`.
2. Monitor: corromper só os driftados, e só após o onset deles — helpers
   puros em `drift_schedule.py` usados pelo monitor do `episode.py`.
3. Retreino: dataset por cliente. Driftado treina com a partição
   corrompida. Limpo treina com a partição limpa — `build_retrain_datasets`
   em `drift_schedule.py`.
4. Schema: campos novos na sumarização — as dimensões do schedule são
   reportadas por configuração em `run_matrix.py`; não houve mudança no
   `validate_pair`.
5. Métricas: linha de disparo e custo da mistura — a grade calcula a linha
   de disparo por par; o custo da mistura usa as métricas existentes
   (`clean_retention_delta`, `corrupted_accuracy_gain`).
6. Registro no registry: família `e08-partial-drift` em
   `experiments/shared/registry.py`.

## Execução da grade

```bash
python -m experiments.e08_partial_drift.run_grid --output-dir output/e08-partial-drift/cifar-10/grid-v1
```

27 pares (3 quoruns × 3 janelas × 3 ritmos), seed 42, cerca de 1 hora.

## Eixos adiados

- Varredura de cobertura (k = 1 a 10): a pergunta original do quorum como
  política de cobertura. Fica para uma rodada posterior.
- Seeds 42 a 46: expande só se a leitura da seed 42 justificar.
- Onset aleatório: exige Monte Carlo (50 trials na literatura). Fora do
  orçamento. Padrões determinísticos declarados são a prática do FedDrift.
- Ação por grupo (retreinar só driftados): desenho FedDrift/DriftGuard.
  Trabalho futuro.
