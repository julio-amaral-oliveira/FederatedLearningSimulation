# S3 — Sensibilidade das conclusões de downtime à duração do round de retreino

**Pedido do revisor (R3, MLOps):** verificar se as conclusões de downtime sobrevivem quando a
duração do round de retreino deixa de ser a virtual (≈8 s, `server.timeout`) e passa a ser
communication-dominated (minutos). Painel pediu sensibilidade d ∈ {8, 80, 800} s.

**Dados:** `results/e09-gradual-drift/cifar-10/grid-v1/<corrupcao>_sev<N>/seed_<S>/quorum_0.3_retrain_rounds_1_window_2_dropout_T_5_ramp_<R>/` (12 grupos × 5 seeds = 60 pares agente/baseline; 59 episódios com decisão, 1 sem).

---

## Regra de aproximação usada (passos 2–3)

Para cada episódio com decisão em t_d:

1. Trajetória stepwise reconstruída a partir de `corrupted_accuracy_history` (regra de
   `comparison_core.compute_downtime`: a acurácia do instante i vale até o instante i+1; o
   intervalo final conta até `end_time`; abaixo de τ = 0.50).
2. Round de duração d: o intervalo [t_d, t_d+d] é avaliado com a acurácia pré-retreino
   (registro `monitor_tick` exatamente em t_d — confirmado que o tick que dispara o detector
   tem `time == t_d`).
3. A partir de t_d+d: acurácia do **primeiro registro pós-retreino com `time >= t_d+d`**
   (registro `monitor_tick` — os ticks pós-retreino já avaliam o modelo retreinado com a
   fração f(v) crescente do teste misto, capturando o ponto 3 do pedido). Se não houver
   registro ≥ t_d+d, usa-se o último registro pós-retreino (não ocorreu: o `episode_end` em
   `end_time` sempre qualifica).
4. Orçamento (semântica do código `episode.py`: `required_seconds = retrain_rounds * timeout`,
   `budget_fits = (end_time - t_d) >= required_seconds`): se t_d + d > `end_time`, o controlador
   teria **pulado o retreino** (`retrain_skipped_budget`) e o agente degenera à baseline
   (downtime do `baseline.json` do mesmo par).
5. Episódio **sem decisão** (detector nunca disparou): o round nunca é executado — o downtime é
   invariante a d (mantido o valor registrado). Ocorrência única: `gaussian_noise_sev3/seed_42/ramp_10`
   (downtime = 540 s = baseline do par; ver "caso limite").
6. Downtime recomputado no horizonte [production_start, end_time] com a mesma regra stepwise;
   agregação por grupo (média sobre 5 seeds). Coluna d=8 usa o valor **registrado**
   (`metrics.downtime_seconds`).

### Validação e limitações

- **Validação d=8:** a reconstrução com d = 8.0 reproduz o downtime registrado com erro ≤ 0.0037 s
  (a duração virtual real do round é 8.0037 s; o nominal é 8.0). 50/60 episódios diferem por
  exatamente esse valor; irrelevante para a comparação.
- **Identidade exata:** para todos os 59 episódios com decisão (nenhum pula em d=80),
  `d80 = d8 + (80 − 8.0037)·[acc(t_d) < τ]` vale com erro ≤ 5.7e-14 — o custo do round estendido é
  exatamente o intervalo [t_d, t_d+80] quando a acurácia no disparo já está abaixo de τ, e **zero**
  quando o detector dispara cedo (acc(t_d) ≥ τ).
- **Limitação da aproximação (passo 2):** o intervalo [t_d, t_d+d] usa acc(t_d), ignorando o
  declínio intra-round da fração do teste misto. Sem impacto na classificação: em todos os 59
  episódios a decisão dispara com acc(t_d) < τ **ou** o valor em t_d já está acima de τ e a
  trajetória pós-t_d permanece acima até o horizonte (fog ramp 20 nas 5 seeds; frosted ramp 20
  nas seeds com detecção precoce, acc(t_d) ≥ τ ⇒ zero tempo abaixo de τ após t_d).
- **Limitação da aproximação (passo 3):** a acurácia de t_d+d é tomada do tick de monitoramento
  (grade de 10 s) mais próximo — quantização ≤ 10 s no início do segmento pós-retreino. A
  degradação pós-retreino com a fração crescente é real mas pequena: acc(t_d+8) − acc(t_d+80)
  tem média +0.0059 (17/59 episódios, máximo +0.043) e **nenhum** episódio cai abaixo de τ no
  tick de t_d+80 — a escolha do registro não altera nenhuma classificação neste grid.
- **Limitação do cenário d=800:** com d=800 > horizonte de produção (600 s), o round não cabe em
  *nenhum* episódio (t_d ≥ 259 s ⇒ t_d + 800 > end_time ≈ 759 s). A coluna d=800 representa,
  portanto, não "agente com round lento" mas **agente cujo retreino é pulado pelo orçamento** —
  degeneração à baseline por política do controlador, não por round longo em si.

---

## Resultados (média sobre 5 seeds, segundos)

| Corrupção | Severidade | Rampa (ticks) | d=8 | d=80 | d=800 | Baseline | Episódios que pulariam (d=800) |
|---|---|---|---|---|---|---|---|
| fog | 4 | 5 | 58.0 | 130.0 | 550.0 | 550.0 | 5/5 |
| fog | 4 | 10 | 24.0 | 96.0 | 504.0 | 504.0 | 5/5 |
| fog | 4 | 20 | 0.0 | 0.0 | 408.0 | 408.0 | 5/5 |
| frosted_glass_blur | 4 | 5 | 68.0 | 140.0 | 560.0 | 560.0 | 5/5 |
| frosted_glass_blur | 4 | 10 | 40.0 | 112.0 | 524.0 | 524.0 | 5/5 |
| frosted_glass_blur | 4 | 20 | 3.2 | 32.0 | 454.0 | 454.0 | 5/5 |
| gaussian_noise | 3 | 5 | 130.0 | 202.0 | 558.0 | 558.0 | 5/5 |
| gaussian_noise | 3 | 10 | 178.4 | 236.0 | 526.0 | 526.0 | 4/5* |
| gaussian_noise | 3 | 20 | 112.0 | 184.0 | 448.0 | 448.0 | 5/5 |
| motion_blur | 1 | 5 | 68.0 | 140.0 | 562.0 | 562.0 | 5/5 |
| motion_blur | 1 | 10 | 50.0 | 122.0 | 530.0 | 530.0 | 5/5 |
| motion_blur | 1 | 20 | 14.4 | 72.0 | 466.0 | 466.0 | 5/5 |
| **Média geral** | | | **62.2** | **122.2** | **507.5** | **507.5** | |

\* gaussian_noise sev3 ramp 10 seed 42: o detector nunca disparou (nenhuma decisão) — o episódio
não entra na contagem de "pularia"; seu downtime (540 s) já é o da baseline do par e é invariante a d.
Nos 12 grupos, d=800 = baseline exatamente: os episódios com decisão pulam (→ baseline) e o único
episódio sem decisão já igualava a baseline do par.

---

## Interpretação

**d = 80 s (round communication-dominated, um round realista):** as conclusões qualitativas
sobrevivem integralmente. A margem agente ≪ baseline permanece enorme: média geral 122.2 s vs
507.5 s da baseline (redução de 76%); em todos os 12 grupos o agente com round de 80 s mantém
downtime muito abaixo da baseline (margem relativa de 55–100%; a menor, gaussian, 236 vs 526). O custo do round
estendido é exatamente o intervalo [t_d, t_d+80] quando o disparo ocorre com acurácia já abaixo
de τ — em média +60 s sobre d=8, explicado pela identidade d80 = d8 + 72 s — e **zero** quando o
detector dispara cedo: fog ramp 20 permanece com downtime zero (acc(t_d) ≈ 0.57 ≥ τ em todas as
seeds), e frosted_glass ramp 20 sobe de 3.2 para 32 s (detecção precoce em parte das seeds). O
caso limite gaussian sobrevive: mesmo com round de 80 s, o retreino recupera a acurácia para
≈0.68 (acima de τ) nos episódios em que dispara — a fraqueza continua sendo a detecção (1 episódio
nunca dispara), não o round. A degradação do teste misto com o tempo (fração f(v) → 1) reduz a
acurácia pós-retreino em ~0.006 em média, sem jamais cruzar τ.

**d = 800 s (round longo demais para o horizonte):** as conclusões qualitativas **não** sobrevivem
no sentido de que o cenário muda de natureza: com round de 800 s > horizonte de produção (600 s),
o orçamento do controlador (`end_time − t_d ≥ round`) falha em **todos** os 59 episódios com
decisão, e a política registrada (`retrain_skipped_budget`) pula o retreino — o agente degenera à
baseline por construção (d=800 = baseline em todos os 12 grupos). Isto é uma consequência da
política de orçamento, não da dinâmica do round: um round que não cabe no horizonte nunca é
iniciado. A leitura correta para o paper é que a garantia de downtime depende de o round caber no
horizonte; para frotas reais (round de minutos), o ponto de ruptura está em d ≈ horizonte − t_d
(≥ 140 s neste grid: a menor folga é end_time − t_d = 759.3 − 619.3 ≈ 140 s), e o mecanismo de
ruptura é a regra de skip, que já está implementada e registrada nos resultados.

---

## Frases prontas (inglês, para o paper)

1. "The downtime conclusions are robust to a 10× longer retraining round: with an 80 s
   communication-dominated round, the agent still cuts downtime from 507.5 s (baseline) to
   122.2 s on average, and the per-group ranking, the zero-downtime cases, and the gaussian
   detection-limited behaviour are unchanged."

2. "The cost of a longer round is exactly the pre-retrain interval it extends: downtime grows
   by (d − 8) s when the trigger fires below the accuracy threshold, and does not grow at all
   when the detector fires early (e.g., fog ramp-20 retains zero downtime at d = 80 s)."

3. "At d = 800 s the round no longer fits within the 600 s production horizon and the
   controller's budget rule (`retrain_skipped_budget`) declines to retrain in all 59 triggered
   episodes, so the agent degenerates to the baseline; the sensitivity boundary is set by the
   budget policy, not by the round's training dynamics."
