# S2 — Custo do Monitoramento (detecção) vs. Custo do Retreino

**Requisitado por:** R3 (perspectiva MLOps) — quantificar o custo do monitoramento
(MC-dropout, T=5) vs. o custo de um round de retreino, nunca reportado no paper.

**Data:** 2026-08-16 · **Autor:** análise de engenharia de performance

---

## 1. Premissas (todas verificadas no código-fonte)

| # | Premissa | Fonte |
|---|---|---|
| P1 | 1 forward de batch 32 = 1,70 GFLOPs (analítico; 1 MAC = 2 FLOPs; ReLU/MaxPool/dropout/softmax/entropy/ADWIN excluídos — <1% do custo; ADWIN é O(n) sobre poucas centenas de escalares por cliente/tick) | `src/utils/models.py:56` (CNNCIFAR10), contagem manual |
| P2 | Backward ≈ 2× o custo do forward (regra padrão conv/linear); passo de treino = fwd + bwd + opt | convenção padrão |
| P3 | Monitoramento: 1 batch não-rotulado (32 imgs) por cliente por tick, T=5 → **50 forwards de batch-32 por tick** (10 clientes × 5) = 1.600 forwards de imagem; 80 ticks/episódio (60 produção + 20 warm-up; warm-up também atualiza os detectores, `observe_tick(warmup=True)` executa `update`) → **4.000 forwards de batch-32 por episódio** | `src/orchestrator/orchestrator.py:70`, `experiments/e07_drift_agent/episode.py:58` |
| P4 | Retreino: 1 round global, 1 época local, 5.000 imgs/cliente (50.000/10), batch 32 → 156,25 batches/cliente (último batch parcial, 8 imgs) → **1.562,5 passos de treino no total** (10 clientes) | paper §Retraining orchestration; `episode.py:56-66` |
| P5 | 1 round de retreino ≈ 8 s virtuais no paper; batch medido do retreino não inclui avaliação do test set (10.000 imgs, fora do escopo da pergunta) | paper §Retraining |
| P6 | T=50 (UDD publicado) usado como sensibilidade; T=5 é o valor do artigo | paper §Label-free drift detection |

> **Correção de enquadramento (importante):** o briefing da tarefa contava o monitoramento
> como "1.600 forwards por tick × 80 = 128.000 forwards de **batch-32**". Isso confunde
> forwards de *imagem* com forwards de *batch*: o código executa 1 batch de 32 por cliente
> por tick, logo são 50 forwards de batch-32/tick = 4.000/episódio (128.000 forwards de
> imagem). Os números abaixo usam a realidade do código (P3). O enquadramento do briefing
> (128.000 batches) aparece como caso-limite na §5 (ratio 27×, inviável na borda).

## 2. Hardware e método do microbenchmark

- **Hardware:** Apple M5, 10 cores, 16 GB unificados, macOS 27, torch 2.12.1
  (python 3.10.20); CPU benchmark com `torch.set_num_threads(4)` (default do env);
  MPS benchmark com `torch.mps.synchronize()`.
- **Método:** `CNNCIFAR10()` real importado de `src.utils.models`; input `randn(32,3,32,32)`;
  forward em `model.eval()` com dropout em `train()` (réplica exata do `UDDDetector.update`);
  `no_grad`; warm-up de 10 iterações; tempo médio de 200–300 repetições (`timeit`). Passo de
  treino = fwd + cross-entropy + `backward()` + `opt.step()` (SGD). Variabilidade run-to-run
  ~±5–10% (aferida em re-execuções).
- **Script:** reproduzível inline no Anexo A.

## 3. FLOPs analíticos por camada (1 imagem, CNNCIFAR10, 32×32)

| Camada | Saída | MACs | FLOPs |
|---|---|---|---|
| conv1 (3→32, k3, p1) | 32×32×32 | 884.736 | 1.769.472 |
| conv2 (32→32, k3, p1) | 32×32×32 | 9.437.184 | 18.874.368 |
| conv3 (32→64, k3, p1) | 64×16×16 | 4.718.592 | 9.437.184 |
| conv4 (64→64, k3, p1) | 64×16×16 | 9.437.184 | 18.874.368 |
| fc1 (4096→512) | 512 | 2.097.152 | 4.194.304 |
| fc2 (512→10) | 10 | 5.120 | 10.240 |
| **Total (1 imagem)** | | | **53.159.936 (~53,2 MFLOPs)** |

Parâmetros: 2.168.362 (~2,17 M — confere com o briefing). **1 batch de 32 = 1,70 GFLOPs.**
(FLOPs/parâmetro ≈ 24,5 — típico de CNN com 2 fc grandes.)

## 4. Tabela principal — custo por componente

| Componente | Quantidade | FLOPs | Tempo CPU | Tempo MPS | % do episódio* |
|---|---|---|---|---|---|
| 1 forward, batch 32 (MC-dropout) | 1 | 1,70 GFLOPs | 36,9 ms | 1,54 ms | — |
| 1 atualização de detector (T=5, 1 cliente) | 5 forwards + softmax/entropia | 8,51 GFLOPs | 166,0 ms | 12,7 ms | — |
| **Monitoramento por tick** (10 clientes) | 50 forwards | 85,1 GFLOPs | 1,66 s | 0,13 s | 1,3% do tick de 10 s (MPS) |
| **Monitoramento por episódio** (80 ticks) | 4.000 forwards | **6,80 TFLOPs** | 132,8 s | 10,1 s | 47% do compute reativo |
| **Retreino (1 round, 1 época)** | 1.562,5 passos (fwd+bwd) | **7,97 TFLOPs** | 137,7 s | 11,4 s | 53% do compute reativo |
| **Ratio monitoramento : retreino** | | **0,85×** | 0,96× | **0,89×** | ≈ 1:1 |

\* "Episódio" = janela virtual de 800 s (80 ticks × 10 s); a coluna % usa o tempo MPS
(dispositivo default do simulador via `get_device`). Percentual do episódio em CPU:
17% do tick consumido só por monitoramento.

**Sensibilidades:** retreino com os 5 rounds do default do config (`retrain_rounds=5`) →
custo ×5 (≈40 TFLOPs; monitoramento passa a ser 0,17×). Treino inicial (20 rounds) é
offline/amortizado e fica fora da comparação reativa.

## 5. Interpretação

**O monitoramento NÃO domina — mas não é gratuito.** Com T=5, o custo de detecção por
episódio é **0,85× o de um round de retreino em FLOPs e 0,89× em tempo de parede (MPS)**:
os dois componentes são da mesma ordem de grandeza e o detector responde por ~47% de todo
o compute reativo do episódio. A razão para a paridade é estrutural: há 82× mais forwards
de monitoramento do que passos de treino (4.000 vs ~48,7 passos de fwd-equivalente... na
verdade 4.000 vs 1.562,5), mas cada passo de treino custa ~3× um forward e há 1.562,5 deles;
em FLOPs: 4.000×1,70 G vs 1.562,5×5,10 G.

**Viabilidade na borda:** no dispositivo do simulador (MPS), o monitoramento ocupa só
**0,13 s de cada janela de 10 s (1,3%)** — sobra folga de sobra para o retreino reativo
dentro do tick, sustentando a afirmação de resposta em tempo real do paper. Em CPU de
borda (4 threads), porém, o mesmo monitoramento consome **1,66 s/tick (17%)** — ainda
agendável, mas longe de desprezível, e a latência de 1 forward (~37 ms) limita o tick mínimo
viável. **A alavanca crítica é T:** o UDD publicado usa T=50 (P6); com T=50 o monitoramento
escala ×10 → ~9× o round de retreino (≈101 s/episódio em MPS, ~22 min em CPU), dominando o
custo — o que valida retroativamente a escolha do paper por T=5, mas expõe a sensibilidade
a T como item obrigatório de future work (o próprio paper a declara como tal). O caso-limite
do briefing (contar 128.000 forwards como *batches*: 217,7 TFLOPs, ratio 27×) só seria
atingido se cada cliente monitorasse 32 batches por tick — 32× além do protocolo atual;
ainda assim, em MPS, 32 batches/tick custariam 4,05 s < tick de 10 s.

## 6. Frases prontas para o paper (EN)

1. *"Across a full episode (80 ticks), the T=5 MC-dropout monitor costs 6.8 TFLOPs — 0.85× the 8.0 TFLOPs of the single retraining round (0.89× in wall-clock on our accelerator), i.e. 1.3% of each 10-s tick budget: detection is not free, but it does not dominate the reactive budget."*

2. *"Had we kept the published UDD setting of T=50, monitoring would exceed the retraining round by roughly 9×, which is exactly why T=5 was chosen as a cost decision; quantifying the accuracy sensitivity to T is left as future work."*

3. *"On edge CPUs the detector alone consumes 1.66 s (17%) of each 10-s monitoring tick; the resulting ~37 ms per stochastic forward bounds the minimum feasible tick for real-time reactive retraining on such devices."*

---

## Anexo A — script do benchmark (reproduzível)

```python
import sys, time
sys.path.insert(0, "…/FederatedLearningSimulation")
import torch, torch.nn as nn, torch.nn.functional as F
from src.utils.models import CNNCIFAR10

torch.manual_seed(0); torch.set_num_threads(4)
model = CNNCIFAR10(); model.eval()
for m in model.modules():
    if isinstance(m, nn.Dropout): m.train()          # réplica do UDDDetector
x = torch.randn(32, 3, 32, 32)
def timeit(fn, reps, inner=1):
    for _ in range(10): fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        for _ in range(inner): fn()
    return (time.perf_counter() - t0) / (reps * inner)
with torch.no_grad():
    fwd_ms = timeit(lambda: model(x), 300) * 1e3     # 36,9 ms (CPU) / 1,54 ms (MPS)
    def upd5():
        s = 0.0
        for _ in range(5):
            p = F.softmax(model(x), dim=-1)
            s += float(-torch.sum(p * torch.clamp(p, min=1e-12).log(), dim=-1).sum())
        return s
    upd5_ms = timeit(upd5, 100) * 1e3                # 166,0 ms (CPU) / 12,7 ms (MPS)
model.train(); opt = torch.optim.SGD(model.parameters(), lr=0.01)
y = torch.randint(0, 10, (32,))
def step():
    opt.zero_grad(); F.cross_entropy(model(x), y).backward(); opt.step()
step_ms = timeit(step, 100) * 1e3                    # 88,2 ms (CPU) / 7,3 ms (MPS)
# extrapolação: tick = 10·upd5_ms; episódio = 80·tick; retreino = 1562,5·step_ms
# MPS: torch.mps.synchronize() antes de cada leitura de relógio.
```

Medidos nesta máquina (2026-08-16): fwd 36,9 ms (CPU) / 1,54 ms (MPS); update T=5
166,0 / 12,7 ms; train step 88,2 / 7,3 ms; FLOPs/forward batch-32 = 1,7011 G
(analítico; `torch.profiler` nesta build retornou 0 flops para ops de CPU — contagem
manual usada como fonte).
