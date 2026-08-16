# S4 — Fact-checker notes: 3 works to add + comparative table (R2)

**Scope:** Sections II-B (Local detection, central decision), II-C (Label-free drift detection), II-D (Retraining orchestration and budgets), II-E (Metrics gap) of `report/e09/main.tex`.
**Date of verification:** 2026-08-16. All metadata below was verified against publisher/arXiv/IEEE records; items not confirmable are marked **não-verificado**.
**Status of paper:** read-only; `main.tex` not modified.

---

## (a) Verification of the 3 works

### 1. PUDD — VERIFIED (all metadata confirmed)

- **Exact title:** "Early Concept Drift Detection via Prediction Uncertainty"
- **Authors (complete):** Pengqian Lu, Jie Lu, Anjin Liu, Guangquan Zhang (Australian AI Institute, University of Technology Sydney)
- **Venue:** AAAI-25, *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 39, no. 18, pp. 19124–19132, 2025 (AAAI-25 Technical Tracks 18). DOI: 10.1609/aaai.v39i18.34105. arXiv:2412.11158 (submitted 15 Dec 2024; "Accepted by AAAI-2025"). Code: github.com/RocStone/PUDD.
- **Method (verified from AAAI proceedings/arXiv):**
  - Detection signal: **PU-index**, a metric derived from the classifier's **prediction uncertainty** (not from error rate).
  - Mechanism: sliding window discarding outdated data → stream split into two samples → **Adaptive PU-index Bucketing** algorithm builds histograms satisfying theoretical conditions → **Pearson's chi-square test** decides drift.
  - **False-positive control: confirmed.** "The p-value from the Pearson's Chi-square test serves as a precise control mechanism for our tolerance to false alarms. By adjusting the significance level (α), we can directly modulate the trade-off between sensitivity and false positive rate."
  - Theoretical results: (1) PU-index detects drift even when error rates remain stable; (2) any error-rate change implies a PU-index change.
- **Relation to UDD (Baier et al., HICSS 2023):** PUDD is **not FL** (single-stream, centralized streaming detection). It is **complementary**, not competing, to UDD: both are label-free uncertainty-based detectors, but UDD feeds MC-dropout Shannon entropy into an ADWIN window test, whereas PUDD tests the full distribution of the PU-index with a chi-square test and explicit α-controlled FPR. **Not verified** whether PUDD's related-work section cites UDD/Baier — the verified material shows its comparisons target error-rate detectors and classic window methods; do not claim PUDD positions itself against UDD.
- **Placement in II-C:** yes — it belongs in II-C as a third label-free method alongside UDD and D3M, best inserted after the D3M sentence (main.tex:194–196). It does **not** belong to the II-B census (not an FL work).

### 2. Adaptive-FedAVG — VERIFIED (title differs from the one in the review request)

- **Exact title:** "Adaptive Federated Learning in Presence of Concept Drift" — **not** "Adaptive Federated Averaging in Unsupervised Situation Awareness". Use the correct title.
- **Authors:** Giuseppe Canonaco, Alex Bergamasco, Alessio Mongelluzzo, Manuel Roveri (Politecnico di Milano)
- **Venue:** 2021 International Joint Conference on Neural Networks (IJCNN 2021), Shenzhen, July 18–22, 2021, pp. 1–7, IEEE. DOI: 10.1109/IJCNN52387.2021.9533710.
- **Method (verified):** **passive, server-side approach — no drift detection at all.** "Following a passive approach, Adaptive-FedAVG … promptly react[s] to concept drift by adapting the learning rate to increase the plasticity of the learning phase" (abstract); the server-side module works "as a learning rate scheduler able to re-calibrate the step size" (Canonaco's thesis, Politecnico di Milano). FedDrift's AISTATS paper corroborates: "adapts to drifts by centrally tuning the learning rate used by all clients as a function of the variability across updates."
- **Census relevance (II-B):** **yes, it is an additional exception.** It is an FL concept-drift work that uses **neither local detection nor server-side detection** — it is detection-free passive adaptation. If added to the reviewed set, the exception list grows from 2 to 3: FLASH (server-side detection), Stallmann (global detector), Adaptive-FedAVG (no detection). See verdict in (d).

### 3. FedSGT — VERIFIED (exists; arXiv ID confirmed)

- **Exact title:** "FedSGT: Exact Federated Unlearning via Sequential Group-based Training"
- **Authors:** Bokang Zhang (Emory), Hong Guan (ASU), Hong Kyu Lee (Emory), Ruixuan Liu (Emory), Jia Zou (ASU), Li Xiong (Emory)
- **Venue:** arXiv:2511.23393, submitted 28 Nov 2025 (v1; v2: 2 Mar 2026). Preprint — no peer-reviewed venue found.
- **What it calls "downtime":** availability, not served-model quality. "Existing exact unlearning methods typically require frequent retraining from scratch, resulting in high communication cost and **long service downtime**"; design goal "G2: Service Maintenance. The framework aims to **reduce the service downtime**". Downtime is the time the service is **offline/unavailable** while retraining to erase data; its metric is the expected number of unlearning requests supported before retraining ("service failure/termination") — i.e., **binary availability**, not time served below a quality threshold.
- **Does it weaken the II-E claim?** **No**, as the claim is scoped ("none of the works we reviewed **[the FL drift literature]** quantifies the time the served model stays below a quality threshold"). FedSGT is unlearning literature, outside the reviewed drift set, and its downtime is availability-based. **However**, a one-sentence acknowledgment distinguishing "availability downtime" (unlearning, zero-downtime MLOps upgrades) from "quality-below-threshold serving time" is **recommended** in II-E — it preempts R2 and sharpens the novelty claim. See (d) and ready-made sentences in (c).
- **Side flag (II-D):** FedSGT also defines a *training budget* (number of PEFT modules/sequences, B) controlling how many deletions are supported before service termination — do not conflate with retraining budgets in the drift literature; if cited near II-D, the different meaning must be stated.

---

## (b) Comparative table

"?" = could not be confirmed from verified sources. Downtime column refers to a *time-below-quality-threshold* metric as used in this paper.

| Work | Detection signal | Detection location | Trigger / aggregation | Retraining budget | Drift regime studied | Downtime metric |
|---|---|---|---|---|---|---|
| FedDrift (Jothimurugesan 2023, AISTATS) | local model test accuracy across models | per-client (local) + server hierarchical clustering | local drift test (threshold δ); multiple-model clustering (isolate drifted clients, lazy merging) | none explicit (cluster reconfiguration) | staggered sudden, recurring; gradual staggered (FMoW) | no |
| FLARE (Chow 2023, IWCMC) | KS test on model-confidence distributions (label-free) | sensor/endpoint (local) | KS statistic increase > φ; mitigation = send recent data to client for retraining | none explicit (scheduling-based) | abrupt drift (MNIST-C corruptions) | no |
| FLAME (Mavromatis 2024, arXiv) | KS-based monitoring, adaptive threshold (window mean+3σ), label-free | endpoint (local) | adaptive threshold on KS stats; selective retraining trigger; prior-concept data retention | none explicit | repeated/recurrent drift in IoT; compared vs ADWIN/KSWIN | no |
| FL-MalDrift (Patel 2026, Sci. Reports 16:1821) | on-device error-rate detectors (ADWIN, DDM, EDDM, HDDM) | device (local) + server participation controller (EWMA-smoothed drift scores) | detector alarms; selective aggregation (FedAvg/FedSGD) filtered by drift scores | none explicit | gradual/continuous evolution of Android malware (temporal drift) | no |
| DriftGuard (Han 2026, arXiv) | device performance degradation reports | local observation; server decision π^t=(Trig,S,θ) | server detects outdated shared params / group degradation; MoE global vs group retraining | **explicit** retraining configuration (devices, parameters); cost reduction ≤83% | asynchronous data drift across devices | no |
| FedDAA (Fu 2025, arXiv) | data prototypes (P(Y\|X)) per client | local (RDLD module) | prototype cluster-assignment change; dynamic client clustering; per-source strategies | none explicit | real + virtual + label drift | no |
| FedPLC (Zhou 2026, Sensors 26(1):283) | label-wise head similarity / community structure (client-reported statistics; no per-client drift test per se) | client-reported, server-side community detection (Louvain) | reactive community reorganization; label-wise community aggregation | none explicit | abrupt + incremental drift on non-IID data | no |
| FLASH (Panchal 2023, ICML) | magnitude of aggregated parameter updates (gradient disparity ‖Δ²−v‖) | **server-side** | gradient-disparity moving average; server-side drift-aware adaptive optimizer (effective-LR boost) | none explicit (client early-stopping) | class-conditional (real) drift, sudden | no |
| FELK / FL-HVD (Chen 2026, IJDSAA 21:46) | per-feature KS-2samp (label-free, distribution-based) | client-side (local) | FELK alarm (early-exit); FDM dual-model adaptation (two server models) | none explicit | virtual drift (asymptotic/gradual), 21 scenarios | no |
| Stallmann 2024 (FUZZ-IEEE) | federated fuzzy Davies-Bouldin index (global cluster fit, label-free) | **global / server-side** | Δ_t vs Δ threshold δ | none (detection only, no adaptation) | sudden global drift | no |
| Adaptive-FedAVG (Canonaco 2021, IJCNN) | **none — passive** (no detection) | **server-side LR scheduler** | none (continuous adaptation) | none | concept drift in non-stationary processes (specific drift types ?) | no |
| PUDD (Lu 2025, AAAI) | PU-index (prediction uncertainty) | n/a — centralized streaming (not FL) | Pearson χ² test, α-controlled FPR | none explicit (retrain classifier on recent data) | early/gradual drift, incremental learning | no |
| **This paper** | per-client UDD: MC-dropout entropy (T=5) + ADWIN, label-free | client (local) | collective quorum policy (central decision) | **explicit:** one global round; reactive retraining on current mixture | gradual drift (covariate shift, drift width) | **yes — time served model stays below quality threshold (primary metric)** |

**Notes for the table:**
- FedPLC's row: the paper's II-B counts it among "local detection with central decision". Verification shows FedPLC has **no explicit per-client drift detector** — it reorganizes label-wise communities at the server from client-reported statistics. The census label is defensible (client-side signal, central decision) but should not imply a per-client statistical test. Flag in II-B if reviewers push.
- FL-MalDrift citation in the paper (Sci. Reports vol. 16, Art. 1821, 2026) was **cross-checked against Nature: correct** (Scientific Reports (2026) 16:1821, DOI 10.1038/s41598-025-31592-z, online 15 Dec 2025).
- Casado et al. 2022 (counted in the census) is not in R2's requested table; note it detects locally (CUSUM-type on classifier confidence) but adapts locally too — the "central decision" label is looser for it.

---

## (c) Ready-made sentences (English, IEEE citations)

Proposed bibitems (matching the paper's thebibliography style — add in `main.tex` only if you choose to accept these notes):

```
\bibitem{lu2025} P.~Lu, J.~Lu, A.~Liu, and G.~Zhang, ``Early concept drift
detection via prediction uncertainty,'' in \emph{Proc. AAAI Conf. on
Artificial Intelligence}, vol.~39, no.~18, pp.~19124--19132, 2025.
\bibitem{canonaco2021} G.~Canonaco, A.~Bergamasco, A.~Mongelluzzo, and
M.~Roveri, ``Adaptive federated learning in presence of concept drift,'' in
\emph{Proc. IEEE Int. Joint Conf. on Neural Networks (IJCNN)},
pp.~1--7, 2021.
\bibitem{zhang2025} B.~Zhang, H.~Guan, H.~K. Lee, R.~Liu, J.~Zou, and
L.~Xiong, ``FedSGT: Exact federated unlearning via sequential group-based
training,'' arXiv:2511.23393, 2025.
```

**Adaptive-FedAVG — II-B (census):**
1. "A third exception is Adaptive-FedAVG, which forgoes drift detection altogether and reactively increases the server-side learning rate to raise the plasticity of the learning phase, a passive alternative to local detection \cite{canonaco2021}."
2. "Adaptive-FedAVG \cite{canonaco2021} adapts the learning rate centrally as a function of the variability across client updates, with no detection signal at all; counting it, nine of the twelve concept-drift FL works we reviewed use local detection."
3. "The passive server-side learning-rate scheduling of Adaptive-FedAVG \cite{canonaco2021} stands in contrast to the detection-driven designs above, but it remains a minority pattern."

**PUDD — II-C (after the D3M sentence, main.tex:194–196):**
1. "In the same label-free family, PUDD detects drift from the prediction uncertainty index (PU-index) of the classifier, comparing the two windows of a sliding stream with a Pearson chi-square test after an adaptive histogram-bucketing step \cite{lu2025}."
2. "Unlike UDD, whose ADWIN stage tracks the mean of the entropy signal, PUDD tests the full distribution of the PU-index and uses the test's significance level as a direct false-positive control \cite{lu2025}."
3. "PUDD is a centralized, single-stream detector \cite{lu2025}; its principled false-positive control is nonetheless the property that our collective-quorum policy emulates across clients."

**FedSGT — II-E (metrics gap):**
1. "Downtime language exists in adjacent ML-systems literature: exact federated unlearning quantifies service downtime as the unavailability incurred when retraining to erase data \cite{zhang2025}, and zero-downtime model upgrades are standard MLOps practice."
2. "These availability notions are distinct from the metric we adopt: they measure the time the service is offline, whereas we measure the time the served model remains online but below a quality threshold."
3. "Among the reviewed FL drift works, none quantifies the time the served model stays below a quality threshold; the closest related notion, service downtime in federated unlearning \cite{zhang2025}, measures availability rather than served-model quality."

---

## (d) Verdicts

### Census "nine of the eleven" (II-B)

**Arithmetic is correct for the set as currently listed** (11 works: Jothimurugesan, FLARE, FLAME, FL-MalDrift, DriftGuard, FedDAA, FedPLC, FELK/FL-HVD, Casado = 9 local; FLASH and Stallmann = 2 exceptions — both verified: FLASH detects server-side from aggregated update magnitude, Stallmann is a global fuzzy-clustering detector).

**The number must change if Adaptive-FedAVG is added**, as R2 requests: the set becomes 12 works, still 9 local, with a **third exception** (Adaptive-FedAVG uses no detection at all — passive server-side LR adaptation). Recommended rewrite: "nine of the twelve concept-drift FL works we reviewed do so. The exceptions are informative: FLASH … Stallmann … and Adaptive-FedAVG, which forgoes detection entirely and adapts the server learning rate passively \cite{canonaco2021}."

**PUDD does not affect the census** (centralized, non-FL — it belongs in II-C only).

### Downtime-metric novelty claim (II-E)

**Claim holds as scoped.** No work in the reviewed FL drift set quantifies time-below-quality-threshold (verified: they report detection latency, retraining cost, communication volume, accuracy dips — never a below-threshold serving-time metric). FedSGT uses "service downtime" but (i) is unlearning literature, outside the drift set, and (ii) means *availability* downtime, not served-model quality. **Acknowledging sentence is recommended** (not strictly necessary): it preempts R2, distinguishes availability vs. quality-below-threshold, and strengthens the novelty claim rather than weakening it. Use sentence 1–2 from (c).

### Additional flags
- Adaptive-FedAVG's correct title is "Adaptive Federated Learning in Presence of Concept Drift" — do not print the working title from the review request.
- FedSGT's budget (PEFT-module budget B for unlearning) must not be conflated with retraining budgets in II-D if cited there.
- FedPLC counts as "local detection" in the census only in the loose sense (client-reported statistics, server-side community detection — no per-client statistical test); pre-empt with a qualifier if space allows.
- FL-MalDrift's existing citation (Sci. Reports 16:1821, 2026) is confirmed correct.
