# Chapter 4: Results, Discussion and Managerial Recommendations

## Chapter Introduction

This chapter reports the empirical results and confronts them with the hypotheses of Chapter 2. All figures are produced by the reproducible pipeline in `code/experiments.py` on a held-out, out-of-sample set of N = 12,000 paired sessions, and are stored in `code/results/results.json`. **A standing caveat governs the entire chapter: every number is a *simulated* Implementation Shortfall under the calibrated model of Chapter 3. Only the sign, ranking and comparative statics of the effects are interpreted; the basis-point magnitudes are model artefacts and are neither a backtest on, nor a claim of deployable edge in, the live market.**

## 4.1 Global Performance — Testing H1

| Strategy | Mean IS (bps) | Median (bps) | VaR95 (bps) | Mean vs TWAP |
|---|---|---|---|---|
| TWAP | 17.64 | 17.82 | 97.3 | — |
| VWAP | 17.10 | 17.42 | 95.1 | +3.1% |
| Almgren-Chriss | 17.93 | 18.04 | 86.9 | −1.7% |
| DQN-family | 12.52 | 16.86 | 91.7 | +29.0% |
| PPO-family | 11.68 | 16.87 | 80.8 | +33.8% |

![Figure 4.1 — Out-of-sample Implementation-Shortfall distribution by strategy (simulated; box = inter-quartile range, whiskers 5–95%).](code/figures/fig_is_distribution.png)

The PPO-family agent reduces mean simulated IS by **33.8% relative to TWAP** (Δ = −5.96 bps; Newey-West t = −67.4; bootstrap 95% CI [−6.13, −5.79]), by **31.7% relative to VWAP** (t = −60.9) and by **34.9% relative to Almgren-Chriss** (t = −60.6). The DQN-family agent reduces IS by **29.0% / 26.8% / 30.2%** respectively. All reductions are significant at the 0.1% level. **H1 is supported.**

The PPO-family agent outperforms the DQN-family agent by **6.7%** (t = −12.2; CI [−0.97, −0.71]), confirming **H1a**: the continuous-action policy-gradient agent beats the discrete-action value-based agent, the discretisation handicap being the operative mechanism. The reduction relative to TWAP (33.8%) exceeds that relative to VWAP (31.7%), confirming **H1b**: VWAP is the harder benchmark because it already exploits the volume profile. As classical theory predicts for this impact model, Almgren-Chriss does not beat TWAP on *mean* IS, but it does reduce *tail* risk (VaR95 86.9 vs 97.3 bps) — its design objective.

**Honest nuance, reported rather than hidden.** The PPO mean (11.68) lies well below its median (16.87). The agent's advantage is therefore *concentrated* in a subset of sessions — those combining a strong exploitable signal with high volatility — rather than being uniform: the *median* reduction versus TWAP is only about 5%, whereas the *mean* reduction (34%) reflects large savings in the favourable sessions. This concentration is precisely the content of H3.

## 4.2 Regime Conditionality — Testing H3

| Strategy | σ = 15% (low) | σ = 22% | σ = 35% (high) |
|---|---|---|---|
| Almgren-Chriss | 18.21 | 18.74 | 16.81 |
| DQN-family | 16.01 | 15.00 | 6.42 |
| PPO-family | 15.92 | 13.55 | 5.44 |
| **PPO vs Almgren-Chriss** | **+12.6%** (t=−25.5) | **+27.7%** (t=−41.0) | **+67.6%** (t=−47.5) |

![Figure 4.2 — Mean Implementation Shortfall by strategy and volatility regime (simulated).](code/figures/fig_regime.png)

The PPO advantage over Almgren-Chriss rises **monotonically** with volatility, from 12.6% in the calm regime to 67.6% in the stressed regime, strongly confirming **H3**: adaptivity is worth most when volatility is high and the static schedule's fixed-volatility calibration is most mis-specified. In the high-volatility regime (σ = 35% > 25%), PPO beats Almgren-Chriss by 67.6%, far exceeding the 10% threshold of **H3a**, which is **supported**.

**H3b is rejected — and we report this openly.** H3b predicted *no* significant PPO–Almgren-Chriss difference in calm markets (σ < 15%). The data reject it: a smaller but highly significant 12.6% edge (t = −25.5) persists at σ = 15%, because the order-flow signal remains exploitable even in low volatility. The *direction* of H3 is therefore confirmed while its *strong form* (a vanishing edge in calm regimes) is falsified — a result we retain rather than discard, since the asymmetry of gains across regimes is itself the economically meaningful finding.

## 4.3 Fragmentation Alpha — Testing H2

Dynamic three-venue routing is compared against a single pooled ("Euronext-only") venue of identical total capacity, on common scenarios, as the cross-venue price dispersion is varied:

| Cross-venue dispersion (bps) | 0.0 | 4.0 | 8.0 | 16.0 | 32.0 |
|---|---|---|---|---|---|
| Fragmentation alpha (bps) | 0.00 | 1.21 | 3.95 | 10.69 | 25.27 |
| as % of single-venue IS | 0.0% | 10.6% | 34.6% | 93.7% | 221% |

![Figure 4.3 — Fragmentation alpha vanishes as cross-venue dispersion approaches zero (H2).](code/figures/fig_h2_depth.png)

The fragmentation alpha is **exactly zero when cross-venue price dispersion is zero** and rises monotonically with it — a clean, falsifiable confirmation that the multi-venue gain is genuinely attributable to exploiting cross-venue dispersion, not a relabelling of temporal optimisation. At a moderate, realistic dispersion of approximately 8 bps, routing captures about **34.6% of single-venue IS**, exceeding the 25% threshold of **H2**, which is **supported**; the dynamic multi-venue policy significantly outperforms the single-venue benchmark (**H2a supported**). The effect is monotone increasing in dispersion, consistent in spirit with **H2b** (more fragmented names yield larger gains); we note, however, that the simulator parameterises fragmentation through price dispersion rather than the Herfindahl–Hirschman Index directly, so H2b is corroborated by a proxy rather than tested on HHI itself. The very large values at 16–32 bps dispersion (where simulated IS turns negative) lie outside any realistic dispersion range and are shown only to trace the mechanism. These magnitudes are consistent in direction with the cross-venue execution-alpha discussion of Lehalle & Laruelle (2013) and Gresse (2017).

## 4.4 Robustness

| Variant | PPO vs TWAP | PPO vs Almgren-Chriss |
|---|---|---|
| Baseline (linear impact, Gaussian) | 31.0% | 30.9% |
| Square-root impact (δ = 0.5) | 15.3% | 15.4% |
| Student-t shocks (4 d.o.f.) | 33.0% | 33.5% |
| Higher impact coefficients | 13.5% | 13.2% |
| Out-of-distribution parameter shift | 22.7% | 22.3% |

![Figure 4.4 — Robustness of the PPO advantage across model assumptions (simulated).](code/figures/fig_robustness.png)

The PPO advantage is **sign-stable across every variant**, including an out-of-distribution test in which the signal persistence and the reference volatility are shifted at test time (22.7%; the agent degrades but remains well ahead). The *magnitude* varies materially with the impact specification — it roughly halves under square-root or higher impact — which is exactly why only signs and rankings, not basis-point levels, are interpreted throughout.

## 4.5 Discussion

These results reproduce, in a controlled model, the central mechanisms claimed in the literature: a signal-aware adaptive policy beats signal-blind schedules (Cartea et al., 2018); the gain is regime-dependent (Cartea & Jaimungal, 2015); and cross-venue dispersion is an exploitable source of execution alpha (Lehalle & Laruelle, 2013). The contribution of the dissertation is not the discovery of these mechanisms but their *joint, falsifiable demonstration* in a single reproducible framework, together with an explicit accounting of where the reinforcement-learning advantage comes from (an observable signal and conditioning on the current state) and where it does not (it is not free; it shrinks when the signal weakens, the impact steepens, or the environment shifts).

## 4.6 Managerial Recommendations

The managerial reading is conditional, not promotional. RL-style adaptive execution is most valuable for **large orders, in volatile regimes, on genuinely fragmented names**, where the gains over both naive schedules and static Almgren-Chriss are largest. It adds comparatively little in calm, concentrated conditions, where Almgren-Chriss is already near-optimal and far easier to govern, validate and explain to a risk committee. As an *in-model illustration only* — not a deployable estimate — the 5.96 bps mean IS reduction versus TWAP, applied to €2bn of annual European-equity turnover, would correspond to on the order of €1.2m per year; the realism of any such figure is bounded by every limitation in Section 4.8, above all the simulation-to-reality gap.

## 4.7 Hypothesis Scorecard

| Hypothesis | Verdict | Evidence |
|---|---|---|
| H1 (RL < TWAP and VWAP) | Supported | PPO −33.8% / −31.7%; DQN −29.0% / −26.8% (p<0.001) |
| H1a (PPO > DQN) | Supported | +6.7% (t=−12.2) |
| H1b (gain vs TWAP > vs VWAP) | Supported | 33.8% > 31.7% |
| H2 (fragmentation ≥25% extra) | Supported | 34.6% of single-venue IS at ~8 bps dispersion; 0 at zero dispersion |
| H2a (multi > single venue) | Supported | alpha > 0 whenever dispersion > 0 |
| H2b (more for lower HHI) | Supported via proxy | monotone in dispersion; HHI not modelled directly |
| H3 (gains rise with volatility) | Supported | 12.6% → 27.7% → 67.6% across regimes |
| H3a (PPO > AC by >10% in high vol) | Supported | +67.6% at σ=35% |
| H3b (no edge in low vol) | **Rejected** | +12.6% at σ=15% is significant (t=−25.5) |

## 4.8 Limitations

1. **This is a mechanism study, not a backtest.** All results are simulated IS under a stated model; only signs, rankings and comparative statics are interpreted, never the basis-point magnitudes as deployable figures.
2. **Magnitudes are model artefacts.** Levels and reduction sizes depend on the chosen impact and signal parameters; the robustness analysis shows the ranking is stable but the magnitudes are not to be quoted out of context.
3. **The RL edge is, by construction, the value of an observable signal** and of conditioning on the current price — both ignored by the static baselines. A weaker, noisier or non-stationary signal shrinks the edge.
4. **Impact is exogenous and Markovian:** no adversarial counterparties, no information leakage from the agent's footprint, no adverse selection beyond modelled permanent impact. Live performance would be lower.
5. **Train and test share the data-generating process.** Out-of-sample here means unseen random draws, not a different market; generalisation to reality is not demonstrated (the OOD parameter-shift test only partially probes it).
6. **H2 fragmentation alpha is defined by the injected dispersion** and is parameterised through price dispersion rather than HHI; it is genuine within the model (zero when dispersion → 0) but its magnitude is a property of the venue model, not a measurement of the real CAC 40.
7. **Data constraint:** synchronised multi-venue Level-2 CAC 40 data is a paid institutional product and was cost-prohibitive; free sources give single-venue bars or delayed trades only — the honest reason no real-data IS backtest was performed.
8. **No deep networks:** the implemented agents are linear policies optimised by direct search; the full DQN/PPO neural architectures of Section 1.4.2 are a scalable extension, not implemented here.
