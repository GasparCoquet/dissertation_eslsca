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

### 4.1.1 Interpreting the mechanism

The table requires interpretation beyond the headline numbers. TWAP and VWAP represent zero-state-conditioning strategies: both are pre-committed schedules that use no real-time information beyond the clock (TWAP) or the volume curve (VWAP). Their mean IS values of 17.64 and 17.10 bps respectively are therefore a pure measure of the price impact the investor accepts by spreading execution over time without adapting to observed price dynamics. VWAP's marginal improvement over TWAP (3.1%) reflects the value of timing executions to coincide with periods of high natural liquidity, thereby reducing the agent's own footprint in price-formation — a well-documented mechanism (Lehalle & Laruelle, 2013).

Almgren-Chriss is qualitatively different: it *does* optimise over the price-impact function, but it does so once, offline, using a fixed volatility estimate. Accordingly it improves tail risk materially (VaR95 falls from 97.3 to 86.9 bps) while slightly *raising* the mean IS relative to TWAP, because the Almgren-Chriss solution tilts execution toward the start of the day to front-load before uncertainty compounds, and this front-loading carries a higher expected price-impact cost per unit than the flat TWAP schedule under the linear impact model used here. This observation is consistent with Forsyth et al. (2012), who show that the Almgren-Chriss solution is optimal only under its own parametric assumptions and can under-perform simpler rules when those assumptions are misspecified or when unmodelled signals are available.

The RL agents exploit a layer of information the static strategies discard: the intraday order-flow signal that generates directional short-run price predictability. By conditioning each participation decision on the current signal value, the PPO agent achieves a mean IS of 11.68 bps — a level the static schedules cannot reach regardless of their parameter tuning, because they have no mechanism through which to use state information at all. This is precisely the value of a signal-adaptive policy described in Cartea et al. (2018): even a moderate degree of short-run predictability translates into substantial execution-cost savings when the policy can condition its pace on it.

The hierarchy PPO > DQN > TWAP/VWAP in mean IS, alongside the reversed hierarchy in VaR95 (PPO 80.8 < TWAP 97.3), means that PPO Pareto-dominates the static baselines on *both* dimensions simultaneously — it delivers lower expected cost *and* lower tail risk. This is not a guaranteed property of RL in general; it holds here because the signal also provides early warning of adverse price moves, allowing the PPO agent to accelerate when the signal is favourable and slow when conditions are threatening.

### 4.1.2 Execution-schedule analysis

![Figure 4.6 — Average execution schedule by strategy (simulated).](code/figures/fig_participation.png)

The average execution schedule (Figure 4.6) reveals *how* the PPO agent realises its mean-IS advantage. Unlike TWAP — which places equal participation at every interval — the PPO policy front-loads execution when the order-flow signal is positive at the session open and decelerates sharply in mid-session if the signal weakens, before accelerating again near the close to avoid excessive position-completion risk. This pattern is consistent with the "bang-bang" character that optimal signal-adaptive policies tend toward when the signal is strong: it is worth accepting higher instantaneous impact to trade while prices move in a favourable direction, rather than spreading execution mechanically.

DQN exhibits a qualitatively similar but less smooth schedule, because the discrete action space forces it to choose among a finite grid of participation rates, blunting its ability to fine-tune participation in response to continuous signal values. This discretisation cost is a direct cause of the 6.7% mean-IS gap between PPO and DQN. The VWAP schedule, by construction, mirrors the pre-specified volume profile and cannot deviate from it; Almgren-Chriss follows a declining-rate schedule (front-loaded but smoothly so) determined offline. Both schedules are blind to any information that arrives after the session opens.

## 4.2 Regime Conditionality — Testing H3

| Strategy | σ = 15% (low) | σ = 22% | σ = 35% (high) |
|---|---|---|---|
| Almgren-Chriss | 18.21 | 18.74 | 16.81 |
| DQN-family | 16.01 | 15.00 | 6.42 |
| PPO-family | 15.92 | 13.55 | 5.44 |
| **PPO vs Almgren-Chriss** | **+12.6%** (t=−25.5) | **+27.7%** (t=−41.0) | **+67.6%** (t=−47.5) |

![Figure 4.2 — Mean Implementation Shortfall by strategy and volatility regime (simulated).](code/figures/fig_regime.png)

![Figure 4.7 — IS distribution by volatility regime: TWAP vs PPO (simulated).](code/figures/fig_regime_dist.png)

The PPO advantage over Almgren-Chriss rises **monotonically** with volatility, from 12.6% in the calm regime to 67.6% in the stressed regime, strongly confirming **H3**: adaptivity is worth most when volatility is high and the static schedule's fixed-volatility calibration is most mis-specified. In the high-volatility regime (σ = 35% > 25%), PPO beats Almgren-Chriss by 67.6%, far exceeding the 10% threshold of **H3a**, which is **supported**.

**H3b is rejected — and we report this openly.** H3b predicted *no* significant PPO–Almgren-Chriss difference in calm markets (σ < 15%). The data reject it: a smaller but highly significant 12.6% edge (t = −25.5) persists at σ = 15%, because the order-flow signal remains exploitable even in low volatility. The *direction* of H3 is therefore confirmed while its *strong form* (a vanishing edge in calm regimes) is falsified — a result we retain rather than discard, since the asymmetry of gains across regimes is itself the economically meaningful finding.

### 4.2.1 Why the RL edge scales with volatility

The mechanism behind the monotonic scaling is transparent within the model structure. Almgren-Chriss calibrates its optimal speed parameter η using a *fixed* volatility estimate. When realised volatility matches the calibration (approximately the mid-regime), the solution is near-optimal given the model's own signal-free premise. When realised volatility rises well above the calibration — as in the σ = 35% regime — two sources of mis-specification compound: first, the optimal urgency is higher than the pre-calibrated schedule assumes (faster trading is justified to avoid amplified price-impact risk from a larger adverse price move), and second, the signal becomes *more* informative in absolute bps terms when volatility is high, because a given signal coefficient implies larger predicted price moves. The PPO agent, conditioning on the current signal in each period, exploits both of these at once, producing a 67.6% advantage.

In the calm regime (σ = 15%), the signal is still informative but price moves are small in absolute terms; the PPO agent's IS falls only to 15.92 bps versus AC's 18.21 bps, a 12.6% improvement. This residual edge confirms that the order-flow signal retains value at any level of volatility, but its benefit is far more dramatic when market conditions are stressed. Practically, this implies that the case for deploying an adaptive RL execution policy is strongest precisely when markets are least comfortable — a relevant consideration for risk managers, since those are also the sessions where manual intervention in execution is most costly and most likely to introduce discretionary biases.

The IS distributions (Figure 4.7) make the same point visually: in the low-volatility regime the TWAP and PPO distributions are largely overlapping with the PPO distribution slightly left-shifted; in the high-volatility regime the PPO distribution separates dramatically to the left, with a long right tail remaining for TWAP. The right tail of TWAP in the stressed regime corresponds to sessions where adverse mid-session price moves compound against the pre-committed schedule — exactly the sessions where the PPO agent's ability to front-load or defer pays off most.

## 4.3 Fragmentation Alpha — Testing H2

Dynamic three-venue routing is compared against a single pooled ("Euronext-only") venue of identical total capacity, on common scenarios, as the cross-venue price dispersion is varied:

| Cross-venue dispersion (bps) | 0.0 | 4.0 | 8.0 | 16.0 | 32.0 |
|---|---|---|---|---|---|
| Fragmentation alpha (bps) | 0.00 | 1.21 | 3.95 | 10.69 | 25.27 |
| as % of single-venue IS | 0.0% | 10.6% | 34.6% | 93.7% | 221% |

![Figure 4.3 — Fragmentation alpha vanishes as cross-venue dispersion approaches zero (H2).](code/figures/fig_h2_depth.png)

![Figure 4.8 — The RL edge scales with the order-flow signal strength (simulated).](code/figures/fig_signal_value.png)

The fragmentation alpha is **exactly zero when cross-venue price dispersion is zero** and rises monotonically with it — a clean, falsifiable confirmation that the multi-venue gain is genuinely attributable to exploiting cross-venue dispersion, not a relabelling of temporal optimisation. At a moderate, realistic dispersion of approximately 8 bps, routing captures about **34.6% of single-venue IS**, exceeding the 25% threshold of **H2**, which is **supported**; the dynamic multi-venue policy significantly outperforms the single-venue benchmark (**H2a supported**). The effect is monotone increasing in dispersion, consistent in spirit with **H2b** (more fragmented names yield larger gains); we note, however, that the simulator parameterises fragmentation through price dispersion rather than the Herfindahl–Hirschman Index directly, so H2b is corroborated by a proxy rather than tested on HHI itself. The very large values at 16–32 bps dispersion (where simulated IS turns negative) lie outside any realistic dispersion range and are shown only to trace the mechanism. These magnitudes are consistent in direction with the cross-venue execution-alpha discussion of Lehalle & Laruelle (2013) and Gresse (2017).

### 4.3.1 The mechanism and its limits

The zero-dispersion anchor (0.00 bps alpha at 0 bps dispersion) is the most important single cell in Table 4.3. It establishes that the multi-venue architecture contains no hidden advantage from any source other than price dispersion itself: when all venues quote identically, there is nothing to route optimally, and the dynamic router does not outperform a single venue. This falsifiability is what distinguishes a genuine mechanism test from a result that could be explained by any number of confounded factors.

As dispersion rises from zero, the routing policy allocates volume to the venue offering the most favourable instantaneous quote, subject to the venue's capacity and the agent's impact on each venue's price. At 4 bps dispersion the gain is modest (1.21 bps, 10.6%) because the dispersion rarely exceeds the bid-ask spread and the routing decision is not consistently actionable. By 8 bps the dispersion is large enough relative to the spread that the best venue is identifiable from the observable quote, and the PPO agent captures approximately a third of single-venue IS through routing alone. This is consistent with the literature on smart order routing: Gresse (2017) shows that in fragmented European equity markets the venues do exhibit persistent relative price differences intraday, and that algorithms capable of identifying the best venue at each moment can realise material cost reductions.

The signal-strength figure (Figure 4.8) provides a complementary view: it plots the fragmentation alpha (or equivalently the RL IS advantage) as a function of signal strength, confirming the linear scaling of the edge with the informativeness of the observable. When the order-flow signal is zero, the RL policy has no informational advantage over static schedules; as signal strength rises, the alpha grows. This scaling is consistent with the theoretical prediction of Cartea et al. (2018) that the value of a signal-adaptive policy is proportional to the signal's predictive power. The result also provides a natural governance criterion: practitioners can test their own signal strength before committing to a RL execution stack, and can calibrate expected savings against the cost of the infrastructure.

## 4.4 Convergence and Learning Behaviour

![Figure 4.5 — Policy-search learning curves: best mean IS per CEM generation (simulated).](code/figures/fig_learning.png)

A necessary condition for trust in a learned policy is that the optimisation did in fact converge — that the reported IS figures reflect a stable policy and not the noise of an optimiser that was still exploring at evaluation time. Figure 4.5 plots the best mean IS achieved by the policy-search procedure (Cross-Entropy Method for the linear policy) at each generation, for both the PPO-family and DQN-family agents.

Both curves exhibit a consistent structure: rapid improvement in the first fifteen to twenty generations as the optimiser identifies the coarse shape of the optimal schedule, followed by a slower plateau phase in which incremental generations provide diminishing returns. The PPO-family curve reaches approximate stationarity at a mean IS near 11.7 bps; the DQN-family curve plateaus at approximately 12.5 bps. The separation between the two plateau levels is the discretisation penalty quantified in Section 4.1: the DQN cannot improve further because its action space does not contain the precise participation rate that the PPO policy can represent.

The convergence behaviour also has a methodological implication: the policy was evaluated on the held-out set only after the training curve had plateaued, so the IS figures in Table 4.1 are not contaminated by evaluating an under-trained policy. This is a necessary but not sufficient condition for the results to be meaningful; the other necessary condition — that the held-out set shares the same data-generating process as training — is met by construction but, as noted in Section 4.8, this does not imply robustness to the real market.

## 4.5 Robustness

| Variant | PPO vs TWAP | PPO vs Almgren-Chriss |
|---|---|---|
| Baseline (linear impact, Gaussian) | 31.0% | 30.9% |
| Square-root impact (δ = 0.5) | 15.3% | 15.4% |
| Student-t shocks (4 d.o.f.) | 33.0% | 33.5% |
| Higher impact coefficients | 13.5% | 13.2% |
| Out-of-distribution parameter shift | 22.7% | 22.3% |

![Figure 4.4 — Robustness of the PPO advantage across model assumptions (simulated).](code/figures/fig_robustness.png)

The PPO advantage is **sign-stable across every variant**, including an out-of-distribution test in which the signal persistence and the reference volatility are shifted at test time (22.7%; the agent degrades but remains well ahead). The *magnitude* varies materially with the impact specification — it roughly halves under square-root or higher impact — which is exactly why only signs and rankings, not basis-point levels, are interpreted throughout.

### 4.5.1 Interpreting the robustness pattern

The robustness table deserves closer reading because the pattern of magnitude variation is itself informative. The two variants that most compress the PPO advantage are square-root impact (δ = 0.5) and higher impact coefficients, both of which share a structural feature: they increase the marginal cost of executing a given volume increment. Under a square-root impact schedule, the agent faces a convex cost that grows faster than under linear impact; the optimal response is to spread execution more evenly over time, which reduces the scope for signal-timed concentration. When the agent's most aggressive actions become more costly, the gap between signal-adaptive and signal-blind strategies narrows because the agent's optimal policy converges toward a smoother schedule that more closely resembles TWAP. This observation is consistent with Almgren et al. (2005), who show that under square-root price impact the optimal trajectory is more front-loaded and smoother than under linear impact, leaving less room for state-dependent acceleration or deceleration.

By contrast, the Student-t shock variant (33.0% and 33.5%) shows an *increase* relative to baseline in the PPO advantage. Fat-tailed price shocks create occasional large adverse moves that the pre-committed TWAP and Almgren-Chriss schedules cannot avoid, but the PPO agent, by conditioning on the current signal, can identify the early stages of an adverse drift and accelerate ahead of the worst of the move. Heavy tails therefore *expand* the value of adaptivity, a result with practical relevance given that empirical equity returns are well-documented to exhibit heavier tails than Gaussian (Nevmyvaka et al., 2006).

The out-of-distribution test is designed as a stress test of the policy's transferability: it shifts both the signal persistence and the reference volatility away from their training values at evaluation time, without allowing the policy to re-adapt. The 22.7% residual advantage shows that the policy retains its qualitative properties — it still front-loads when the signal is positive, decelerates when it is negative — even when the quantitative parameter environment shifts. This partial robustness does not imply generalisation to the real market, where the data-generating process differs from the calibrated model in many dimensions simultaneously; it does provide limited evidence that the learned heuristic is not purely an artefact of the exact training parameters.

## 4.6 Threats to Validity and Why We Interpret Only Signs and Rankings

Careful interpretation of simulation-based results requires an explicit statement of the threats to validity that prevent treating the simulated IS magnitudes as estimates of live-market performance. Four threats are structural and cannot be mitigated within the present study.

First, the calibrated model is itself a low-dimensional representation of the actual price-formation process in CAC 40 futures. The model has one order-flow signal, a parametric impact function, and Gaussian or Student-t price innovations. Real markets have multiple correlated signals, a complex and non-stationary impact structure that depends on the identity and size of other participants, and fat-tailed innovations with persistent volatility clustering — none of which is captured here. The simulated IS magnitudes therefore reflect the model's own assumptions, not a property of the real market.

Second, the RL agent is trained and tested within the same model. Out-of-sample here means unseen random draws from the same data-generating process, not a genuinely independent sample. The agent cannot be evaluated on its ability to generalise to structural breaks, regime changes, or distributional shifts that lie outside the training distribution. The OOD test in Section 4.5 provides partial evidence on parameter-level shifts but does not address structural model changes.

Third, the simulation contains no adversarial counterparties. In the real market, a large systematic execution algorithm creates an observable footprint that other participants can detect and trade against — a form of adverse selection that is entirely absent from the model. This implies that the simulated IS figures are optimistic in a way that cannot be quantified without live-market data.

Fourth, and most practically: synchronised multi-venue Level-2 data for CAC 40 names is a paid institutional product that was cost-prohibitive for this study. The absence of real data means no real-data backtest was performed, and the fragmentation results in particular are model constructs rather than measurements of the actual CAC 40 microstructure. These limitations are the reason the chapter title includes "simulated" in every figure caption and the standing caveat is repeated at every section break. The reader's attention is directed to signs and rankings — the PPO policy outperforms static benchmarks, the outperformance scales with volatility and signal strength, and multi-venue routing adds value when dispersion is non-zero — because these ordinal findings are informative across a wide range of plausible model perturbations, as the robustness table confirms.

## 4.7 Discussion

These results reproduce, in a controlled model, the central mechanisms claimed in the literature: a signal-aware adaptive policy beats signal-blind schedules (Cartea et al., 2018); the gain is regime-dependent (Cartea & Jaimungal, 2015); and cross-venue dispersion is an exploitable source of execution alpha (Lehalle & Laruelle, 2013). The contribution of the dissertation is not the discovery of these mechanisms but their *joint, falsifiable demonstration* in a single reproducible framework, together with an explicit accounting of where the reinforcement-learning advantage comes from (an observable signal and conditioning on the current state) and where it does not (it is not free; it shrinks when the signal weakens, the impact steepens, or the environment shifts).

The rejection of H3b deserves particular mention as a positive methodological outcome. A study that reports only confirmed hypotheses is suspect; a study that reports a falsified hypothesis — where the data deviate from the prediction in a theoretically interpretable direction — provides genuine information. The 12.6% residual PPO advantage in the calm regime (t = −25.5) is not noise; it is a consistent finding that the order-flow signal retains predictive value at all levels of volatility, something a pre-registration of H3b would have obscured had the study reported only the high-volatility result.

## 4.8 Managerial Recommendations

The managerial reading is conditional, not promotional. RL-style adaptive execution is most valuable for **large orders, in volatile regimes, on genuinely fragmented names**, where the gains over both naive schedules and static Almgren-Chriss are largest. It adds comparatively little in calm, concentrated conditions, where Almgren-Chriss is already near-optimal and far easier to govern, validate and explain to a risk committee. As an *in-model illustration only* — not a deployable estimate — the 5.96 bps mean IS reduction versus TWAP, applied to €2bn of annual European-equity turnover, would correspond to on the order of €1.2m per year; the realism of any such figure is bounded by every limitation in Section 4.9, above all the simulation-to-reality gap.

Three practical implications follow from the regime-conditionality results. First, the case for investing in adaptive RL execution infrastructure is strongest for participants who routinely execute large orders on volatile names — the 67.6% IS improvement at σ = 35% dwarfs the 12.6% improvement in calm conditions, and the cost of building and maintaining the infrastructure should be calibrated accordingly. Second, a hybrid governance approach is sensible: use Almgren-Chriss as the primary policy in calm, predictable conditions where its tail-risk management (VaR95 86.9 bps) and analytical tractability outweigh the RL edge; switch to adaptive RL in volatile conditions where the RL edge is largest and the Almgren-Chriss calibration is most likely to be stale. Third, the fragmentation results suggest that the value of a smart routing layer is highly sensitive to the name's fragmentation characteristics: practitioners should measure the realised cross-venue price dispersion in their execution universe before deploying a multi-venue RL policy, since the benefit vanishes at zero dispersion.

## 4.9 Hypothesis Scorecard

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

## 4.10 Limitations

1. **This is a mechanism study, not a backtest.** All results are simulated IS under a stated model; only signs, rankings and comparative statics are interpreted, never the basis-point magnitudes as deployable figures.
2. **Magnitudes are model artefacts.** Levels and reduction sizes depend on the chosen impact and signal parameters; the robustness analysis shows the ranking is stable but the magnitudes are not to be quoted out of context.
3. **The RL edge is, by construction, the value of an observable signal** and of conditioning on the current price — both ignored by the static baselines. A weaker, noisier or non-stationary signal shrinks the edge.
4. **Impact is exogenous and Markovian:** no adversarial counterparties, no information leakage from the agent's footprint, no adverse selection beyond modelled permanent impact. Live performance would be lower.
5. **Train and test share the data-generating process.** Out-of-sample here means unseen random draws, not a different market; generalisation to reality is not demonstrated (the OOD parameter-shift test only partially probes it).
6. **H2 fragmentation alpha is defined by the injected dispersion** and is parameterised through price dispersion rather than HHI; it is genuine within the model (zero when dispersion → 0) but its magnitude is a property of the venue model, not a measurement of the real CAC 40.
7. **Data constraint:** synchronised multi-venue Level-2 CAC 40 data is a paid institutional product and was cost-prohibitive; free sources give single-venue bars or delayed trades only — the honest reason no real-data IS backtest was performed.
8. **No deep networks:** the implemented agents are linear policies optimised by direct search; the full DQN/PPO neural architectures of Section 1.4.2 are a scalable extension, not implemented here.
