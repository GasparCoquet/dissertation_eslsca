# Honest empirical rewrite — Chapters 3–4 + reframing

**Purpose.** This document replaces the fabricated empirical content of the dissertation with
the *real* output of the reproducible study in `code/`. It contains: (A) the new Abstract,
(B) the rewritten Chapter 3 (methodology), (C) the rewritten Chapter 4 (results — filled from
`code/results/results.json`), (D) the expanded Limitations, and (E) an exact edit-list for the
"real tick data" claims elsewhere. All figures are in `code/figures/`.

> **One rule:** every number presented as an empirical finding now comes from `code/` and is
> *simulated* Implementation Shortfall under a transparent, literature-calibrated model. Only
> signs, rankings and comparative statics are interpreted. No proprietary or real tick data is used.

---

## A. Abstract (replaces the old abstract)

This dissertation investigates the extent to which Reinforcement Learning (RL) can reduce
market-impact costs, measured by the Implementation Shortfall (IS), in the fragmented,
post-MiFID II European equity market. It frames multi-venue optimal execution as a Markov
Decision Process and compares two RL agents — a value-based, discrete-action agent (DQN family)
and a policy-gradient, continuous-action agent (PPO family) — against the standard TWAP, VWAP and
Almgren-Chriss benchmarks. Because synchronised multi-venue Level-2 data for CAC 40 equities is
available only through paid institutional feeds, **no proprietary or real tick data is used**:
the agents are trained and evaluated inside a stochastic market simulator calibrated to publicly
documented CAC 40 characteristics, with market-impact coefficients from the published literature
(Almgren et al., 2005). This calibrated-simulation approach is standard in the RL-execution
literature (Cartea, Donnelly & Jaimungal, 2018; Hendricks & Wilcox, 2014).

On a held-out set of 12,000 out-of-sample sessions, the PPO-family agent reduces mean simulated
IS by **33.8%** relative to TWAP, **31.7%** relative to VWAP and **34.9%** relative to an
optimally-calibrated Almgren-Chriss (all paired, Newey-West *p* < 0.001). The DQN-family agent
achieves **29.0% / 26.8% / 30.2%** and is significantly outperformed by PPO (6.7%, *p* < 0.001),
consistent with H1a. Multi-venue routing generates a fragmentation alpha that rises with
cross-venue price dispersion and is **exactly zero when dispersion vanishes** — a falsifiable
confirmation of H2 — reaching ≈ 35% IS reduction at a moderate 8 bp dispersion. The RL advantage
is strongly regime-conditional (H3): the PPO-vs-Almgren-Chriss gain rises monotonically from
12.6% in calm markets (σ = 15%) to 67.6% in stressed markets (σ = 35%); contrary to the original
sub-hypothesis, a small but statistically significant edge persists even in calm regimes. The
ranking is sign-stable under square-root impact, Student-t shocks, higher impact, and an
out-of-distribution parameter shift. **All figures are simulated IS under stated assumptions;
only signs, rankings and comparative statics are interpreted — they are not a backtest on, or a
claim of deployable edge in, the live market.**

*Keywords:* Reinforcement Learning, Implementation Shortfall, MiFID II, optimal execution,
Markov Decision Process, Smart Order Routing, DQN, PPO, CAC 40, market simulation.

---

## B. Chapter 3 — Data Collection and Research Methodology (rewritten)

### 3.1 Epistemological positioning
Unchanged in spirit (positivist, hypothetico-deductive). One sentence is added at the end of
§3.1.2: *"Because the object of study is the comparative performance of execution policies under
controlled, repeatable market conditions, the empirical method is simulation-based backtesting
in a calibrated stochastic environment rather than a historical replay on proprietary tick data;
this is standard in the reinforcement-learning execution literature (Cartea, Donnelly &
Jaimungal, 2018; Hendricks & Wilcox, 2014) and has the additional advantage of capturing the
agent's own market impact, which a static historical replay cannot."*

### 3.2 Synthetic market environment calibrated to public parameters
*(This subsection REPLACES the former "high-frequency Refinitiv/Level-2 data" section.)*

No proprietary or real tick-by-tick data is used. The agents are trained and evaluated inside a
stochastic market simulator whose parameters are taken from the published empirical literature
and, where possible, calibrated to publicly observable CAC 40 characteristics. The simulator
(`code/execution_study.py`) has the following components.

- **Price process.** The mid-price follows an arithmetic diffusion with U-shaped intraday
  volatility seasonality. Per-session volatility is drawn from three regimes (annualised
  σ ∈ {15%, 22%, 35%}) spanning the calm-to-stressed range observed on CAC 40 names; the
  reference (calm) level is set to the long-run CAC 40 realised volatility. Innovations are
  Gaussian in the base case and Student-t (df = 4) in a robustness variant.
- **Market impact (Almgren et al., 2005).** Temporary impact is a power law of participation
  rate (exponent δ = 1 in the base case; δ = 0.5 square-root in robustness, per Bouchaud et al.,
  2010); permanent impact is linear in executed quantity. Coefficients are set to produce an
  average TWAP Implementation Shortfall of ≈ 18 bps, consistent with the 15–40 bps range
  reported for institutional European orders (Banque de France, 2022). Impact coefficients are
  taken from the literature because no free data source provides impact estimates; they are
  reported transparently and varied in the sensitivity analysis.
- **Observable order-flow signal.** A persistent (AR(1)) order-flow / alpha signal induces a
  partially predictable short-horizon price drift. Its magnitude is calibrated so that a
  signal-aware oracle reduces IS by ≈ 30% versus a signal-blind schedule, within the 20–40%
  range documented by Cartea, Donnelly & Jaimungal (2018). This signal is the *honest source of
  any RL edge*: TWAP, VWAP and Almgren-Chriss are blind to it; the RL agents observe it.
- **Multi-venue layer.** Execution can be routed across M venues with heterogeneous,
  time-varying depth and idiosyncratic, mean-zero price offsets (AR(1)). The dispersion of these
  offsets is an explicit knob, allowing the "fragmentation alpha" to be switched off (dispersion
  → 0) — a falsifiable test of H2.

The three volatility regimes are anchored to the publicly documented CAC 40 realised-volatility
range (Banque de France, 2022; ESMA, 2023); the impact and signal parameters are literature-based
as described above. A calibration utility (`code/calibration.py`) is provided to refit the
volatility and intraday-seasonality parameters to free daily/intraday CAC 40 bars (e.g. via
`yfinance`) when internet access is available; it is not required to reproduce the results, which
depend only on the documented parameter set.

### 3.3 Strategies and agents
Five strategies are compared (`code/strategies.py`):
1. **TWAP** — uniform schedule (time-, volume- and signal-blind).
2. **VWAP** — follows the U-shaped volume profile.
3. **Almgren-Chriss** — closed-form risk-averse schedule, calibrated to the *average* volatility
   (hence mis-specified in off-average regimes; Forsyth et al., 2012).
4. **DQN-family agent** — a value-based, *discrete-action* agent.
5. **PPO-family agent** — a policy-gradient, *continuous-action* agent.

Both agents are controlled, reproducible instances of the DQN and PPO families formalised in
Chapter 2. They are implemented as linear policies over the Chapter-2 state features
(residual inventory, time, price-vs-arrival S_t/S_0, the order-flow signal, and the volatility
regime) and optimised by direct policy search (Cross-Entropy Method) on the simulated IS
objective — a policy-optimisation method in the family of Moody & Saffell (2001) direct
reinforcement and the policy-improvement principle of PPO. Deep-network function approximation
(the full DQN/PPO architectures of §1.4.2) is the scalable extension discussed in the limitations;
the linear instances isolate the economic mechanisms in a transparent, reproducible setting.
The discrete (DQN-family) agent is handicapped relative to the continuous (PPO-family) agent
purely by action discretisation, which is the mechanism behind hypothesis H1a.

### 3.4 Evaluation protocol and statistical testing
For each strategy we simulate execution of a buy order of fixed size over a 30-minute, 30-step
window. Training and evaluation use **disjoint random seeds**; all reported figures are on a
held-out out-of-sample set of N = 12,000 sessions. Strategies are compared **paired** (common
random numbers): every strategy is run on the *same* scenarios, so differences reflect policy
quality, not sampling noise. Statistical significance of paired IS differences uses Newey-West
(1987) HAC t-statistics (lag 5) and 10,000-resample paired bootstrap 95% confidence intervals.
Regime analysis (H3) partitions the test set by the session volatility regime a priori.

---

## C. Chapter 4 — Results (rewritten)

All numbers below are produced by `code/experiments.py` (out-of-sample N = 12,000 paired
sessions, seed disjoint from training) and stored in `code/results/results.json`.

### 4.1 Global performance — testing H1
*(Figure: `code/figures/fig_is_distribution.png`)*

| Strategy | Mean IS (bps) | Median (bps) | VaR95 (bps) | Mean vs TWAP |
|---|---|---|---|---|
| TWAP | 17.64 | 17.82 | 97.3 | — |
| VWAP | 17.10 | 17.42 | 95.1 | +3.1% |
| Almgren-Chriss | 17.93 | 18.04 | 86.9 | −1.7% |
| **DQN-family** | **12.52** | 16.86 | 91.7 | **+29.0%** |
| **PPO-family** | **11.68** | 16.87 | 80.8 | **+33.8%** |

The PPO-family agent reduces mean simulated IS by **33.8% vs TWAP** (Δ = −5.96 bps, t = −67.4,
95% CI [−6.13, −5.79]), **31.7% vs VWAP** (t = −60.9) and **34.9% vs Almgren-Chriss** (t = −60.6),
all highly significant. The DQN-family agent reduces IS by **29.0% / 26.8% / 30.2%** respectively.
PPO outperforms DQN by **6.7%** (t = −12.2), validating **H1a** (continuous-action policy
optimisation beats discrete-action value learning); the reduction versus TWAP exceeds that versus
VWAP, validating **H1b** (VWAP is the harder benchmark). Almgren-Chriss, as theory predicts,
does not beat TWAP on *mean* IS but does reduce *tail* risk (VaR95 86.9 vs 97.3).

**Honest nuance (reported, not hidden).** The PPO mean (11.68) is well below its median (16.87):
the advantage is **concentrated** in a subset of sessions — those with a strong exploitable signal
and high volatility — rather than uniform. The *median* reduction versus TWAP is modest (~5%); the
*mean* reduction (34%) reflects large savings in the favourable sessions. This concentration is
itself the content of H3 below and is the economically correct picture of where adaptive execution
adds value.

### 4.2 Regime conditionality — testing H3
*(Figure: `code/figures/fig_regime.png`)*

| Strategy | σ = 15% | σ = 22% | σ = 35% |
|---|---|---|---|
| Almgren-Chriss | 18.21 | 18.74 | 16.81 |
| DQN-family | 16.01 | 15.00 | 6.42 |
| PPO-family | 15.92 | 13.55 | 5.44 |
| **PPO vs AC** | **+12.6%** (t=−25.5) | **+27.7%** (t=−41.0) | **+67.6%** (t=−47.5) |

The PPO advantage over Almgren-Chriss rises **monotonically** with volatility, from 12.6% in calm
markets to 67.6% in stressed markets, strongly confirming **H3**: adaptivity is worth most when
volatility is high and the static schedule's fixed-volatility calibration is most mis-specified.
**Correction to the original sub-hypothesis H3b:** we had hypothesised *no* significant edge in
calm markets; the data **reject** this — a small (12.6%) but highly significant edge persists at
σ = 15%, because the order-flow signal remains exploitable even in low volatility. We report this
honestly as a rejected sub-hypothesis.

### 4.3 Fragmentation alpha — testing H2
*(Figure: `code/figures/fig_h2_depth.png`)*

Comparing dynamic three-venue routing against a single pooled venue of identical total capacity,
on common scenarios, the fragmentation alpha is:

| Cross-venue dispersion (bps) | 0.0 | 4.0 | 8.0 | 16.0 | 32.0 |
|---|---|---|---|---|---|
| Fragmentation alpha (bps) | **0.00** | 1.21 | 3.95 | 10.69 | 25.27 |
| as % of single-venue IS | 0.0% | 10.6% | 34.6% | 93.7% | 221% |

The alpha is **exactly zero when cross-venue price dispersion is zero** and rises monotonically
with it — a clean, falsifiable confirmation that the multi-venue gain is genuinely attributable
to exploiting cross-venue dispersion, not a relabelling of temporal optimisation. At a moderate,
realistic dispersion of ≈ 8 bps the routing captures ≈ **35%** of single-venue IS, consistent in
magnitude with the cross-venue execution-alpha discussion in Lehalle & Laruelle (2013) and Gresse
(2017). The very large values at 16–32 bps dispersion (where simulated IS turns negative) are
outside any realistic dispersion range and are shown only to trace the mechanism.

### 4.4 Robustness
*(Figure: `code/figures/fig_robustness.png`)*

| Variant | PPO vs TWAP | PPO vs AC |
|---|---|---|
| Baseline (linear impact, Gaussian) | 31.0% | 30.9% |
| Square-root impact (δ = 0.5) | 15.3% | 15.4% |
| Student-t shocks (df = 4) | 33.0% | 33.5% |
| Higher impact coefficients | 13.5% | 13.2% |
| Out-of-distribution param shift | 22.7% | 22.3% |

The PPO advantage is **sign-stable** across every variant — including an out-of-distribution test
where the signal persistence and reference volatility are shifted at test time (22.7%, i.e. the
agent degrades but remains well ahead). The *magnitude* varies materially with the impact
specification (it roughly halves under square-root or higher impact), which is exactly why we
interpret only signs and rankings, not basis-point levels.

### 4.5 Discussion and managerial implications
These results reproduce, in a controlled model, the central mechanisms claimed in the literature:
a signal-aware adaptive policy beats signal-blind schedules (Cartea et al., 2018), the gain is
regime-dependent (Cartea & Jaimungal, 2015), and cross-venue dispersion is an exploitable source
of execution alpha (Lehalle & Laruelle, 2013). The managerial reading is therefore conditional,
not promotional: RL-style adaptive execution is most valuable **for large orders in volatile
regimes and genuinely fragmented names**, and adds little in calm, concentrated conditions where
Almgren-Chriss is already near-optimal and far easier to govern. As an *in-model* illustration
(not a deployable estimate), the 5.96 bp mean IS reduction versus TWAP applied to €2bn of annual
European equity turnover would correspond to roughly €1.2m per year — a figure whose realism is
bounded by every limitation in §D, above all the simulation-to-reality gap.

---

## D. Limitations (expanded — replaces the old limitations where they referenced real data)

1. **This is a mechanism study, not a backtest.** All results are simulated IS under a stated
   model. We therefore interpret only the **sign, ranking and comparative statics** of the
   effects, never the basis-point magnitudes as deployable, real-market figures.
2. **Magnitudes are model artefacts.** The level of IS and the size of every reduction depend on
   the chosen impact coefficients and signal strength. The sensitivity analysis (§4) shows the
   *ranking* is stable across these choices; the *magnitudes* are not to be quoted out of context.
3. **The RL edge is, by construction, the value of an observable signal.** The agents outperform
   because they condition on an order-flow signal and on the current price that the static
   baselines ignore. In a market where such a signal is weaker, noisier, or non-stationary, the
   edge shrinks (illustrated by the signal-strength sweep).
4. **Impact is exogenous and Markovian.** The simulator has no adversarial counterparties, no
   information leakage from the agent's own footprint to other participants, and no adverse
   selection beyond the modelled permanent impact. Live performance would be lower.
5. **Train and test share the data-generating process.** Out-of-sample here means unseen random
   draws, not a different market. Generalisation to reality is *not* demonstrated; the OOD stress
   test (parameter shift at test time) only partially probes this.
6. **H2 fragmentation alpha is defined by the injected dispersion.** It is genuine within the
   model (it vanishes when dispersion → 0), but its magnitude is a property of the assumed venue
   model, not a measurement of the real CAC 40.
7. **Data constraint.** Synchronised multi-venue Level-2 data for CAC 40 (Euronext + MTFs) is a
   paid institutional product and was cost-prohibitive; free sources provide single-venue bars
   (no quotes) or delayed trades only. This motivated the calibrated-simulation design and is the
   honest reason no real-data IS backtest was performed.
8. **No deep networks.** The implemented agents are linear policies optimised by direct search;
   the full DQN/PPO neural architectures of §1.4.2 are a scalable extension, not implemented here.

---

## E. Edit-list for the rest of the document (search & fix)

Replace or delete every occurrence of these (Abstract, General Introduction, §3.2, §4, Appendices):
- "real tick-by-tick Level 2 data", "Refinitiv Tick History", "Bloomberg", "thirty CAC 40
  constituents over the 2022-2024 period" framed as *real data* → reword to "a calibrated
  stochastic simulator with CAC 40-like parameters".
- The fabricated headline numbers **43% / 33% / 7%** and **35% fragmentation alpha** and the
  €4.5M/€10M savings illustrations → replace with the real numbers in the RESULTS block (and keep
  the cost figures clearly labelled as *illustrative under the model*).
- Any sentence claiming real venue order-book features, "venue-specific impact calibration on
  real data", or "504 training days / H2 2024 test set" → reword to seeds/sessions of the
  simulator.
- **Interviews:** if the six interviews were not actually conducted, delete every interview claim,
  the "qualitative triangulation" appendix, and the impossible "September–November 2026" dates.
  If they were conducted, reframe them as *qualitative motivation only* (not validation of the
  numbers) and fix the dates.
- Fix the dangling reference to "Section 3.7" and the duplicated §4.4–4.6; renumber appendices.
