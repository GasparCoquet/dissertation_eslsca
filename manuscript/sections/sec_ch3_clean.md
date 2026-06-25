# Chapter 3: Data Collection and Research Methodology

## Chapter Introduction

This chapter specifies the empirical method used to test the hypotheses of Chapter 2. Section 3.1 states the epistemological positioning. Section 3.2 describes the calibrated stochastic market simulator that serves as the data-generating environment, and explains why a simulation-based method — rather than a historical replay on proprietary tick data — is the appropriate and, given data-access constraints, the only feasible design. Section 3.3 specifies the benchmark strategies and the two reinforcement-learning agents. Section 3.4 sets out the evaluation protocol and the statistical tests.

## 3.1 Epistemological Positioning

The study adopts a positivist paradigm and a hypothetico-deductive approach: testable hypotheses are derived from theory (Chapter 2) and confronted with empirical evidence under conditions designed to allow falsification. Because the object of study is the *comparative* performance of execution policies under controlled, repeatable market conditions, the empirical method is simulation-based backtesting in a calibrated stochastic environment rather than a historical replay on proprietary tick data. This choice is standard in the reinforcement-learning execution literature (Nevmyvaka, Feng & Kearns, 2006; Hendricks & Wilcox, 2014; Cartea, Donnelly & Jaimungal, 2018) and carries a decisive methodological advantage: it captures the agent's own market impact and the market's reaction to its actions, which a static historical replay — where the tape is fixed regardless of what the agent does — cannot represent.

## 3.2 A Calibrated Market Simulator

### 3.2.1 Rationale and data constraint

No proprietary or real tick-by-tick data is used in this study. Synchronised multi-venue Level-2 data for CAC 40 equities (Euronext Paris together with the relevant MTFs) is available only through paid institutional feeds, the cost of which was prohibitive for this dissertation; freely available sources provide either single-venue daily/intraday bars without quote-level depth, or delayed trade prints, neither of which supports a credible multi-venue Implementation-Shortfall backtest. Rather than overstate the evidence, the study therefore adopts a transparent, reproducible simulator whose parameters are taken from the published literature and anchored to publicly documented CAC 40 characteristics. The full pipeline is openly reproducible from the accompanying `code/` directory.

### 3.2.2 Price, volatility and intraday seasonality

The mid-price follows an arithmetic diffusion over a 30-minute execution window discretised into K = 30 one-minute steps (indexed k = 0, 1, …, K−1). The per-step mid-price update is:

```
S_{k+1} = S_k  +  κ · n_k  +  θ · a_k · (σ / σ_ref)  +  σ_step · season_k · z_k
```

where:

- **S_k** is the mid-price at step k; **S_0** is the arrival (pre-trade) price.
- **n_k** is the number of shares executed at step k (the agent's action). The term **κ · n_k** represents permanent price impact: each unit of executed quantity shifts the fundamental price permanently upward (for a buy order) by the coefficient κ, following Almgren et al. (2005).
- **a_k** is the observable order-flow signal (see §3.2.4). The term **θ · a_k · (σ / σ_ref)** is a signal-driven drift: when the signal is positive, the mid-price drifts upward, penalising a delayed buy; when negative, it drifts downward, rewarding patience. Scaling by σ / σ_ref ensures that the signal's exploitable value increases with session volatility, providing the mechanism behind Hypothesis H3. σ_ref is a normalisation constant set to the central volatility regime (22% annualised).
- **z_k ~ N(0, 1)** (base case) or **z_k ~ t(4)** (robustness variant) is the idiosyncratic noise innovation.
- **σ_step** is the per-step volatility, derived from the annualised session volatility σ by the standard scaling: σ_step = σ · √(Δt), where Δt = 1/252 · 1/390 denotes one trading minute as a fraction of a year (390 trading minutes per day, 252 days per year).
- **season_k ∈ (0, 1]** is the U-shaped intraday seasonality multiplier, taking its peak values at k = 0 (open) and k = K−1 (close) and its minimum at the midday trough. This is parameterised as season_k = 1 − β · sin²(π · k / (K−1)) with β ∈ (0, 1) controlling the depth of the U, consistent with the empirical seasonality documented in European equity markets (Bouchaud et al., 2010).

Each session draws its annualised σ from one of three regimes — {15%, 22%, 35%} — spanning the calm-to-stressed range documented for CAC 40 names (Banque de France, 2022; ESMA, 2023), with the central level set to the long-run CAC 40 realised volatility.

### 3.2.3 Market-impact calibration

Market impact follows the Almgren et al. (2005) two-component decomposition. The **temporary impact** of trading n_k shares at step k is the instantaneous cost incurred over and above the mid-price, modelled as a power law of the per-step participation rate u_k = n_k / V_k (shares traded as a fraction of market volume at that step):

```
h(u_k) = a_temp · u_k^δ
```

In the base specification δ = 1 (linear temporary impact). A robustness variant uses δ = 0.5, corresponding to the square-root law of Bouchaud et al. (2010), which is empirically dominant for large institutional orders. The execution price received at step k is therefore S_k + h(u_k) (for a buy order paying the spread and temporary impact), so the cost of executing n_k shares at step k is n_k · [S_k + h(u_k)].

The **permanent impact** of executing a total quantity X over the window shifts the equilibrium price by an amount proportional to the order size relative to average daily volume D:

```
g = a_perm · (X / D)
```

This permanent shift is assumed to materialise linearly across steps proportional to n_k, contributing κ · n_k to each S_{k+1} update (with κ = a_perm / D · S_0 in price units).

The primary performance metric is **Implementation Shortfall (IS)**, the difference between the paper portfolio value and the actual portfolio value, expressed in basis points:

```
IS_bps = [ Σ_{k=0}^{K-1} n_k · P_k  −  X · S_0 ] / (X · S_0) · 10^4
```

where P_k = S_k + h(u_k) is the all-in execution price at step k, X = Σ n_k is the total order size, and S_0 is the arrival mid-price. A higher IS_bps indicates a worse outcome for the buyer.

The impact coefficients a_temp and a_perm are calibrated jointly so that the average Implementation Shortfall of a TWAP schedule (a uniform, participation-rate-agnostic benchmark) is approximately **18 basis points**. This lies within the 15–40 bps range reported for institutional European orders by Banque de France (2022) and is consistent with the academic literature on European equity market impact. Because no freely accessible data source provides direct impact estimates for the target universe, these coefficients are literature-based; they are reported transparently in Table B.1 (Appendix B) and subjected to a ±50% sensitivity analysis in Section 4.4.

### 3.2.4 Observable order-flow signal

The order-flow signal a_k evolves as a first-order autoregressive process:

```
a_{k+1} = ρ · a_k + ε_k,     ε_k ~ N(0, σ_ε²)
```

with |ρ| < 1 ensuring stationarity. The AR(1) structure is the minimal parameterisation that produces the short-horizon serial correlation and mean-reversion of Order Flow Imbalance (OFI), the quantity defined in Chapter 2 as the signed difference between buyer-initiated and seller-initiated volume at the top of the book. Cartea, Donnelly & Jaimungal (2018) and Bouchaud et al. (2010) both document that OFI is positively autocorrelated at the intraday horizon, with autocorrelation decaying over minutes — a pattern captured here by setting ρ ≈ 0.7 (corresponding to a half-life of roughly 2 minutes at the one-minute step frequency).

The signal enters the price dynamics via the drift term θ · a_k · (σ / σ_ref) described in §3.2.2: a positive a_k forecasts a near-term price increase, creating an urgency cost for a delayed buyer. The signal is **observable only to the reinforcement-learning agents**; the TWAP, VWAP and Almgren-Chriss benchmarks are signal-blind by construction. This asymmetry is deliberate: it creates a falsifiable performance wedge attributable exclusively to signal exploitation. The signal's magnitude is calibrated so that a hypothetical oracle — a policy with perfect foresight of every future a_k — reduces IS by approximately 30% relative to the signal-blind TWAP, within the 20–40% range documented by Cartea, Donnelly & Jaimungal (2018) for informed execution strategies.

Because the signal is observable to the agent but not to the econometrician observing market prices, it proxies the private order-flow information that institutional traders routinely condition on but cannot share externally. This mirrors the real-world setting described in Chapter 1.

### 3.2.5 Multi-venue routing and the fragmentation layer

In the three-venue specification, each venue j ∈ {1, 2, 3} has its own depth D_j and a time-varying idiosyncratic price offset c_j relative to the common mid-price S_k. The per-step execution cost of trading n_{j,k} shares at venue j is:

```
cost_{j,k} = c_{j,k} · n_{j,k}  +  b_j · n_{j,k}²
```

where the **linear term** c_{j,k} captures the venue-specific price offset (bid-ask spread component and inter-venue price dispersion) and the **quadratic term** b_j · n_{j,k}² captures the temporary impact local to that venue's order book:

```
b_j = a_temp · S_k / D_j
```

so that venues with deeper books (larger D_j) exhibit lower quadratic impact, as expected from a price-impact model in which the market absorbs volume at lower marginal cost when depth is ample.

The optimal instantaneous venue split for a given step-level child order of size n_k solves the convex quadratic programme:

```
min_{n_{j,k}}   Σ_j [ c_{j,k} · n_{j,k}  +  b_j · n_{j,k}² ]
subject to       Σ_j n_{j,k} = n_k,     n_{j,k} ≥ 0
```

The Karush-Kuhn-Tucker (KKT) conditions for this problem yield a **water-filling** solution: at optimum, all venues receiving positive flow share the same effective marginal cost, λ* (the shadow price of the capacity constraint), with:

```
n_{j,k}* = max( 0,  (λ* − c_{j,k}) / (2 b_j) )
```

and λ* chosen so that Σ_j n_{j,k}* = n_k. In practice this is solved efficiently in closed form by sorting venues by their adjusted cost and iterating over candidate active sets, with O(J log J) complexity for J venues. The full pseudocode is given in Appendix A (Algorithm A.3).

The **dispersion knob** governs the variance of the offsets c_{j,k}: setting it to zero makes all venues price-equivalent and collapses the routing problem to a pure depth-weighting exercise, eliminating any price-improvement benefit of fragmentation. This allows H2 — that RL routing adds value specifically through fragmentation alpha rather than through added aggregate liquidity — to be tested by comparing performance with dispersion switched on versus off. The **single-venue benchmark** is constructed with the same aggregate depth D_total = Σ_j D_j as the pooled three-venue setting, ensuring the comparison isolates routing value rather than a liquidity-capacity effect.

## 3.3 Strategies and Agents

### 3.3.1 Benchmarks

Three non-learning benchmarks are implemented. **TWAP** executes a uniform n_k = X/K shares at each step, blind to time-of-day volume profile, signal and volatility regime. **VWAP** distributes execution proportionally to the expected volume at each step, following the U-shaped volume profile v_k (normalised so that Σ_k v_k = 1): n_k = X · v_k; this eliminates participation-rate spikes at high-volume periods but remains signal-blind. **Almgren-Chriss** implements the closed-form risk-averse schedule derived by Almgren et al. (2005), which minimises the mean-variance trade-off E[IS] + λ · Var[IS] for a given risk-aversion parameter λ. The schedule front-loads execution monotonically as λ increases, converging to TWAP as λ → 0. It is calibrated to the *average* volatility (22% annualised) and is therefore mis-specified when a session's realised volatility departs from that average — a source of its under-performance in stressed regimes, consistent with the analysis in Forsyth et al. (2012).

The Almgren-Chriss implementation is validated by confirming that it reproduces the theoretical efficient frontier across a grid of λ values: the (E[IS], Std[IS]) pairs trace out the expected convex curve, with front-loading increasing monotonically. This internal validation provides confidence that the benchmark is correctly implemented before any RL comparison is drawn.

![Figure 3.1 — Almgren-Chriss efficient frontier: front-loading increases with risk aversion λ, validating the implementation.](code/figures/fig_ac_frontier.png)

### 3.3.2 Reinforcement-learning agents

Two RL agents are implemented as controlled, reproducible instances of the value-based and policy-gradient families. Both share the same **state vector** s_k = (q_k, τ_k, S_k/S_0, a_k, σ_regime), where q_k = (X − Σ_{i<k} n_i)/X is the residual inventory fraction, τ_k = (K−k)/K is the residual time fraction, S_k/S_0 is the price relative to arrival, a_k is the order-flow signal, and σ_regime ∈ {low, mid, high} is the session volatility indicator.

The **DQN-family agent** is value-based: it maps states to a Q-value over a discrete grid of M action levels {0, Δn, 2Δn, …, q_k · X} (with M = 20 in the base case), selecting the action that maximises estimated Q. The discrete grid is the defining constraint of this family: it cannot represent action values between grid points, which handicaps it relative to a continuous policy in regimes where the optimal action falls strictly between two grid nodes. This discretisation gap is the mechanism hypothesised in H1a to explain any performance difference from the continuous agent. Full pseudocode for the DQN-family training loop is given in Appendix A (Algorithm A.1).

The **PPO-family agent** is policy-gradient: it directly parameterises a continuous action distribution over n_k ∈ [0, q_k · X] and optimises policy parameters by gradient ascent on an importance-weighted objective with a clipping mechanism that prevents destabilising policy updates (Schulman et al., 2017). The clipped surrogate objective is:

```
L^CLIP(θ) = E_k [ min( r_k(θ) · Â_k,  clip(r_k(θ), 1−ε, 1+ε) · Â_k ) ]
```

where r_k(θ) = π_θ(a_k|s_k) / π_{θ_old}(a_k|s_k) is the importance ratio and Â_k is the advantage estimate. Full pseudocode is given in Appendix A (Algorithm A.2).

Both agents' policies are represented as **linear functions of the state features** (rather than deep neural networks), enabling exact reproducibility and interpretable weight inspection. The policy parameters are optimised by the **Cross-Entropy Method (CEM)**, a population-based, gradient-free optimiser in the family of Moody & Saffell (2001) direct reinforcement: at each generation a population of parameter vectors is sampled from a Gaussian, evaluated on the IS objective over a batch of simulated episodes, and the top-k elite vectors are used to update the sampling distribution's mean and covariance. CEM is well-suited to the relatively low-dimensional linear policy class and avoids the sensitivity to learning-rate tuning that characterises gradient-based methods. Deep neural-network function approximation — the full DQN/PPO architectures with experience replay and actor-critic value networks — is the natural scalable extension discussed in Chapter 5 (Limitations); the linear-policy instances used here are designed to isolate the economic mechanisms (signal exploitation, venue routing, regime adaptation) in a transparent, reproducible setting. All training hyper-parameters — population size, elite fraction, number of generations, action grid resolution — are reported in Appendix B (Table B.1).

![Figure 3.2 — Average execution schedule by strategy (simulated).](code/figures/fig_participation.png)

## 3.4 Evaluation Protocol and Statistical Testing

Each strategy executes a fixed buy order of size X = 1% of average daily volume over the 30-minute window. Training and evaluation use **disjoint random seeds**: all policy optimisation is performed on a dedicated training set, and no training episode appears in the evaluation set. All reported figures are computed on a held-out, out-of-sample set of **N = 12,000 sessions**, stratified equally across the three volatility regimes (4,000 sessions each), ensuring sufficient power for the regime-conditional tests required by H3.

Strategies are compared using a **common-random-number (paired) design**: every strategy is run on the *same* 12,000 scenarios (same price paths, same signal realisations, same volume profiles), differing only in the actions taken. This eliminates scenario-sampling variance from the performance differences, so that paired IS differences Δ_i = IS_i^{strategy A} − IS_i^{strategy B} reflect policy quality rather than sampling noise. The paired design substantially reduces the standard error of comparisons relative to an unpaired design with the same N, which is particularly important when performance gaps are narrow (as expected for intra-RL comparisons).

The primary test statistic for each paired comparison is a **Newey-West (1987) HAC t-statistic**, which accounts for the serial correlation and heteroskedasticity present in IS differences across sessions simulated with correlated price processes. The variance estimator takes the schematic form:

```
V^NW = γ_0 + 2 · Σ_{l=1}^{L} w_l · γ_l
```

where γ_l = N^{-1} Σ_{i=l+1}^{N} Δ̂_i · Δ̂_{i-l} is the l-th sample autocovariance of the demeaned paired differences Δ̂_i = Δ_i − Δ̄, and w_l = 1 − l/(L+1) are the Bartlett (triangular) kernel weights that ensure positive semi-definiteness of the variance estimate. The lag truncation parameter is set to L = 5, following the rule-of-thumb L = O(T^{1/4}) for T ≈ 30 steps per episode and N episodes. The HAC t-statistic is t^{NW} = √N · Δ̄ / √V^{NW}, referred to a standard-normal distribution under the null of zero mean difference.

The parametric HAC test is complemented by a **paired bootstrap**: 10,000 resample draws (with replacement) of the N paired differences yield an empirical distribution of Δ̄* values, from which 95% confidence intervals are read as the 2.5th and 97.5th percentiles. The bootstrap makes no parametric distributional assumption and is robust to the fat tails of IS under stressed-volatility scenarios (Student-t innovations variant).

For **H3** (regime-conditional performance), the evaluation set is partitioned *a priori* by volatility regime (low / mid / high), and the IS comparison is conducted separately within each partition. The test of H3 is then a test of whether the performance advantage of the signal-aware RL agents, measured as ΔIS_bps = IS^{TWAP} − IS^{RL}, is statistically and economically larger in the high-volatility regime than in the low-volatility regime. A regime-interaction contrast is constructed as ΔIS^{high} − ΔIS^{low} and tested with the same Newey-West / bootstrap procedure.

All results are presented with **point estimates, 95% confidence intervals, and Newey-West p-values**. The null hypothesis for each primary comparison is that the two strategies have equal expected IS (two-sided test at the 5% level). No data-mining correction is applied, as the number of pre-registered comparisons is small (five main hypotheses, each with a single primary comparison); the absence of such a correction is acknowledged as a limitation in Chapter 5.
