# Chapter 3: Data Collection and Research Methodology

## Chapter Introduction

This chapter specifies the empirical method used to test the hypotheses of Chapter 2. Section 3.1 states the epistemological positioning. Section 3.2 describes the calibrated stochastic market simulator that serves as the data-generating environment, and explains why a simulation-based method — rather than a historical replay on proprietary tick data — is the appropriate and, given data-access constraints, the only feasible design. Section 3.3 specifies the benchmark strategies and the two reinforcement-learning agents. Section 3.4 sets out the evaluation protocol and the statistical tests.

## 3.1 Epistemological Positioning

The study adopts a positivist paradigm and a hypothetico-deductive approach: testable hypotheses are derived from theory (Chapter 2) and confronted with empirical evidence under conditions designed to allow falsification. Because the object of study is the *comparative* performance of execution policies under controlled, repeatable market conditions, the empirical method is simulation-based backtesting in a calibrated stochastic environment rather than a historical replay on proprietary tick data. This choice is standard in the reinforcement-learning execution literature (Nevmyvaka, Feng & Kearns, 2006; Hendricks & Wilcox, 2014; Cartea, Donnelly & Jaimungal, 2018) and carries a decisive methodological advantage: it captures the agent's own market impact and the market's reaction to its actions, which a static historical replay — where the tape is fixed regardless of what the agent does — cannot represent.

## 3.2 A Calibrated Market Simulator

### 3.2.1 Rationale and data constraint

No proprietary or real tick-by-tick data is used in this study. Synchronised multi-venue Level-2 data for CAC 40 equities (Euronext Paris together with the relevant MTFs) is available only through paid institutional feeds, the cost of which was prohibitive for this dissertation; freely available sources provide either single-venue daily/intraday bars without quote-level depth, or delayed trade prints, neither of which supports a credible multi-venue Implementation-Shortfall backtest. Rather than overstate the evidence, the study therefore adopts a transparent, reproducible simulator whose parameters are taken from the published literature and anchored to publicly documented CAC 40 characteristics. The full pipeline is openly reproducible from the accompanying `code/` directory.

### 3.2.2 Price, volatility and intraday seasonality

The mid-price follows an arithmetic diffusion over a 30-minute execution window discretised into thirty one-minute steps. Each session's volatility is drawn from one of three regimes — annualised σ ∈ {15%, 22%, 35%} — spanning the calm-to-stressed range documented for CAC 40 names (Banque de France, 2022; ESMA, 2023), with the central level set to the long-run CAC 40 realised volatility. Both volatility and traded volume follow a U-shaped intraday seasonality (higher at the open and the close). Innovations are Gaussian in the base case and Student-t (4 degrees of freedom) in a robustness variant.

### 3.2.3 Market-impact calibration

Market impact follows the Almgren et al. (2005) decomposition: a temporary component that is a power law of the participation rate (exponent δ = 1 in the base case; δ = 0.5, the square-root law of Bouchaud et al. (2010), in robustness) and a permanent component linear in executed quantity. The coefficients are set so that the average TWAP Implementation Shortfall is approximately 18 basis points, within the 15–40 bps range reported for institutional European orders (Banque de France, 2022). Because no free data source provides impact estimates, these coefficients are literature-based; they are reported transparently and varied in the sensitivity analysis of Section 4.4.

### 3.2.4 Observable order-flow signal

A persistent, observable order-flow signal (an AR(1) proxy for the Order Flow Imbalance of the Chapter-2 state vector) induces a partially predictable short-horizon price drift. Its magnitude is calibrated so that a signal-aware oracle reduces IS by approximately 30% relative to a signal-blind schedule, within the 20–40% range documented by Cartea, Donnelly & Jaimungal (2018). This signal is the honest source of any reinforcement-learning advantage: the TWAP, VWAP and Almgren-Chriss benchmarks are blind to it by construction, whereas the RL agents observe it. The signal's price effect scales with session volatility, so its exploitable value is larger in volatile regimes — the mechanism underlying H3.

### 3.2.5 Multi-venue layer

Execution may be routed across three venues with heterogeneous, time-varying depth and idiosyncratic, mean-zero price offsets (the inter-venue spread of the Chapter-2 state vector), modelled as AR(1) processes. The dispersion of these offsets is an explicit control knob: setting it to zero removes all cross-venue price differences, which allows the multi-venue "fragmentation alpha" to be switched off and thus provides a falsifiable test of H2. The single-venue ("Euronext-only") benchmark is constructed with the *same total capacity* as the pooled three venues, so the comparison isolates the value of routing across dispersed venues rather than merely adding liquidity.

## 3.3 Strategies and Agents

### 3.3.1 Benchmarks

Three non-learning benchmarks are implemented: **TWAP** (a uniform schedule, blind to time-of-day, volume and signal); **VWAP** (which follows the U-shaped volume profile); and **Almgren-Chriss**, the closed-form risk-averse schedule, calibrated to the *average* volatility and therefore mis-specified when a session's realised volatility departs from that average (Forsyth et al., 2012). The Almgren-Chriss implementation is validated by confirming that it reproduces the efficient frontier — converging to TWAP as the risk-aversion parameter tends to zero and front-loading monotonically as it increases.

### 3.3.2 Reinforcement-learning agents

Two agents are implemented as controlled, reproducible instances of the DQN and PPO families formalised in Chapter 2. Both are policies over the Chapter-2 state features — residual inventory, remaining time, price relative to arrival (S/S₀), the order-flow signal, and the volatility regime — and both are optimised by direct policy search (the Cross-Entropy Method) on the simulated Implementation-Shortfall objective, a policy-optimisation method in the family of Moody & Saffell (2001) direct reinforcement and of the policy-improvement principle of PPO. The **DQN-family agent** is value-based with a *discrete* action grid; the **PPO-family agent** is policy-gradient with a *continuous* action. The discrete agent is handicapped relative to the continuous one purely by action discretisation — the mechanism behind H1a. Deep neural-network function approximation (the full DQN/PPO architectures of Section 1.4.2) is the scalable extension discussed in the Limitations; the linear-policy instances used here isolate the economic mechanisms in a transparent, reproducible setting. Training hyper-parameters are reported in Appendix B.

## 3.4 Evaluation Protocol and Statistical Testing

Each strategy executes a fixed buy order over the 30-minute window. Training and evaluation use **disjoint random seeds**; all reported figures are computed on a held-out, out-of-sample set of **N = 12,000 sessions**. Strategies are compared **paired** (common random numbers): every strategy is run on the *same* scenarios, so differences reflect policy quality, not sampling noise. The statistical significance of paired IS differences is assessed with Newey-West (1987) HAC t-statistics (lag 5) and with 10,000-resample paired-bootstrap 95% confidence intervals. For H3, the test set is partitioned by volatility regime *a priori*.
