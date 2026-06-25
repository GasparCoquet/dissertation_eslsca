# Appendix B: Simulator and Agent Hyperparameters

All values below are the actual settings used by the reproducible pipeline (`code/execution_study.py`, `code/strategies.py`, `code/experiments.py`). The study is deterministic given the fixed seeds.

## B.1 Market simulator

| Parameter | Value |
|---|---|
| Execution window | 30 minutes, 30 steps of 1 minute |
| Arrival mid-price | 100.0 (units arbitrary; IS reported in bps) |
| Order size | 10% average participation of per-step volume |
| Volatility regimes (annualised σ) | 15%, 22%, 35% |
| Almgren-Chriss reference σ | 22% |
| Temporary impact coefficient (a_temp) | 0.008 (δ = 1 base; δ = 0.5 robustness) |
| Permanent impact coefficient (a_perm) | 0.002 |
| Intraday seasonality | U-shaped (volume and volatility) |
| Order-flow signal | AR(1), persistence ρ = 0.65 |
| Signal drift coefficient (θ) | 0.0005, scaled by σ / σ_ref |
| Innovations | Gaussian (base); Student-t, 4 d.o.f. (robustness) |
| Venues | 3 (multi-venue); 1 pooled (single-venue benchmark) |
| Depth heterogeneity / dispersion | depth std 0.15; offset dispersion swept 0–32 bps |

## B.2 PPO-family agent (continuous action, policy-gradient)

| Parameter | Value |
|---|---|
| Policy | m = softplus(θ · features); execution fraction f = min(1, m / (N − k)) |
| State features | bias, order-flow signal, price-vs-arrival 100·(S/S₀ − 1), normalised time, volatility-regime flag, residual inventory |
| Optimiser | Cross-Entropy Method (direct policy search) |
| Generations / population / elite | 22 / 28 / top 25% |
| Evaluation batch per generation | 600 paired scenarios (common random numbers) |
| Initial exploration σ | 0.6 |

## B.3 DQN-family agent (discrete action, value-based)

| Parameter | Value |
|---|---|
| Action grid (TWAP-rate multipliers) | {0.0, 0.5, 1.0, 1.5} |
| Policy | greedy argmax of linear score W · features over the discrete grid |
| Optimiser | Cross-Entropy Method on the scoring weights |
| Generations / population / elite | 20 / 36 / top 20% |
| Evaluation batch per generation | 420 paired scenarios |
| Initial exploration σ | 0.7 |

## B.4 Evaluation protocol

| Parameter | Value |
|---|---|
| Out-of-sample test set | 12,000 sessions, seed disjoint from training |
| Comparison | paired (common random numbers across all strategies) |
| Significance test | Newey-West (1987) HAC t-statistic, lag 5 |
| Confidence intervals | paired bootstrap, 10,000 resamples |
| Almgren-Chriss validation | reproduces efficient frontier (TWAP as λ → 0; front-loading as λ rises) |
