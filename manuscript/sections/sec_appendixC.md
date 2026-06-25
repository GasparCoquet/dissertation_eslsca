# Appendix C: Full Numerical Results

All values are simulated Implementation Shortfall (basis points) on the held-out out-of-sample set of N = 12,000 paired sessions. Reproducible from `code/experiments.py` (`code/results/results.json`).

## C.1 Out-of-sample IS distribution (H1)

| Strategy | Mean | Median | Std | VaR95 |
|---|---|---|---|---|
| TWAP | 17.64 | 17.82 | 49.76 | 97.3 |
| VWAP | 17.10 | 17.42 | 48.42 | 95.1 |
| Almgren-Chriss | 17.93 | 18.04 | 42.76 | 86.9 |
| DQN | 12.52 | 16.86 | 54.34 | 91.7 |
| PPO | 11.68 | 16.87 | 48.78 | 80.8 |

## C.2 Pairwise reductions, significance and confidence intervals

| Comparison | Reduction | Δ (bps) | Newey-West t | 95% CI (bps) |
|---|---|---|---|---|
| PPO vs TWAP | 33.8% | -5.96 | -67.4 | [-6.13, -5.79] |
| PPO vs VWAP | 31.7% | -5.42 | -60.9 | [-5.59, -5.24] |
| PPO vs Almgren-Chriss | 34.9% | -6.25 | -60.6 | [-6.45, -6.05] |
| DQN vs TWAP | 29.0% | -5.12 | -56.8 | [-5.30, -4.94] |
| DQN vs VWAP | 26.8% | -4.58 | -50.0 | [-4.76, -4.40] |
| DQN vs Almgren-Chriss | 30.2% | -5.41 | -38.5 | [-5.68, -5.14] |
| PPO vs DQN | 6.7% | -0.84 | -12.2 | [-0.97, -0.71] |

## C.3 Mean IS by volatility regime (H3)

| Strategy | σ=15% | σ=22% | σ=35% |
|---|---|---|---|
| TWAP | 17.98 | 18.69 | 16.21 |
| VWAP | 17.43 | 18.15 | 15.67 |
| Almgren-Chriss | 18.21 | 18.74 | 16.81 |
| DQN | 16.01 | 15.00 | 6.42 |
| PPO | 15.92 | 13.55 | 5.44 |

PPO versus Almgren-Chriss by regime:

| Regime | PPO | AC | Reduction | t |
|---|---|---|---|---|
| σ=15% | 15.92 | 18.21 | 12.6% | -25.5 |
| σ=22% | 13.55 | 18.74 | 27.7% | -41.0 |
| σ=35% | 5.44 | 16.81 | 67.6% | -47.5 |

## C.4 Fragmentation alpha vs cross-venue dispersion (H2)

| Dispersion (bps) | Multi-venue IS | Single-venue IS | Alpha (bps) | Alpha (%) | t |
|---|---|---|---|---|---|
| 0 | 11.41 | 11.41 | 0.00 | 0.0% | 57 |
| 4 | 10.20 | 11.41 | 1.21 | 10.6% | 303 |
| 8 | 7.47 | 11.41 | 3.95 | 34.6% | 352 |
| 16 | 0.72 | 11.41 | 10.69 | 93.7% | 408 |
| 32 | -13.86 | 11.41 | 25.27 | 221.4% | 456 |

## C.5 Robustness across model assumptions

| Variant | PPO IS | TWAP IS | PPO vs TWAP | PPO vs AC |
|---|---|---|---|---|
| Baseline (linear, Gaussian) | 13.25 | 19.20 | 31.0% | 30.9% |
| Square-root impact | 31.00 | 36.59 | 15.3% | 15.4% |
| Student-t shocks | 12.32 | 18.38 | 33.0% | 33.5% |
| Higher impact | 24.46 | 28.29 | 13.5% | 13.2% |
| Out-of-distribution shift | 14.75 | 19.09 | 22.7% | 22.3% |
