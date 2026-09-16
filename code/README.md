# Reproducible empirical pipeline (simulation-based)

This directory produces **all** the empirical results in the dissertation's Chapter 4.
No proprietary or real tick data is used: the market is a calibrated stochastic simulator, and
results are **simulated** Implementation Shortfall (only signs, rankings and comparative statics are
interpreted). See Chapter 3 of the dissertation for the methodology and Section 4.10 for the full caveats.

## Files
| File | Purpose |
|---|---|
| `execution_study.py` | Market simulator: price + U-shape seasonality + Almgren impact + AR(1) order-flow signal + multi-venue routing. |
| `strategies.py` | Baselines (TWAP, VWAP, Almgren-Chriss) + RL agents (PPO-family policy-gradient, DQN-family discrete value). Both are **linear policies trained by Cross-Entropy search**, not neural networks; the deep architectures of section 1.4.2 are a stated extension, not implemented here. |
| `experiments.py` | Runs H1/H2/H3 + robustness, writes `results/` and `figures/`. |
| `calibration.py` | Optional: re-anchor volatility regimes to free Yahoo data (needs internet + `yfinance`). |
| `make_report.py` | Renders the empirical supplement PDF from `results/`. |

## Reproduce
```bash
pip install numpy scipy pandas matplotlib fpdf2
python experiments.py      # trains agents, writes results/ + figures/  (a few minutes)
python make_report.py      # builds the empirical supplement PDF
```
Results are deterministic given the fixed seeds in `experiments.py`.

## Headline (out-of-sample, 12,000 sessions; simulated IS)

The signal, not the learning, does most of the work. Table 4.1 of the dissertation,
mean IS in bps:

| Strategy | mean | median | VaR95 | vs TWAP |
|---|---|---|---|---|
| Almgren-Chriss | 17.93 | 18.04 | 86.9 | −1.7% |
| **Signal-TWAP (heuristic, non-learning)** | **11.83** | **16.71** | **85.9** | **+32.9%** |
| DQN-family | 12.52 | 16.86 | 91.7 | +29.0% |
| PPO-family | 11.68 | 16.87 | 80.8 | +33.8% |

**Read the Signal-TWAP row against the PPO row.** A trivial signal-proportional rule
(scale the TWAP rate by `max(0, 1 + 0.5 * signal)`, coefficient fixed in advance on a
separate seed) already captures a 32.9% reduction versus TWAP. The learned policy
adds about **1.3%** on top of it. Almost the entire headline gain is the value of
conditioning on an order-flow signal, which any signal-aware rule captures; the value
of the RL policy specifically is that 1.3% residual and, more cleanly, the **6.7%**
PPO-over-DQN margin (H1a). `signal_benchmark.py` reproduces the row.

- Fragmentation alpha = **0** at zero cross-venue dispersion, rising with dispersion (H2, falsifiable).
- PPO-vs-AC gain rises **12.6% → 67.6%** from calm to stressed regimes (H3).
- Sign-stable under sqrt impact, Student-t, higher impact, and out-of-distribution shift.

## Honesty / scope
This is a **mechanism study**, not a backtest. Magnitudes are model artefacts; the agents are
linear policies (not deep nets); train/test share the data-generating process. Full caveats in the
Limitations section of the supplement.
