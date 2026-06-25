# Reproducible empirical pipeline (honest, simulation-based)

This directory produces **all** the empirical results in the dissertation's revised Chapter 4.
No proprietary or real tick data is used — the market is a calibrated stochastic simulator, and
results are **simulated** Implementation Shortfall (only signs/rankings/comparative statics are
interpreted). See `../REWRITE_Ch3-4_HONEST.md` for the write-up and `../Dissertation_Coquet_Ch3-4_EMPIRICAL.pdf`
for the rendered supplement.

## Files
| File | Purpose |
|---|---|
| `execution_study.py` | Market simulator: price + U-shape seasonality + Almgren impact + AR(1) order-flow signal + multi-venue routing. |
| `strategies.py` | Baselines (TWAP, VWAP, Almgren-Chriss) + RL agents (PPO-family policy-gradient, DQN-family discrete value), trained by Cross-Entropy policy search. |
| `experiments.py` | Runs H1/H2/H3 + robustness, writes `results/` and `figures/`. |
| `calibration.py` | Optional: re-anchor volatility regimes to free Yahoo data (needs internet + `yfinance`). |
| `make_report.py` | Renders the empirical supplement PDF from `results/`. |

## Reproduce
```bash
pip install numpy scipy pandas matplotlib fpdf2
python experiments.py      # trains agents, writes results/ + figures/  (a few minutes)
python make_report.py      # builds ../Dissertation_Coquet_Ch3-4_EMPIRICAL.pdf
```
Results are deterministic given the fixed seeds in `experiments.py`.

## Headline (out-of-sample, 12,000 sessions; simulated IS)
- PPO-family vs TWAP **−33.8%**, vs VWAP **−31.7%**, vs Almgren-Chriss **−34.9%** (all p<0.001).
- DQN-family vs TWAP **−29.0%**; PPO beats DQN by **6.7%** (H1a).
- Fragmentation alpha = **0** at zero cross-venue dispersion, rising with dispersion (H2, falsifiable).
- PPO-vs-AC gain rises **12.6% → 67.6%** from calm to stressed regimes (H3).
- Sign-stable under sqrt impact, Student-t, higher impact, and out-of-distribution shift.

## Honesty / scope
This is a **mechanism study**, not a backtest. Magnitudes are model artefacts; the agents are
linear policies (not deep nets); train/test share the data-generating process. Full caveats in the
Limitations section of the supplement.
