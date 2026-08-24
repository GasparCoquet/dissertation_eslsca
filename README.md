# Reinforcement Learning and Implementation Shortfall

MBA dissertation, ESLSCA 2025-2026. **Can a learned execution policy beat TWAP, VWAP
and Almgren-Chriss on implementation shortfall - and if it can, is the learning what
did it?**

- **[`Dissertation_Coquet_FINAL.pdf`](Dissertation_Coquet_FINAL.pdf)** - the full text, 87 pages.
- **[`code/`](code/)** - the pipeline that produces every empirical result in Chapter 4, with a README of its own.

## The finding, including the part that cuts against the thesis

On a 12,000-session out-of-sample protocol the PPO-family agent reduces mean IS by
**33.8%** versus TWAP. That number on its own is misleading, so the dissertation
prices it against a benchmark built specifically to undercut it:

| Strategy | mean IS (bps) | vs TWAP |
|---|---|---|
| Almgren-Chriss | 17.93 | −1.7% |
| **Signal-TWAP** (trivial, non-learning) | **11.83** | **+32.9%** |
| DQN-family | 12.52 | +29.0% |
| PPO-family | 11.68 | +33.8% |

A one-line signal-proportional rule, with its single coefficient fixed in advance on a
separate seed, captures 32.9 of those 33.8 points. **The reinforcement learning adds
about 1.3%.** Almost all of the headline is the value of conditioning on an order-flow
signal, not the value of the policy that conditions on it. The cleaner policy-specific
margin is the 6.7% PPO-over-DQN gap.

## Scope, stated plainly

This is a mechanism study, not a backtest.

- The market is a **calibrated stochastic simulator** - price process, U-shape volume
  seasonality, Almgren impact, AR(1) order-flow signal, multi-venue routing. No
  proprietary data, no real tick data, no Level 2. Institutional-grade microstructure
  data was cost-prohibitive, which is the honest reason no real-data backtest exists here.
- The agents are **linear policies trained by Cross-Entropy search**, not neural
  networks. The deep PPO/DQN architectures discussed in section 1.4.2 are a stated
  scalable extension, not something implemented and claimed.
- Train and test share the data-generating process, so magnitudes are model artefacts.
  Only signs, rankings and comparative statics are interpreted.

Results are deterministic given the fixed seeds in `code/experiments.py`.
