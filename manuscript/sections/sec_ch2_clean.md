# Chapter 2: Conceptual Framework and Research Hypotheses

## Chapter Introduction

This chapter translates the theoretical insights of Chapter 1 into an original conceptual framework and testable research hypotheses. Section 2.1 provides the formal MDP formalisation of the multi-venue optimal execution problem, defining state space, action space, transition dynamics, and reward function with precision and completeness. Section 2.2 develops a conceptual model of the relationships between key variables identified in the literature and maps the MDP components to the conceptual model. Section 2.3 derives the research hypotheses from the conceptual model and specifies the empirical tests used to evaluate them.

## 2.1 MDP Formalisation of the Execution Problem

### 2.1.1 Justification and Scope

The formalisation of the optimal execution problem as a Markov Decision Process is both theoretically justified and practically necessary. Theoretically, the MDP framework is appropriate whenever three conditions hold: decisions are sequential (each execution slice affects the state of the market and hence subsequent decision conditions), the environment is stochastic (price movements, order arrivals, and liquidity migrations are random), and the current state contains sufficient information for optimal decision-making (Markov property). All three conditions hold to a good approximation for intraday execution over 30-minute horizons. Practically, the MDP formalisation is necessary because the multi-venue execution problem, with its high-dimensional continuous state and action spaces, cannot be solved analytically — a numerical or learning-based approach is required.

The scope of our MDP covers the execution of a single buy-side equity order from initial decision to complete fill, with the agent making allocation decisions at one-minute intervals. We abstract away from the real-time sub-second dynamics of individual order routing, focusing instead on the higher-level tactical execution decisions (how much to execute in total this minute, and how to allocate across venues) that are the primary determinants of IS for institutional orders. This level of abstraction is consistent with the practice of institutional execution algorithms, which typically operate in a "parent order management" mode on minute-level slices rather than reacting to every tick.

*Figure 4: MDP architecture for the multi-venue optimal execution problem, adapted from Sutton & Barto (2018) and Cartea et al. (2018).*

### 2.1.2 State Space, Action Space and Reward Function

The state vector **s** comprises seven components capturing all information relevant to the execution decision:

- **Normalised residual inventory** q = Q/Q₀ ∈ [0,1] measures the fraction of the initial position still to execute, encoding urgency and expected terminal penalty.
- **Normalised remaining time** τ = (T−t)/T ∈ [0,1], combined with q, enables the agent to assess execution pace.
- **Normalised mid price** S/S₀ encodes the opportunity cost evolution since the initial decision.
- **Short-window realised volatility** σ̂ (5-minute rolling window) proxies current timing risk.
- **Order Flow Imbalance** OFI = (VA_bid − VA_ask)/(VA_bid + VA_ask) provides a short-term directional price signal (Cont et al., 2014).
- **Per-venue available liquidity vector** (L¹, ..., Lⁿ) captures instantaneous cross-venue execution opportunities.
- **Inter-venue spread vector** (ΔS^{ij}) signals cross-venue price arbitrage opportunities.

The action **a** = (v¹, ..., vⁿ) is a continuous vector of per-venue execution rates, subject to: non-negativity, total rate cap v_max, and a no-overfill constraint. The instantaneous reward function captures direct price costs, venue-specific quadratic temporary impact, permanent impact, and adaptive inventory risk penalisation respectively. A terminal penalty completes the formulation, penalising residual inventory at horizon T.

## 2.2 Conceptual Model and Variable Relationships

The conceptual model positions Implementation Shortfall as the dependent variable, driven by three categories of factors. Execution strategy variables — algorithm choice, parameters, and SOR design — are under the decision-maker's control and constitute the primary research levers. Market condition variables — volatility σ, aggregate liquidity L, OFI, fragmentation HHI — are exogenous but observed and incorporated into the agent's state. Order characteristics — size, urgency, direction — are determined upstream and define the execution problem parameters.

Two mediating mechanisms connect these variables to IS outcomes. The market impact mechanism (Almgren et al., 2005; Kyle, 1985) determines how execution rate and venue choice translate into price impact through venue-specific η and κ parameters. The multi-venue liquidity dynamics mechanism (Menkveld, 2013; Gresse, 2017) determines how liquidity migrates across venues in response to the agent's activity and HFT market-maker behaviour. The RL agent implicitly learns both mechanisms through interaction with the calibrated simulator, achieving an adaptive policy that classical models cannot provide.

## 2.3 Research Hypotheses

Three research hypotheses are derived from the conceptual model, each addressing a distinct dimension of the central research question:

**H1:** A deep RL agent (DQN or PPO) trained on a calibrated stochastic simulator of European market microstructure achieves statistically significant IS reduction relative to TWAP and VWAP in an out-of-sample evaluation.

> **H1a:** PPO achieves greater reduction than DQN, owing to its superior handling of continuous action spaces.
>
> **H1b:** IS reduction relative to TWAP exceeds reduction relative to VWAP, as VWAP is a more sophisticated benchmark.

**H2:** Multi-venue fragmentation constitutes a statistically significant source of exploitable execution alpha for an RL agent with dynamic SOR, contributing at least 25% of total IS reduction beyond what a single-venue RL agent achieves.

> **H2a:** The multi-venue RL agent significantly outperforms a version constrained to Euronext Paris only.
>
> **H2b:** The multi-venue contribution is larger for stocks with higher fragmentation (lower HHI).

**H3:** RL agent IS gains are conditional on market regime, being statistically more pronounced in high-volatility periods (σ > 25% annualised) than in low-volatility periods (σ < 15%), consistent with the adaptive value of RL under non-stationary conditions.

> **H3a:** In high-volatility regimes, PPO outperforms Almgren-Chriss by more than 10%.
>
> **H3b:** In low-volatility regimes, the PPO–Almgren-Chriss difference is not statistically significant.
