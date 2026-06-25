# Chapter 2: Conceptual Framework and Research Hypotheses

## Chapter Introduction

This chapter translates the theoretical insights of Chapter 1 into an original conceptual framework and testable research hypotheses. Section 2.1 provides the formal MDP formalisation of the multi-venue optimal execution problem, defining the state space, action space, transition dynamics, and reward function with precision and completeness, and derives the two learning approaches used in the empirical study. Section 2.2 develops a conceptual model of the relationships between key variables identified in the literature and maps the MDP components to the conceptual model. Section 2.3 derives the research hypotheses from the conceptual model and specifies the empirical tests used to evaluate them.

## 2.1 MDP Formalisation of the Execution Problem

### 2.1.1 Justification and Scope

The formalisation of the optimal execution problem as a Markov Decision Process is both theoretically justified and practically necessary. An MDP is the appropriate representation whenever decision-making is sequential, the environment is stochastic, and the current state is sufficient for optimal decision-making — the so-called Markov property. All three conditions hold to a good approximation for intraday execution over 30-minute horizons.

**Sequential decisions.** The execution of an institutional order is irreversibly decomposed into a sequence of smaller slices. Each slice executed at time t removes inventory, moves the mid-price through permanent impact, and depletes visible liquidity at the venues used, thereby altering the state of the problem for every subsequent slice. The optimal rate at time t therefore depends not only on current market conditions but also on all future decision points — a property that makes single-shot optimisation inadequate and sequential decision-making essential (Sutton & Barto, 2018).

**Stochastic environment.** Between any two decision points, the mid-price evolves according to a stochastic diffusion driven by noise traders, informed flow, and market-maker repositioning. Order arrival rates at each venue are Poisson processes whose intensities are themselves time-varying. Liquidity migrates across venues in response to HFT market-maker behaviour (Menkveld, 2013). No deterministic model can adequately capture these dynamics; the framework must accommodate probabilistic transitions.

**Markov property.** The state vector s_t is constructed so that the conditional distribution of s_{t+1} given (s_t, a_t) is identical to its distribution given (s_0, a_0, ..., s_t, a_t) — that is, the past trajectory beyond the current state is not informative about future transitions once s_t is known. This holds by construction in the calibrated simulator used in Chapter 4: the price process is Markovian, the residual inventory is a sufficient statistic for urgency, and the rolling volatility estimate captures the relevant short-window history without requiring the full path.

We define the MDP as the tuple (S, A, P, r, γ, T), where:

- **S** is the continuous state space (a bounded subset of R^7, described in Section 2.1.2);
- **A** is the continuous action space of per-venue execution rate vectors;
- **P : S × A × S → [0, 1]** is the stochastic transition kernel encoding market dynamics;
- **r : S × A → R** is the immediate reward function;
- **γ ∈ [0, 1]** is the discount factor (set to 1 in the undiscounted finite-horizon formulation);
- **T** is the finite execution horizon (30 minutes, with decision steps at one-minute intervals, giving T = 30 steps).

The scope of the MDP covers the execution of a single buy-side equity order from initial decision to complete fill, with the agent making allocation decisions at one-minute intervals. We abstract away from real-time sub-second dynamics, focusing on the higher-level tactical execution decisions — how much to execute in total at each step and how to allocate across venues — that are the primary determinants of IS for institutional orders. This level of abstraction is consistent with institutional execution algorithm practice, which typically operates in a "parent order management" mode on minute-level slices rather than reacting to every tick.

*Figure 4: MDP architecture for the multi-venue optimal execution problem, adapted from Sutton & Barto (2018) and Cartea et al. (2018).*

### 2.1.2 State Space, Action Space, Transition Dynamics, and Reward Function

**State vector.** The state at time t is the seven-dimensional vector:

s_t = (q_t, τ_t, S_t/S_0, σ̂_t, OFI_t, L_t^1, ..., L_t^n, ΔS_t^{ij})

where:

- **Normalised residual inventory** q_t = Q_t/Q_0 ∈ [0, 1] measures the fraction of the initial position still to execute, encoding both urgency and the expected terminal penalty from leaving shares unexecuted.
- **Normalised remaining time** τ_t = (T − t)/T ∈ [0, 1], combined with q_t, enables the agent to assess whether execution pace is ahead or behind a neutral schedule.
- **Normalised mid-price** S_t/S_0 encodes the opportunity cost relative to the initial decision price, allowing the agent to condition on adverse or favourable price drift.
- **Short-window realised volatility** σ̂_t (5-minute rolling window) proxies current timing risk and allows the agent to slow down in high-volatility conditions.
- **Order Flow Imbalance** OFI_t = (VA_bid − VA_ask)/(VA_bid + VA_ask) ∈ [−1, 1] provides a short-term directional price signal consistent with the microstructural evidence of Cont et al. (2014).
- **Per-venue available liquidity vector** (L_t^1, ..., L_t^n) captures instantaneous cross-venue execution opportunities, where L_t^k is the available depth within a fixed spread band at venue k.
- **Inter-venue spread differential** ΔS_t^{ij} = S_t^i − S_t^j signals cross-venue price arbitrage opportunities that an intelligent SOR can exploit.

**Action space and constraints.** The action a_t = (v_t^1, ..., v_t^n) is a continuous vector of per-venue execution rates expressed as fractions of the initial order size. Three constraints are imposed: non-negativity (v_t^k ≥ 0 for all k, ruling out short-selling), total rate cap (Σ_k v_t^k ≤ v_max, preventing overly aggressive execution), and a no-overfill constraint (Σ_k v_t^k ≤ q_t, ruling out execution beyond residual inventory). The continuous nature of A motivates the choice of PPO over DQN for the primary agent, as discussed in Section 2.1.3.

**Transition dynamics.** The mid-price at the next step is driven by three forces: permanent impact from execution, a signal-driven drift term, and idiosyncratic diffusion:

S_{t+1} = S_t + η · (Σ_k v_t^k) · S_0 + μ(OFI_t) · Δt + σ̂_t · S_t · √Δt · ε_{t+1}

where η is the permanent impact coefficient (Kyle, 1985; Almgren et al., 2005), μ(OFI_t) is a drift function of the order flow imbalance (capturing short-term price predictability), Δt is the length of one time step, and ε_{t+1} ~ N(0, 1) is the idiosyncratic price shock. Available liquidity at each venue evolves according to a mean-reverting process with stochastic shocks that are correlated across venues, reflecting the empirical finding of liquidity co-migration documented by Gresse (2017) and Menkveld (2013).

**Reward function.** The immediate reward at step t combines execution revenue net of all price costs:

r_t = −[ κ · (Σ_k v_t^k)^2 · Δt + (S_t − S_0) · (Σ_k v_t^k) · Q_0 + λ · q_t^2 · σ̂_t^2 ]

The three terms represent, respectively: the quadratic temporary impact cost (with coefficient κ measuring the venue-average instantaneous impact), the price drift cost relative to the arrival price S_0 (capturing the opportunity cost of adverse price movement during execution), and an adaptive inventory risk penalty (with coefficient λ weighting the urgency-adjusted variance of holding residual inventory). A terminal penalty r_T = −Φ · q_T^2 is applied at horizon T, penalising any residual inventory q_T > 0 and ensuring the agent has a strong incentive to complete the order.

**Objective and Bellman equation.** The agent seeks a policy π : S → A that maximises the expected cumulative reward:

J(π) = E_π [ Σ_{t=0}^{T} r_t ]

The optimal value function V*(s) = max_π J(π | s_0 = s) satisfies the Bellman optimality equation:

V*(s_t) = max_{a_t ∈ A} { r(s_t, a_t) + E[ V*(s_{t+1}) | s_t, a_t ] }

Because S is continuous and high-dimensional, the exact Bellman equation cannot be solved by dynamic programming on a discretised grid. This motivates the use of function approximation, implemented via deep neural networks in the DQN and PPO algorithms described in Section 2.1.3.

The Implementation Shortfall incurred by a given execution trajectory (n_1, ..., n_T) — where n_t = Σ_k v_t^k · Q_0 shares are executed at step t at an average fill price P_t — is:

IS = Σ_t n_t (P_t − S_0) + (Q_0 − Σ_t n_t)(S_T − S_0)

Minimising IS is equivalent to maximising J(π) under the correspondence between the price cost terms and the reward function above, providing a direct link between the MDP objective and the empirical performance metric used in Chapter 4.

### 2.1.3 Value-Based and Policy-Gradient Learning Approaches

Two families of deep RL algorithms are used to approximate the solution of the MDP, motivated by different treatments of the action space.

**Value-based learning: DQN.** Deep Q-Network learning (Sutton & Barto, 2018) estimates the action-value function Q(s, a) — the expected cumulative reward from taking action a in state s and following the optimal policy thereafter. The Q-learning target for a transition (s_t, a_t, r_t, s_{t+1}) is:

y_t = r_t + γ · max_{a'} Q_θ̄(s_{t+1}, a')

where θ̄ denotes the parameters of a periodically frozen target network. The network Q_θ is trained by minimising the mean squared Bellman error:

L(θ) = E[ (y_t − Q_θ(s_t, a_t))^2 ]

DQN requires a discrete action space, so the continuous per-venue rate vector is quantised into a finite grid of execution levels. This discretisation is implementationally simple but introduces approximation error, particularly for fine-grained allocation decisions across multiple venues. This limitation motivates hypothesis H1a.

**Policy-gradient learning: PPO.** Proximal Policy Optimisation operates directly on a parameterised stochastic policy π_θ(a | s), outputting a continuous Gaussian distribution over the action vector. PPO maximises a clipped surrogate objective that prevents destabilising large policy updates:

L^{CLIP}(θ) = E[ min( ρ_t(θ) · Â_t, clip(ρ_t(θ), 1 − ε, 1 + ε) · Â_t ) ]

where ρ_t(θ) = π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t) is the probability ratio, Â_t is the generalised advantage estimate, and ε is the clipping threshold (typically 0.2). By operating in the native continuous action space without discretisation, PPO can represent fine-grained multi-venue allocations that DQN approximates only coarsely. This difference is the theoretical foundation for H1a: PPO's continuous policy representation should translate into lower IS through more precise venue allocation, particularly when liquidity is unevenly distributed across venues.

## 2.2 Conceptual Model and Variable Relationships

The conceptual model maps the MDP components onto the causal structure governing IS outcomes, distinguishing controllable, exogenous, and characteristic variables and identifying the two mediating mechanisms through which they operate.

**Dependent variable.** Implementation Shortfall (IS) is the dependent variable, defined as the value-weighted average fill price minus the arrival mid-price, expressed in basis points. IS captures both the explicit cost of market impact and the opportunity cost of price drift during execution, making it the most comprehensive and theoretically grounded transaction cost measure available (Almgren et al., 2005).

**Controllable strategy variables.** These are under the decision-maker's direct control and constitute the primary research levers: (i) algorithm choice (TWAP, VWAP, DQN, PPO, Almgren-Chriss); (ii) algorithm parameters (risk aversion λ, urgency schedule, clipping threshold ε); and (iii) Smart Order Router design (number of venues, routing logic, update frequency). In the MDP language, controllable variables determine the policy π and the structure of the action space A.

**Exogenous market-condition variables.** These are observable but not controllable and constitute the conditioning environment within which the policy operates: (i) intraday realised volatility σ, driving the diffusion term in the price transition and the inventory risk component of the reward; (ii) aggregate market liquidity L, determining the capacity for large execution without adverse price impact; (iii) Order Flow Imbalance OFI, providing the short-term directional signal incorporated into the state vector; and (iv) market fragmentation HHI, measuring the concentration of trading volume across venues and thus the potential alpha available from dynamic SOR. Higher fragmentation (lower HHI) implies greater cross-venue price and liquidity dispersion, amplifying the value of an intelligent routing policy.

**Order characteristics.** Order size (as a fraction of average daily volume, ADV%), urgency (time horizon T), and direction (buy vs. sell) are determined upstream by the portfolio manager and define the parameters of the execution problem. Large, urgent buy orders in rising markets face the most adverse execution conditions; the RL agent's adaptive policy is expected to add most value precisely in these cases.

**Mediating mechanisms.** Two mechanisms connect the upstream variables to IS outcomes.

The *market impact mechanism* (Kyle, 1985; Almgren et al., 2005) determines how execution rate and venue choice translate into price impact through the permanent impact coefficient η and the temporary impact coefficient κ. Permanent impact is irreversible: each share executed at step t shifts S permanently by η per unit, imposing a cost on all subsequent slices. Temporary impact is reversible: the spread-crossing and top-of-book depletion induced by aggressive execution at step t dissipate by t+1 as market makers repost. An RL agent that learns these dynamics can time execution to minimise the cascade of permanent impact, trading off against the inventory risk of waiting.

The *multi-venue liquidity dynamics mechanism* (Menkveld, 2013; Gresse, 2017) determines how visible and latent liquidity migrates across venues in response to the agent's activity and HFT market-maker behaviour. When the agent exhausts top-of-book liquidity at the primary venue, market makers on alternative venues may widen spreads in anticipation of further order flow — a "footprint" effect that raises the effective cost of continued one-venue execution. A dynamic SOR that detects this migration and routes subsequent slices to venues with restored liquidity can materially reduce average IS. The RL agent learns this mechanism implicitly through the liquidity state variables (L_t^1, ..., L_t^n) and the inter-venue spread differentials (ΔS_t^{ij}).

## 2.3 Research Hypotheses

Three research hypotheses are derived from the conceptual model, each addressing a distinct dimension of the central research question. A brief theoretical justification precedes each hypothesis to make explicit the causal link to the model.

**Theoretical justification for H1.** The MDP formalisation establishes that the optimal execution policy is state-dependent and non-linear in σ̂_t, OFI_t, and q_t — properties that TWAP and VWAP, by construction, cannot exploit (TWAP ignores all state variables; VWAP conditions only on volume profile). A deep RL agent that approximates V*(s) through neural function approximation should therefore achieve lower IS than either benchmark in expectation across a diverse out-of-sample evaluation set.

**H1:** A deep RL agent (DQN or PPO) trained on a calibrated stochastic simulator of European market microstructure achieves statistically significant IS reduction relative to TWAP and VWAP in an out-of-sample evaluation.

> **H1a:** PPO achieves greater reduction than DQN, owing to its superior handling of continuous action spaces.
>
> **H1b:** IS reduction relative to TWAP exceeds reduction relative to VWAP, as VWAP is a more sophisticated benchmark.

**Theoretical justification for H2.** When the action space is extended from a single venue to n venues, the feasible policy set expands: the agent can exploit cross-venue price differentials (ΔS_t^{ij} > 0), shift flow to venues with deeper available liquidity, and reduce the footprint-driven spread widening that occurs when execution is concentrated at one venue. A single-venue agent, by definition, cannot access these additional degrees of freedom. The incremental IS reduction attributable to dynamic SOR is therefore strictly non-negative in expectation, and strictly positive whenever fragmentation creates exploitable dispersion — a condition that is more likely the lower the HHI.

**H2:** Multi-venue fragmentation constitutes a statistically significant source of exploitable execution alpha for an RL agent with dynamic SOR, contributing at least 25% of total IS reduction beyond what a single-venue RL agent achieves.

> **H2a:** The multi-venue RL agent significantly outperforms a version constrained to Euronext Paris only.
>
> **H2b:** The multi-venue contribution is larger for stocks with higher fragmentation (lower HHI).

**Theoretical justification for H3.** The value of adaptive policy-making is greatest when the environment is non-stationary and the cost of sub-optimal decisions is high. In high-volatility regimes, the inventory risk term λ · q_t^2 · σ̂_t^2 in the reward function rises sharply, making the timing and pace of execution highly consequential. Static benchmarks (TWAP, VWAP, Almgren-Chriss with fixed parameters) cannot adjust to these within-regime fluctuations, while an RL policy that conditions on σ̂_t can modulate urgency dynamically. In low-volatility regimes, the inventory penalty is small, execution pace matters less, and the adaptive advantage of RL shrinks toward zero.

**H3:** RL agent IS gains are conditional on market regime, being statistically more pronounced in high-volatility periods (σ > 25% annualised) than in low-volatility periods (σ < 15%), consistent with the adaptive value of RL under non-stationary conditions.

> **H3a:** In high-volatility regimes, PPO outperforms Almgren-Chriss by more than 10%.
>
> **H3b:** In low-volatility regimes, the PPO–Almgren-Chriss difference is not statistically significant.
