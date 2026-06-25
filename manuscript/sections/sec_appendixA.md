# Appendix A: Model Equations and Algorithm Pseudocode

This appendix collects the formal specification of the simulator and the learning algorithms. It mirrors exactly the reproducible implementation in the accompanying `code/` directory (`execution_study.py`, `strategies.py`).

## A.1 Market dynamics

For an execution of X shares over K = 30 one-minute steps, the mid-price evolves as

S_{k+1} = S_k + κ·n_k + θ·a_k·(σ/σ_ref) + σ_step·season_k·z_k,

where n_k is the quantity executed in step k, κ the permanent-impact coefficient, θ·a_k the drift from the observable order-flow signal, σ_step = σ_ann/√(252·510)·S_0 the per-step volatility, season_k the U-shaped intraday multiplier, and z_k an i.i.d. standardised innovation (Gaussian in the base case, Student-t with 4 degrees of freedom in robustness).

The observable signal follows an AR(1) process a_{k+1} = ρ·a_k + ε_k, ρ = 0.65.

The execution price paid in step k includes temporary impact:

P_k = S_k + a_temp·(n_k / V̄_k)^δ · S_0,

with V̄_k = V̄·season_k the step volume, δ the impact exponent (1 in the base case, 0.5 in robustness). The Implementation Shortfall, in basis points of arrival notional, is

IS = [ Σ_k n_k·P_k − X·S_0 ] / (X·S_0) · 10^4.

## A.2 Multi-venue routing (water-filling)

Given a step quantity n to split across J venues with depths D_j and idiosyncratic offsets c_j, the router solves

minimise Σ_j ( c_j·n_j + b_j·n_j² ), b_j = a_temp·S_k / D_j, subject to Σ_j n_j = n, n_j ≥ 0.

The KKT stationarity condition gives n_j* = max(0, (ν − c_j) / (2·b_j)), with the multiplier ν chosen so that Σ_j n_j* = n (closed-form water-filling over the active set). The single-venue benchmark pools the total capacity Σ_j D_j and sets all offsets to zero.

## A.3 Cross-Entropy Method (direct policy search)

```
Input: regimes, generations G, population P, elite fraction f, batch B
Initialise mean vector mu, std vector sigma
for g = 1..G:
    draw B paired evaluation scenarios (common random numbers)
    sample P parameter vectors  theta_i ~ Normal(mu, diag(sigma^2))
    for each i:  J_i = mean over the B scenarios of IS( policy(theta_i) )
    elite = the f*P parameter vectors with the lowest J
    mu    = mean(elite);  sigma = std(elite)
return mu
```

## A.4 PPO-family (continuous-action) policy

The policy maps state features φ(s) to a TWAP-rate multiplier and an execution fraction:

m(s) = softplus(θ·φ(s)),  f(s) = min(1, m(s)/(K − k)),

with φ(s) = [1, signal a_k, 100·(S_k/S_0 − 1), normalised time, volatility-regime flag, residual inventory]. Parameters θ are optimised by the Cross-Entropy Method of A.3. This is a policy-optimisation method in the family of Moody & Saffell (2001) and the policy-improvement principle of PPO (Schulman et al., 2017).

## A.5 DQN-family (discrete-action) policy

The agent scores each discrete TWAP-rate multiplier in {0, 0.5, 1.0, 1.5} with a linear function of the same features and acts greedily:

a*(s) = argmax_a ( W_a · φ(s) ),  f(s) = min(1, multiplier(a*) / (K − k)).

The scoring weights W are optimised by the Cross-Entropy Method. The agent is value-based with a discrete action grid, in the family of DQN (Mnih et al., 2015); the discretisation handicap relative to A.4 is the mechanism behind hypothesis H1a.

## A.6 Almgren-Chriss benchmark

Holdings follow the closed-form trajectory x_k = X·sinh(κ̃(K−k)) / sinh(κ̃K), with κ̃ = √(λ·σ_step² / η), η = a_temp·S_0/V̄ and risk-aversion λ. The schedule converges to TWAP as λ → 0 and front-loads as λ increases (validated in Figure 3.1).
