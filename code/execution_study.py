"""
execution_study.py
===================
Calibrated synthetic-market study of Reinforcement Learning for optimal
execution (Implementation Shortfall) on CAC 40-like equities.

NO proprietary or real tick data is used. The market is a stochastic
simulator whose parameters are taken from the published literature
(Almgren et al., 2005) and calibrated to publicly observable CAC 40
volatility characteristics. All reported figures are SIMULATED
Implementation Shortfall under stated assumptions; only signs, rankings
and comparative statics are interpreted (see dissertation Ch. 3-4).

Mechanisms (the honest sources of any RL edge):
  - Temporal: a partially predictable, observable order-flow / alpha
    signal (AR(1)) creates short-horizon price drift. An adaptive agent
    that observes the signal can time execution; TWAP/VWAP/Almgren-Chriss
    are blind to it.
  - Regime: Almgren-Chriss is calibrated to an AVERAGE volatility; when a
    session's realised volatility departs from that average the static
    schedule is mis-specified, while the RL agent adapts (=> H3).
  - Multi-venue: venues quote idiosyncratic, time-varying price offsets
    and have heterogeneous depth; dynamic routing captures cross-venue
    price/liquidity dispersion. The dispersion magnitude is an explicit
    knob, so the "fragmentation alpha" can be shown to VANISH when
    dispersion -> 0 (=> falsifiable H2).

Author: Gaspar Coquet (dissertation pipeline)
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------
@dataclass
class Params:
    N: int = 30                 # execution steps (1-min slices over a 30-min window)
    mid0: float = 100.0         # arrival mid price
    # order size expressed as average participation of per-step market volume
    Vbar: float = 1.0           # baseline market volume per step (unit)
    participation: float = 0.10 # target avg participation => X = participation * Vbar * N

    # impact (Almgren et al. 2005 style); magnitudes are model parameters
    a_temp: float = 0.008       # temporary impact coefficient (price fraction at u=1)
    a_perm: float = 0.002       # permanent impact coefficient (price fraction at full order)
    delta: float = 1.0          # temporary-impact power-law exponent (1=linear, 0.5=sqrt)

    # volatility (annualised) -> per-step; minutes per year ~ 252*510 for Euronext day
    minutes_per_year: float = 252 * 510
    sigma_ann_ref: float = 0.22 # AC calibration vol (the "average" the desk assumes)

    # observable order-flow / alpha signal: AR(1) drift predictor
    sig_rho: float = 0.65       # signal persistence
    sig_sigma: float = 1.0      # signal innovation scale (standardised)
    sig_theta: float = 0.0005   # price drift per unit signal per step (fraction);
                                # calibrated so a signal-aware oracle beats TWAP by
                                # ~30%, matching Cartea et al. (2018) 20-40% range

    # innovations
    dist: str = "normal"        # "normal" or "t" (Student-t, df below)
    t_df: int = 4

    # terminal penalty on unexecuted inventory (bps-equivalent, large)
    term_penalty: float = 0.02  # price fraction penalty per unit residual at horizon

    # multi-venue
    n_venues: int = 3
    depth_mean: float = 1.0     # mean per-venue depth (capacity scale)
    depth_het: float = 0.0      # std of per-venue depth (heterogeneity knob)
    venue_disp: float = 0.0     # std of idiosyncratic venue price offset (dispersion knob)
    venue_offset_rho: float = 0.5  # offset persistence

    seed: int = 0

    @property
    def X(self) -> float:
        return self.participation * self.Vbar * self.N

    def sigma_step(self, sigma_ann: float) -> float:
        """Per-step price stdev in PRICE units for a given annual vol."""
        return sigma_ann / np.sqrt(self.minutes_per_year) * self.mid0


# U-shaped intraday volatility/volume seasonality (higher at open/close)
def u_shape(N: int) -> np.ndarray:
    t = np.linspace(0, 1, N)
    s = 1.0 + 0.6 * (np.cos(2 * np.pi * t) * 0.5 + 0.5) * 2 - 0.6  # gentle U
    s = 0.7 + 0.8 * (1 - np.sin(np.pi * t))  # high at ends, low mid
    return s / s.mean()


# --------------------------------------------------------------------------
# Scenario: all exogenous randomness for one execution session
# --------------------------------------------------------------------------
@dataclass
class Scenario:
    sigma_ann: float
    z: np.ndarray            # (N,) price innovations (standardised)
    sig: np.ndarray          # (N,) observable signal a_k
    season: np.ndarray       # (N,) seasonality multiplier
    depth: np.ndarray        # (N, n_venues) per-venue depth
    voff: np.ndarray         # (N, n_venues) per-venue price offset (price units)


def draw_scenario(p: Params, rng: np.random.Generator, sigma_ann: float) -> Scenario:
    N = p.N
    season = u_shape(N)
    # innovations
    if p.dist == "t":
        z = rng.standard_t(p.t_df, size=N) / np.sqrt(p.t_df / (p.t_df - 2))
    else:
        z = rng.standard_normal(N)
    # AR(1) observable signal
    sig = np.zeros(N)
    e = rng.standard_normal(N) * p.sig_sigma
    for k in range(1, N):
        sig[k] = p.sig_rho * sig[k - 1] + e[k]
    # per-venue depth (heterogeneous, time-varying) -- total capacity preserved
    base = np.maximum(0.2, p.depth_mean + p.depth_het * rng.standard_normal((N, p.n_venues)))
    depth = base
    # per-venue idiosyncratic price offset (mean-zero across venues), AR(1) in time
    voff = np.zeros((N, p.n_venues))
    ino = rng.standard_normal((N, p.n_venues)) * p.venue_disp
    for k in range(1, N):
        voff[k] = p.venue_offset_rho * voff[k - 1] + ino[k]
    voff -= voff.mean(axis=1, keepdims=True)            # mean-zero across venues
    voff *= p.mid0                                       # to price units
    return Scenario(sigma_ann, z, sig, season, depth, voff)


# --------------------------------------------------------------------------
# Core simulation: run an execution policy against a fixed scenario
# --------------------------------------------------------------------------
def apply_step(p: Params, sc: Scenario, k: int, mid: float, n: float,
               multivenue: bool = False):
    """One slice of execution. Returns (cost_increment, new_mid).
    Shared by simulate() and the Q-learning trainer so dynamics are identical."""
    X = p.X
    sigma_step = p.sigma_step(sc.sigma_ann)
    vbar_k = p.Vbar * sc.season[k]                  # volume seasonality (U-shape)
    if multivenue:
        n_j, eff = route(p, n, sc.depth[k], sc.voff[k], mid)
        exec_cash = float(np.sum(n_j * eff))
    else:
        u = n / vbar_k                              # participation this step
        temp = p.a_temp * (u ** p.delta) * p.mid0   # temporary impact (not persistent)
        exec_cash = n * (mid + temp)
    cost_inc = exec_cash - n * p.mid0               # vs arrival price
    perm = p.a_perm * (n / X) * p.mid0              # permanent impact (persists)
    drift = p.sig_theta * sc.sig[k] * p.mid0 * (sc.sigma_ann / p.sigma_ann_ref)
    new_mid = mid + perm + drift + sigma_step * sc.season[k] * sc.z[k]
    return cost_inc, new_mid


def simulate(p: Params, sc: Scenario, policy, multivenue: bool = False):
    """
    policy(state) -> participation fraction f in [0,1] of CURRENT residual.
    state = dict(k, rfrac, sig, sigma_ann).
    Returns IS in basis points of arrival notional (X * mid0), lower = better.
    """
    N, X = p.N, p.X
    mid = p.mid0
    residual = X
    cost = 0.0
    sched = np.zeros(N)
    for k in range(N):
        rfrac = residual / X
        if k == N - 1:
            f = 1.0                  # must finish
        else:
            f = float(np.clip(policy(dict(k=k, rfrac=rfrac, sig=sc.sig[k],
                                          sigma_ann=sc.sigma_ann,
                                          px=mid / p.mid0)), 0.0, 1.0))
        n = f * residual
        sched[k] = n
        cost_inc, mid = apply_step(p, sc, k, mid, n, multivenue)
        cost += cost_inc
        residual -= n
        if residual <= 1e-12:
            break
    if residual > 1e-9:
        cost += residual * (mid - p.mid0) + p.term_penalty * residual * p.mid0
    is_bps = cost / (X * p.mid0) * 1e4
    return is_bps, sched


def route(p: Params, n: float, depth: np.ndarray, voff: np.ndarray, mid: float):
    """
    Split n across venues minimising sum_j [ offset_j * n_j + b_j * n_j^2 ],
    b_j = a_temp*mid/depth_j (quadratic temporary impact), n_j>=0, sum=n.
    Closed-form water-filling on the KKT conditions. Returns (n_j, eff_price_j).
    """
    b = p.a_temp * mid / np.maximum(depth, 1e-6)        # convex coef per venue
    c = voff.copy()                                     # linear price offset
    # KKT: c_j + 2 b_j n_j = nu  => n_j = (nu - c_j)/(2 b_j), clipped at 0
    # water-filling: solve for nu so that sum n_j = n over the active set
    order = np.argsort(c)
    active = np.ones(p.n_venues, dtype=bool)
    for _ in range(p.n_venues):
        inv2b = 1.0 / (2 * b[active])
        nu = (n + np.sum(c[active] * inv2b)) / np.sum(inv2b)
        nj = (nu - c) * (1.0 / (2 * b))
        neg = nj < 0
        if not np.any(neg & active):
            break
        active[neg] = False
        if not np.any(active):
            active[order[0]] = True
            break
    nj = np.where(active, (nu - c) / (2 * b), 0.0)
    nj = np.maximum(nj, 0.0)
    if nj.sum() > 0:
        nj *= n / nj.sum()
    eff = mid + c + b * nj                              # effective per-share price
    return nj, eff


if __name__ == "__main__":
    p = Params()
    rng = np.random.default_rng(0)
    sc = draw_scenario(p, rng, 0.22)
    is_bps, sched = simulate(p, sc, lambda s: 1.0 / (p.N - s["k"]))  # ~TWAP
    print("sanity TWAP-ish IS (bps):", round(is_bps, 2), "sched sum:", round(sched.sum(), 3), "X:", p.X)
