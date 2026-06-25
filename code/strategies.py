"""
strategies.py
=============
Execution strategies evaluated in the study:
  - TWAP            : uniform schedule (time-blind, volume-blind, signal-blind)
  - VWAP            : follows the (U-shaped) volume profile
  - Almgren-Chriss  : closed-form risk-averse schedule, calibrated to an
                      AVERAGE volatility (sigma_ann_ref) -> mis-specified in
                      off-average regimes (Forsyth et al. 2012; => H3)
  - QLearningAgent  : tabular Q-learning over a discretised state
                      (Nevmyvaka, Feng & Kearns 2006), observes the order-flow
                      signal and current regime that the static models ignore.
"""
from __future__ import annotations
import numpy as np
from execution_study import Params, Scenario, apply_step, draw_scenario

# --------------------------------------------------------------------------
# Static / analytical baselines (returned as policies: state -> fraction)
# --------------------------------------------------------------------------
def twap_policy(p: Params):
    def pol(s):
        return 1.0 / (p.N - s["k"])          # uniform shares
    return pol


def vwap_policy(p: Params):
    season = None
    def pol(s):
        nonlocal season
        if season is None:
            from execution_study import u_shape
            season = u_shape(p.N)
        k = s["k"]
        remaining_profile = season[k:].sum()
        return season[k] / remaining_profile  # fraction of residual matching volume profile
    return pol


def almgren_chriss_schedule(p: Params, lam: float = 0.53) -> np.ndarray:
    """Closed-form AC holdings -> per-step fractions of original X.
    kappa = sqrt(lam * sigma_step^2 / eta_lin), eta_lin = a_temp*mid0/Vbar."""
    N = p.N
    sigma_step = p.sigma_step(p.sigma_ann_ref)
    eta_lin = p.a_temp * p.mid0 / p.Vbar
    kappa = np.sqrt(max(lam, 1e-9) * sigma_step ** 2 / eta_lin)
    j = np.arange(N + 1)
    if kappa * N < 1e-6:
        x = p.X * (1 - j / N)                 # -> TWAP as lam -> 0
    else:
        x = p.X * np.sinh(kappa * (N - j)) / np.sinh(kappa * N)
    n = x[:-1] - x[1:]                         # shares per step
    return n


def ac_policy(p: Params, lam: float = 0.53):
    n = almgren_chriss_schedule(p, lam)
    cum = np.concatenate([[0], np.cumsum(n)])
    def pol(s):
        k = s["k"]
        residual = p.X - cum[k]
        if residual <= 1e-12:
            return 0.0
        return n[k] / residual
    return pol


# --------------------------------------------------------------------------
# Tabular Q-learning agent
# --------------------------------------------------------------------------
# Actions are MULTIPLES of the current TWAP rate 1/(N-k): m=1 reproduces TWAP
# exactly; m<1 decelerates, m>1 accelerates. f = min(1, m/(N-k)) of residual.
ACTIONS = np.array([0.0, 0.5, 0.75, 1.0, 1.3, 1.6])       # TWAP-rate multipliers

def mult_to_frac(m, k, N):
    return min(1.0, m / (N - k))

class QLearningAgent:
    def __init__(self, p: Params, n_rbins=6, sig_edges=(-1.5, -0.5, 0.5, 1.5),
                 px_edges=(-0.004, -0.0015, 0.0015, 0.004),
                 vol_thresh=0.27, alpha=0.15, gamma=1.0):
        self.p = p
        self.n_rbins = n_rbins
        self.sig_edges = np.array(sig_edges)
        self.px_edges = np.array(px_edges)          # bins of (mid/mid0 - 1)
        self.vol_thresh = vol_thresh
        self.alpha = alpha
        self.gamma = gamma
        self.nA = len(ACTIONS)
        # Q[k, rbin, sigbin, pxbin, volbin, action]
        self.Q = np.zeros((p.N, n_rbins, len(sig_edges) + 1,
                           len(px_edges) + 1, 2, self.nA))

    def _idx(self, k, rfrac, sig, px, sigma_ann):
        rb = min(self.n_rbins - 1, int(rfrac * self.n_rbins))
        sb = int(np.searchsorted(self.sig_edges, sig))
        pb = int(np.searchsorted(self.px_edges, px - 1.0))   # price vs arrival
        vb = 1 if sigma_ann >= self.vol_thresh else 0
        return k, rb, sb, pb, vb

    def act_index(self, state, eps):
        idx = self._idx(state["k"], state["rfrac"], state["sig"],
                        state["px"], state["sigma_ann"])
        if np.random.random() < eps:
            return np.random.randint(self.nA)
        return int(np.argmax(self.Q[idx]))

    def greedy_policy(self):
        def pol(s):
            idx = self._idx(s["k"], s["rfrac"], s["sig"], s["px"], s["sigma_ann"])
            m = ACTIONS[int(np.argmax(self.Q[idx]))]
            return mult_to_frac(m, s["k"], self.p.N)
        return pol

    def _value(self, idx, k):
        """State value for bootstrapping. At the terminal step only the
        forced full-liquidation action is feasible, so use ITS value (not the
        max over never-updated actions, which would be optimistically 0)."""
        if k == self.p.N - 1:
            return self.Q[idx + (self.nA - 1,)]
        return float(np.max(self.Q[idx]))

    def train(self, regimes, n_episodes=120_000, eps0=1.0, eps1=0.05,
              seed=12345, multivenue=False):
        p = self.p
        rng = np.random.default_rng(seed)
        for ep in range(n_episodes):
            eps = eps1 + (eps0 - eps1) * max(0, 1 - ep / (0.8 * n_episodes))
            sigma_ann = regimes[rng.integers(len(regimes))]
            sc = draw_scenario(p, rng, sigma_ann)
            mid = p.mid0
            residual = p.X
            for k in range(p.N):
                rfrac = residual / p.X
                px = mid / p.mid0
                state = dict(k=k, rfrac=rfrac, sig=sc.sig[k], px=px, sigma_ann=sigma_ann)
                if k == p.N - 1:
                    a = self.nA - 1                  # force finish
                else:
                    a = self.act_index(state, eps)
                f = 1.0 if k == p.N - 1 else mult_to_frac(ACTIONS[a], k, p.N)
                n = f * residual
                idx = self._idx(k, rfrac, sc.sig[k], px, sigma_ann)
                cost_inc, mid = apply_step(p, sc, k, mid, n, multivenue)
                residual -= n
                r = -cost_inc / (p.X * p.mid0) * 1e4     # reward in -bps
                # bootstrap from next state's value (or terminal forced value)
                if k < p.N - 1 and residual > 1e-12:
                    nxt = self._idx(k + 1, residual / p.X, sc.sig[k + 1],
                                    mid / p.mid0, sigma_ann)
                    target = r + self.gamma * self._value(nxt, k + 1)
                else:
                    tail = 0.0
                    if residual > 1e-9:                 # terminal penalty into reward
                        tp_cost = residual * (mid - p.mid0) + p.term_penalty * residual * p.mid0
                        tail = -tp_cost / (p.X * p.mid0) * 1e4
                    target = r + tail
                qa = self.Q[idx + (a,)]
                self.Q[idx + (a,)] = qa + self.alpha * (target - qa)
                if residual <= 1e-12:
                    break
        return self


# --------------------------------------------------------------------------
# Policy-gradient agent (direct policy optimisation via Cross-Entropy Method)
# A continuous-action policy optimiser in the family of Moody & Saffell (2001)
# direct reinforcement and the policy-optimisation principle of PPO. The policy
# maps state features to a TWAP-rate multiplier; parameters are optimised
# directly on the simulated IS objective.
# --------------------------------------------------------------------------
def _softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)

class PolicyGradientAgent:
    """Linear-Gaussian policy m(state)=softplus(theta.features); theta optimised
    by CEM on mean simulated IS with common-random-number paired evaluation."""
    FEAT_DIM = 6

    def __init__(self, p: Params, vol_thresh=0.27):
        self.p = p
        self.vol_thresh = vol_thresh
        self.theta = np.zeros(self.FEAT_DIM)
        self.theta[0] = 0.541        # softplus(0.541)=1.0 => starts at TWAP

    def features(self, s):
        p = self.p
        return np.array([
            1.0,
            s["sig"],
            100.0 * (s["px"] - 1.0),
            (s["k"] / (p.N - 1)) - 0.5,
            1.0 if s["sigma_ann"] >= self.vol_thresh else -1.0,
            s["rfrac"] - 0.5,
        ])

    def policy_from(self, theta):
        def pol(s):
            m = _softplus(float(theta @ self.features(s)))
            return min(1.0, m / (self.p.N - s["k"]))
        return pol

    def greedy_policy(self):
        return self.policy_from(self.theta)

    def train(self, regimes, iters=22, pop=28, elite_frac=0.25, batch=600,
              sigma0=0.6, seed=2024, multivenue=False):
        from execution_study import draw_scenario, simulate
        rng = np.random.default_rng(seed)
        mu = self.theta.copy()
        sig = np.full(self.FEAT_DIM, sigma0)
        n_elite = max(2, int(pop * elite_frac))
        for it in range(iters):
            # common scenarios this generation (paired -> low-variance ranking)
            eval_rng = np.random.default_rng(10_000 + it)
            scs = [draw_scenario(self.p, eval_rng, regimes[eval_rng.integers(len(regimes))])
                   for _ in range(batch)]
            pops = mu + sig * rng.standard_normal((pop, self.FEAT_DIM))
            J = np.empty(pop)
            for i in range(pop):
                pol = self.policy_from(pops[i])
                J[i] = np.mean([simulate(self.p, sc, pol, multivenue)[0] for sc in scs])
            elite = pops[np.argsort(J)[:n_elite]]
            mu = elite.mean(0)
            sig = elite.std(0) + 1e-3
        self.theta = mu
        return self


# Discrete multipliers for the value-based / DQN-family agent (coarser action
# grid => the discretisation handicap that motivates H1a, PPO > DQN).
DISC_ACTIONS = np.array([0.0, 0.5, 1.0, 1.5])

class DiscreteValueAgent:
    """Value-based, DISCRETE-action agent: scores each discrete multiplier with
    a linear function of state features and acts greedily (argmax). The scoring
    weights are optimised by direct policy search (CEM) on simulated IS. This is
    a controlled, stable instance of the DQN family (value-based + discrete
    actions) from Chapter 2; it is handicapped vs the continuous policy-gradient
    agent purely by action discretisation."""
    FEAT_DIM = 6

    def __init__(self, p: Params, vol_thresh=0.27):
        self.p = p
        self.vol_thresh = vol_thresh
        self.nA = len(DISC_ACTIONS)
        self.W = np.zeros((self.nA, self.FEAT_DIM))
        self.W[2, 0] = 1.0          # bias toward the m=1.0 (TWAP) action initially

    def features(self, s):
        p = self.p
        return np.array([1.0, s["sig"], 100.0 * (s["px"] - 1.0),
                         (s["k"] / (p.N - 1)) - 0.5,
                         1.0 if s["sigma_ann"] >= self.vol_thresh else -1.0,
                         s["rfrac"] - 0.5])

    def policy_from(self, W):
        def pol(s):
            scores = W @ self.features(s)
            m = DISC_ACTIONS[int(np.argmax(scores))]
            return min(1.0, m / (self.p.N - s["k"]))
        return pol

    def greedy_policy(self):
        return self.policy_from(self.W)

    def train(self, regimes, iters=26, pop=44, elite_frac=0.2, batch=500,
              sigma0=0.7, seed=4242, multivenue=False):
        from execution_study import draw_scenario, simulate
        rng = np.random.default_rng(seed)
        d = self.nA * self.FEAT_DIM
        mu = self.W.flatten().copy()
        sig = np.full(d, sigma0)
        n_elite = max(2, int(pop * elite_frac))
        for it in range(iters):
            eval_rng = np.random.default_rng(20_000 + it)
            scs = [draw_scenario(self.p, eval_rng, regimes[eval_rng.integers(len(regimes))])
                   for _ in range(batch)]
            pops = mu + sig * rng.standard_normal((pop, d))
            J = np.empty(pop)
            for i in range(pop):
                pol = self.policy_from(pops[i].reshape(self.nA, self.FEAT_DIM))
                J[i] = np.mean([simulate(self.p, sc, pol, multivenue)[0] for sc in scs])
            elite = pops[np.argsort(J)[:n_elite]]
            mu = elite.mean(0); sig = elite.std(0) + 1e-3
        self.W = mu.reshape(self.nA, self.FEAT_DIM)
        return self

