"""
experiments.py
==============
Runs the full empirical study and writes results (JSON + CSV) and figures.

All numbers are SIMULATED Implementation Shortfall (bps) under the calibrated
model of execution_study.py. Only signs, rankings and comparative statics are
interpreted (see dissertation Ch. 3-4 and the limitations section).

Outputs (in code/results/ and code/figures/):
  results.json         all headline numbers used in the dissertation
  table_h1.csv         per-strategy IS distribution (H1)
  table_regime.csv     per-strategy IS by volatility regime (H3)
  fig_is_distribution.png, fig_regime.png, fig_h2_depth.png,
  fig_learning.png, fig_robustness.png
"""
from __future__ import annotations
import json, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from execution_study import Params, draw_scenario, simulate
from strategies import (twap_policy, vwap_policy, ac_policy,
                        PolicyGradientAgent, DiscreteValueAgent)

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results"); os.makedirs(RES, exist_ok=True)
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)

REGIMES = [0.15, 0.22, 0.35]
TRAIN_REGIMES = REGIMES
N_TEST = 12000           # out-of-sample sessions for evaluation
TEST_SEED = 20240705     # held-out seed, disjoint from training seeds


# --------------------------------------------------------------------------
# Evaluation helpers
# --------------------------------------------------------------------------
def eval_policies(p, policies, regimes=REGIMES, n=N_TEST, seed=TEST_SEED,
                  multivenue=False):
    """Paired evaluation: every policy sees the SAME set of out-of-sample
    scenarios (common random numbers) so differences are not noise."""
    rng = np.random.default_rng(seed)
    sigs = [regimes[i] for i in rng.integers(len(regimes), size=n)]
    scns = []
    rng2 = np.random.default_rng(seed + 1)
    for sa in sigs:
        scns.append(draw_scenario(p, rng2, sa))
    out = {name: np.empty(n) for name in policies}
    regs = np.array(sigs)
    for j, sc in enumerate(scns):
        for name, pol in policies.items():
            out[name][j] = simulate(p, sc, pol, multivenue)[0]
    return out, regs


def newey_west_tstat(d, lag=5):
    """HAC (Newey-West) t-stat for mean of paired differences d (H0: mean=0)."""
    d = np.asarray(d, float); n = len(d); m = d.mean(); e = d - m
    g0 = np.dot(e, e) / n
    s = g0
    for l in range(1, lag + 1):
        w = 1 - l / (lag + 1)
        cov = np.dot(e[l:], e[:-l]) / n
        s += 2 * w * cov
    se = np.sqrt(s / n)
    return m / se if se > 0 else 0.0


def paired_bootstrap_ci(d, B=10000, seed=0):
    rng = np.random.default_rng(seed); d = np.asarray(d, float); n = len(d)
    means = d[rng.integers(0, n, size=(B, n))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summary(v):
    return dict(mean=float(v.mean()), median=float(np.median(v)),
                std=float(v.std()), p95=float(np.percentile(v, 95)),
                var95=float(np.percentile(v, 95)))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    t0 = time.time()
    p = Params()
    out = {"params": {k: getattr(p, k) for k in
           ["N", "mid0", "participation", "a_temp", "a_perm", "delta",
            "sigma_ann_ref", "sig_theta", "sig_rho"]},
           "regimes": REGIMES, "n_test": N_TEST}

    # ---- Train agents ----
    print("[1/6] training agents ...")
    pg = PolicyGradientAgent(p).train(TRAIN_REGIMES, seed=2024,
                                      iters=22, pop=28, batch=600)
    dqn = DiscreteValueAgent(p).train(TRAIN_REGIMES, seed=4242,
                                      iters=20, pop=36, batch=420)
    out["pg_theta"] = pg.theta.tolist()

    policies = {
        "TWAP": twap_policy(p),
        "VWAP": vwap_policy(p),
        "Almgren-Chriss": ac_policy(p),
        "DQN": dqn.greedy_policy(),
        "PPO": pg.greedy_policy(),
    }

    # ---- H1: global out-of-sample performance ----
    print("[2/6] H1 out-of-sample evaluation ...")
    res, regs = eval_policies(p, policies)
    out["H1"] = {name: summary(v) for name, v in res.items()}
    # pairwise vs each benchmark
    out["H1_tests"] = {}
    for a in ["PPO", "DQN"]:
        for b in ["TWAP", "VWAP", "Almgren-Chriss"]:
            d = res[a] - res[b]
            lo, hi = paired_bootstrap_ci(d)
            out["H1_tests"][f"{a}_vs_{b}"] = dict(
                delta=float(d.mean()), pct=float(100 * (1 - res[a].mean() / res[b].mean())),
                tstat=float(newey_west_tstat(d)), ci=[lo, hi])
    d = res["PPO"] - res["DQN"]
    out["H1_tests"]["PPO_vs_DQN"] = dict(delta=float(d.mean()),
        pct=float(100 * (1 - res["PPO"].mean() / res["DQN"].mean())),
        tstat=float(newey_west_tstat(d)), ci=list(paired_bootstrap_ci(d)))

    # ---- H3: regime-conditional performance ----
    print("[3/6] H3 regime analysis ...")
    reg_tbl = {}
    for name, v in res.items():
        reg_tbl[name] = {f"{r:.2f}": float(v[regs == r].mean()) for r in REGIMES}
    out["H3"] = reg_tbl
    # PPO vs AC gap by regime + significance in low/high
    out["H3_tests"] = {}
    for r in REGIMES:
        mask = regs == r
        d = res["PPO"][mask] - res["Almgren-Chriss"][mask]
        out["H3_tests"][f"{r:.2f}"] = dict(
            ppo=float(res["PPO"][mask].mean()), ac=float(res["Almgren-Chriss"][mask].mean()),
            pct=float(100 * (1 - res["PPO"][mask].mean() / res["Almgren-Chriss"][mask].mean())),
            tstat=float(newey_west_tstat(d)))

    # ---- H2: multi-venue fragmentation alpha + depth-heterogeneity knob ----
    print("[4/6] H2 multi-venue fragmentation ...")
    out["H2"] = run_h2(p)

    # ---- Robustness ----
    print("[5/6] robustness ...")
    out["robustness"] = run_robustness(p, pg)

    # ---- Figures + tables ----
    print("[6/6] figures + tables ...")
    make_figures(p, res, regs, out)
    write_tables(res, regs, out)

    with open(os.path.join(RES, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"done in {time.time()-t0:.0f}s. results in {RES}")
    print_headline(out)
    return out


def run_h2(p):
    """Fragmentation alpha: dynamic multi-venue routing vs a single pooled venue
    with the SAME total capacity, as a function of cross-venue price dispersion.
    Paired (common random numbers): the single-venue world is built from the
    same scenario by pooling depth and zeroing venue offsets, so only the venue
    structure differs. Alpha must vanish as dispersion -> 0 (falsifiable)."""
    from execution_study import Scenario
    disp_grid = [0.0, 0.0004, 0.0008, 0.0016, 0.0032]
    rows = []
    for disp in disp_grid:
        pm = Params(n_venues=3, venue_disp=disp, depth_het=0.15)
        ps = Params(n_venues=1, venue_disp=0.0)
        rng = np.random.default_rng(777)
        sigs = [REGIMES[i] for i in rng.integers(len(REGIMES), size=5000)]
        rng2 = np.random.default_rng(778)
        d = np.empty(len(sigs)); im_all = np.empty(len(sigs)); is_all = np.empty(len(sigs))
        for t, sa in enumerate(sigs):
            sc = draw_scenario(pm, rng2, sa)
            # pooled single-venue scenario: same price path, summed depth, no offset
            sc_s = Scenario(sa, sc.z, sc.sig, sc.season,
                            sc.depth.sum(axis=1, keepdims=True),
                            np.zeros((pm.N, 1)))
            im = simulate(pm, sc, twap_policy(pm), multivenue=True)[0]
            isg = simulate(ps, sc_s, twap_policy(ps), multivenue=True)[0]
            im_all[t] = im; is_all[t] = isg; d[t] = isg - im
        rows.append(dict(dispersion=disp, is_multi=float(im_all.mean()),
                         is_single=float(is_all.mean()),
                         alpha_bps=float(d.mean()),
                         alpha_pct=float(100 * (1 - im_all.mean() / is_all.mean())),
                         tstat=float(newey_west_tstat(d))))
    return {"by_dispersion": rows}


def run_robustness(p, pg):
    """Re-evaluate PPO vs baselines under alternative model assumptions."""
    checks = {}
    variants = {
        "baseline": Params(),
        "sqrt_impact": Params(delta=0.5),
        "student_t": Params(dist="t", t_df=4),
        "high_impact": Params(a_temp=0.012, a_perm=0.003),
        "ood_shift": Params(sig_rho=0.5, sigma_ann_ref=0.18),  # test-time param shift
    }
    for name, pv in variants.items():
        res, _ = eval_policies(pv, {"TWAP": twap_policy(pv),
                                    "Almgren-Chriss": ac_policy(pv),
                                    "PPO": pg.greedy_policy()}, n=5000, seed=99)
        checks[name] = dict(
            ppo_vs_twap=float(100 * (1 - res["PPO"].mean() / res["TWAP"].mean())),
            ppo_vs_ac=float(100 * (1 - res["PPO"].mean() / res["Almgren-Chriss"].mean())),
            ppo_is=float(res["PPO"].mean()), twap_is=float(res["TWAP"].mean()))
    return checks


# --------------------------------------------------------------------------
# Figures and tables
# --------------------------------------------------------------------------
ORDER = ["TWAP", "VWAP", "Almgren-Chriss", "DQN", "PPO"]
COLORS = {"TWAP": "#9e9e9e", "VWAP": "#7e7e7e", "Almgren-Chriss": "#5b5b5b",
          "DQN": "#4c78a8", "PPO": "#c44e52"}

def make_figures(p, res, regs, out):
    # Fig: IS distribution (box)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    data = [res[k] for k in ORDER]
    bp = ax.boxplot(data, labels=ORDER, showfliers=False, patch_artist=True,
                    whis=(5, 95))
    for patch, k in zip(bp["boxes"], ORDER):
        patch.set_facecolor(COLORS[k]); patch.set_alpha(0.8)
    ax.set_ylabel("Implementation Shortfall (bps)")
    ax.set_title("Out-of-sample IS distribution by strategy (simulated)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_is_distribution.png"), dpi=150)
    plt.close(fig)

    # Fig: regime
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(REGIMES)); w = 0.16
    for i, k in enumerate(ORDER):
        means = [res[k][regs == r].mean() for r in REGIMES]
        ax.bar(x + (i - 2) * w, means, w, label=k, color=COLORS[k])
    ax.set_xticks(x); ax.set_xticklabels([f"sigma={int(r*100)}%" for r in REGIMES])
    ax.set_ylabel("Mean IS (bps)"); ax.set_title("IS by volatility regime (simulated)")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_regime.png"), dpi=150)
    plt.close(fig)

    # Fig: H2 depth/dispersion
    rows = out["H2"]["by_dispersion"]
    fig, ax = plt.subplots(figsize=(7, 4.3))
    xs = [r["dispersion"] * 1e4 for r in rows]
    ys = [r["alpha_bps"] for r in rows]
    ax.plot(xs, ys, "o-", color="#c44e52")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Cross-venue price dispersion (bps)")
    ax.set_ylabel("Fragmentation alpha (bps)")
    ax.set_title("Multi-venue routing alpha vanishes as dispersion -> 0 (H2)")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_h2_depth.png"), dpi=150)
    plt.close(fig)

    # Fig: robustness
    rb = out["robustness"]
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    names = list(rb.keys())
    ax.bar(np.arange(len(names)) - 0.2, [rb[n]["ppo_vs_twap"] for n in names], 0.4,
           label="PPO vs TWAP", color="#c44e52")
    ax.bar(np.arange(len(names)) + 0.2, [rb[n]["ppo_vs_ac"] for n in names], 0.4,
           label="PPO vs Almgren-Chriss", color="#4c78a8")
    ax.set_xticks(np.arange(len(names))); ax.set_xticklabels(names, rotation=20, fontsize=8)
    ax.set_ylabel("IS reduction (%)"); ax.set_title("Robustness of PPO advantage (simulated)")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_robustness.png"), dpi=150)
    plt.close(fig)


def write_tables(res, regs, out):
    import csv
    with open(os.path.join(RES, "table_h1.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["strategy", "mean", "median", "std", "VaR95", "pct_vs_TWAP"])
        tw = res["TWAP"].mean()
        for k in ORDER:
            v = res[k]
            w.writerow([k, f"{v.mean():.2f}", f"{np.median(v):.2f}", f"{v.std():.2f}",
                        f"{np.percentile(v,95):.2f}", f"{100*(1-v.mean()/tw):.1f}"])
    with open(os.path.join(RES, "table_regime.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["strategy"] + [f"sigma_{int(r*100)}" for r in REGIMES])
        for k in ORDER:
            w.writerow([k] + [f"{res[k][regs==r].mean():.2f}" for r in REGIMES])


def print_headline(out):
    print("\n================ HEADLINE (simulated IS) ================")
    for k in ORDER:
        s = out["H1"][k]
        print(f"  {k:16s} mean={s['mean']:6.2f}  median={s['median']:6.2f}  VaR95={s['var95']:6.1f}")
    print("  ---- H1 reductions ----")
    for kk, vv in out["H1_tests"].items():
        print(f"  {kk:28s} {vv['pct']:+5.1f}%  (t={vv['tstat']:.1f})")
    print("  ---- H3 PPO vs AC by regime ----")
    for r, vv in out["H3_tests"].items():
        print(f"  sigma={r}: PPO {vv['ppo']:.2f} vs AC {vv['ac']:.2f} -> {vv['pct']:+.1f}% (t={vv['tstat']:.1f})")
    print("  ---- H2 fragmentation alpha by dispersion ----")
    for row in out["H2"]["by_dispersion"]:
        print(f"  disp={row['dispersion']*1e4:.1f}bps: alpha={row['alpha_bps']:.2f}bps ({row['alpha_pct']:+.1f}%)")


if __name__ == "__main__":
    main()
