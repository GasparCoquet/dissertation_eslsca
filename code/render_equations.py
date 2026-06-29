"""
render_equations.py
Render the dissertation's display equations as properly typeset math images
(LaTeX-style, Computer Modern) via matplotlib mathtext, so the manuscript shows
real superscripts/subscripts/symbols instead of ASCII. Output: code/figures/eq_*.png

Contract with build_dissertation.py: images whose basename starts with "eq_" are
embedded centred at NATURAL size (rendered at EQ_DPI), with no caption.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["mathtext.fontset"] = "cm"   # Computer Modern (classic LaTeX look)

HERE = os.path.dirname(__file__)
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
EQ_DPI = 200
FONTSIZE = 15

EQUATIONS = {
    # ---------------- Chapter 1 ----------------
    "eq_is_perold": r"\mathrm{IS}=\sum_t n_t\,(S_t-S_0)+\left(N-\sum_t n_t\right)(S_{\mathrm{end}}-S_0)+\sum_t c_t",
    "eq_impact_power": r"g(v)=\eta\,(v/V)^{\delta}",
    "eq_ac_dynamics": r"\tilde S_k=S_{k-1}+\sigma\sqrt{\tau}\,\xi_k-\kappa\,n_k-\eta\,(n_k/\tau)",
    "eq_ac_traj": r"x_k=X_0\,\frac{\sinh\left(\tilde\kappa\,(T-t_k)\right)}{\sinh(\tilde\kappa\,T)}",
    "eq_opt_policy": r"\pi^{*}=\arg\max_{\pi}\ \mathbb{E}_{\pi}\!\left[\sum_{t=0}^{T}\gamma^{t}\,R(s_t,a_t)\right]",
    "eq_dqn_loss": r"L(\theta)=\mathbb{E}\!\left[\left(r_t+\gamma\max_{a'}Q(s_{t+1},a';\theta^{-})-Q(s_t,a_t;\theta)\right)^{2}\right]",
    "eq_ppo_clip": r"L^{\mathrm{CLIP}}(\theta)=\mathbb{E}_t\!\left[\min\!\left(\rho_t(\theta)\hat A_t,\ \mathrm{clip}(\rho_t(\theta),1-\epsilon,1+\epsilon)\,\hat A_t\right)\right]",
    # ---------------- Chapter 2 ----------------
    "eq_state2": r"s_t=\left(q_t,\ \tau_t,\ S_t/S_0,\ \hat\sigma_t,\ \mathrm{OFI}_t,\ L_t^{1},\dots,L_t^{n},\ \Delta S_t^{ij}\right)",
    "eq_trans2": r"S_{t+1}=S_t+\eta\left(\sum_k v_t^{k}\right)S_0+\mu(\mathrm{OFI}_t)\,\Delta t+\hat\sigma_t\,S_t\sqrt{\Delta t}\,\varepsilon_{t+1}",
    "eq_reward2": r"r_t=-\left[\kappa\left(\sum_k v_t^{k}\right)^{2}\Delta t+(S_t-S_0)\left(\sum_k v_t^{k}\right)Q_0+\lambda\,q_t^{2}\,\hat\sigma_t^{2}\right]",
    "eq_obj2": r"J(\pi)=\mathbb{E}_{\pi}\!\left[\sum_{t=0}^{T} r_t\right]",
    "eq_bellman2": r"V^{*}(s_t)=\max_{a_t\in A}\left\{r(s_t,a_t)+\mathbb{E}\left[V^{*}(s_{t+1})\mid s_t,a_t\right]\right\}",
    "eq_is2": r"\mathrm{IS}=\sum_t n_t\,(P_t-S_0)+\left(Q_0-\sum_t n_t\right)(S_T-S_0)",
    "eq_dqn_target2": r"y_t=r_t+\gamma\max_{a'}Q_{\bar\theta}(s_{t+1},a')",
    "eq_dqn_mse2": r"L(\theta)=\mathbb{E}\left[(y_t-Q_{\theta}(s_t,a_t))^{2}\right]",
    "eq_ppo2": r"L^{\mathrm{CLIP}}(\theta)=\mathbb{E}\!\left[\min\!\left(\rho_t(\theta)\hat A_t,\ \mathrm{clip}(\rho_t(\theta),1-\epsilon,1+\epsilon)\,\hat A_t\right)\right]",
    # ---------------- Chapter 3 ----------------
    "eq_price3": r"S_{k+1}=S_k+\kappa\,n_k+\theta\,a_k\,(\sigma/\sigma_{\mathrm{ref}})+\sigma_{\mathrm{step}}\,\mathrm{season}_k\,z_k",
    "eq_temp3": r"h(u_k)=a_{\mathrm{temp}}\,u_k^{\delta}",
    "eq_perm3": r"\kappa\,n_k=a_{\mathrm{perm}}\,\frac{n_k}{X}\,S_0\qquad\mathrm{(total\ over\ }X\mathrm{:}\ a_{\mathrm{perm}}\,S_0)",
    "eq_isbps3": r"\mathrm{IS}_{\mathrm{bps}}=\frac{\sum_{k=0}^{K-1} n_k P_k-X S_0}{X S_0}\times 10^{4}",
    "eq_ar1_3": r"a_{k+1}=\rho\,a_k+\varepsilon_k,\qquad \varepsilon_k\sim\mathcal{N}(0,\sigma_\varepsilon^{2}),\quad \rho=0.65",
    "eq_venuecost3": r"\mathrm{cost}_{j,k}=c_{j,k}\,n_{j,k}+b_j\,n_{j,k}^{2}",
    "eq_bj3": r"b_j=a_{\mathrm{temp}}\,S_k/D_j",
    "eq_route3": r"\min_{n_{j,k}}\ \sum_j\left[c_{j,k}\,n_{j,k}+b_j\,n_{j,k}^{2}\right]\quad\mathrm{s.t.}\ \sum_j n_{j,k}=n_k,\ \ n_{j,k}\geq 0",
    "eq_kkt3": r"n_{j,k}^{*}=\max\!\left(0,\ \frac{\lambda^{*}-c_{j,k}}{2\,b_j}\right)",
    "eq_nw3": r"V^{\mathrm{NW}}=\gamma_0+2\sum_{l=1}^{L} w_l\,\gamma_l",
}


def main():
    ok, fail = 0, []
    for name, tex in EQUATIONS.items():
        try:
            fig = plt.figure(figsize=(0.1, 0.1))
            fig.text(0.0, 0.0, "$" + tex + "$", fontsize=FONTSIZE)
            path = os.path.join(FIG, name + ".png")
            fig.savefig(path, dpi=EQ_DPI, bbox_inches="tight", pad_inches=0.06,
                        facecolor="white")
            plt.close(fig)
            ok += 1
        except Exception as e:
            fail.append((name, str(e)[:120]))
            plt.close("all")
    print("rendered %d equations" % ok)
    for n, e in fail:
        print("  FAIL", n, "->", e)
    # clean stray test file
    t = os.path.join(FIG, "_eqtest.png")
    if os.path.exists(t):
        os.remove(t)
    return not fail


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
