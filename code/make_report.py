"""
make_report.py
Render the honest empirical supplement (Abstract + Ch.3 + Ch.4 + Limitations)
to a self-contained PDF with the result tables and the four figures embedded.
Pure fpdf2 (no LaTeX / LibreOffice needed).
"""
import os, json
from fpdf import FPDF

HERE = os.path.dirname(__file__)
FIG = os.path.join(HERE, "figures")
RES = os.path.join(HERE, "results")
OUT = os.path.join(os.path.dirname(HERE), "Dissertation_Coquet_Ch3-4_EMPIRICAL.pdf")

R = json.load(open(os.path.join(RES, "results.json")))

class PDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Times", "I", 9)
        self.set_text_color(120)
        self.cell(0, 8, "Research Dissertation - Gaspar Coquet - ESLSCA MBA 2026  |  Empirical supplement (simulated)",
                  align="C")
        self.set_text_color(0)
        self.ln(10)
    def footer(self):
        self.set_y(-15)
        self.set_font("Times", "", 9)
        self.set_text_color(120)
        self.cell(0, 10, str(self.page_no()), align="C")
        self.set_text_color(0)

pdf = PDF(format="A4")
pdf.set_margins(25, 20, 25)
pdf.set_auto_page_break(True, 20)

def H(txt, size=14, sp=3, top=2):
    pdf.ln(top); pdf.set_font("Times", "B", size)
    pdf.multi_cell(0, 7, txt); pdf.ln(sp)

def P(txt, size=11.5):
    pdf.set_font("Times", "", size)
    pdf.multi_cell(0, 6.2, txt, align="J"); pdf.ln(1.5)

def table(headers, rows, widths=None, fs=10):
    pdf.set_font("Times", "B", fs)
    avail = 160
    if widths is None:
        widths = [avail / len(headers)] * len(headers)
    pdf.set_fill_color(235, 235, 235)
    for h, w in zip(headers, widths):
        pdf.cell(w, 7, h, border=1, align="C", fill=True)
    pdf.ln()
    pdf.set_font("Times", "", fs)
    for r in rows:
        for c, w in zip(r, widths):
            pdf.cell(w, 6.5, str(c), border=1, align="C")
        pdf.ln()
    pdf.ln(2)

def figure(name, w=160, cap=None):
    path = os.path.join(FIG, name)
    if os.path.exists(path):
        x = (210 - w) / 2
        pdf.image(path, x=x, w=w)
        if cap:
            pdf.set_font("Times", "I", 9); pdf.set_text_color(90)
            pdf.multi_cell(0, 5, cap, align="C"); pdf.set_text_color(0)
        pdf.ln(2)

# ---------------- Title ----------------
pdf.add_page()
pdf.ln(30)
pdf.set_font("Times", "B", 18)
pdf.multi_cell(0, 9, "Reinforcement Learning and Implementation Shortfall on Fragmented "
                     "European Equity Markets", align="C")
pdf.ln(6)
pdf.set_font("Times", "", 13)
pdf.multi_cell(0, 7, "Empirical Supplement - Honest, Simulation-Based Results\n"
                     "(replaces the provisional Chapter 4)", align="C")
pdf.ln(10)
pdf.set_font("Times", "", 12)
pdf.multi_cell(0, 7, "Gaspar Coquet\nESLSCA Business School Paris - MBA Trading & Financial Markets\n"
                     "Academic Year 2025-2026", align="C")
pdf.ln(14)
pdf.set_font("Times", "I", 10.5); pdf.set_text_color(90)
pdf.multi_cell(0, 5.5,
    "All results in this document are SIMULATED Implementation Shortfall under a transparent, "
    "literature-calibrated market model. No proprietary or real tick data is used. Only the sign, "
    "ranking and comparative statics of the effects are interpreted; the basis-point magnitudes are "
    "model artefacts and are not a backtest on, or a claim of deployable edge in, the live market. "
    "The full pipeline is reproducible from the accompanying code/ directory.", align="J")
pdf.set_text_color(0)

# ---------------- Abstract ----------------
pdf.add_page()
H("Abstract")
t = R["H1_tests"]
P(f"This dissertation investigates the extent to which Reinforcement Learning (RL) can reduce "
  f"market-impact costs, measured by the Implementation Shortfall (IS), in the fragmented "
  f"post-MiFID II European equity market. It frames multi-venue optimal execution as a Markov "
  f"Decision Process and compares two RL agents - a value-based discrete-action agent (DQN family) "
  f"and a policy-gradient continuous-action agent (PPO family) - against TWAP, VWAP and "
  f"Almgren-Chriss. Because synchronised multi-venue Level-2 data for CAC 40 equities is available "
  f"only through paid institutional feeds, no proprietary or real tick data is used: the agents are "
  f"trained and evaluated inside a stochastic market simulator calibrated to publicly documented "
  f"CAC 40 characteristics, with impact coefficients from Almgren et al. (2005). This "
  f"calibrated-simulation approach is standard in the RL-execution literature (Cartea, Donnelly & "
  f"Jaimungal, 2018; Hendricks & Wilcox, 2014).")
P(f"On 12,000 out-of-sample sessions, the PPO-family agent reduces mean simulated IS by "
  f"{t['PPO_vs_TWAP']['pct']:.1f}% versus TWAP, {t['PPO_vs_VWAP']['pct']:.1f}% versus VWAP and "
  f"{t['PPO_vs_Almgren-Chriss']['pct']:.1f}% versus Almgren-Chriss (all paired, Newey-West p < 0.001). "
  f"The DQN-family agent achieves {t['DQN_vs_TWAP']['pct']:.1f}% / {t['DQN_vs_VWAP']['pct']:.1f}% / "
  f"{t['DQN_vs_Almgren-Chriss']['pct']:.1f}% and is significantly outperformed by PPO "
  f"({t['PPO_vs_DQN']['pct']:.1f}%, p < 0.001), consistent with H1a. Multi-venue routing generates a "
  f"fragmentation alpha that rises with cross-venue price dispersion and is exactly zero when "
  f"dispersion vanishes - a falsifiable confirmation of H2. The RL advantage is strongly "
  f"regime-conditional (H3): the PPO-vs-Almgren-Chriss gain rises monotonically from "
  f"{R['H3_tests']['0.15']['pct']:.1f}% in calm markets to {R['H3_tests']['0.35']['pct']:.1f}% in "
  f"stressed markets; contrary to the original sub-hypothesis, a small but significant edge persists "
  f"even in calm regimes. The ranking is sign-stable under square-root impact, Student-t shocks and "
  f"an out-of-distribution parameter shift.")

# ---------------- Chapter 3 ----------------
pdf.add_page()
H("Chapter 3 (revised) - Methodology: a calibrated market simulator")
P("Epistemological stance unchanged (positivist, hypothetico-deductive). Because the object of "
  "study is the comparative performance of execution policies under controlled, repeatable market "
  "conditions, the empirical method is simulation-based backtesting in a calibrated stochastic "
  "environment rather than a historical replay on proprietary tick data. This is standard in the "
  "RL-execution literature (Cartea et al., 2018; Hendricks & Wilcox, 2014) and additionally captures "
  "the agent's own market impact, which a static historical replay cannot.")
H("3.2 Synthetic market environment calibrated to public parameters", 12.5)
P("No proprietary or real tick data is used. The simulator (code/execution_study.py) comprises: "
  "(i) an arithmetic mid-price with U-shaped intraday volatility/volume seasonality, with "
  "per-session volatility drawn from three regimes (annualised sigma in {15%, 22%, 35%}) anchored to "
  "publicly documented CAC 40 realised-volatility ranges; (ii) Almgren et al. (2005) temporary "
  "(power-law) and permanent (linear) market impact, with coefficients set to give an average TWAP "
  "IS of about 18 bps, within the 15-40 bps range reported for institutional European orders "
  "(Banque de France, 2022); (iii) a persistent (AR(1)) observable order-flow signal inducing "
  "partially predictable short-horizon drift, calibrated so a signal-aware oracle gains about 30% "
  "(within the 20-40% range of Cartea et al., 2018) - this signal, invisible to the static "
  "baselines, is the honest source of any RL edge; and (iv) a multi-venue layer with heterogeneous, "
  "time-varying depth and idiosyncratic mean-zero price offsets whose dispersion is an explicit knob, "
  "so the fragmentation alpha can be switched off (a falsifiable test of H2).")
H("3.3 Strategies and agents", 12.5)
P("Five strategies: TWAP, VWAP, Almgren-Chriss (closed-form, calibrated to the average volatility "
  "and therefore mis-specified off-average), a DQN-family value-based discrete-action agent, and a "
  "PPO-family policy-gradient continuous-action agent. Both agents are controlled, reproducible "
  "instances of the DQN/PPO families of Chapter 2: linear policies over the Chapter-2 state features "
  "(residual inventory, time, price-vs-arrival S_t/S_0, the order-flow signal, the volatility "
  "regime), optimised by direct policy search (Cross-Entropy Method) on the simulated IS objective - "
  "a policy-optimisation method in the family of Moody & Saffell (2001) and the policy-improvement "
  "principle of PPO. Deep-network function approximation is the scalable extension (see Limitations); "
  "the discrete agent is handicapped only by action discretisation, the mechanism behind H1a.")
H("3.4 Evaluation protocol", 12.5)
P("Each strategy executes a fixed buy order over a 30-minute, 30-step window. Training and "
  "evaluation use disjoint seeds; all figures are out-of-sample on N = 12,000 sessions. Strategies "
  "are compared paired (common random numbers), with Newey-West (1987) HAC t-statistics (lag 5) and "
  "10,000-resample paired-bootstrap 95% confidence intervals. H3 partitions the test set by "
  "volatility regime a priori.")

# ---------------- Chapter 4 ----------------
pdf.add_page()
H("Chapter 4 (revised) - Results")
H1 = R["H1"]
def pct_vs_twap(k):
    tw = H1["TWAP"]["mean"]; return f"{100*(1-H1[k]['mean']/tw):+.1f}%"
H("4.1 Global performance - testing H1", 12.5)
table(["Strategy", "Mean IS", "Median", "VaR95", "vs TWAP"],
      [[k, f"{H1[k]['mean']:.2f}", f"{H1[k]['median']:.2f}", f"{H1[k]['var95']:.1f}",
        ("-" if k=="TWAP" else pct_vs_twap(k))]
       for k in ["TWAP","VWAP","Almgren-Chriss","DQN","PPO"]],
      widths=[42, 30, 28, 28, 32])
P(f"PPO reduces mean simulated IS by {t['PPO_vs_TWAP']['pct']:.1f}% vs TWAP (t = {t['PPO_vs_TWAP']['tstat']:.0f}), "
  f"{t['PPO_vs_VWAP']['pct']:.1f}% vs VWAP and {t['PPO_vs_Almgren-Chriss']['pct']:.1f}% vs Almgren-Chriss, all "
  f"highly significant. DQN reduces IS by {t['DQN_vs_TWAP']['pct']:.1f}% / {t['DQN_vs_VWAP']['pct']:.1f}% / "
  f"{t['DQN_vs_Almgren-Chriss']['pct']:.1f}%. PPO beats DQN by {t['PPO_vs_DQN']['pct']:.1f}% (t = {t['PPO_vs_DQN']['tstat']:.0f}), "
  f"validating H1a; the reduction vs TWAP exceeds that vs VWAP, validating H1b. Almgren-Chriss does not beat TWAP on "
  f"MEAN IS (as theory predicts for this impact model) but does cut TAIL risk (VaR95 86.9 vs 97.3).")
P("Honest nuance (reported, not hidden): the PPO mean (11.68) is well below its median (16.87). The advantage is "
  "CONCENTRATED in sessions with a strong exploitable signal and high volatility, not uniform; the median reduction "
  "vs TWAP is only ~5%, while the mean reduction (34%) reflects large savings in the favourable sessions. This "
  "concentration is precisely the content of H3.")
figure("fig_is_distribution.png", 150, "Figure 4.1 - Out-of-sample IS distribution by strategy (simulated; box = IQR, whiskers 5-95%).")

pdf.add_page()
H("4.2 Regime conditionality - testing H3", 12.5)
h3 = R["H3"]; h3t = R["H3_tests"]
table(["Strategy", "sigma=15%", "sigma=22%", "sigma=35%"],
      [[k, f"{h3[k]['0.15']:.2f}", f"{h3[k]['0.22']:.2f}", f"{h3[k]['0.35']:.2f}"]
       for k in ["Almgren-Chriss","DQN","PPO"]] +
      [["PPO vs AC", f"{h3t['0.15']['pct']:+.1f}%", f"{h3t['0.22']['pct']:+.1f}%", f"{h3t['0.35']['pct']:+.1f}%"]],
      widths=[44, 38, 38, 38])
P(f"The PPO advantage over Almgren-Chriss rises monotonically with volatility, from "
  f"{h3t['0.15']['pct']:.1f}% (calm) to {h3t['0.35']['pct']:.1f}% (stressed), strongly confirming H3. "
  f"Correction to sub-hypothesis H3b: we had hypothesised NO significant edge in calm markets; the data "
  f"REJECT this - a small ({h3t['0.15']['pct']:.1f}%) but highly significant edge persists at sigma = 15%, "
  f"because the order-flow signal remains exploitable even in low volatility. We report this honestly as a "
  f"rejected sub-hypothesis.")
figure("fig_regime.png", 150, "Figure 4.2 - Mean IS by strategy and volatility regime (simulated).")

H("4.3 Fragmentation alpha - testing H2", 12.5)
rows = R["H2"]["by_dispersion"]
table(["Dispersion (bps)"] + [f"{r['dispersion']*1e4:.0f}" for r in rows],
      [["Alpha (bps)"] + [f"{r['alpha_bps']:.2f}" for r in rows],
       ["% of single-venue IS"] + [f"{r['alpha_pct']:.0f}%" for r in rows]],
      widths=[46] + [22.8]*len(rows), fs=9)
P("Comparing dynamic three-venue routing against a single pooled venue of identical total capacity on common "
  "scenarios, the fragmentation alpha is EXACTLY ZERO when cross-venue price dispersion is zero and rises "
  "monotonically with it - a clean, falsifiable confirmation that the multi-venue gain genuinely comes from "
  "exploiting cross-venue dispersion, not relabelled temporal optimisation. At a moderate ~8 bp dispersion the "
  "routing captures ~35% of single-venue IS, consistent with Lehalle & Laruelle (2013) and Gresse (2017). The very "
  "large values at 16-32 bp dispersion (simulated IS turning negative) are outside any realistic range and only "
  "trace the mechanism.")
figure("fig_h2_depth.png", 140, "Figure 4.3 - Fragmentation alpha vanishes as cross-venue dispersion -> 0 (H2).")

pdf.add_page()
H("4.4 Robustness", 12.5)
rb = R["robustness"]
labels = {"baseline":"Baseline (linear, Gaussian)","sqrt_impact":"Square-root impact",
          "student_t":"Student-t shocks","high_impact":"Higher impact","ood_shift":"Out-of-distribution shift"}
table(["Variant","PPO vs TWAP","PPO vs AC"],
      [[labels[k], f"{rb[k]['ppo_vs_twap']:.1f}%", f"{rb[k]['ppo_vs_ac']:.1f}%"] for k in labels],
      widths=[80, 40, 40])
P("The PPO advantage is sign-stable across every variant, including an out-of-distribution test where the signal "
  "persistence and reference volatility are shifted at test time (the agent degrades but stays well ahead). The "
  "magnitude varies materially with the impact specification (it roughly halves under square-root or higher impact), "
  "which is exactly why only signs and rankings are interpreted, not basis-point levels.")
figure("fig_robustness.png", 150, "Figure 4.4 - Robustness of the PPO advantage across model assumptions (simulated).")

H("4.5 Discussion and managerial implications", 12.5)
P("These results reproduce, in a controlled model, the central mechanisms claimed in the literature: a signal-aware "
  "adaptive policy beats signal-blind schedules (Cartea et al., 2018), the gain is regime-dependent (Cartea & "
  "Jaimungal, 2015), and cross-venue dispersion is an exploitable source of execution alpha (Lehalle & Laruelle, "
  "2013). The managerial reading is conditional, not promotional: RL-style adaptive execution is most valuable for "
  "large orders in volatile regimes and genuinely fragmented names, and adds little in calm, concentrated conditions "
  "where Almgren-Chriss is already near-optimal and far easier to govern. As an in-model illustration (NOT a "
  "deployable estimate), the 5.96 bp mean IS reduction vs TWAP on EUR 2bn annual turnover would correspond to about "
  "EUR 1.2m per year - a figure bounded by every limitation below, above all the simulation-to-reality gap.")

# ---------------- Limitations ----------------
pdf.add_page()
H("Limitations and boundary conditions")
for i, txt in enumerate([
  "This is a mechanism study, not a backtest. All results are simulated IS under a stated model; we interpret only "
  "the sign, ranking and comparative statics of effects, never the bp magnitudes as deployable figures.",
  "Magnitudes are model artefacts. The IS level and every reduction depend on the chosen impact and signal "
  "parameters; the robustness analysis shows the RANKING is stable, but the MAGNITUDES must not be quoted out of context.",
  "The RL edge is, by construction, the value of an observable signal and of conditioning on current price - both "
  "ignored by the static baselines. A weaker, noisier or non-stationary signal shrinks the edge.",
  "Impact is exogenous and Markovian: no adversarial counterparties, no information leakage from the agent's "
  "footprint, no adverse selection beyond modelled permanent impact. Live performance would be lower.",
  "Train and test share the data-generating process. Out-of-sample here means unseen draws, not a different market; "
  "generalisation to reality is not demonstrated (the OOD parameter-shift test only partially probes it).",
  "H2 fragmentation alpha is defined by the injected dispersion: genuine within the model (zero when dispersion -> 0) "
  "but its magnitude is a property of the venue model, not a measurement of the real CAC 40.",
  "Data constraint: synchronised multi-venue Level-2 CAC 40 data is a paid institutional product and was "
  "cost-prohibitive; free sources give single-venue bars (no quotes) or delayed trades only - the honest reason no "
  "real-data IS backtest was performed.",
  "No deep networks: the implemented agents are linear policies optimised by direct search; the full DQN/PPO neural "
  "architectures of section 1.4.2 are a scalable extension, not implemented here.",
]):
    pdf.set_font("Times", "B", 11.5); pdf.cell(7, 6.2, f"{i+1}.")
    pdf.set_font("Times", "", 11.5); pdf.multi_cell(0, 6.2, txt, align="J"); pdf.ln(0.5)

H("Note on integration with the main dissertation", 12.5)
P("This supplement replaces the provisional Chapter 4 and the data section of Chapter 3. Across the rest of the "
  "document, every reference to 'real tick-by-tick Level 2 / Refinitiv / Bloomberg' data must be reworded to the "
  "calibrated simulator; the fabricated 43% / 33% / 7% headline figures and the impossible-dated interviews must be "
  "removed; and the dangling 'Section 3.7' reference, duplicated sections 4.4-4.6, and appendix ordering must be "
  "fixed. See REWRITE_Ch3-4_HONEST.md for the exact edit-list.")

pdf.output(OUT)
print("written", OUT, "pages:", pdf.page_no())
