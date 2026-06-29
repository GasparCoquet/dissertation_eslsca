"""
real_data_check.py
Limited REAL-DATA sanity check using free public data (Yahoo Finance).

Free OHLCV bars carry no order book, so a real Implementation Shortfall (which is
dominated by market impact) cannot be reconstructed from them. What free data CAN
anchor are the two simulator calibration choices that drive the results:
  (1) the volatility regimes {15%, 22%, 35%} used in the simulator, and
  (2) the U-shaped intraday volume profile assumed by VWAP and the seasonality term.

Outputs: code/figures/fig_real_vol.png, fig_real_volume_profile.png
         code/results/real_data_check.json
"""
import os, json, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yfinance as yf
warnings.filterwarnings("ignore")

HERE = os.path.dirname(__file__)
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
RES = os.path.join(HERE, "results"); os.makedirs(RES, exist_ok=True)

NAMES = ["MC.PA", "OR.PA", "SAN.PA", "AIR.PA", "BNP.PA", "TTE.PA", "SU.PA", "AI.PA"]


def annualised_vol(close):
    r = np.log(close.astype(float)).diff().dropna()
    return float(r.std()) * np.sqrt(252) * 100.0


def main():
    out = {"names": NAMES}

    # ---- (1) realized volatility per name + pooled rolling 21d vol ----
    per_name, rolling_all = {}, []
    for t in NAMES:
        try:
            h = yf.Ticker(t).history(period="2y", interval="1d", auto_adjust=True)
            if len(h) < 60:
                continue
            c = h["Close"]
            per_name[t] = round(annualised_vol(c), 1)
            r = np.log(c.astype(float)).diff().dropna()
            roll = r.rolling(21).std() * np.sqrt(252) * 100.0
            rolling_all.append(roll.dropna().values)
        except Exception as e:
            print("skip", t, type(e).__name__)
    roll = np.concatenate(rolling_all) if rolling_all else np.array([])
    out["realised_vol_pct_per_name"] = per_name
    if roll.size:
        out["rolling21_vol_pct"] = dict(
            min=round(float(roll.min()), 1), p10=round(float(np.percentile(roll, 10)), 1),
            median=round(float(np.median(roll)), 1), p90=round(float(np.percentile(roll, 90)), 1),
            max=round(float(roll.max()), 1),
            frac_in_15_35=round(float(np.mean((roll >= 15) & (roll <= 35))), 3))
    print("per-name annualised vol (%):", per_name)
    print("rolling 21d vol summary:", out.get("rolling21_vol_pct"))

    # plot vol distribution with the three regime lines
    if roll.size:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.hist(roll, bins=60, color="#4c78a8", alpha=0.85)
        for v, lab in [(15, "15%"), (22, "22%"), (35, "35%")]:
            ax.axvline(v, color="#c44e52", lw=1.4)
            ax.text(v, ax.get_ylim()[1]*0.92, lab, color="#c44e52", ha="center", fontsize=9)
        ax.set_xlabel("Annualised 21-day realised volatility (%)")
        ax.set_ylabel("Frequency (CAC 40 names, 2y)")
        ax.set_title("Real CAC 40 realised volatility vs simulator regimes")
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_real_vol.png"), dpi=150)
        plt.close(fig)

    # ---- (2) intraday volume profile (U-shape) ----
    prof = {}
    for t in NAMES[:4]:
        try:
            h = yf.Ticker(t).history(period="5d", interval="15m", auto_adjust=True)
            if not len(h):
                continue
            h = h.copy()
            h["mod"] = h.index.strftime("%H:%M")
            g = h.groupby("mod")["Volume"].mean()
            g = g / g.sum()
            for k, v in g.items():
                prof.setdefault(k, []).append(float(v))
        except Exception as e:
            print("intraday skip", t, type(e).__name__)
    if prof:
        keys = sorted(prof.keys())
        vals = [float(np.mean(prof[k])) for k in keys]
        out["intraday_volume_profile"] = {k: round(v, 4) for k, v in zip(keys, vals)}
        u = vals[0] + vals[-1] > 2 * (sum(vals)/len(vals))  # ends heavier than mean
        out["u_shape_confirmed"] = bool(u)
        print("intraday U-shape (ends>mean):", u)
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        ax.bar(range(len(keys)), vals, color="#4c78a8")
        ax.set_xticks(range(0, len(keys), max(1, len(keys)//8)))
        ax.set_xticklabels([keys[i] for i in range(0, len(keys), max(1, len(keys)//8))], fontsize=8)
        ax.set_xlabel("Time of day (Europe/Paris)")
        ax.set_ylabel("Share of daily volume")
        ax.set_title("Real CAC 40 intraday volume profile (U-shape)")
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_real_volume_profile.png"), dpi=150)
        plt.close(fig)

    with open(os.path.join(RES, "real_data_check.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", os.path.join(RES, "real_data_check.json"))


if __name__ == "__main__":
    main()
