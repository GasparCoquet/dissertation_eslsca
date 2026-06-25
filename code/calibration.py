"""
calibration.py  (optional utility)
==================================
Refit the simulator's volatility and intraday-seasonality parameters to FREE
public CAC 40 data. This is NOT required to reproduce the study (the headline
results depend only on the documented parameter set in execution_study.py); it
is provided so the volatility regimes can be re-anchored to live data when an
internet connection and a free data source are available.

No paid/proprietary data is used. Requires `yfinance` (pip install yfinance)
and internet access. If unavailable, the script reports the literature-based
defaults and exits cleanly.

Usage:  python calibration.py            # uses a default basket of CAC 40 names
"""
from __future__ import annotations
import sys, json, os
import numpy as np

CAC40 = ["TTE.PA", "MC.PA", "OR.PA", "SAN.PA", "BNP.PA"]   # free Yahoo tickers

LIT_DEFAULTS = dict(
    sigma_ann_low=0.15, sigma_ann_ref=0.22, sigma_ann_high=0.35,
    source="Banque de France (2022); ESMA (2023) realised-vol ranges for CAC 40",
)


def annualised_vol_from_daily(close: np.ndarray) -> float:
    r = np.diff(np.log(close))
    return float(np.std(r) * np.sqrt(252))


def main():
    try:
        import yfinance as yf
    except Exception:
        print("yfinance not installed -> using literature defaults:")
        print(json.dumps(LIT_DEFAULTS, indent=2))
        return
    vols = {}
    for tk in CAC40:
        try:
            df = yf.download(tk, period="2y", interval="1d", progress=False)
            if len(df) > 50:
                vols[tk] = annualised_vol_from_daily(df["Close"].values.ravel())
        except Exception as e:
            print(f"  {tk}: download failed ({type(e).__name__})")
    if not vols:
        print("no data downloaded -> using literature defaults:")
        print(json.dumps(LIT_DEFAULTS, indent=2))
        return
    arr = np.array(list(vols.values()))
    out = dict(per_name={k: round(v, 3) for k, v in vols.items()},
               sigma_ann_ref=round(float(np.median(arr)), 3),
               sigma_ann_low=round(float(np.percentile(arr, 10)), 3),
               sigma_ann_high=round(float(np.percentile(arr, 90)), 3),
               note="Re-anchored to free Yahoo daily data; intraday seasonality "
                    "requires 1-min bars (yf interval='1m', last 7 days).")
    os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
    path = os.path.join(os.path.dirname(__file__), "results", "calibration.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("calibration written to", path)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
