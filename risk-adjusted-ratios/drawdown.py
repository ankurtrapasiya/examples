# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "matplotlib"]
# ///
"""Drawdown depth and duration for the five tickers, and a chart of SPY's, for the drawdown
section of the ratios post.  Run:  uv run examples/risk-adjusted-ratios/drawdown.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chartstyle as cs  # noqa: E402
import market_data as md  # noqa: E402

OUT = cs.out_dir("risk-adjusted-ratios")
K = np.sqrt(252)

px = md.prices()
r = md.daily_returns(px)
print(f"{'ticker':6} {'max drawdown':>13} {'peak':>11} {'trough':>11} {'recovered':>11} {'days under water':>17} {'longest':>8} {'sharpe':>7}")
for t in md.TICKERS:
    w = (1 + r[t]).cumprod(); peak = w.cummax(); dd = w / peak - 1
    trough = dd.idxmin(); pk = w.loc[:trough].idxmax()
    after = w.loc[trough:]; rec = after[after >= peak.loc[trough]].index
    rec_date = rec[0] if len(rec) else None
    under = (dd < 0)
    # longest stretch below the previous peak
    runs, cur = [], 0
    for v in under:
        cur = cur + 1 if v else 0; runs.append(cur)
    ex = r[t] - 0.02 / 252
    print(f"{t:6} {dd.min():13.1%} {str(pk.date()):>11} {str(trough.date()):>11} {str(rec_date.date()) if rec_date is not None else 'never':>11} {len(w.loc[pk:rec_date]) - 1 if rec_date is not None else -1:17d} {max(runs):8d} {K * ex.mean() / ex.std():7.2f}")

spy = (1 + r["SPY"]).cumprod(); peak = spy.cummax(); dd = spy / peak - 1
cs.apply()
fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 3.8 * 2))
cs.title(a1, "SPY: value of 1 dollar, and the highest value so far", 1)
a1.plot(spy.index, spy, color=cs.MAROON, label="value of 1 dollar in SPY")
a1.plot(peak.index, peak, color=cs.GREY, lw=1, label="highest value so far (the peak)")
a1.fill_between(spy.index, spy, peak, color=cs.MAROON, alpha=0.12, lw=0)
a1.set_xlabel("Date"); a1.set_ylabel("Value (dollars)"); a1.legend()
cs.title(a2, "The drawdown: how far below the peak, day by day", 2)
a2.fill_between(dd.index, 100 * dd, 0, color=cs.MAROON, alpha=0.25, lw=0)
a2.plot(dd.index, 100 * dd, color=cs.MAROON, lw=1, label="drawdown from the peak (%)")
a2.axhline(100 * dd.min(), color=cs.BLUE, lw=1, ls="--", label=f"maximum drawdown, {100 * dd.min():.1f}%")
a2.set_xlabel("Date"); a2.set_ylabel("Below the peak (%)"); a2.legend()
cs.save(fig, OUT / "drawdown.svg")
