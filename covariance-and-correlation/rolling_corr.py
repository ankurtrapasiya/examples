# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "matplotlib"]
# ///
"""Rolling correlations and the crash-year effect, for the "correlations rise in a crash" section
of the covariance post.  Run:  uv run examples/covariance-and-correlation/rolling_corr.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chartstyle as cs  # noqa: E402
import market_data as md  # noqa: E402

OUT = cs.out_dir("covariance-and-correlation")
r = md.daily_returns(md.prices())
stocks = ["SPY", "NVDA", "MS", "OXY"]
for yr in ["2020", "2021", "2022", "2023", "2024", "2025"]:
    sub = r.loc[yr, stocks]; c = sub.corr().values[np.triu_indices(4, 1)].mean()
    mix = sub.mean(axis=1)
    print(f"{yr}: avg pairwise corr {c:.2f}; sigma of each " + " ".join(f"{t}:{sub[t].std():.2%}" for t in stocks) + f"; equal mix {mix.std():.2%}; avg of the three stocks {sub[['NVDA','MS','OXY']].std().mean():.2%}")
pairs = [("SPY", "NVDA"), ("SPY", "GLD"), ("MS", "OXY")]
roll = {f"{a}-{b}": r[a].rolling(60).corr(r[b]) for a, b in pairs}
for k, s in roll.items():
    print(f"rolling 60-day {k}: min {s.min():.2f} (on {s.idxmin().date()}), max {s.max():.2f} (on {s.idxmax().date()}), on 2020-03-31 {s.loc['2020-03-31']:.2f}")

cs.apply()
fig, ax = plt.subplots(1, 1, figsize=(8, 3.8))
cs.title(ax, "Correlation is not a constant: 60-day rolling correlation of three pairs")
for (k, s), col in zip(roll.items(), [cs.MAROON, cs.BLUE, cs.AMBER]):
    ax.plot(s.index, s, color=col, lw=1.2, label=k)
ax.axhline(0, color=cs.GREY, lw=0.8)
ax.set_xlabel("Date (end of the 60-day window)"); ax.set_ylabel("Correlation of daily returns"); ax.legend()
cs.save(fig, OUT / "rolling-correlation.svg")
