# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "matplotlib"]
# ///
"""The "two tools" chart for the post "The data is part of the model": a fitted line with its
misses, and how squaring the misses hands most of the total to the biggest one.

The five SPY months are the ones the linear regression post works by hand (this month's
volatility against next month's, both % a year, rounded to whole points).
Run:  uv run examples/data-is-part-of-the-model/two_tools.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chartstyle as cs  # noqa: E402

OUT = cs.out_dir("data-is-part-of-the-model")
months = ["Jun 2020", "Aug 2020", "Oct 2020", "Dec 2020", "Feb 2021"]
x = np.array([20, 13, 24, 16, 17], dtype=float)    # this month's volatility
y = np.array([30, 8, 21, 8, 16], dtype=float)      # next month's

slope = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()
intercept = y.mean() - slope * x.mean()
pred = intercept + slope * x
miss = y - pred
print(f"line: next = {intercept:.2f} + {slope:.2f} x this month")
for m, xi, yi, pi, e in zip(months, x, y, pred, miss):
    print(f"  {m}: actual {yi:.0f}, predicted {pi:.2f}, miss {e:+.2f}, squared {e * e:.2f}")
print(f"MAE {np.abs(miss).mean():.2f}  MSE {(miss ** 2).mean():.2f}  RMSE {np.sqrt((miss ** 2).mean()):.2f}")
share_abs = np.abs(miss) / np.abs(miss).sum() * 100
share_sq = miss ** 2 / (miss ** 2).sum() * 100
print("share of the total, plain / squared:", np.round(share_abs), np.round(share_sq, 1))

cs.apply()
fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 3.8 * 2))

grid = np.linspace(11, 26, 50)
a1.plot(grid, intercept + slope * grid, color=cs.BLUE, lw=1.6, label="the fitted line")
for xi, yi, pi in zip(x, y, pred):
    a1.plot([xi, xi], [pi, yi], color=cs.AMBER, lw=1.4, ls=":", label="_miss")
a1.plot([], [], color=cs.AMBER, lw=1.4, ls=":", label="a miss: actual minus predicted")
a1.scatter(x, y, s=46, color=cs.MAROON, zorder=3, label="five SPY months")
for m, xi, yi, e in zip(months, x, y, miss):
    a1.annotate(f"{m}\nmiss {e:+.1f}", (xi, yi), textcoords="offset points", xytext=(8, -4),
                fontsize=8.5, color=cs.INK)
a1.set_xlim(11, 27)
a1.set_xlabel("This month's volatility (% a year)")
a1.set_ylabel("Next month's volatility (% a year)")
a1.legend()
cs.title(a1, "A fitted line, and how far each month is from it", 1)

pos = np.arange(len(months))
a2.bar(pos - 0.2, share_abs, width=0.4, color=cs.BLUE, label="misses counted as they are (MAE)")
a2.bar(pos + 0.2, share_sq, width=0.4, color=cs.MAROON, label="misses squared first (MSE, RMSE)")
a2.set_xticks(pos, months)
a2.set_xlabel("Month")
a2.set_ylabel("Share of the total miss (%)")
a2.legend()
cs.title(a2, "Squaring hands the biggest miss most of the total", 2)

cs.save(fig, OUT / "two-tools.svg")
