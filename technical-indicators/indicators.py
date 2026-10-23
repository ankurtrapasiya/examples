# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "matplotlib"]
# ///
"""Numbers and chart for the post "Indicators: a price pattern turned into a number".

Three rolling statistics of SPY's price (a 20-day and a 100-day simple moving average, the
20-day Bollinger position, the 20-day momentum), what the lookback does to lag and to how often a
crossover flips, one Bollinger value worked by hand, and how much three indicators built from the
same 20 days of prices agree with each other. Concept only: no trading rule is run here.
Run:  uv run examples/technical-indicators/indicators.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chartstyle as cs  # noqa: E402
import market_data as md  # noqa: E402

OUT = cs.out_dir("technical-indicators")
HAND_DAY = "2020-03-16"   # the day worked by hand in the post


def p(label, value):
    print(f"{label:<66} {value}")


px = md.prices(["SPY"])["SPY"]
sma20, sma100 = px.rolling(20).mean(), px.rolling(100).mean()
sd20 = px.rolling(20).std()
boll = (px - sma20) / (2 * sd20)            # how many two-standard-deviation bands the price is from its average
mom20 = px / px.shift(20) - 1               # 20-day momentum: the return over the last 20 days
pct = lambda x: f"{100 * x:+.2f}%"  # noqa: E731

# --- one Bollinger value by hand ------------------------------------------------------------
w = px.loc[:HAND_DAY].iloc[-20:]
print(f"Bollinger position on {HAND_DAY}, by hand")
p("  last 20 closes, first and last", f"{w.iloc[0]:.2f} ... {w.iloc[-1]:.2f}")
p("  20-day average of the close", f"{w.mean():.2f}")
p("  20-day standard deviation of the close", f"{w.std():.2f}")
p("  close minus average", f"{w.iloc[-1] - w.mean():+.2f}")
p("  divided by two standard deviations", f"{(w.iloc[-1] - w.mean()) / (2 * w.std()):+.2f}")
p("  20-day momentum the same day", pct(mom20[HAND_DAY]))
p("  price over the 20-day average, minus one", pct(px[HAND_DAY] / sma20[HAND_DAY] - 1))

# --- lag: how long after the 2020 low each average turned up --------------------------------
low = px["2020"].idxmin()
up20 = sma20[low:].diff().gt(0).idxmax()
up100 = sma100[low:].diff().gt(0).idxmax()
bd = lambda a, b: int(np.busday_count(a.date(), b.date()))  # noqa: E731
print()
p("2020 price low", low.date())
p("20-day average first rose on", f"{up20.date()}  ({bd(low, up20)} trading days later)")
p("100-day average first rose on", f"{up100.date()}  ({bd(low, up100)} trading days later)")
p("first 20-day value exists on", px.index[19].date())
p("first 100-day value exists on", px.index[99].date())

# --- noise: how often a crossover flips, for different fast windows ---------------------------
print()
print("Crossover flips, 2019 to 2023 (fast average crosses the 100-day average)")
flips = {}
for fast in (5, 10, 20, 50):
    side = np.sign(px.rolling(fast).mean() - sma100).dropna()
    flips[fast] = int((side.diff().abs() > 0).sum())
    p(f"  fast window {fast:>3} days", f"{flips[fast]} flips")

# --- redundancy: three indicators from the same 20 days -------------------------------------
ind = pd.DataFrame({"sma_ratio": px / sma20 - 1, "bollinger": boll, "momentum_20d": mom20}).dropna()
print()
print("Correlation between the three 20-day indicators, 2019 to 2023")
print(ind.corr().round(2).to_string())

# --- chart ----------------------------------------------------------------------------------
cs.apply()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 3.8 * 2))
win = slice("2020-01-01", "2020-07-31")
ax1.plot(px[win].index, px[win], color=cs.MAROON, label="SPY close")
ax1.plot(sma20[win].index, sma20[win], color=cs.BLUE, label="20-day average")
ax1.plot(sma100[win].index, sma100[win], color=cs.GREY, label="100-day average")
for d, c in ((low, cs.MAROON), (up20, cs.BLUE), (up100, cs.GREY)):
    ax1.axvline(d, color=c, lw=0.8, ls="--")
ax1.set_xlabel("Date (2020)")
ax1.set_ylabel("Price (dollars)")
cs.title(ax1, "The low, then the day each average turned up (dashed)", 1)
ax1.legend()

ax2.plot(ind[win].index, ind.loc[win, "bollinger"], color=cs.MAROON, label="Bollinger position (two-standard-deviation units)")
ax2.plot(ind[win].index, 10 * ind.loc[win, "momentum_20d"], color=cs.BLUE, label="20-day momentum (return, times 10)")
ax2.axhline(0, color=cs.GREY, lw=0.8)
ax2.set_xlabel("Date (2020)")
ax2.set_ylabel("Indicator value")
cs.title(ax2, "Two indicators built from the same 20 days move together", 2)
ax2.legend()

cs.save(fig, OUT / "indicators.svg")
print(f"\nchart -> {OUT / 'indicators.svg'}")
