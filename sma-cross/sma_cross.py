# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "matplotlib"]
# ///
"""Numbers and chart for the post "The SMA cross: two averages, and the day they swap places".

The 50-day and 200-day simple moving averages of SPY's close, every day they crossed from 2020 to
2023, how far the price had already moved by the time each cross arrived, and how many crosses
Murphy's faster pairs (5/20, 10/50) produce on the same data. Concept only: no trading rule is run.
Run:  uv run examples/sma-cross/sma_cross.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chartstyle as cs  # noqa: E402
import market_data as md  # noqa: E402

OUT = cs.out_dir("sma-cross")


def p(label, value):
    print(f"{label:<60} {value}")


def crosses(fast, slow):
    """Days the fast average moved from one side of the slow one to the other. +1 = golden, -1 = death."""
    side = np.sign(fast - slow).dropna()
    flips = side[side.diff().fillna(0) != 0]
    return flips


px = md.prices(["SPY"])["SPY"]
sma50, sma200 = px.rolling(50).mean(), px.rolling(200).mean()

# --- one day by hand: the 2020 death cross ------------------------------------------------------
x = crosses(sma50, sma200)
first = x.index[0]
print(f"The 50/200 cross worked by hand on {first.date()}")
p("  mean of the last 50 closes", f"{px.loc[:first].iloc[-50:].mean():.2f}")
p("  mean of the last 200 closes", f"{px.loc[:first].iloc[-200:].mean():.2f}")
prev = px.index[px.index.get_loc(first) - 1]
p(f"  the day before ({prev.date()}): 50-day, 200-day", f"{sma50[prev]:.2f}, {sma200[prev]:.2f}")

# --- every 50/200 cross, and what the price had done by then ------------------------------------
print()
print("Every 50/200 cross on SPY, 2020 to 2025")
print(f"  {'date':<12}{'kind':<8}{'close':>9}{'from recent extreme':>22}{'since':>13}")
for d, s in x.items():
    kind = "golden" if s > 0 else "death"
    past = px.loc[:d].iloc[-120:]                 # the previous six months
    ext = past.idxmax() if s < 0 else past.idxmin()
    move = px[d] / px[ext] - 1
    print(f"  {d.date()!s:<12}{kind:<8}{px[d]:>9.2f}{100 * move:>21.1f}%{ext.date()!s:>13}")

# --- faster pairs flip more --------------------------------------------------------------------
print()
print("Number of crosses, 2020 to 2025, by pair of windows")
for f, s in ((5, 20), (10, 50), (50, 200)):
    n = len(crosses(px.rolling(f).mean(), px.rolling(s).mean()))
    p(f"  {f}-day over {s}-day", f"{n} crosses")

# --- chart ----------------------------------------------------------------------------------
cs.apply()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 3.8 * 2))
ax1.plot(px.index, px, color=cs.MAROON, lw=0.8, label="SPY close")
ax1.plot(sma50.index, sma50, color=cs.BLUE, label="50-day average")
ax1.plot(sma200.index, sma200, color=cs.GREY, label="200-day average")
for d, s in x.items():
    ax1.axvline(d, color=cs.GREEN if s > 0 else cs.AMBER, lw=0.8, ls="--")
ax1.set_xlabel("Date")
ax1.set_ylabel("Price (dollars)")
cs.title(ax1, "Golden crosses (green) and death crosses (amber), 2020 to 2025", 1)
ax1.legend()

win = slice("2025-01-01", "2025-09-30")
ax2.plot(px[win].index, px[win], color=cs.MAROON, label="SPY close")
ax2.plot(sma50[win].index, sma50[win], color=cs.BLUE, label="50-day average")
ax2.plot(sma200[win].index, sma200[win], color=cs.GREY, label="200-day average")
for d, s in x[win].items():
    ax2.axvline(d, color=cs.GREEN if s > 0 else cs.AMBER, lw=0.8, ls="--")
ax2.set_xlabel("Date (2025)")
ax2.set_ylabel("Price (dollars)")
cs.title(ax2, "2025 up close: the death cross after the fall, the golden cross after the recovery", 2)
ax2.legend()

cs.save(fig, OUT / "sma_cross.svg")
print(f"\nchart -> {OUT / 'sma_cross.svg'}")
