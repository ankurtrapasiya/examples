# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "matplotlib"]
# ///
"""Numbers and chart for the post "Bollinger Bands: an average with a margin that breathes".

The 20-day simple moving average of SPY's close with bands two standard deviations above and
below it, one day worked by hand, how often the close sits outside the bands, how the band width
moved through 2020, and the link between %B and the "Bollinger position" of the earlier
indicators post. Concept only: no trading rule is run here.
Run:  uv run examples/bollinger-bands/bollinger.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chartstyle as cs  # noqa: E402
import market_data as md  # noqa: E402

OUT = cs.out_dir("bollinger-bands")
HAND_DAY = "2020-03-16"
N, D = 20, 2


def p(label, value):
    print(f"{label:<60} {value}")


px = md.prices(["SPY"])["SPY"]
mid = px.rolling(N).mean()
sd = px.rolling(N).std()              # sample standard deviation (n-1), pandas' default
upper, lower = mid + D * sd, mid - D * sd
pct_b = (px - lower) / (upper - lower)
width = (upper - lower) / mid

# --- one day by hand -------------------------------------------------------------------------
w = px.loc[:HAND_DAY].iloc[-N:]
print(f"Bollinger Bands on {HAND_DAY}, by hand ({N} days, {D} standard deviations)")
p("  close", f"{w.iloc[-1]:.2f}")
p("  middle band: mean of the last 20 closes", f"{w.mean():.2f}")
p("  standard deviation of those closes", f"{w.std():.2f}")
p("  upper band: middle + 2 sd", f"{w.mean() + D * w.std():.2f}")
p("  lower band: middle - 2 sd", f"{w.mean() - D * w.std():.2f}")
p("  %B: (close - lower) / (upper - lower)", f"{pct_b[HAND_DAY]:.3f}")
p("  position in two-sd units (earlier post)", f"{(w.iloc[-1] - w.mean()) / (D * w.std()):+.2f}")
p("  band width: (upper - lower) / middle", f"{100 * width[HAND_DAY]:.1f}%")
p("  population sd would be", f"{w.std(ddof=0):.2f}")

# --- how often the close is outside the bands ------------------------------------------------
ok = pct_b.dropna()
print()
p("days with a band value", len(ok))
p("close above the upper band", f"{(ok > 1).sum()}  ({100 * (ok > 1).mean():.1f}%)")
p("close below the lower band", f"{(ok < 0).sum()}  ({100 * (ok < 0).mean():.1f}%)")
p("inside the bands", f"{100 * ((ok >= 0) & (ok <= 1)).mean():.1f}%")

# --- the width through 2020 -------------------------------------------------------------------
w20 = width["2020"]
print()
p("narrowest band width in 2020-2025", f"{100 * width.min():.1f}% on {width.idxmin().date()}")
p("widest band width in 2020-2025", f"{100 * width.max():.1f}% on {width.idxmax().date()}")
p("band width on 2020-02-19 (the high before the crash)", f"{100 * width['2020-02-19']:.1f}%")
p("band width on 2020-03-16", f"{100 * width['2020-03-16']:.1f}%")
p("band width on 2020-08-31", f"{100 * width['2020-08-31']:.1f}%")

# --- chart ----------------------------------------------------------------------------------
cs.apply()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 3.8 * 2))
win = slice("2020-01-01", "2020-08-31")
ax1.plot(px[win].index, px[win], color=cs.MAROON, label="SPY close")
ax1.plot(mid[win].index, mid[win], color=cs.BLUE, label="20-day average (middle band)")
ax1.plot(upper[win].index, upper[win], color=cs.GREY, label="upper band (average + 2 sd)")
ax1.plot(lower[win].index, lower[win], color=cs.GREY, label="lower band (average - 2 sd)")
ax1.set_xlabel("Date (2020)")
ax1.set_ylabel("Price (dollars)")
cs.title(ax1, "The bands open in the March crash and close again by summer", 1)
ax1.legend()

ax2.plot(pct_b[win].index, pct_b[win], color=cs.MAROON, label="%B (0 = lower band, 1 = upper band)")
ax2.axhline(1, color=cs.GREY, lw=0.8, ls="--")
ax2.axhline(0, color=cs.GREY, lw=0.8, ls="--")
ax2.set_xlabel("Date (2020)")
ax2.set_ylabel("%B (fraction of the band)")
cs.title(ax2, "Where the close sits inside the band, day by day", 2)
ax2.legend()

cs.save(fig, OUT / "bollinger.svg")
print(f"\nchart -> {OUT / 'bollinger.svg'}")
