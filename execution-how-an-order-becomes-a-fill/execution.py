# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "matplotlib"]
# ///
"""Numbers and chart for the post "Execution: how an order becomes a fill, and what it costs on the way".

Two parts. Made-up data: one stock with a made-up day (six trading intervals, a volume profile,
a price path and an order book), and a made-up order to buy 20,000 shares, to show benchmarks,
TWAP and VWAP schedules, slicing against one big market order, the passive-or-aggressive choice,
queue priority, order book imbalance and the micro-price. The price file has no intraday prices,
volume or quotes, so every number in that part is made up and labelled so. Real data: the five
tickers' daily returns, to measure how far a price wanders while an order waits, and to check the
square-root-of-time rule that the post uses for horizons shorter than a day.
Run:  uv run examples/execution-how-an-order-becomes-a-fill/execution.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chartstyle as cs  # noqa: E402
import market_data as md  # noqa: E402

OUT = cs.out_dir("execution-how-an-order-becomes-a-fill")


def p(label, value):
    print(f"{label:<58} {value}")


# --- made-up day ------------------------------------------------------------------------------------
Q = 20_000                                     # parent order: buy 20,000 shares
ARRIVAL = 50.00                                # mid price when the decision is made
HALF = 0.02                                    # half the spread: bid 49.98, ask 50.02
VOL = np.array([25, 15, 10, 10, 15, 25]) / 100 * 1_000_000   # a U-shaped day, 1 million shares
MID = np.array([50.00, 50.04, 50.06, 50.05, 50.08, 50.12])   # the mid in each interval (made up)

twap = np.full(6, Q / 6)
vwap = Q * VOL / VOL.sum()
print("== schedules (shares per interval)")
p("volume profile", VOL.astype(int).tolist())
p("TWAP slices", np.round(twap).astype(int).tolist())
p("VWAP slices", np.round(vwap).astype(int).tolist())

print("== benchmarks (every slice crosses the spread: fills at mid + 2 cents)")
fill = MID + HALF
market_vwap = (MID * VOL).sum() / VOL.sum()            # what the market paid, at the mid
market_twap = MID.mean()
p("market VWAP over the day", f"{market_vwap:.4f}")
p("market TWAP over the day", f"{market_twap:.4f}")
for name, q in (("TWAP schedule", twap), ("VWAP schedule", vwap)):
    avg = (fill * q).sum() / q.sum()
    slip = avg - ARRIVAL                               # a buy: paying more is a cost
    p(f"{name}: average fill", f"{avg:.4f}")
    p(f"{name}: vs arrival (cents, $, bps)",
      f"{100 * slip:.2f}c  ${slip * Q:,.0f}  {1e4 * slip / ARRIVAL:.1f} bps")
    p(f"{name}: vs market VWAP (cents)", f"{100 * (avg - market_vwap):+.2f}")
    p(f"{name}: vs market TWAP (cents)", f"{100 * (avg - market_twap):+.2f}")

print("== one market order against six slices (2,000 shares at every cent, book refills between slices)")
DEPTH = 2_000


def walk(shares, first=HALF * 100):
    """Average cents above the mid paid by a market buy that empties level after level."""
    full, rest = divmod(shares, DEPTH)
    cents = sum(DEPTH * (first + k) for k in range(int(full))) + rest * (first + full)
    return cents / shares


p("one order of 20,000: average cents above the mid", f"{walk(Q):.2f}")
p("six slices of 3,333: average cents above the mid", f"{walk(Q / 6):.2f}")
p("cents saved per share by slicing", f"{walk(Q) - walk(Q / 6):.2f}")
p("dollars saved on the order", f"${(walk(Q) - walk(Q / 6)) / 100 * Q:,.0f}")

print("== passive or aggressive (one share, cents against buying at the ask now)")
SAVE, MISS = 4, 6    # filled at the bid saves the whole spread; a miss buys later 6 cents higher
for pf in (0.4, 0.6, 0.8):
    p(f"fill probability {pf:.0%}: expected gain (cents)", f"{pf * SAVE - (1 - pf) * MISS:+.1f}")
p("break-even fill probability", f"{MISS / (SAVE + MISS):.0%}")

print("== queue priority at the best bid (price first, then time)")
AHEAD, MINE, SELL = 12_000, 5_000, 15_000
filled = min(max(SELL - AHEAD, 0), MINE)
p("shares ahead, mine, incoming market sell", f"{AHEAD:,} {MINE:,} {SELL:,}")
p("my shares filled", f"{filled:,}")

print("== the book's imbalance")
db, ds, pb, ps = 8_000, 2_000, 49.98, 50.02
obi = (db - ds) / (db + ds)
I = db / (db + ds)
p("OBI", f"{obi:.2f}")
p("I = d_b / (d_b + d_s)", f"{I:.2f}")
p("micro-price", f"{I * ps + (1 - I) * pb:.3f}")
p("mid", f"{(pb + ps) / 2:.3f}")

# --- real data: how far the price wanders while you wait --------------------------------------------
try:
    px = md.prices()
except (OSError, KeyError) as e:
    sys.exit(f"cannot read the prices file data/prices.csv: {e}")
r = md.daily_returns(px)
DAYS = np.arange(1, 11)
print("== spread of the d-day return (%), measured on overlapping windows, and sigma * sqrt(d)")
measured = {t: np.array([100 * (px[t] / px[t].shift(d) - 1).std() for d in DAYS]) for t in px.columns}
for t in px.columns:
    s1 = 100 * r[t].std()
    p(f"{t}: 1 day / 5 days measured / 5 days sqrt rule", f"{s1:.2f} / {measured[t][4]:.2f} / {s1 * np.sqrt(5):.2f}")
s_spy = r["SPY"].std()
p("SPY-like stock at $50: one interval (1/6 day), cents", f"{100 * 50 * s_spy * np.sqrt(1 / 6):.0f}")
p("SPY-like stock at $50: one day, cents", f"{100 * 50 * s_spy:.0f}")

# --- chart ------------------------------------------------------------------------------------------
cs.apply()
fig, axes = plt.subplots(2, 1, figsize=(8, 3.8 * 2))
ax = axes[0]
k = np.arange(1, 7)
ax.plot(k, vwap, marker="o", color=cs.MAROON, label="VWAP schedule, follows the volume forecast")
ax.plot(k, twap, marker="o", color=cs.BLUE, label="TWAP schedule, equal slices")
ax.set_xticks(k)
ax.set_ylim(0, 6_000)
ax.set_xlabel("Trading interval of the made-up day (1 = the open, 6 = the close)")
ax.set_ylabel("Shares bought in the interval")
cs.title(ax, "Two ways to slice 20,000 shares over a made-up day", 1)
ax.legend()

ax = axes[1]
for t, c in (("NVDA", cs.MAROON), ("SPY", cs.BLUE)):
    ax.plot(DAYS, measured[t], marker="o", linestyle="none", color=c, label=f"{t}, measured")
    ax.plot(DAYS, 100 * r[t].std() * np.sqrt(DAYS), color=c, alpha=0.6,
            label=f"{t}, one-day move times the square root of the days")
ax.set_xticks(DAYS)
ax.set_ylim(0, measured["NVDA"].max() * 1.15)
ax.set_xlabel("Days the order waits")
ax.set_ylabel("Typical price move, one standard deviation (%)")
cs.title(ax, "How far the price wanders while an order waits, 2020 to 2025", 2)
ax.legend()
cs.save(fig, OUT / "execution.svg")
