# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "matplotlib"]
# ///
"""Numbers and chart for the post "Frictions and capacity: a strategy that works small can fail big".

Two parts. Real data: the moving-average crossover (5/20 and 50/200 days, hold the ticker or cash)
on the blog's five tickers, charged a cost per position change, to show how trading frequency
multiplies the cost drag. Made-up data: one small company's stock with a made-up order book,
equal depth at every cent, and a made-up rule with a fixed edge per round trip, to show how the
cost per trade grows with the fund's size until the edge is gone (capacity), and what happens
when three funds trade the same signal at once (crowding). The price file has no volume and no
quotes, so every number in the second part is made up and labelled so.
Run:  uv run examples/frictions-and-capacity-small-can-fail-big/capacity.py
"""
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chartstyle as cs  # noqa: E402
import market_data as md  # noqa: E402

OUT = cs.out_dir("frictions-and-capacity-small-can-fail-big")


def p(label, value):
    print(f"{label:<62} {value}")


# --- part 1, real data: the crossover's cost drag on five tickers ---------------------------------
try:
    px = md.prices()
except (OSError, KeyError) as e:
    sys.exit(f"cannot read the prices file data/prices.csv: {e}")
r = md.daily_returns(px)
years = len(r) / md.TRADING_DAYS
COSTS = (0.0, 0.001, 0.005, 0.01)


def crossover(t, fast, slow):
    """1 on days the rule holds the ticker (yesterday's fast average above the slow one), else 0."""
    s = px[t]
    pos = (s.rolling(fast).mean() > s.rolling(slow).mean()).astype(float).shift(1)
    pos = pos.reindex(r.index).fillna(0)
    changes = pos.diff().abs().fillna(0)          # 1 on each day the position changes
    return pos, changes


def yearly(pos, changes, t, cost):
    """Compound return a year after charging `cost` on every position change."""
    net = pos * r[t] - changes * cost
    return (1 + net).prod() ** (1 / years) - 1


def breakeven(pos, changes, t):
    """The cost per change at which the rule's whole gain is gone (bisection)."""
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if yearly(pos, changes, t, mid) > 0 else (lo, mid)
    return lo


p("years of daily returns", f"{years:.2f}  ({r.index[0].date()} to {r.index[-1].date()})")
print("\nCrossover, hold or cash; return a year at each cost per position change")
fast_curves = {}
for fast, slow in ((5, 20), (50, 200)):
    for t in md.TICKERS:
        pos, ch = crossover(t, fast, slow)
        ys = [yearly(pos, ch, t, c) for c in COSTS]
        n = int(ch.sum())
        p(f"  {fast}/{slow} {t}: {n} changes, {n / years:.1f} a year",
          "  ".join(f"{100 * c:.1f}%: {100 * y:+.1f}%" for c, y in zip(COSTS, ys))
          + f"   break-even {100 * breakeven(pos, ch, t):.2f}%"
          + f"   drag at 0.5% ~ {100 * n / years * 0.005:.1f}% a year")
        if fast == 5:
            grid = np.linspace(0, 0.01, 41)
            fast_curves[t] = (grid, [yearly(pos, ch, t, c) for c in grid])
for t in md.TICKERS:
    p(f"  buy and hold {t}, a year", f"{100 * ((1 + r[t]).prod() ** (1 / years) - 1):+.1f}%")

# --- part 2, made-up data: one small stock, one rule, four fund sizes ----------------------------
PRICE = 20.00          # made up: the stock's mid price, dollars
TICK = 0.01            # one cent between price levels
DEPTH = 20_000         # made up: shares waiting at every cent, on both sides of the book
VOLUME = 500_000       # made up: shares traded on an average day (10 million dollars)
EDGE = 0.005           # made up: the rule's gross gain per round trip, 0.5%
TRIPS = 20             # made up: round trips a year


def side_cost(shares):
    """Average distance from the mid, as a share of the price, for a market order that walks a
    book with DEPTH shares at every cent: the first DEPTH shares fill one cent away, the next
    DEPTH two cents away, and so on. The book is assumed to refill before the next order."""
    full, rest = divmod(shares, DEPTH)
    cents = DEPTH * full * (full + 1) / 2 + rest * (full + 1)   # sum of (level x shares at it)
    return cents * TICK / shares / PRICE


def round_trip(fund):
    shares = fund / PRICE
    return 2 * side_cost(shares)              # buy walks the asks, sell walks the bids


def yearly_profit(fund, crowd=1):
    """Dollars a year for one fund when `crowd` funds of the same size send the same order at the
    same moment and share the walk up the book equally."""
    cost = 2 * side_cost(crowd * fund / PRICE)
    return TRIPS * (EDGE - cost) * fund


print("\nMade-up stock: mid $20.00, 20,000 shares at every cent, 500,000 shares a day")
print("Made-up rule: +0.5% gross per round trip, 20 round trips a year")
for fund in (100_000, 1_000_000, 10_000_000, 100_000_000):
    shares = fund / PRICE
    rt = round_trip(fund)
    last_level = math.ceil(shares / DEPTH)
    head = f"{100 * shares / VOLUME:.0f}% of a day; walks {last_level} cents; round trip {100 * rt:.2f}%"
    if shares > VOLUME:          # more than a whole day's trading: not one order any more
        tail = f"; {shares / VOLUME:.0f} days of volume, the last share {last_level} cents above the mid: not tradable"
    else:
        tail = (f"; net per trip {100 * (EDGE - rt):+.2f}%; a year {100 * TRIPS * (EDGE - rt):+.1f}%"
                f" = ${yearly_profit(fund):+,.0f}")
    p(f"  fund ${fund:,.0f}: order {shares:,.0f} shares", head + tail)
p("  order of 1% of a day's volume", f"{VOLUME // 100:,} shares = ${VOLUME // 100 * PRICE:,.0f}")

funds = np.geomspace(50_000, 10_000_000, 400)
alone = np.array([yearly_profit(f) for f in funds])
three = np.array([yearly_profit(f, crowd=3) for f in funds])
best = funds[alone.argmax()]
zero = funds[np.argmax((alone < 0) & (funds > best))]
p("  fund size with the most dollars a year (alone)", f"${best:,.0f}  -> ${alone.max():,.0f} a year")
p("  fund size where the yearly profit reaches zero (alone)", f"${zero:,.0f}")
best3 = funds[three.argmax()]
zero3 = funds[np.argmax((three < 0) & (funds > best3))]
p("  same, each of three funds trading together: best", f"${best3:,.0f}  -> ${three.max():,.0f} a year")
p("  same, each of three funds: zero", f"${zero3:,.0f}")
for fund in (1_000_000,):
    rt3 = 2 * side_cost(3 * fund / PRICE)
    p(f"  ${fund:,.0f} fund, alone vs one of three",
      f"round trip {100 * round_trip(fund):.2f}% vs {100 * rt3:.2f}%; "
      f"a year ${yearly_profit(fund):+,.0f} vs ${yearly_profit(fund, 3):+,.0f}")

# --- chart -----------------------------------------------------------------------------------------
cs.apply()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 3.8 * 2))
for t, col in zip(md.TICKERS, (cs.MAROON, cs.BLUE, cs.GREY, cs.PURPLE, cs.AMBER)):
    grid, ys = fast_curves[t]
    ax1.plot(100 * grid, 100 * np.array(ys), color=col, label=t)
ax1.axhline(0, color=cs.GREY, lw=0.8, ls="--", label="_zero")
ax1.set_xlabel("Cost charged per position change (% of the trade)")
ax1.set_ylabel("Return of the 5/20 crossover (% a year)")
cs.title(ax1, "The 5/20 crossover on five tickers, by cost per trade", 1)
ax1.legend()

# each line stops a little past its zero, so the losses far beyond do not flatten the picture
a, b = funds <= 4.5e6, funds <= 1.6e6
ax2.plot(funds[a] / 1e6, alone[a] / 1e3, color=cs.MAROON, label="one fund trading alone")
ax2.plot(funds[b] / 1e6, three[b] / 1e3, color=cs.BLUE, label="each of three funds trading the same signal")
ax2.axhline(0, color=cs.GREY, lw=0.8, ls="--", label="_zero")
ax2.set_xscale("log")
ax2.set_xticks([0.1, 0.3, 1, 3], ["0.1", "0.3", "1", "3"])
ax2.set_xlabel("Size of the fund (millions of dollars, log scale)")
ax2.set_ylabel("Profit after costs (thousands of dollars a year)")
cs.title(ax2, "A made-up rule on a made-up small stock, by fund size", 2)
ax2.legend()

try:
    cs.save(fig, OUT / "capacity.svg")
except OSError as e:
    sys.exit(f"cannot write the chart: {e}")
