# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "matplotlib"]
# ///
"""Numbers and chart for the post "What a trade really costs".

A toy order book walked by market orders of growing size, and a cost model (commission plus
impact) applied to the 20/100-day crossover rule on SPY and to a rule that trades every day.
Run:  uv run examples/what-a-trade-costs/costs.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chartstyle as cs  # noqa: E402
import market_data as md  # noqa: E402

OUT = cs.out_dir("what-a-trade-costs")


def p(label, value):
    print(f"{label:<66} {value}")


print("\n== a toy order book ==")
asks = [(100.05, 300), (100.10, 500), (100.20, 1000), (100.35, 2000), (100.60, 4000)]
bids = [(99.95, 400), (99.90, 600), (99.80, 1200), (99.65, 2500), (99.40, 5000)]
p("best bid / best ask / spread / mid", f"{bids[0][0]:.2f} / {asks[0][0]:.2f} / {asks[0][0] - bids[0][0]:.2f} / {(asks[0][0] + bids[0][0]) / 2:.3f}")
p("relative spread", f"{(asks[0][0] - bids[0][0]) / ((asks[0][0] + bids[0][0]) / 2):.3%}")


def walk(book, qty):
    left, cost = qty, 0.0
    for price, size in book:
        take = min(size, left); cost += take * price; left -= take
        if left == 0:
            return cost / qty
    return None


sizes = [100, 300, 800, 1800, 3800, 7800]
fills = []
for q in sizes:
    avg = walk(asks, q)
    fills.append(avg)
    p(f"market buy {q:5} shares: average fill / slippage vs best ask", f"{avg:.4f} / {avg - asks[0][0]:.4f} ({(avg - asks[0][0]) / asks[0][0]:.3%})")
p("round trip at the quotes, 100 shares: pay the spread", f"{asks[0][0] - bids[0][0]:.2f} a share = {(asks[0][0] - bids[0][0]) / ((asks[0][0] + bids[0][0]) / 2):.2%}")

print("\n== a cost model: commission plus impact ==")
COMM, IMPACT = 10.0, 0.005
for side, sign in [("buy", 1), ("sell", -1)]:
    px_ = 100 * (1 + sign * IMPACT)
    p(f"{side} 1,000 at 100 with $10 commission and 0.5% impact: price / cash", f"{px_:.2f} / {-sign * 1000 * px_ - COMM:+,.2f}")
p("round trip at an unchanged price: cost", f"{2 * 1000 * 100 * IMPACT + 2 * COMM:,.0f} = {(2 * 1000 * 100 * IMPACT + 2 * COMM) / (1000 * 100):.2%}")

print("\n== costs on two rules, SPY 2020-2025 ==")
px = md.prices()["SPY"]
r = md.daily_returns(md.prices())["SPY"]
fast, slow = px.rolling(20).mean(), px.rolling(100).mean()
pos_slow = (fast > slow).astype(float).shift(1).fillna(0).reindex(r.index).fillna(0)
pos_fast = (px > px.shift(1)).astype(float).shift(1).fillna(0).reindex(r.index).fillna(0)   # buy after an up day, sell after a down day
results = {}
for name, pos in [("20/100 crossover", pos_slow), ("yesterday-up rule", pos_fast)]:
    trades = pos.diff().abs().fillna(0)
    p(f"{name}: position changes, 2020 to 2025", f"{int(trades.sum())}")
    for c in [0.0, 0.001, 0.005, 0.01]:
        strat = pos * r - trades * c
        tot = np.prod(1 + strat) - 1
        results[(name, c)] = (1 + strat).cumprod()
        p(f"   net of {c:.1%} per position change: total return", f"{tot:+.1%}")
p("buy and hold SPY", f"{np.prod(1 + r) - 1:+.1%}")

print("\n== break-even: edge needed to pay for turnover ==")
for turnover, cost in [(0.4, 0.001), (0.1, 0.001), (0.4, 0.0005)]:
    p(f"daily turnover {turnover:.0%} at {cost:.2%} per dollar traded: daily cost", f"{turnover * cost:.3%} = {turnover * cost * 252:.1%} a year")

# ------------------------------------------------------------------ chart ----
cs.apply()
fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(8, 3.8 * 3))

cs.title(a1, "A toy order book: shares waiting at each price", 1)
a1.barh([b[0] for b in bids], [-b[1] for b in bids], height=0.04, color=cs.GREEN, label="bids: buyers waiting")
a1.barh([a[0] for a in asks], [a[1] for a in asks], height=0.04, color=cs.MAROON, label="asks: sellers waiting")
a1.axhline((asks[0][0] + bids[0][0]) / 2, color=cs.GREY, lw=1, ls="--", label="mid price 100.00")
a1.set_xlabel("Shares waiting (bids drawn to the left, asks to the right)")
a1.set_ylabel("Price (dollars)")
a1.set_xticks([-5000, -2500, 0, 2500, 5000]); a1.set_xticklabels(["5,000", "2,500", "0", "2,500", "5,000"])
a1.legend()

cs.title(a2, "Average fill price of a market buy as the order grows", 2)
a2.plot(sizes, fills, color=cs.MAROON, marker="o", label="average price paid per share")
a2.axhline(asks[0][0], color=cs.BLUE, lw=1, ls="--", label="best ask, 100.05: what a small order pays")
a2.set_xscale("log"); a2.set_xticks(sizes); a2.set_xticklabels([f"{s:,}" for s in sizes])
a2.set_xlabel("Order size (shares, log scale)")
a2.set_ylabel("Average fill price (dollars)")
a2.legend()

cs.title(a3, "Two rules on SPY under four cost levels: the busy rule is the one costs kill", 3)
for (name, c), series in results.items():
    if c in (0.0, 0.005):
        col = cs.MAROON if "cross" in name else cs.BLUE
        a3.plot(series.index, 100 * (series - 1), color=col, lw=1.6 if c == 0 else 1.0, ls="-" if c == 0 else ":",
                label=f"{name}, {'no costs' if c == 0 else '0.5% per position change'}")
a3.plot(r.index, 100 * ((1 + r).cumprod() - 1), color=cs.GREY, lw=1, label="buy and hold SPY")
a3.set_xlabel("Date")
a3.set_ylabel("Return since the start (%)")
a3.legend()
cs.save(fig, OUT / "costs.svg")
