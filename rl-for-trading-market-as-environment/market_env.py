# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "matplotlib"]
# ///
"""Numbers and chart for the post "RL for trading: the market as an environment, and where that breaks".

Arithmetic only. No agent runs and nothing is learned by a Q-table. A fixed family of rules,
"hold SPY tomorrow if it fell over the last N days, cash otherwise", stands in for a learned
policy so the environment's parts can be shown: target-position orders, a dollar reward with
costs charged through cash, picking N on 2020-2022 and testing the frozen choice on 2023-2025
once, and how the market's day-to-day behaviour changed between the two periods.
Run:  uv run examples/rl-for-trading-market-as-environment/market_env.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chartstyle as cs  # noqa: E402
import market_data as md  # noqa: E402

OUT = cs.out_dir("rl-for-trading-market-as-environment")
TRAIN, TEST = ("2020", "2022"), ("2023", "2025")
LOOKBACKS = (1, 2, 3, 5, 10, 20)
COST = 0.001                      # 0.10% of the value traded, every time the holding changes
SHARES, CASH0 = 100, 100_000.0    # the dollar example: 100 shares of SPY on a $100,000 account


def p(label, value):
    print(f"{label:<70} {value}")


try:
    px = md.prices(["SPY"])["SPY"]
except (OSError, KeyError) as e:
    sys.exit(f"cannot read the prices file data/prices.csv: {e}")
r = md.daily_returns(px.to_frame())["SPY"]


def holding(n):
    """1 on day t if SPY fell over the n days to the close of t-1, else 0. Known the evening before."""
    return (px / px.shift(n) - 1 < 0).astype(float).shift(1).reindex(r.index).fillna(0)


def score(n, cost, period):
    h = holding(n).loc[period[0]:period[1]]
    rr = r.loc[period[0]:period[1]]
    trades = h.diff().abs().fillna(h.iloc[0])      # the first day starts from cash
    net = h * rr - cost * trades
    years = len(rr) / md.TRADING_DAYS
    return {"gross": 100 * ((1 + h * rr).prod() - 1), "net": 100 * ((1 + net).prod() - 1),
            "trades": trades.sum() / years, "n_trades": int(trades.sum()), "in_mkt": 100 * h.mean(), "daily_net": net}


# --- 1. orders from target positions ------------------------------------------------------------
print("Orders under the two meanings of an action, shares of SPY")
for cur, tgt in ((0, 100), (100, 100), (-100, 100), (100, 0)):
    p(f"  target-position: holding {cur:+}, action says hold {tgt:+}", f"order {tgt - cur:+}")
held = 0
for k in range(3):
    held += SHARES
p("  incremental: action 'buy' three evenings running", f"holding {held:+}")
print()

# --- 2. one day's reward in dollars ---------------------------------------------------------------
d0, d1 = pd.Timestamp("2022-12-30"), pd.Timestamp("2023-01-03")
p0, p1 = px[d0], px[d1]
cash = CASH0 - SHARES * p0 - COST * SHARES * p0
v0, v1 = CASH0, cash + SHARES * p1
print(f"Buy {SHARES} SPY at the close of {d0.date()}, value at the close of {d1.date()}")
p("  prices", f"{p0:.2f} -> {p1:.2f}")
p("  cash after buying and paying the cost", f"{cash:,.2f}")
p("  cost paid", f"{COST * SHARES * p0:,.2f}")
p("  value = cash + shares x price, before and after", f"{v0:,.2f} -> {v1:,.2f}")
p("  reward = change in value", f"{v1 - v0:+,.2f}  ({100 * (v1 - v0) / v0:+.3f}%)")
p("  same day without the cost", f"{SHARES * (p1 - p0):+,.2f}")
print()

# --- 3. the training period, gross and net -------------------------------------------------------
bh = {per: 100 * ((1 + r.loc[per[0]:per[1]]).prod() - 1) for per in (TRAIN, TEST)}
print(f"Training period {TRAIN[0]}-{TRAIN[1]}: hold after an N-day fall, cost {100 * COST:.2f}% per trade")
p("  buy and hold", f"{bh[TRAIN]:+.1f}%")
rows = {n: score(n, COST, TRAIN) for n in LOOKBACKS}
print(f"  {'N':>3} {'gross %':>8} {'net %':>8} {'trades/yr':>9} {'days in %':>9}")
for n, s in rows.items():
    print(f"  {n:>3} {s['gross']:>+8.1f} {s['net']:>+8.1f} {s['trades']:>9.1f} {s['in_mkt']:>9.1f}")
best_gross = max(rows, key=lambda n: rows[n]["gross"])
best_net = max(rows, key=lambda n: rows[n]["net"])
p("  best N before costs / after costs", f"{best_gross} / {best_net}")
g = rows[best_gross]
p(f"  N={best_gross}: trades in total, and the cost per trade that wipes out the gross gain",
  f"{g['n_trades']}, {100 * (np.log1p(g['gross'] / 100) / g['n_trades']):.3f}%")
print()

# --- 4. the frozen choice on the test period, once -----------------------------------------------
t = score(best_net, COST, TEST)
t1 = score(best_gross, COST, TEST)
print(f"Test period {TEST[0]}-{TEST[1]}, N frozen at {best_net} (picked on training, after costs)")
p("  buy and hold", f"{bh[TEST]:+.1f}%")
p(f"  N={best_net} gross / net / trades a year / days in market",
  f"{t['gross']:+.1f}% / {t['net']:+.1f}% / {t['trades']:.1f} / {t['in_mkt']:.1f}%")
p(f"  (for comparison, N={best_gross} gross / net)", f"{t1['gross']:+.1f}% / {t1['net']:+.1f}%")
test_all = {n: score(n, COST, TEST)["net"] for n in LOOKBACKS}
p("  every N after costs on the test period (not used to choose)",
  "  ".join(f"N={n}: {v:+.1f}" for n, v in test_all.items()))
print()

# --- 5. how the market changed --------------------------------------------------------------------
ac = r.groupby(r.index.year).apply(lambda x: x.autocorr(1))
yr1 = pd.DataFrame({"h": holding(best_gross), "r": r})
gross1 = yr1.groupby(yr1.index.year).apply(lambda d: 100 * ((1 + d.h * d.r).prod() - 1))
bh_y = r.groupby(r.index.year).apply(lambda x: 100 * ((1 + x).prod() - 1))
print("By year: correlation of today's SPY return with yesterday's; N=1 rule gross; buy and hold")
for y in ac.index:
    p(f"  {y}", f"{ac[y]:+.3f}   N=1 gross {gross1[y]:+6.1f}%   buy and hold {bh_y[y]:+6.1f}%")

# --- chart -----------------------------------------------------------------------------------------
cs.apply()
fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 3.8 * 2))
h1 = holding(best_gross)
tr1 = h1.diff().abs().fillna(h1.iloc[0])
a1.plot(100 * ((1 + r).cumprod() - 1), color=cs.GREY, label="Buy and hold SPY")
a1.plot(100 * ((1 + h1 * r).cumprod() - 1), color=cs.BLUE, label="Hold after a down day, no costs")
a1.plot(100 * ((1 + h1 * r - COST * tr1).cumprod() - 1), color=cs.MAROON,
        label="Hold after a down day, 0.10% per trade")
a1.axvline(pd.Timestamp("2023-01-01"), color=cs.INK, lw=0.8, ls="--")
a1.text(pd.Timestamp("2022-12-01"), a1.get_ylim()[1] * 0.92, "training  ", ha="right", fontsize=9)
a1.text(pd.Timestamp("2023-02-01"), a1.get_ylim()[1] * 0.92, "  test", ha="left", fontsize=9)
a1.set_ylabel("Growth since January 2020 (%)")
a1.set_xlabel("Date")
cs.title(a1, "One rule, before and after costs", 1)
a1.legend()
cols = [cs.MAROON if v < -0.1 else cs.GREY for v in ac.values]
a2.bar(ac.index.astype(str), ac.values, color=cols)
a2.axhline(0, color=cs.INK, lw=0.8)
a2.set_ylabel("Correlation with the day before")
a2.set_xlabel("Year")
cs.title(a2, "How much SPY's daily move reversed, by year", 2)
try:
    cs.save(fig, OUT / "market_env.svg")
except OSError as e:
    sys.exit(f"cannot write the chart to {OUT}: {e}")
