# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "matplotlib"]
# ///
"""Numbers and chart for the post "States and rewards: what you tell the learner decides what it learns".

Arithmetic only. Three SPY indicators (Bollinger position, 20-day return, 20-day volatility) are
cut into quartile bins by hand, one evening's readings become one state label, and the number of
labels a bin choice creates is compared with the number the 2020-2025 days actually land in.
Then one day's reward for an agent in cash and one already long, and the ten days of the January
2023 episode scored four ways under the rule "long after a down day, cash after an up day".
No agent runs and nothing is learned.
Run:  uv run examples/states-and-rewards-what-the-learner-learns/states_rewards.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chartstyle as cs  # noqa: E402
import market_data as md  # noqa: E402

OUT = cs.out_dir("states-and-rewards-what-the-learner-learns")
IND = {"bollinger": "Bollinger position", "ret_20d": "20-day return", "vol_20d": "20-day volatility"}
EVENING = "2022-12-30"            # the close before day 1 of the January 2023 episode
START, DAYS = "2023-01-03", 10
COST, BONUS, GAMMA = 0.10, 0.15, 0.9   # % of the trade; % per trade; discount


def p(label, value):
    print(f"{label:<72} {value}")


# The three indicators as in market_data.features, built here so no day is lost to its targets.
try:
    px = md.prices(["SPY"])["SPY"]
except (OSError, KeyError) as e:
    sys.exit(f"cannot read the prices file data/prices.csv: {e}")
f = pd.DataFrame({
    "bollinger": (px - px.rolling(20).mean()) / (2 * px.rolling(20).std()),
    "ret_20d": px / px.shift(20) - 1,
    "vol_20d": px.pct_change().rolling(20).std(),
}).dropna()
p("SPY days with all three indicators", f"{len(f)}  ({f.index[0].date()} to {f.index[-1].date()})")
print()

# --- one evening cut into bins --------------------------------------------------------------
q = f.quantile([0.25, 0.5, 0.75])
x = f.loc[EVENING]
bins = {c: int((x[c] > q[c]).sum()) for c in IND}   # how many thresholds the value is above: 0..3
print(f"Quartile thresholds over all days, and the readings on {EVENING}")
for c, name in IND.items():
    p(f"  {name}: 25/50/75% = {q[c].iloc[0]:.4f} / {q[c].iloc[1]:.4f} / {q[c].iloc[2]:.4f}",
      f"value {x[c]:.4f} -> bin {bins[c]}")
b1, b2, b3 = bins.values()
p("  state label as digits", f"{b1}{b2}{b3}")
p(f"  state number = {b1} x 16 + {b2} x 4 + {b3}", b1 * 16 + b2 * 4 + b3)
p(f"  with holdings (cash 0, long 1): long -> 64 + {b1 * 16 + b2 * 4 + b3}", 64 + b1 * 16 + b2 * 4 + b3)
print()

# --- too few bins: two halves of the Bollinger position -------------------------------------
med = f["bollinger"].median()
low = f["bollinger"][f["bollinger"] <= med]
p("Two bins: lower half of the Bollinger position spans", f"{low.min():.2f} ({low.idxmin().date()}) to {low.max():.2f}")
print()

# --- states possible versus states visited --------------------------------------------------
rows = []
for b in (2, 3, 4, 5, 10):
    lab = sum(pd.qcut(f[c], b, labels=False).astype(int) * b ** (2 - k) for k, c in enumerate(IND))
    n = lab.value_counts()
    rows.append((b, b ** 3, len(n), int((n >= 10).sum()), int(n[n < 10].sum()), n.median()))
print("Bins per indicator, three indicators, quantile bins over all days")
print(f"  {'bins':>4} {'possible':>8} {'visited':>8} {'>=10 visits':>11} {'days in <10':>11} {'median visits':>13}")
for r in rows:
    print(f"  {r[0]:>4} {r[1]:>8} {r[2]:>8} {r[3]:>11} {r[4]:>11} {r[5]:>13.1f}")
p("Seven features at 10 bins each: possible states", f"{10 ** 7:,}")
print()

# --- one day, two holdings ------------------------------------------------------------------
r = 100 * md.daily_returns(px.to_frame())["SPY"]
i0 = r.index.get_loc(pd.Timestamp(START))
ret = r.iloc[i0:i0 + DAYS].values
d1 = ret[0]
p(f"Day 1 reward, cost {COST}% per trade (rows: holding coming in; cols: go long / go cash)", "")
p("  in cash", f"go long {d1 - COST:+.2f}   go cash {0:+.2f}")
p("  already long", f"go long {d1:+.2f}   go cash {-COST:+.2f}")
print()

# --- ten days, four rewards -----------------------------------------------------------------
prev = r.iloc[i0 - 1:i0 - 1 + DAYS].values      # yesterday's return, the Dyna post's state
h = (prev <= 0).astype(int)                       # long after a down day, cash after an up day
h_in = np.r_[0, h[:-1]]                           # holding coming into each day; starts in cash
trade = np.abs(h - h_in)
price_change = ret
account = h * ret + 0.0                           # + 0.0 turns -0.0 into 0.0
after_cost = account - COST * trade
realised = np.zeros(DAYS)
run = 1.0
for t in range(DAYS):
    if h[t]:
        run *= 1 + ret[t] / 100
    if h_in[t] and not h[t]:                      # the step that sells books the gain
        realised[t] = 100 * (run - 1)
        run = 1.0
print("Ten days of the January 2023 episode, rule: long after a down day, cash after an up day")
print(f"  {'day':>3} {'holding':>7} {'trade':>5} {'price chg':>9} {'account':>8} {'realised':>8} {'after cost':>10}")
for t in range(DAYS):
    print(f"  {t + 1:>3} {'long' if h[t] else 'cash':>7} {'yes' if trade[t] else '':>5} "
          f"{price_change[t]:>+9.2f} {account[t]:>+8.2f} {realised[t]:>+8.2f} {after_cost[t]:>+10.2f}")
p("  totals: price change / account / realised / after cost",
  f"{price_change.sum():+.2f} / {account.sum():+.2f} / {realised.sum():+.2f} / {after_cost.sum():+.2f}")
p("  trades, days with a zero realised reward", f"{trade.sum()}, {(realised == 0).sum()}")
p(f"  open position at day 3's loss {ret[2]:+.2f}% booked 10 steps later, discounted at {GAMMA}",
  f"{GAMMA ** 10 * ret[2]:+.2f}")
print()

# --- a bonus per trade ----------------------------------------------------------------------
p(f"Bonus {BONUS}% per trade, cost {COST}%: reward per trade", f"{BONUS - COST:+.2f}")
p("  ten days, rule above: after cost / with bonus",
  f"{after_cost.sum():+.2f} / {after_cost.sum() + BONUS * trade.sum():+.2f}")
p("  trade every day (10 trades): money lost to costs / reward added by the bonus",
  f"{-COST * 10:+.2f} / {BONUS * 10:+.2f}")

# --- chart ----------------------------------------------------------------------------------
cs.apply()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 3.8 * 2))
ax1.plot(f.index, f["bollinger"], color=cs.GREY, lw=0.8, label="SPY's Bollinger position")
for i, k in enumerate(q.index):
    ax1.axhline(q["bollinger"][k], color=cs.BLUE, lw=1.2, ls="--",
                label="quartile thresholds, the bin edges" if i == 0 else f"_q{k}")
ax1.plot([pd.Timestamp(EVENING)], [x["bollinger"]], "o", color=cs.MAROON, ms=6,
         label=f"{EVENING}, bin {bins['bollinger']}")
ax1.set_xlabel("Trading day (2020 to 2025)")
ax1.set_ylabel("Bollinger position (bands at -1 and +1)")
cs.title(ax1, "One indicator cut into four bins at its quartiles", 1)
ax1.legend()

bs = [row[0] for row in rows]
ax2.plot(bs, [row[1] for row in rows], color=cs.GREY, marker="o", label="states the bins create")
ax2.plot(bs, [row[2] for row in rows], color=cs.BLUE, marker="o", label="states at least one day lands in")
ax2.plot(bs, [row[3] for row in rows], color=cs.MAROON, marker="o", label="states with ten days or more")
ax2.set_yscale("log")
ax2.set_xticks(bs)
ax2.set_xlabel("Bins per indicator (three indicators)")
ax2.set_ylabel("Number of states (log scale)")
cs.title(ax2, "More bins create states faster than the days can fill them", 2)
ax2.legend()
try:
    cs.save(fig, OUT / "states_rewards.svg")
except OSError as e:
    sys.exit(f"cannot write the chart: {e}")
