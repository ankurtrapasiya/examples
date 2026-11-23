# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "matplotlib"]
# ///
"""Numbers and chart for the post "Comparing learners: change one thing at a time, or the result
says nothing".

Arithmetic and fixed rules only. Nothing is learned: no tree is fitted and no Q-table is updated.
The script shows (1) a label and a reward computed for the same evenings of January 2023,
(2) four fixed long-or-cash rules on SPY scored by hit rate and by return, (3) how changing the
period, the cost or the start date changes which rule looks best, and (4) the metric set and the
year-by-year results for the same rules.
Run:  uv run examples/comparing-learners-one-change-at-a-time/compare.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chartstyle as cs  # noqa: E402
import market_data as md  # noqa: E402

OUT = cs.out_dir("comparing-learners-one-change-at-a-time")
COST = 0.001          # 0.10% of the value traded, each time the holding changes
H, THETA = 5, 0.01    # the label: long if the next 5 days return more than 1%
K = np.sqrt(md.TRADING_DAYS)


def show(label, value):
    print(f"{label:<66} {value}")


try:
    px = md.prices(["SPY"])["SPY"]
except (OSError, KeyError) as e:
    sys.exit(f"cannot read the prices file data/prices.csv: {e}")
r = md.daily_returns(px.to_frame())["SPY"]


def decided_evening_before(signal):
    """Holding for day t is decided at the close of t-1, from data up to that close only."""
    return signal.astype(float).shift(1).reindex(r.index).fillna(0)


ma50 = px.rolling(50).mean()
RULES = {
    "Hold SPY every day": pd.Series(1.0, index=r.index),
    "Hold after a down day": decided_evening_before(px.pct_change() < 0),
    "Hold after an up day": decided_evening_before(px.pct_change() > 0),
    "Hold while above its 50-day average": decided_evening_before(px > ma50),
}
FIRST = r.index[r.index > ma50.first_valid_index()][0]   # first day every rule can decide


def score(h, start, end=None, cost=COST):
    h, rr = h.loc[start:end], r.loc[start:end]
    trades = h.diff().abs().fillna(h.iloc[0])            # the first day starts from cash
    net = h * rr - cost * trades
    value = (1 + net).cumprod()
    hit = (((h == 1) & (rr > 0)) | ((h == 0) & (rr <= 0))).mean()
    return {"hit": 100 * hit, "ret": 100 * (value.iloc[-1] - 1), "vol": 100 * K * net.std(),
            "sharpe": K * net.mean() / net.std(), "dd": 100 * (value / value.cummax() - 1).min(),
            "trades": trades.sum() / len(rr) * md.TRADING_DAYS, "value": value}


# --- 1. a label and a reward for the same evenings -------------------------------------------------
fwd = px.shift(-H) / px - 1
print(f"Label: long if SPY returns more than {THETA:.0%} over the next {H} days. Reward: the next day's return,")
print(f"less {COST:.2%} if the holding changes. Evenings of January 2023.")
evenings = px.loc["2023-01-03":"2023-01-11"].index
for t in evenings:
    nxt = r.index[r.index > t][0]
    label = "long" if fwd[t] > THETA else "cash"
    show(f"  {t.date()}: next 5 days {100 * fwd[t]:+.2f}% -> label {label}; next day {nxt.date()}",
         f"{100 * r[nxt]:+.2f}%; long from cash {100 * (r[nxt] - COST):+.2f}%, long from long {100 * r[nxt]:+.2f}%, cash 0.00%")
print()

# --- 2. hit rate against return ---------------------------------------------------------------------
print(f"Four rules on SPY from {FIRST.date()} (the first day all four can decide) to {r.index[-1].date()}")
for name, h in RULES.items():
    g, n = score(h, FIRST, cost=0), score(h, FIRST)
    show(f"  {name}: hit rate / before costs / after {COST:.2%} / trades a year",
         f"{g['hit']:.1f}% / {g['ret']:+.1f}% / {n['ret']:+.1f}% / {n['trades']:.0f}")
print()

# --- 3. change one condition -----------------------------------------------------------------------
print("Hold after a down day against holding, return and Sharpe ratio")
for period in (("2020", "2022"), ("2023", "2025")):
    for cost in (0, COST):
        a, b = score(RULES["Hold after a down day"], *period, cost), score(RULES["Hold SPY every day"], *period, cost)
        show(f"  {period[0]} to {period[1]}, cost {cost:.2%}",
             f"down-day rule {a['ret']:+.1f}% ({a['sharpe']:.2f}), holding {b['ret']:+.1f}% ({b['sharpe']:.2f})")
print()
trend, hold = RULES["Hold while above its 50-day average"], RULES["Hold SPY every day"]
for start in (r.index[0], FIRST):
    a, b = score(trend, start), score(hold, start)
    show(f"  start {start.date()}: 50-day rule / holding / gap, after costs",
         f"{a['ret']:+.1f}% / {b['ret']:+.1f}% / {b['ret'] - a['ret']:.1f} points")
show("  holding SPY from the first close to the day the 50-day average exists",
     f"{100 * ((1 + r.loc[:ma50.first_valid_index()]).prod() - 1):+.1f}%")
print()

# --- 4. the metric set, and year by year -----------------------------------------------------------
print("2023 to 2025, after costs: return / volatility / Sharpe / worst fall / trades a year")
for name in ("Hold SPY every day", "Hold while above its 50-day average", "Hold after a down day"):
    s = score(RULES[name], "2023")
    show(f"  {name}", f"{s['ret']:+.1f}% / {s['vol']:.1f}% / {s['sharpe']:.2f} / {s['dd']:.1f}% / {s['trades']:.0f}")
print()
years = [str(y) for y in range(2021, 2026)]
yearly = pd.DataFrame({name: [score(h, y, y)["ret"] for y in years] for name, h in RULES.items()}, index=years)
print("Return after costs, by year (%)")
for y, row in yearly.iterrows():
    show(f"  {y}: " + " / ".join(f"{v:+.1f}" for v in row), f"best: {row.idxmax()}")
print()

# --- chart -----------------------------------------------------------------------------------------
cs.apply()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 3.8 * 2))
colours = [cs.BLUE, cs.MAROON, cs.GREY, cs.PURPLE]
for (name, h), c in zip(RULES.items(), colours):
    v = score(h, FIRST)["value"]
    ax1.plot(v.index, 100 * (v - 1), color=c, label=name)
ax1.axhline(0, color=cs.GREY, lw=0.8)
ax1.set_ylabel("Growth since 16 March 2020 (%)")
ax1.set_xlabel("Date")
ax1.legend()
cs.title(ax1, "Four fixed rules on SPY after a 0.10% cost per trade", 1)
x = np.arange(len(years))
w = 0.2
for i, ((name, _), c) in enumerate(zip(RULES.items(), colours)):
    ax2.bar(x + (i - 1.5) * w, yearly[name], width=w, color=c, label=name)
ax2.axhline(0, color=cs.GREY, lw=0.8)
ax2.set_xticks(x, years[:-1] + [f"{years[-1]} (to {r.index[-1]:%b})"])
ax2.set_xlabel("Year")
ax2.set_ylabel("Return after costs (%)")
ax2.legend()
cs.title(ax2, "The same rules year by year: no rule wins every year", 2)
cs.save(fig, OUT / "compare.svg")
print("chart:", OUT / "compare.svg")
