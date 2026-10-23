# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "matplotlib"]
# ///
"""Numbers and chart for the post "Reinforcement learning: learning from reward, not from labels".

The same ten trading days of SPY seen two ways: as supervised rows (features -> label) and as
one reinforcement-learning episode (state, action, reward, next state), with the delayed-reward
arithmetic: the total reward of the episode and the reward still to come after each day.
Concept only: no learner is trained here.
Run:  uv run examples/reinforcement-learning-reward-not-labels/episode.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chartstyle as cs  # noqa: E402
import market_data as md  # noqa: E402

OUT = cs.out_dir("reinforcement-learning-reward-not-labels")
START, DAYS = "2023-01-03", 10


def p(label, value):
    print(f"{label:<60} {value}")


px = md.prices(["SPY"])["SPY"]
r = md.daily_returns(px.to_frame())["SPY"]
pct = lambda x: f"{100 * x:+.2f}%"  # noqa: E731

# --- the same days as supervised rows: five past returns -> next day's return ----------------
print("Supervised rows (features: last five daily returns; label: the next day's return)")
i0 = r.index.get_loc(pd.Timestamp(START))
for i in range(i0, i0 + 3):
    feats = ", ".join(pct(x) for x in r.iloc[i - 5:i])
    p(f"  {r.index[i].date()}  features [{feats}]", f"label {pct(r.iloc[i])}")

# --- the same days as one episode: long SPY from the first day, hold to the last --------------
ep = r.iloc[i0:i0 + DAYS]                      # reward each day = that day's return on a long position
total = ep.sum()
to_go = ep[::-1].cumsum()[::-1]                # reward still to come from day t to the end (S&B eq. 3.7)
print()
print(f"One episode: long SPY on {ep.index[0].date()}, hold {DAYS} days to {ep.index[-1].date()}")
for d, (day, rew) in enumerate(ep.items(), start=1):
    p(f"  day {d:>2}  {day.date()}  reward {pct(rew)}", f"still to come {pct(to_go[day])}")
p("Total reward of the episode (sum of the ten)", pct(total))
p("Best single day", f"{pct(ep.max())} on {ep.idxmax().date()}")
p("Worst single day", f"{pct(ep.min())} on {ep.idxmin().date()}")
p("Days with a negative reward", int((ep < 0).sum()))
p("Reward still to come after day 5 (days 6 to 10)", pct(to_go.iloc[5]))

# --- chart: daily rewards with the running total, then the reward still to come ---------------
cs.apply()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 3.8 * 2))
x = np.arange(1, DAYS + 1)

ax1.bar(x, 100 * ep.values, color=cs.MAROON, width=0.6, label="reward that day (long SPY)")
ax1.plot(x, 100 * ep.cumsum().values, color=cs.BLUE, marker="o", label="running total since day 1")
ax1.axhline(0, color=cs.GREY, lw=0.8)
ax1.set_xticks(x)
ax1.set_xlabel("Day of the episode (trading days from 2023-01-03)")
ax1.set_ylabel("Reward (% of the position)")
cs.title(ax1, "Ten daily rewards and their running total", 1)
ax1.legend()

ax2.bar(x, 100 * to_go.values, color=cs.PURPLE, width=0.6, label="reward still to come from that day to day 10")
ax2.axhline(0, color=cs.GREY, lw=0.8)
ax2.set_xticks(x)
ax2.set_xlabel("Day of the episode (trading days from 2023-01-03)")
ax2.set_ylabel("Reward from that day to the end (% of the position)")
cs.title(ax2, "What the decision made on each day is answerable for", 2)
ax2.legend()

cs.save(fig, OUT / "episode.svg")
print(f"\nchart -> {OUT / 'episode.svg'}")
