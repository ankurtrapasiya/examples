# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "matplotlib"]
# ///
"""Numbers and chart for the post "MDPs: a state, an action, a reward and the value of what follows".

The ten-day SPY episode from the reinforcement-learning post, now discounted: the weight each
later reward gets for three discount factors, the discounted return of the episode, the discounted
reward-to-go from each day, and a check of the recursion G_t = R_{t+1} + gamma * G_{t+1}
(Sutton and Barto, equation 3.9). Arithmetic only: nothing is learned here.
Run:  uv run examples/mdp-state-action-reward-value/discounting.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chartstyle as cs  # noqa: E402
import market_data as md  # noqa: E402

OUT = cs.out_dir("mdp-state-action-reward-value")
START, DAYS = "2023-01-03", 10
GAMMAS = (1.0, 0.9, 0.5)


def p(label, value):
    print(f"{label:<66} {value}")


px = md.prices(["SPY"])["SPY"]
r = md.daily_returns(px.to_frame())["SPY"]
i0 = r.index.get_loc(pd.Timestamp(START))
rewards = r.iloc[i0:i0 + DAYS].values          # R_1 ... R_10, in percent after * 100
pct = lambda x: f"{100 * x:+.2f}%"  # noqa: E731


def to_go(rews, gamma):
    """Discounted reward still to come from each step: G_t = R_{t+1} + gamma * G_{t+1}, G_T = 0."""
    g = np.zeros(len(rews) + 1)
    for t in range(len(rews) - 1, -1, -1):
        g[t] = rews[t] + gamma * g[t + 1]
    return g[:-1]


print("Weight on a reward k days ahead, gamma^k")
for k in (0, 1, 2, 5, 9, 20, 60):
    p(f"  k = {k:>2}", "   ".join(f"gamma {g}: {g**k:.3f}" for g in GAMMAS))

print()
print("The ten rewards (long SPY from 2023-01-03):", ", ".join(pct(x) for x in rewards))
for g in GAMMAS:
    tg = to_go(rewards, g)
    p(f"Discounted return of the episode, gamma {g}", pct(tg[0]))
    p(f"  check: R_1 + gamma * G_2 = {pct(rewards[0])} + {g} x {pct(tg[1])}", pct(rewards[0] + g * tg[1]))

print()
p("Infinite +1 per day, gamma 0.9: sum = 1 / (1 - 0.9)", f"{1 / (1 - 0.9):.1f}")
p("Infinite +1 per day, gamma 0.5: sum = 1 / (1 - 0.5)", f"{1 / (1 - 0.5):.1f}")
p("Days until a reward counts for less than half, gamma 0.9", int(np.ceil(np.log(0.5) / np.log(0.9))))
p("Days until a reward counts for less than half, gamma 0.99", int(np.ceil(np.log(0.5) / np.log(0.99))))

# --- chart ----------------------------------------------------------------------------------
cs.apply()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 3.8 * 2))
k = np.arange(0, 21)
for g, c in zip(GAMMAS, (cs.GREY, cs.MAROON, cs.BLUE)):
    ax1.plot(k, g**k, color=c, marker="o", ms=3, label=f"discount factor {g}")
ax1.set_xlabel("Days ahead the reward arrives")
ax1.set_ylabel("Weight the reward gets (1 = full)")
cs.title(ax1, "How much a reward k days away counts today", 1)
ax1.legend()

x = np.arange(1, DAYS + 1)
width = 0.27
for i, (g, c) in enumerate(zip(GAMMAS, (cs.GREY, cs.MAROON, cs.BLUE))):
    ax2.bar(x + (i - 1) * width, 100 * to_go(rewards, g), width=width, color=c, label=f"discount factor {g}")
ax2.axhline(0, color=cs.GREY, lw=0.8)
ax2.set_xticks(x)
ax2.set_xlabel("Day of the episode (trading days from 2023-01-03)")
ax2.set_ylabel("Discounted reward still to come (% of the position)")
cs.title(ax2, "The same ten rewards, summed with three discount factors", 2)
ax2.legend()

cs.save(fig, OUT / "discounting.svg")
print(f"\nchart -> {OUT / 'discounting.svg'}")
