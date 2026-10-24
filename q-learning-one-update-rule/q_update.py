# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "matplotlib"]
# ///
"""Numbers and chart for the post "Q-learning: one update rule, repeated".

One Q-learning update worked by hand on the first day of the January 2023 SPY episode, then the
same single table cell updated again and again: towards one fixed target (how fast the gap
closes for each learning rate) and towards the ten daily targets of the episode (how much a
large learning rate chases each day's reward). Arithmetic on a made-up starting table; nothing is
learned and no policy is run.
Run:  uv run examples/q-learning-one-update-rule/q_update.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chartstyle as cs  # noqa: E402
import market_data as md  # noqa: E402

OUT = cs.out_dir("q-learning-one-update-rule")
START, DAYS = "2023-01-03", 10
GAMMA = 0.9
ALPHAS = (0.2, 0.5, 0.9)

# A made-up starting table, in percent of the position: two states (cash, long), two actions
# (go long, go cash). These are the learner's current guesses before it sees day 1.
Q = pd.DataFrame({"go long": [0.50, 0.80], "go cash": [0.00, 0.10]}, index=["cash", "long"])


def p(label, value):
    print(f"{label:<66} {value}")


px = md.prices(["SPY"])["SPY"]
r = md.daily_returns(px.to_frame())["SPY"]
i0 = r.index.get_loc(pd.Timestamp(START))
rewards = 100 * r.iloc[i0:i0 + DAYS].values      # R_1 ... R_10 in percent

print("Starting Q-table (made up, % of the position)")
print(Q.to_string())
print()

# --- one update: morning of day 1, state cash, action go long, reward R_1, next state long ----
old = Q.loc["cash", "go long"]
best_next = Q.loc["long"].max()
target = rewards[0] + GAMMA * best_next
error = target - old
p("Day-1 reward (long SPY on 2023-01-03)", f"{rewards[0]:+.2f}%")
p("Best entry in the next state's row (long)", f"{best_next:.2f}%")
p(f"Target = reward + {GAMMA} x best next", f"{target:+.2f}%")
p("Error = target - old estimate", f"{error:+.2f}")
for a in (0.2, 0.9, 1.0):
    p(f"New estimate at alpha {a}: old + alpha x error", f"{old + a * error:.3f}%")
    p(f"  same thing as a blend: (1 - alpha) x old + alpha x target", f"{(1 - a) * old + a * target:.3f}%")
print()

# --- the same cell, the same target, again and again ---------------------------------------
N = 10
print(f"Gap to a fixed target left after n updates, as a share of the starting gap: (1 - alpha)^n")
for n in (1, 2, 5, 10):
    p(f"  n = {n:>2}", "   ".join(f"alpha {a}: {(1 - a) ** n:.3f}" for a in ALPHAS))
print()

# --- the same cell updated once per day of the episode, with the rest of the table frozen ----
# Pretend the learner is long on every morning of the episode, chooses "go long" (stay), gets
# the day's return, and lands in state long again. The next-row max is held at its starting
# value so that only the learning rate is at work.
targets = rewards + GAMMA * best_next
print("Daily targets, reward + 0.9 x 0.80:", ", ".join(f"{t:+.2f}" for t in targets))
paths = {}
for a in (0.2, 0.9):
    q = Q.loc["long", "go long"]
    path = [q]
    for t in targets:
        q = q + a * (t - q)
        path.append(q)
    paths[a] = np.array(path)
    p(f"Estimate after each day at alpha {a}", " ".join(f"{x:.2f}" for x in path[1:]))
p("Mean of the ten targets", f"{targets.mean():.2f}%")
p("Biggest day-to-day move of the estimate, alpha 0.2", f"{np.abs(np.diff(paths[0.2])).max():.2f}")
p("Biggest day-to-day move of the estimate, alpha 0.9", f"{np.abs(np.diff(paths[0.9])).max():.2f}")
print()

print("Weight a constant alpha gives a target seen k updates ago, alpha x (1 - alpha)^k")
for k in (0, 1, 2, 5):
    p(f"  k = {k}", "   ".join(f"alpha {a}: {a * (1 - a) ** k:.3f}" for a in (0.2, 0.9)))

# --- chart ----------------------------------------------------------------------------------
cs.apply()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 3.8 * 2))
n = np.arange(0, N + 1)
for a, c in zip(ALPHAS, (cs.BLUE, cs.GREY, cs.MAROON)):
    ax1.plot(n, old + (target - old) * (1 - (1 - a) ** n), color=c, marker="o", ms=3,
             label=f"learning rate {a}")
ax1.axhline(target, color=cs.GREY, lw=0.8, ls="--", label="_target")
ax1.set_xlabel("Number of updates of the same cell (same target each time)")
ax1.set_ylabel("Estimate in the cell (% of the position)")
cs.title(ax1, "One cell moved towards one fixed target, again and again", 1)
ax1.legend()

days = np.arange(0, DAYS + 1)
ax2.plot(days[1:], targets, color=cs.GREY, marker="s", ms=4, lw=0, label="target that day: reward + 0.9 × best next")
ax2.plot(days, paths[0.2], color=cs.BLUE, marker="o", ms=3, label="estimate, learning rate 0.2")
ax2.plot(days, paths[0.9], color=cs.MAROON, marker="o", ms=3, label="estimate, learning rate 0.9")
ax2.axhline(0, color=cs.GREY, lw=0.8)
ax2.set_xticks(days)
ax2.set_xlabel("Day of the episode (0 = before the first update)")
ax2.set_ylabel("Estimate in the cell (% of the position)")
cs.title(ax2, "The same cell fed the ten daily targets of January 2023", 2)
ax2.legend()

cs.save(fig, OUT / "q_update.svg")
print(f"\nchart -> {OUT / 'q_update.svg'}")
