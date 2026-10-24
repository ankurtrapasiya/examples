# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "matplotlib"]
# ///
"""Numbers and chart for the post "Dyna: learn a model of the world, then practise in it".

The ten-day SPY episode from January 2023 (long from 2023-01-03), used three ways: as ten real
Q-learning updates; as the raw material of a model, a table of what reward and next state
followed each state-action pair; and as the source of one planning update, the same Q-update run
on a remembered transition instead of a fresh one. Then the arithmetic of how many updates n
planning steps per real step buy, and what a last-outcome model does with a state that pays
differently each time. Arithmetic on a made-up starting table; no agent runs, nothing is learned
beyond the single updates worked by hand.
Run:  uv run examples/dyna-learn-a-model-then-practise/dyna.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chartstyle as cs  # noqa: E402
import market_data as md  # noqa: E402

OUT = cs.out_dir("dyna-learn-a-model-then-practise")
START, DAYS = "2023-01-03", 10
ALPHA, GAMMA = 0.2, 0.9
PLANNING_STEPS = (0, 5, 50)

# A made-up starting table, in percent of the position. The state is what yesterday did
# (SPY up or down), the action is what the learner holds today.
Q = pd.DataFrame({"go long": [0.50, 0.30], "go cash": [0.00, 0.10]},
                 index=["yesterday up", "yesterday down"])


def p(label, value):
    print(f"{label:<70} {value}")


px = md.prices(["SPY"])["SPY"]
r = md.daily_returns(px.to_frame())["SPY"]
i0 = r.index.get_loc(pd.Timestamp(START))
rewards = 100 * r.iloc[i0:i0 + DAYS].values                 # R_1 ... R_10 in percent
state_of = lambda ret: "yesterday up" if ret > 0 else "yesterday down"  # noqa: E731
states = [state_of(100 * r.iloc[i0 - 1 + t]) for t in range(DAYS + 1)]  # S_0 ... S_10

print("Starting Q-table (made up, % of the position)")
print(Q.to_string())
print()

# --- the ten real steps, and the model they leave behind ------------------------------------
# Each real step is (state, action, reward, next state). The learner is long every day, so the
# action is always "go long". Sutton and Barto's Dyna-Q model stores, for each state-action
# pair, the last reward and next state seen (section 8.2, "assuming deterministic environment").
print("The ten real steps (state before the day, action, reward, state after the day)")
rows = []
last_outcome = {}      # (state, action) -> (reward, next state): the book's deterministic model
all_outcomes = {}      # (state, action) -> list of (reward, next state): what a count model keeps
for t in range(DAYS):
    s, a, rew, s2 = states[t], "go long", rewards[t], states[t + 1]
    rows.append((t + 1, s, a, rew, s2))
    last_outcome[(s, a)] = (rew, s2)
    all_outcomes.setdefault((s, a), []).append((rew, s2))
    p(f"  day {t + 1:>2}: {s:<14} {a}", f"reward {rew:+.2f}%  ->  {s2}")
print()

print("The model after ten days, last-outcome version (one row per state-action pair seen)")
for (s, a), (rew, s2) in last_outcome.items():
    p(f"  Model({s}, {a})", f"reward {rew:+.2f}%, next state {s2}")
p("Pairs the model knows about", f"{len(last_outcome)} of {Q.size}")
print()

# --- the first real update ------------------------------------------------------------------
s, a, rew, s2 = states[0], "go long", rewards[0], states[1]
old = Q.loc[s, a]
target = rew + GAMMA * Q.loc[s2].max()
p(f"Day 1, real step: state {s}, action {a}, reward {rew:+.2f}%, next {s2}", "")
p(f"  target = {rew:+.2f} + {GAMMA} x {Q.loc[s2].max():.2f}", f"{target:+.2f}")
p(f"  new estimate at alpha {ALPHA}: {old:.2f} + {ALPHA} x ({target - old:+.2f})", f"{old + ALPHA * (target - old):.3f}")
Q_after_real = Q.copy()
Q_after_real.loc[s, a] = old + ALPHA * (target - old)
print()

# --- one planning step: the same update on a remembered transition --------------------------
# After day 3 the learner has three real steps in its model. A planning step picks one of the
# remembered pairs at random and runs the identical update with the model's reward and next
# state, never touching the market. Here: the pair recorded on day 3.
s3, a3, rew3, s2_3 = states[2], "go long", rewards[2], states[3]
old3 = Q.loc[s3, a3]
target3 = rew3 + GAMMA * Q.loc[s2_3].max()
p(f"Planning step on the day-3 memory: Model({s3}, {a3}) = reward {rew3:+.2f}%, next {s2_3}", "")
p(f"  target = {rew3:+.2f} + {GAMMA} x {Q.loc[s2_3].max():.2f}", f"{target3:+.2f}")
p(f"  new estimate at alpha {ALPHA}: {old3:.2f} + {ALPHA} x ({target3 - old3:+.2f})", f"{old3 + ALPHA * (target3 - old3):.3f}")
p("  market data read during this step", "none")
print()

# --- how many updates n planning steps buy --------------------------------------------------
print("Updates after each real step, 1 real + n planned per step")
for n in PLANNING_STEPS:
    p(f"  n = {n:>2}: after {DAYS} real steps", f"{DAYS * (1 + n)} updates")
print()

# --- the same state, different outcomes: what each kind of model keeps ----------------------
key = ("yesterday down", "go long")
seen = all_outcomes[key]
rews = np.array([x[0] for x in seen])
nxt = [x[1] for x in seen]
p(f"Times the pair ({key[0]}, {key[1]}) was seen", len(seen))
p("  rewards seen", ", ".join(f"{x:+.2f}%" for x in rews))
p("  last-outcome model returns", f"{last_outcome[key][0]:+.2f}%, next {last_outcome[key][1]}")
p("  average reward over all the times seen", f"{rews.mean():+.2f}%")
p("  spread of those rewards (standard deviation)", f"{rews.std(ddof=0):.2f}%")
for s2 in ("yesterday up", "yesterday down"):
    p(f"  share of the times the next state was {s2}", f"{nxt.count(s2)}/{len(nxt)} = {nxt.count(s2) / len(nxt):.2f}")
print()

key_up = ("yesterday up", "go long")
rews_up = np.array([x[0] for x in all_outcomes[key_up]])
p(f"Times the pair ({key_up[0]}, {key_up[1]}) was seen", len(rews_up))
p("  rewards seen", ", ".join(f"{x:+.2f}%" for x in rews_up))
p("  last-outcome model returns", f"{last_outcome[key_up][0]:+.2f}%")
p("  average reward over all the times seen", f"{rews_up.mean():+.2f}%")

# --- chart ----------------------------------------------------------------------------------
cs.apply()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 3.8 * 2))
steps = np.arange(0, DAYS + 1)
for n, c in zip(PLANNING_STEPS, (cs.GREY, cs.BLUE, cs.MAROON)):
    ax1.plot(steps, steps * (1 + n), color=c, marker="o", ms=3, label=f"{n} planning steps per real step")
ax1.set_xlabel("Real steps taken (trading days from 2023-01-03)")
ax1.set_ylabel("Q-table updates performed (real plus planned)")
cs.title(ax1, "Ten real days, and how many updates they become", 1)
ax1.legend()

days_down = [row[0] for row in rows if row[1] == key[0]]
ax2.bar(days_down, rews, color=cs.BLUE, width=0.6, label="reward seen after a down day, long SPY")
ax2.axhline(rews.mean(), color=cs.MAROON, lw=1.2, ls="--", label=f"average of all of them, {rews.mean():+.2f}%")
ax2.axhline(last_outcome[key][0], color=cs.GREY, lw=1.2, ls=":", label=f"what a last-outcome model keeps, {last_outcome[key][0]:+.2f}%")
ax2.axhline(0, color=cs.GREY, lw=0.8)
ax2.set_xticks(steps[1:])
ax2.set_xlabel("Day of the episode (trading days from 2023-01-03)")
ax2.set_ylabel("Reward that day (% of the position)")
cs.title(ax2, "One state, several outcomes: what the model should remember", 2)
ax2.legend()

cs.save(fig, OUT / "dyna.svg")
print(f"\nchart -> {OUT / 'dyna.svg'}")
