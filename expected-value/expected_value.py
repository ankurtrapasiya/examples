# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "matplotlib"]
# ///
"""Numbers and chart for the post "Expected value".

Run:  uv run examples/expected-value/expected_value.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chartstyle as cs  # noqa: E402
import market_data as md  # noqa: E402

OUT = cs.out_dir("expected-value")

# --- 1. one bet on red at American roulette: 18 red, 18 black, 2 green -------------------
p_win = 18 / 38
ev_bet = p_win * 1 + (1 - p_win) * (-1)
print(f"P(win) = 18/38 = {p_win:.4f}; EV of a $1 bet = {p_win:.4f}*(+1) + {1 - p_win:.4f}*(-1) = {ev_bet:.4f}")
print(f"  1,000 bets a night -> expected result ${1000 * ev_bet:,.2f}")

# --- 2. one day holding SPY, from the 2020-2025 record --------------------------------------
r = md.daily_returns(md.prices(["SPY"]))["SPY"]
up = r > 0
p_up, avg_up, avg_down = up.mean(), r[up].mean(), r[~up].mean()
ev_day = p_up * avg_up + (1 - p_up) * avg_down
print(f"\nSPY: P(up) = {p_up:.4f}, average up day {avg_up:+.4%}, average down day {avg_down:+.4%}")
print(f"  EV of one day = {p_up:.4f}*{avg_up:.4%} + {1 - p_up:.4f}*({avg_down:.4%}) = {ev_day:+.4%}  (plain mean {r.mean():+.4%})")
print(f"  $10,000 for one day: expected {10_000 * ev_day:+,.2f}; 252 days: {10_000 * ev_day * 252:+,.0f}")

# --- 3. expected loss on a loan -----------------------------------------------------------
pd_, lgd, ead = 0.02, 0.45, 1_000_000
el = pd_ * lgd * ead
print(f"\nEL = PD*LGD*EAD = {pd_}*{lgd}*{ead:,} = ${el:,.0f}")

# --- 4. law of large numbers: the roulette bet, spin by spin ------------------------------
rng = np.random.default_rng(1)
spins = np.where(rng.random(5_000) < p_win, 1.0, -1.0)
running = np.cumsum(spins) / np.arange(1, len(spins) + 1)
print(f"\nrunning average per $1 bet: after 10 spins {running[9]:+.3f}, 100 {running[99]:+.3f}, "
      f"1,000 {running[999]:+.4f}, 5,000 {running[-1]:+.4f}; expected {ev_bet:+.4f}")

# a night of 1,000 bets, 20,000 nights
nights = np.where(rng.random((20_000, 1000)) < p_win, 1.0, -1.0).sum(axis=1)
print(f"1,000 bets a night, 20,000 nights: average ${nights.mean():,.1f} (expected ${1000 * ev_bet:,.1f}); "
      f"5% of nights lose more than ${-np.percentile(nights, 5):,.0f}; worst night ${nights.min():,.0f}; "
      f"share of winning nights {(nights > 0).mean():.1%}")

# --- 5. same EV, different risk ----------------------------------------------------------
a = np.array([10_000, 10_000])
b = np.array([0, 20_000])
print(f"\nsame EV: sure thing {a.mean():,.0f} vs coin flip {b.mean():,.0f}; std 0 vs {b.std():,.0f}")

# --- chart ----------------------------------------------------------------------------------
cs.apply()
fig, axes = plt.subplots(2, 1, figsize=(8, 7.6))
ax = axes[0]
ax.plot(np.arange(1, len(running) + 1), running, color=cs.MAROON, lw=1.2)
ax.axhline(ev_bet, color=cs.INK, ls="--", lw=0.8)
ax.text(len(running) * 0.3, ev_bet + 0.045, f"expected value −${abs(ev_bet):.3f} per bet", fontsize=9)
ax.set_xscale("log")
ax.set_ylim(-0.4, 0.4)
ax.set_xlabel("Number of $1 bets on red so far (log scale)")
ax.set_ylabel("Average result per bet so far ($)")
cs.title(ax, "One run of 5,000 bets: the average settles on −5.26 cents", 1)

ax = axes[1]
ax.hist(nights, bins=60, color=cs.GREY)
ax.axvline(1000 * ev_bet, color=cs.MAROON, lw=1.4)
ax.text(1000 * ev_bet + 8, ax.get_ylim()[1] * 0.85, f"expected −${abs(1000 * ev_bet):,.0f}", color=cs.MAROON, fontsize=9)
ax.set_xlabel("Result of one night of 1,000 bets ($)")
ax.set_ylabel("Number of simulated nights (of 20,000)")
cs.title(ax, "20,000 nights: a single night can land far from the expected −$53", 2)
cs.save(fig, OUT / "expected-value.svg")
