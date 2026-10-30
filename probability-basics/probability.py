# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "matplotlib"]
# ///
"""Numbers and chart for the post "Probability basics".

Counts come from SPY's daily closes, 2020-2025 (examples/data/prices.csv); the Bayes
example uses stated assumptions. Run:  uv run examples/probability-basics/probability.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chartstyle as cs  # noqa: E402
import market_data as md  # noqa: E402

OUT = cs.out_dir("probability-basics")

px = md.prices(["SPY"])["SPY"]
r = px.pct_change().dropna()
up = r > 0
n = len(r)

# --- 1. a probability from counts ---------------------------------------------------------
print(f"{n} trading days, {up.sum()} up days: P(up) = {up.sum()}/{n} = {up.mean():.4f}")

# --- 2. complement: at least one down day in a five-day week ------------------------------
p_up = up.mean()
print(f"P(five up days in a row, if independent) = {p_up:.4f}^5 = {p_up ** 5:.4f}; "
      f"P(at least one down day) = {1 - p_up ** 5:.4f}")
weeks = up.groupby(np.arange(n) // 5).agg(["sum", "count"])
weeks = weeks[weeks["count"] == 5]
print(f"actual: of {len(weeks)} five-day blocks, {(weeks['sum'] < 5).mean():.4f} had at least one down day")

# --- 3. addition rule: a down day or a big day (move over 2%) ------------------------------
down, big = ~up, r.abs() > 0.02
print(f"P(down) = {down.sum()}/{n} = {down.mean():.4f}; P(big) = {big.sum()}/{n} = {big.mean():.4f}; "
      f"P(down and big) = {(down & big).sum()}/{n} = {(down & big).mean():.4f}")
print(f"P(down or big) = {down.mean():.4f} + {big.mean():.4f} - {(down & big).mean():.4f} = "
      f"{down.mean() + big.mean() - (down & big).mean():.4f}  (count {(down | big).sum()})")

# --- 4. conditional probability: a signal and the next day ---------------------------------
# signal at yesterday's close: price above its 20-day average
sig = (px > px.rolling(20).mean()).shift(1)
mask = sig.notna() & px.rolling(20).mean().shift(1).notna()
sig, rr = sig[mask].astype(bool), up.reindex(sig[mask].index)
a, b = (sig & rr).sum(), (sig & ~rr).sum()
c, d = (~sig & rr).sum(), (~sig & ~rr).sum()
N = a + b + c + d
print(f"\n2x2 over {N} days:  signal&up {a}, signal&down {b}, nosignal&up {c}, nosignal&down {d}")
print(f"P(up | signal) = {a}/{a + b} = {a / (a + b):.4f};  P(up | no signal) = {c}/{c + d} = {c / (c + d):.4f}")
print(f"P(signal | up) = {a}/{a + c} = {a / (a + c):.4f}  (not the same question)")
print(f"P(signal) = {(a + b) / N:.4f}; P(up and signal) = {a / N:.4f} = P(signal)*P(up|signal) = {(a + b) / N * a / (a + b):.4f}")

# independence check: does yesterday's direction change today's odds?
prev = up.shift(1).iloc[1:].astype(bool)
cur = up.iloc[1:]
print(f"P(up | yesterday up) = {cur[prev].mean():.4f}; P(up | yesterday down) = {cur[~prev].mean():.4f}; P(up) = {p_up:.4f}")
bigdown = r < -0.02
pb, cb = bigdown.shift(1).iloc[1:].astype(bool), bigdown.iloc[1:]
print(f"P(drop over 2%) = {bigdown.mean():.4f}; P(drop over 2% | yesterday was one) = {cb[pb].mean():.4f} "
      f"({cb[pb].sum()} of {pb.sum()})")

# --- 5. Bayes: a backtest that looks good -------------------------------------------------
prior, hit_rate, false_alarm = 0.01, 0.80, 0.05      # assumptions, stated in the post
p_pass = hit_rate * prior + false_alarm * (1 - prior)
posterior = hit_rate * prior / p_pass
print(f"\nP(good backtest) = {hit_rate}*{prior} + {false_alarm}*{1 - prior:.2f} = {p_pass:.4f}")
print(f"P(real edge | good backtest) = {hit_rate * prior:.4f} / {p_pass:.4f} = {posterior:.4f}")
ideas = 10_000
real = round(ideas * prior); lucky = ideas - real
true_pass, false_pass = round(real * hit_rate), round(lucky * false_alarm)
print(f"out of {ideas:,} ideas: {real} real -> {true_pass} pass; {lucky:,} no edge -> {false_pass} pass; "
      f"real share = {true_pass}/{true_pass + false_pass} = {true_pass / (true_pass + false_pass):.2%}")

# simulation check
rng = np.random.default_rng(0)
has_edge = rng.random(5_000_000) < prior
passed = np.where(has_edge, rng.random(5_000_000) < hit_rate, rng.random(5_000_000) < false_alarm)
print(f"simulated: {has_edge[passed].mean():.4f}")

# --- chart: the passes, real and lucky ----------------------------------------------------
cs.apply()
fig, ax = plt.subplots(figsize=(7, 2.6))
ax.barh(["ideas that pass\nthe backtest"], [true_pass], color=cs.MAROON, label=f"real edge: {true_pass}")
ax.barh(["ideas that pass\nthe backtest"], [false_pass], left=[true_pass], color=cs.GREY, label=f"luck or overfitting: {false_pass}")
ax.set_xlabel(f"Number of ideas, out of {ideas:,} tested")
ax.legend(fontsize=9, loc="lower right")
ax.set_title(f"{true_pass + false_pass} backtests look good; {true_pass / (true_pass + false_pass):.0%} are real", fontsize=11)
cs.save(fig, OUT / "backtests.svg")
