# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "matplotlib"]
# ///
"""Numbers and chart for the post "Sharpe, Sortino and Information ratio".

Definitions used (each is spelled out in the post):
  Sharpe   = sqrt(252) * mean(r - rf) / std(r - rf, ddof=1)
  Sortino  = sqrt(252) * mean(r - T)  / sqrt(mean(min(r - T, 0)^2))   (T = target, here rf)
  Info     = sqrt(252) * mean(r - b)  / std(r - b, ddof=1)            (b = benchmark)
Run:  uv run examples/risk-adjusted-ratios/ratios.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chartstyle as cs  # noqa: E402
import market_data as md  # noqa: E402

OUT = cs.out_dir("risk-adjusted-ratios")
K = np.sqrt(252)


def sharpe(r, rf=0.0):
    ex = r - rf
    return K * ex.mean() / ex.std(ddof=1)


def downside_dev(r, target=0.0):
    return np.sqrt(np.mean(np.minimum(r - target, 0) ** 2))


def sortino(r, target=0.0):
    return K * (r - target).mean() / downside_dev(r, target)


def info_ratio(r, b):
    a = r - b
    return K * a.mean() / a.std(ddof=1)


# --- 1. by hand: SPY's first five days of 2020 (percent, risk-free 0), NVDA as the "fund" -----
rets = md.daily_returns(md.prices())
five = (rets.iloc[:5] * 100).round(1)
r = five["SPY"].to_numpy()
print("SPY returns", r, "mean", round(r.mean(), 2), "sd", round(r.std(ddof=1), 3))
print(f"daily Sharpe {r.mean() / r.std(ddof=1):.4f}, annualised {sharpe(r):.2f}")
neg = np.minimum(r, 0)
print("downside parts", neg, "squared", np.round(neg ** 2, 2), "mean of squares", round(np.mean(neg ** 2), 3),
      "downside dev", round(downside_dev(r), 3))
print(f"daily Sortino {r.mean() / downside_dev(r):.4f}, annualised {sortino(r):.2f}")
fund, bench = five["NVDA"].to_numpy(), r
act = fund - bench
print("NVDA", fund, "SPY", bench, "active", np.round(act, 1), "mean", round(act.mean(), 2), "sd", round(act.std(ddof=1), 3))
print(f"daily IR {act.mean() / act.std(ddof=1):.4f}, annualised {info_ratio(fund, bench):.2f}")

# --- 1b. five years, risk-free 2% a year (roughly the average bill rate 2020-2025) ------------
rf_d = (1 + 0.02) ** (1 / 252) - 1
print("\n2020-2025, risk-free 2%:")
for t in rets:
    x = rets[t].to_numpy()
    line = f"  {t:4s} return {x.mean() * 252:5.1%}  sd {x.std(ddof=1) * K:5.1%}  Sharpe {sharpe(x, rf_d):.2f}  Sortino {sortino(x, rf_d):.2f}"
    if t != "SPY":
        a = x - rets["SPY"].to_numpy()
        line += f"  vs SPY: active {a.mean() * 252:+.1%}/yr, tracking error {a.std(ddof=1) * K:.1%}, IR {info_ratio(x, rets['SPY'].to_numpy()):.2f}"
    print(line)

# --- 2. risk-free rate: annual 4% to daily -----------------------------------------------
rf_annual = 0.04
rf_daily = (1 + rf_annual) ** (1 / 252) - 1
print(f"\nrisk-free 4% a year -> {rf_daily:.6%} a day (252nd root); simple /252 gives {rf_annual / 252:.6%}")

# --- 3. same Sharpe, different Sortino ------------------------------------------------
rng = np.random.default_rng(21)
n = 252
mu, sd = 0.0006, 0.01


def standardise(x):
    return mu + sd * (x - x.mean()) / x.std(ddof=1)


symmetric = standardise(rng.normal(0, 1, n))
skewed_up = standardise(rng.exponential(1, n) - 1)        # many small losses, few big gains
skewed_down = standardise(-(rng.exponential(1, n) - 1))   # many small gains, few big losses
print("\nthree funds, same mean and sd (risk-free 4%):")
for name, x in [("symmetric", symmetric), ("big gains, small losses", skewed_up), ("big losses, small gains", skewed_down)]:
    print(f"  {name:24s} mean {x.mean():.4%} sd {x.std(ddof=1):.4%}  Sharpe {sharpe(x, rf_daily):.3f}  "
          f"Sortino {sortino(x, rf_daily):.3f}  worst day {x.min():.2%}  best day {x.max():.2%}")

# --- 4. information ratio: fund vs benchmark -----------------------------------------
def active(alpha_yr, te_yr):
    """Made-up daily active returns with exactly this yearly mean and tracking error."""
    x = rng.normal(0, 1, n)
    return alpha_yr / 252 + (te_yr / K) * (x - x.mean()) / x.std(ddof=1)


bench_y = rng.normal(0.0004, 0.011, n)
manager_a = bench_y + active(0.01, 0.02)     # beats the index by 1%/yr, stays close to it
manager_b = bench_y + active(0.03, 0.12)     # beats it by 3%/yr, but wanders far from it
for name, x in [("A: close to index", manager_a), ("B: big bets", manager_b)]:
    a = x - bench_y
    print(f"  {name:18s} active return {a.mean() * 252:.2%}/yr, tracking error {a.std(ddof=1) * K:.2%}/yr, "
          f"IR {info_ratio(x, bench_y):.3f}")

# --- chart: the two skewed funds -------------------------------------------------------
cs.apply()
fig, ax = plt.subplots(figsize=(8, 3.6))
bins = np.linspace(-0.035, 0.05, 50) * 100
ax.hist(skewed_up * 100, bins=bins, color=cs.MAROON, alpha=0.75, label="big gains, small losses")
ax.hist(skewed_down * 100, bins=bins, color=cs.GREY, alpha=0.75, label="big losses, small gains")
ax.set_xlabel("Daily return (%)")
ax.set_ylabel("Number of days (of 252)")
cs.title(ax, "Two made-up funds with the same mean, standard deviation and Sharpe ratio")
ax.legend(fontsize=9)
cs.save(fig, OUT / "skew.svg")
