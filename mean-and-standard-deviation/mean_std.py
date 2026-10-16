# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "matplotlib"]
# ///
"""Numbers and chart for the post "Mean and standard deviation".

Run:  uv run examples/mean-and-standard-deviation/mean_std.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chartstyle as cs  # noqa: E402
import market_data as md  # noqa: E402

OUT = cs.out_dir("mean-and-standard-deviation")

# --- 1. prices to daily returns: SPY, first trading days of 2019 ---------------------
spy = md.prices(["SPY"])["SPY"]
first = spy.iloc[:6]
print("SPY closes:", [f"{v:.2f}" for v in first])
r = first.values[1:] / first.values[:-1] - 1
print("daily returns %:", np.round(r * 100, 2))

# --- 2. mean and standard deviation by hand, on those five days (rounded to 0.1%) ------
rr = np.round(r * 100, 1)                             # percent, one decimal, so a reader can follow
m = rr.mean()
dev = rr - m
print("returns", rr, "mean", round(m, 2))
print("deviations", np.round(dev, 2), "squared", np.round(dev ** 2, 2), "sum", round((dev ** 2).sum(), 2))
var_s = (dev ** 2).sum() / (len(rr) - 1)
print(f"sample variance = {(dev ** 2).sum():.2f}/{len(rr) - 1} = {var_s:.3f}, sample std = {np.sqrt(var_s):.2f}")
print(f"population std (divide by n) = {np.sqrt((dev ** 2).sum() / len(rr)):.2f}")
print("numpy std ddof=1:", rr.std(ddof=1).round(2), " numpy default ddof=0:", rr.std().round(2))

# --- 2b. five years of real data: average daily return and daily risk, annualised -------
rets = md.daily_returns(md.prices())
print("\n2019-2023, per ticker:")
for t in rets:
    x = rets[t]
    print(f"  {t:4s} daily mean {x.mean():.4%}  daily sd {x.std(ddof=1):.3%}  "
          f"annual mean {x.mean() * 252:.1%}  annual sd {x.std(ddof=1) * np.sqrt(252):.1%}  "
          f"$100 -> ${100 * np.prod(1 + x):.0f}")

# --- 3. two funds, same mean, different spread ------------------------------------
rng = np.random.default_rng(3)
days = 252
def fund(sd):
    """Made-up daily returns with exactly mean 0.04% and the given standard deviation."""
    x = rng.normal(0, 1, days)
    return 0.0004 + sd * (x - x.mean()) / x.std(ddof=1)


calm, wild = fund(0.005), fund(0.015)
for name, x in [("calm", calm), ("wild", wild)]:
    print(f"{name}: daily mean {x.mean():.4%}, daily std {x.std(ddof=1):.4%}, "
          f"annualised mean {x.mean() * 252:.2%}, annualised std {x.std(ddof=1) * np.sqrt(252):.2%}, "
          f"$100 becomes ${100 * np.prod(1 + x):.2f}")

# --- 4. why sqrt(252): simulate yearly returns from independent days ------------------
daily_sd = 0.01
years = rng.normal(0, daily_sd, (20_000, 252)).sum(axis=1)
print(f"daily sd {daily_sd:.2%}; sd of 252-day sums {years.std(ddof=1):.4%}; "
      f"daily sd * sqrt(252) = {daily_sd * np.sqrt(252):.4%}")

# --- 5. 68-95-99.7 on normal data vs fat-tailed data -------------------------------
z = rng.normal(0, 1, 1_000_000)
t = rng.standard_t(3, 1_000_000)
t = t / t.std()
for name, x in [("normal", z), ("fat-tailed (t, 3 df)", t)]:
    shares = [np.mean(np.abs(x) <= k) for k in (1, 2, 3)]
    beyond4 = np.mean(np.abs(x) > 4)
    print(f"{name}: within 1/2/3 sd = {shares[0]:.1%} / {shares[1]:.1%} / {shares[2]:.2%}; beyond 4 sd {beyond4:.4%}")

# --- chart ----------------------------------------------------------------------------
cs.apply()
fig, axes = plt.subplots(2, 1, figsize=(8, 7.6))
ax = axes[0]
for x, c, label in [(wild, cs.GREY, "wild fund"), (calm, cs.MAROON, "calm fund")]:
    ax.plot(100 * np.cumprod(1 + x), color=c, lw=1.2, label=label)
ax.set_xlabel("Trading day of the year (1–252)")
ax.set_ylabel("Value of $100 invested ($)")
cs.title(ax, "Two made-up funds with the same average daily return (0.04%)", 1)
ax.legend(fontsize=9)

ax = axes[1]
bins = np.linspace(-0.05, 0.05, 41)
ax.hist(wild * 100, bins=bins * 100, color=cs.GREY, alpha=0.8, label="wild: sd 1.5%")
ax.hist(calm * 100, bins=bins * 100, color=cs.MAROON, alpha=0.8, label="calm: sd 0.5%")
ax.set_xlabel("Daily return (%)")
ax.set_ylabel("Number of days (of 252)")
cs.title(ax, "The same two funds: very different spread of daily returns", 2)
ax.legend(fontsize=9)
cs.save(fig, OUT / "two-funds.svg")
