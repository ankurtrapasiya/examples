# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "matplotlib"]
# ///
"""Numbers and chart for the post "Returns compound".

Growth factors, CAGR, arithmetic against geometric mean, volatility drag and recovery, on the
blog's shared prices (SPY, NVDA, MS, OXY, GLD, 2020-2025).
Run:  uv run examples/returns-compound/returns.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chartstyle as cs  # noqa: E402
import market_data as md  # noqa: E402

OUT = cs.out_dir("returns-compound")


def p(label, value):
    print(f"{label:<60} {value}")


px = md.prices()
r = md.daily_returns(px)
years = (px.index[-1] - px.index[0]).days / 365.25

print("\n== growth factors ==")
for t in md.TICKERS:
    total = px[t].iloc[-1] / px[t].iloc[0] - 1
    summed = r[t].sum()
    cagr = (1 + total) ** (1 / years) - 1
    p(f"{t}: product of growth factors - 1 / sum of daily returns", f"{total:+.1%} / {summed:+.1%}")
    p(f"{t}: CAGR / total divided by years", f"{cagr:.2%} / {total / years:.2%}")

print("\n== arithmetic vs geometric, yearly ==")
yearly = px.resample("YE").last().pct_change().dropna()
yearly.loc[pd.Timestamp("2020-12-31")] = px.loc["2020"].iloc[-1] / px.iloc[0] - 1
yearly = yearly.sort_index()
for t in md.TICKERS:
    a = yearly[t].mean()
    g = np.prod(1 + yearly[t]) ** (1 / len(yearly)) - 1
    p(f"{t}: yearly returns", " ".join(f"{x:+.0%}" for x in yearly[t]))
    p(f"{t}: arithmetic mean / geometric mean / gap / sigma^2/2", f"{a:.2%} / {g:.2%} / {a - g:.2%} / {yearly[t].std(ddof=0) ** 2 / 2:.2%}")

print("\n== volatility drag, daily ==")
for t in md.TICKERS:
    mu = r[t].mean() * 252
    sig2 = r[t].var() * 252
    geo = np.prod(1 + r[t]) ** (252 / len(r)) - 1
    p(f"{t}: mean x 252 / mean - sigma^2/2 / actual compound rate", f"{mu:.2%} / {mu - sig2 / 2:.2%} / {geo:.2%}")

print("\n== recovery ==")
for loss in [0.10, 0.20, 0.30, 0.337, 0.50, 0.80]:
    p(f"lose {loss:.1%}: gain needed to get back", f"{1 / (1 - loss) - 1:.1%}")
wealth = (1 + r["SPY"]).cumprod()
dd = wealth / wealth.cummax() - 1
p("SPY max drawdown 2020-2025 / recovery needed", f"{dd.min():.1%} / {1 / (1 + dd.min()) - 1:.1%}")

print("\n== rule of 72 ==")
for rate in [0.03, 0.06, 0.09, 0.12]:
    p(f"{rate:.0%}: rule of 72 / exact ln2 / ln(1+r)", f"{72 / (100 * rate):.1f} / {np.log(2) / np.log(1 + rate):.1f} years")

# ------------------------------------------------------------------ chart ----
cs.apply()
fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(8, 3.8 * 3))

cs.title(a1, "Two paths with the same average return: +50% then −50%, and +5% then −5%", 1)
paths = {"+50% then −50%": [100, 150, 75], "+5% then −5%": [100, 105, 99.75], "0% then 0%": [100, 100, 100]}
for (name, v), col in zip(paths.items(), [cs.MAROON, cs.BLUE, cs.GREY]):
    a1.plot([0, 1, 2], v, color=col, marker="o", label=f"{name}: ends at {v[-1]:g}")
a1.set_xticks([0, 1, 2]); a1.set_xticklabels(["start", "after year 1", "after year 2"])
a1.set_xlabel("Time")
a1.set_ylabel("Value of 100 dollars")
a1.legend()

cs.title(a2, "SPY, 2020 to 2025: adding daily returns against multiplying growth factors", 2)
a2.plot(r.index, 100 * ((1 + r["SPY"]).cumprod() - 1), color=cs.MAROON, label="multiply growth factors (the real account)")
a2.plot(r.index, 100 * r["SPY"].cumsum(), color=cs.BLUE, label="add daily returns (wrong)")
a2.set_xlabel("Date")
a2.set_ylabel("Return since the start (%)")
a2.legend()

cs.title(a3, "The gain needed to recover a loss grows faster than the loss", 3)
losses = np.linspace(0, 0.9, 91)
a3.plot(100 * losses, 100 * (1 / (1 - losses) - 1), color=cs.MAROON, label="gain needed to get back to even")
a3.plot(100 * losses, 100 * losses, color=cs.GREY, lw=1, ls="--", label="the loss itself, for comparison")
a3.scatter([33.7], [100 * (1 / (1 - 0.337) - 1)], color=cs.BLUE, zorder=3, label="SPY's worst fall of 2020 to 2025, 33.7%")
a3.set_xlabel("Loss (%)")
a3.set_ylabel("Gain needed to recover (%)")
a3.set_ylim(0, 400)
a3.legend()
cs.save(fig, OUT / "compounding.svg")
