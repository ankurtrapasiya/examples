# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "matplotlib"]
# ///
"""Numbers and chart for the post "Portfolio construction: the frontier meets the real constraints".

Arithmetic only, no optimiser: three sizing rules that turn the same forecasts into weights,
a hand-chosen long-only portfolio measured against SPY (active return, tracking error,
information ratio), each position's share of the portfolio's risk, and how far the weights move
when one forecast is nudged or all of them are shrunk towards their mean.
Run:  uv run examples/portfolio-construction-real-constraints/construction.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chartstyle as cs  # noqa: E402
import market_data as md  # noqa: E402

OUT = cs.out_dir("portfolio-construction-real-constraints")
T = md.TICKERS
r = md.daily_returns(md.prices())                      # daily returns, 2019-2023
mu = r.mean() * md.TRADING_DAYS                        # yearly average return
sd = r.std() * np.sqrt(md.TRADING_DAYS)                # yearly standard deviation
cov = r.cov() * md.TRADING_DAYS                        # yearly covariance matrix
corr = r.corr()


def p(label, value):
    print(f"{label:<60} {value}")


def row(w):
    return "  ".join(f"{t} {100 * x:+5.1f}%" for t, x in zip(T, w))


print("Inputs, 2019-2023, yearly")
for t in T:
    p(f"  {t}: average return, standard deviation", f"{100 * mu[t]:5.1f}%  {100 * sd[t]:5.1f}%")
print("Correlation matrix")
print(corr.round(2).to_string())
off = corr.values[np.triu_indices(5, 1)]
p("Average correlation between the four stocks", f"{corr.loc[T[:4], T[:4]].values[np.triu_indices(4, 1)].mean():.2f}")
p("Average correlation of GLD with the four stocks", f"{corr.loc['GLD', T[:4]].mean():.2f}")

# --- 1. forecasts -> weights ------------------------------------------------------------------
print("\n1. Three sizing rules on the same made-up forecasts (yearly return above the benchmark)")
f = pd.Series([0.02, 0.05, 0.01, -0.01, 0.00], index=T)      # the forecasts: made up
p("  forecasts", row(f))
c = 4.0                                                       # one scale for all: 1% forecast -> 4% weight
w_lin = c * f
w_sign = 0.1 * np.sign(f)                                     # every live forecast gets the same 10%
w_risk = f / sd**2
w_risk = w_risk * (w_lin.abs().sum() / w_risk.abs().sum())    # same gross as the linear rule, to compare
for name, w in (("linear, 4 x forecast", w_lin), ("sign, 10% each way", w_sign), ("risk-adjusted, forecast / variance", w_risk)):
    p(f"  {name}", row(w))
    p("     gross, net", f"{100 * w.abs().sum():.0f}%  {100 * w.sum():+.0f}%")
p("  AAPL / SPY weight ratio, linear rule", f"{w_lin['AAPL'] / w_lin['SPY']:.2f}")
p("  AAPL / SPY weight ratio, risk-adjusted rule", f"{w_risk['AAPL'] / w_risk['SPY']:.2f}")
p("  variance ratio AAPL / SPY", f"{sd['AAPL']**2 / sd['SPY']**2:.2f}")

# --- 2. a long-only portfolio against its benchmark ---------------------------------------------
print("\n2. A hand-chosen long-only portfolio against SPY")
w = pd.Series([0.40, 0.20, 0.20, 0.10, 0.10], index=T)
b = pd.Series([1.0, 0, 0, 0, 0], index=T)                      # the benchmark is SPY itself
rp = r @ w
active = rp - r["SPY"]
p("  weights", row(w))
p("  active weights (portfolio minus benchmark)", row(w - b))
p("  active share, half the sum of absolute active weights", f"{100 * 0.5 * (w - b).abs().sum():.0f}%")
p("  portfolio yearly return, standard deviation", f"{100 * md.TRADING_DAYS * rp.mean():.1f}%  {100 * np.sqrt(md.TRADING_DAYS) * rp.std():.1f}%")
p("  SPY yearly return, standard deviation", f"{100 * mu['SPY']:.1f}%  {100 * sd['SPY']:.1f}%")
ar = md.TRADING_DAYS * active.mean()
te = np.sqrt(md.TRADING_DAYS) * active.std()
p("  active return a year", f"{100 * ar:+.1f}%")
p("  tracking error a year", f"{100 * te:.1f}%")
p("  information ratio", f"{ar / te:.2f}")

# --- 3. who is spending the risk ------------------------------------------------------------------
print("\n3. Each position's share of the portfolio's risk")
sig_p = np.sqrt(w @ cov @ w)
mcr = (cov @ w) / sig_p                                       # marginal: how sigma_p moves per unit of weight
rc = w * mcr                                                  # contribution: sums to sigma_p
p("  portfolio standard deviation (from the covariance matrix)", f"{100 * sig_p:.1f}%")
p("  weighted average of the five standard deviations", f"{100 * (w @ sd):.1f}%")
for t in T:
    p(f"  {t}: weight, risk contribution, share of risk", f"{100 * w[t]:4.0f}%  {100 * rc[t]:5.1f}%  {100 * rc[t] / sig_p:4.0f}%")
p("  sum of contributions", f"{100 * rc.sum():.1f}%")
w4 = pd.Series([0.25, 0.25, 0.25, 0.25, 0.0], index=T)
p("  four stocks equal weight: std, average of their stds", f"{100 * np.sqrt(w4 @ cov @ w4):.1f}%  {100 * (w4 @ sd):.1f}%")
w5 = pd.Series([0.2] * 5, index=T)
p("  five assets equal weight: std, average of their stds", f"{100 * np.sqrt(w5 @ cov @ w5):.1f}%  {100 * (w5 @ sd):.1f}%")

# --- 4. the inputs are estimates ------------------------------------------------------------------
print("\n4. Nudging one forecast, and shrinking all of them")
f2 = f.copy()
f2["JPM"] = 0.02                                                # JPM forecast 1% -> 2%
p("  linear weights, JPM forecast 1%", row(c * f))
p("  linear weights, JPM forecast 2%", row(c * f2))
f3 = f.copy()
f3["AAPL"], f3["SPY"] = 0.045, 0.025
p("  risk-adjusted AAPL/SPY ratio, forecasts 5% and 2%", f"{(f['AAPL'] / sd['AAPL']**2) / (f['SPY'] / sd['SPY']**2):.2f}")
p("  risk-adjusted AAPL/SPY ratio, forecasts 4.5% and 2.5%", f"{(f3['AAPL'] / sd['AAPL']**2) / (f3['SPY'] / sd['SPY']**2):.2f}")
fs = 0.5 * f                                                    # shrink every forecast halfway towards "no view" (zero)
p("  forecasts shrunk halfway towards zero", row(fs))
p("  linear weights from the shrunk forecasts, gross", f"{row(c * fs)}   {100 * (c * fs).abs().sum():.0f}%")
w_cap = w_lin.clip(upper=0.15)                                  # a 15% cap on any one name
p("  linear weights with a 15% cap", f"{row(w_cap)}   displaced {100 * (w_lin - w_cap).sum():.0f}%")

# --- chart -----------------------------------------------------------------------------------------
cs.apply()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 3.8 * 2))
x = np.arange(5)
width = 0.27
for i, (name, ww, col) in enumerate((("linear: 4 × forecast", w_lin, cs.MAROON), ("sign: 10% each way", w_sign, cs.BLUE),
                                     ("risk-adjusted: forecast ÷ variance", w_risk, cs.GREY))):
    ax1.bar(x + (i - 1) * width, 100 * ww.values, width=width, color=col, label=name)
ax1.axhline(0, color=cs.GREY, lw=0.8)
ax1.set_xticks(x, T)
ax1.set_xlabel("Ticker (forecasts +2, +5, +1, −1, 0 percent a year)")
ax1.set_ylabel("Weight (% of the portfolio)")
cs.title(ax1, "The same five forecasts, sized by three rules", 1)
ax1.legend()

ax2.bar(x - 0.18, 100 * w.values, width=0.36, color=cs.BLUE, label="weight, % of money")
ax2.bar(x + 0.18, 100 * (rc / sig_p).values, width=0.36, color=cs.MAROON, label="share of the portfolio's risk")
ax2.set_xticks(x, T)
ax2.set_xlabel("Ticker in the 40/20/20/10/10 portfolio")
ax2.set_ylabel("Share (% of the total)")
cs.title(ax2, "Where the money is and where the risk is", 2)
ax2.legend()

cs.save(fig, OUT / "construction.svg")
print(f"\nchart -> {OUT / 'construction.svg'}")
