# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "matplotlib"]
# ///
"""Numbers and chart for the post "Beta is a slope".

Beta and alpha by regressing each ticker's daily return on SPY's, CAPM's required return, rolling
beta, and a dollar-neutral against a beta-neutral pair. Blog data: SPY, NVDA, GS, XOM, GLD,
2020-2025.  Run:  uv run examples/beta-is-a-slope/beta.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chartstyle as cs  # noqa: E402
import market_data as md  # noqa: E402

OUT = cs.out_dir("beta-is-a-slope")
RF = 0.02


def p(label, value):
    print(f"{label:<64} {value}")


px = md.prices()
r = md.daily_returns(px)
m = r["SPY"]

print("\n== beta and alpha, daily regression on SPY, 2020-2025 ==")
rows = {}
for t in ["NVDA", "GS", "XOM", "GLD"]:
    y = r[t]
    beta = y.cov(m) / m.var()
    alpha_d = y.mean() - beta * m.mean()
    rho = y.corr(m)
    resid = y - (alpha_d + beta * m)
    r2 = 1 - resid.var() / y.var()
    se = np.sqrt(resid.var(ddof=2) / ((m - m.mean()) ** 2).sum())
    rows[t] = dict(beta=beta, alpha=alpha_d * 252, rho=rho, r2=r2, sig=y.std() * np.sqrt(252), se=se)
    p(f"{t}: beta / rho x sigma_i/sigma_m", f"{beta:.2f} / {rho:.2f} x {y.std() / m.std():.2f} = {rho * y.std() / m.std():.2f}")
    p(f"{t}: alpha (daily x 252) / R^2 / beta standard error", f"{alpha_d * 252:+.1%} / {r2:.2f} / {se:.3f}")
p("SPY sigma (annual)", f"{m.std() * np.sqrt(252):.1%}")

print("\n== CAPM: required return vs what happened ==")
rm = (1 + m).prod() ** (252 / len(m)) - 1
p("market (SPY) compound annual return", f"{rm:.1%}")
for t, d in rows.items():
    actual = (1 + r[t]).prod() ** (252 / len(r)) - 1
    required = RF + d["beta"] * (rm - RF)
    p(f"{t}: required Rf + beta(Rm - Rf) / actual / Jensen's alpha", f"{required:.1%} / {actual:.1%} / {actual - required:+.1%}")

print("\n== rolling one-year beta ==")
roll = {t: r[t].rolling(252).cov(m) / m.rolling(252).var() for t in ["NVDA", "GS", "XOM", "GLD"]}
for t, s in roll.items():
    p(f"{t}: rolling 252-day beta, min / max", f"{s.min():.2f} / {s.max():.2f}")

print("\n== hedging a long NVDA book against SPY, 2024 ==")
tr, te = r.loc[:"2023"], r.loc["2024"]
b_a = tr["NVDA"].cov(tr["SPY"]) / tr["SPY"].var()
p("NVDA beta estimated on 2020-2023", f"{b_a:.2f}")
long_only = te["NVDA"]
dollar = te["NVDA"] - te["SPY"]
beta_n = te["NVDA"] - b_a * te["SPY"]
for name, s in [("long NVDA only", long_only), ("dollar neutral: short $1 SPY per $1 NVDA", dollar), (f"beta neutral: short ${b_a:.2f} SPY per $1 NVDA", beta_n)]:
    p(f"2024 {name}: sum of daily P&L / daily sigma / beta to SPY", f"{s.sum():+.1%} / {s.std():.2%} / {s.cov(te['SPY']) / te['SPY'].var():+.2f}")
p("2024 SPY return", f"{(1 + te['SPY']).prod() - 1:+.1%}")

# ------------------------------------------------------------------ chart ----
cs.apply()
fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(8, 3.8 * 3))

cs.title(a1, "NVDA's daily return against SPY's: beta is the slope of the fitted line", 1)
a1.scatter(100 * m, 100 * r["NVDA"], s=6, color=cs.GREY, alpha=0.5, rasterized=True, label="one trading day")
xs = np.linspace(m.min(), m.max(), 50)
d = rows["NVDA"]
a1.plot(100 * xs, 100 * (d["alpha"] / 252 + d["beta"] * xs), color=cs.MAROON, label=f"fitted line, slope {d['beta']:.2f}")
a1.plot(100 * xs, 100 * xs, color=cs.BLUE, lw=1, ls="--", label="slope 1: moving exactly with the market")
a1.set_xlabel("SPY daily return (%)")
a1.set_ylabel("NVDA daily return (%)")
a1.legend()

cs.title(a2, "Beta is not a constant: one-year rolling beta of four tickers", 2)
for t, col in zip(["NVDA", "GS", "XOM", "GLD"], [cs.MAROON, cs.BLUE, cs.AMBER, cs.GREEN]):
    a2.plot(roll[t].index, roll[t], color=col, label=f"{t}")
a2.axhline(1, color=cs.GREY, lw=1, ls="--", label="beta 1: the market itself")
a2.set_xlabel("Date (end of the one-year window)")
a2.set_ylabel("Beta to SPY")
a2.legend()

cs.title(a3, "2024: a long NVDA book hedged by dollars, or by beta", 3)
for name, s, col in [("long NVDA, no hedge", long_only, cs.GREY), ("short 1 dollar of SPY per dollar of NVDA", dollar, cs.BLUE),
                     (f"short {b_a:.2f} dollars of SPY per dollar of NVDA (beta neutral)", beta_n, cs.MAROON)]:
    a3.plot(s.index, 100 * s.cumsum(), color=col, label=name)
a3.axhline(0, color=cs.GREY, lw=0.8)
a3.set_xlabel("Date")
a3.set_ylabel("Cumulative profit (% of the NVDA position)")
a3.legend()
cs.save(fig, OUT / "beta.svg")
