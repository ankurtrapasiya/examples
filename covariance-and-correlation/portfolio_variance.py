# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "matplotlib"]
# ///
"""Numbers and chart for the post "Covariance, correlation and portfolio variance".

Run:  uv run examples/covariance-and-correlation/portfolio_variance.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chartstyle as cs  # noqa: E402
import market_data as md  # noqa: E402

OUT = cs.out_dir("covariance-and-correlation")

# --- 1. covariance and correlation by hand: SPY and GLD, first five days of 2019 ---------
rets = md.daily_returns(md.prices())
five = (rets[["SPY", "GLD"]].iloc[:5] * 100).round(1)
a, b = five["SPY"].to_numpy(), five["GLD"].to_numpy()     # daily % returns, one decimal
print("dates", [d.date().isoformat() for d in five.index])
am, bm = a.mean(), b.mean()
da, db = a - am, b - bm
print("SPY", a, " GLD", b)
print("means", round(am, 2), round(bm, 2))
print("dev SPY", np.round(da, 2), " dev GLD", np.round(db, 2), " products", np.round(da * db, 2), "sum", round((da * db).sum(), 2))
cov = (da * db).sum() / (len(a) - 1)
sa, sb = a.std(ddof=1), b.std(ddof=1)
corr = cov / (sa * sb)
print(f"cov = {(da * db).sum():.2f}/{len(a) - 1} = {cov:.3f}")
print(f"sd SPY = {sa:.3f}, sd GLD = {sb:.3f}, corr = {cov:.3f}/({sa:.3f}*{sb:.3f}) = {corr:.3f}")
print("numpy cov", np.cov(a, b)[0, 1].round(3), " numpy corrcoef", np.corrcoef(a, b)[0, 1].round(3))

# --- 1b. five years: the correlation matrix and the SPY-GLD pair -------------------------
print("\ncorrelation matrix 2019-2023:")
print(rets.corr().round(2))
ann_sd = rets.std(ddof=1) * np.sqrt(252)
print("annual sd:", ann_sd.round(4).to_dict())
rho_sg = rets["SPY"].corr(rets["GLD"])
print(f"SPY-GLD correlation over five years {rho_sg:.3f}")

# --- 2. two-asset portfolio variance: 60% SPY, 40% GLD, five-year numbers ----------------
s1, s2, rho = round(float(ann_sd["SPY"]), 3), round(float(ann_sd["GLD"]), 3), round(float(rho_sg), 2)
w1, w2 = 0.6, 0.4
var_p = w1**2 * s1**2 + w2**2 * s2**2 + 2 * w1 * w2 * rho * s1 * s2
print(f"\nsd SPY {s1}, sd GLD {s2}, rho {rho}")
print(f"terms: {w1**2 * s1**2:.6f} + {w2**2 * s2**2:.6f} + {2 * w1 * w2 * rho * s1 * s2:.6f} = {var_p:.6f}")
print(f"60/40 portfolio sd = sqrt({var_p:.6f}) = {np.sqrt(var_p):.2%}")
print(f"weighted average of the two sds = {w1 * s1 + w2 * s2:.2%}  (what you'd get with correlation +1)")
real = (0.6 * rets["SPY"] + 0.4 * rets["GLD"]).std(ddof=1) * np.sqrt(252)
print(f"actual sd of a daily-rebalanced 60/40 over the five years {real:.2%}")

# matrix form
w = np.array([w1, w2])
S = np.array([[s1**2, rho * s1 * s2], [rho * s1 * s2, s2**2]])
print("matrix form w'Sw =", round(float(w @ S @ w), 6))

# minimum-variance mix of the two
w_min = (s2**2 - rho * s1 * s2) / (s1**2 + s2**2 - 2 * rho * s1 * s2)
sd_min = np.sqrt(w_min**2 * s1**2 + (1 - w_min)**2 * s2**2 + 2 * w_min * (1 - w_min) * rho * s1 * s2)
print(f"minimum-variance weight in SPY {w_min:.4f}, sd {sd_min:.2%} (GLD alone {s2:.1%})")

# --- 3. many assets: the floor set by average correlation -------------------------------
for r in [0.0, 0.3]:
    for n in [1, 10, 100]:
        sd_eq = np.sqrt(s1**2 / n + (1 - 1 / n) * r * s1**2)
        print(f"  {n:3d} equal-weight assets, sd 20% each, correlation {r}: portfolio sd {sd_eq:.2%}")

# --- chart -------------------------------------------------------------------------------
cs.apply()
fig, ax = plt.subplots(figsize=(8, 4))
ws = np.linspace(0, 1, 201)
styles = [(1.0, cs.GREY, "-"), (0.5, cs.GREY, "--"), (rho, cs.MAROON, "-"), (-0.5, cs.INK, "-"), (-1.0, cs.MAROON, "--")]
for r, c, ls in styles:
    sd = np.sqrt(ws**2 * s1**2 + (1 - ws)**2 * s2**2 + 2 * ws * (1 - ws) * r * s1 * s2)
    ax.plot(ws * 100, sd * 100, color=c, ls=ls, lw=1.4, label=f"correlation {r:+.1f}")
ax.scatter([60], [np.sqrt(var_p) * 100], color=cs.MAROON, zorder=3)
ax.annotate(f"60/40 SPY/GLD at {rho:+.2f}", (60, np.sqrt(var_p) * 100), xytext=(64, 4), fontsize=9,
            arrowprops=dict(arrowstyle="-", color=cs.INK, lw=0.6))
ax.set_xlabel("Share of money in SPY, the rest in GLD (%)")
ax.set_ylabel("Portfolio risk: standard deviation (% a year)")
ax.legend(fontsize=9, loc="upper left")
ax.set_ylim(0, 23)
cs.save(fig, OUT / "portfolio-risk.svg")
