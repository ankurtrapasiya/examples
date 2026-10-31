# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "matplotlib"]
# ///
"""Numbers and chart for the post "Portfolio optimisation".

No optimiser here on purpose: we draw many random long-only portfolios and look at
where they land. Inputs are the 2020-2025 yearly figures of five real tickers.
Run:  uv run examples/portfolio-optimisation/random_portfolios.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chartstyle as cs  # noqa: E402
import market_data as md  # noqa: E402

OUT = cs.out_dir("portfolio-optimisation")

rets = md.daily_returns(md.prices())
names = list(rets.columns)                           # SPY, NVDA, GS, XOM, GLD
mu = (rets.mean() * 252).to_numpy()                  # yearly expected return, from the five years
sd = (rets.std(ddof=1) * np.sqrt(252)).to_numpy()    # yearly standard deviation
corr = rets.corr().to_numpy()
cov = np.outer(sd, sd) * corr
N = len(names)
rf = 0.02
assert np.all(np.linalg.eigvalsh(cov) > 0)


def stats(W, mu_=mu):
    ret = W @ mu_
    vol = np.sqrt(np.einsum("ij,jk,ik->i", W, cov, W))
    return ret, vol, (ret - rf) / vol


rng = np.random.default_rng(17)
W = rng.dirichlet(np.ones(N), 50_000)                # random weights, each row sums to 1, all >= 0
ret, vol, shp = stats(W)
for i, n in enumerate(names):
    print(f"{n:16s} return {mu[i]:.1%}  sd {sd[i]:.1%}  Sharpe {(mu[i] - rf) / sd[i]:.3f}")

best, low = np.argmax(shp), np.argmin(vol)
fmt = lambda w: ", ".join(f"{n} {x:.0%}" for n, x in zip(names, w))
print(f"\nbest Sharpe of 50,000 random mixes: {shp[best]:.3f}  return {ret[best]:.2%}  sd {vol[best]:.2%}")
print("   weights:", fmt(W[best]))
print(f"lowest risk of 50,000 random mixes: sd {vol[low]:.2%}  return {ret[low]:.2%}  Sharpe {shp[low]:.3f}")
print("   weights:", fmt(W[low]))
eq = np.full((1, N), 1 / N)
r_eq, v_eq, s_eq = stats(eq)
print(f"equal weights: return {r_eq[0]:.2%}  sd {v_eq[0]:.2%}  Sharpe {s_eq[0]:.3f}")

# sensitivity: nudge one expected return by five percentage points
for t, delta in [("GLD", -0.05), ("NVDA", -0.05)]:
    mu2 = mu.copy()
    mu2[names.index(t)] += delta
    _, _, shp2 = stats(W, mu2)
    b2 = np.argmax(shp2)
    print(f"\nif {t}'s expected return is {mu[names.index(t)]:.1%} {delta:+.0%} = {mu2[names.index(t)]:.1%}:")
    print("   best mix becomes:", fmt(W[b2]))

# --- chart -------------------------------------------------------------------------------------
cs.apply()
fig, ax = plt.subplots(figsize=(8, 5.2))
grey_to_maroon = LinearSegmentedColormap.from_list("g2m", [cs.GREY, cs.MAROON_LIGHT, cs.MAROON])
ax.scatter(vol * 100, ret * 100, c=shp, cmap=grey_to_maroon, s=2, vmin=shp.min(), vmax=shp.max(), rasterized=True)   # 50,000 dots: a bitmap inside the SVG, not 50,000 paths
ax.scatter(sd * 100, mu * 100, color=cs.INK, s=25, zorder=3)
for i, n in enumerate(names):
    ax.annotate(n, (sd[i] * 100, mu[i] * 100), xytext=(5, -3), textcoords="offset points", fontsize=8)
ax.scatter([vol[best] * 100], [ret[best] * 100], color=cs.INK, edgecolors="white", s=70, zorder=4, label="best Sharpe found")
ax.scatter([vol[low] * 100], [ret[low] * 100], color=cs.BLUE, s=60, zorder=4, label="lowest risk found")
x_max, y_max = sd.max() * 100 + 4, mu.max() * 100 + 5     # axes hold every ticker and every mix; nothing is clipped
xs = np.linspace(0, x_max, 2)
ax.plot(xs, rf * 100 + shp[best] * xs, color=cs.MAROON, lw=0.9, ls="--", label="capital market line (cash + best mix)")
ax.scatter([0], [rf * 100], color=cs.MAROON, s=15)
ax.set_xlim(0, x_max)
ax.set_ylim(0, y_max)
ax.set_xlabel("Risk: standard deviation of yearly return (% a year)")
ax.set_ylabel("Expected return (% a year)")
cs.title(ax, "50,000 random mixes of five tickers, 2020–2025 (redder = higher Sharpe ratio)")
ax.legend(fontsize=8, loc="lower right")
cs.save(fig, OUT / "frontier.svg")
