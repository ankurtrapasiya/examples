# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "scikit-learn", "matplotlib"]
# ///
"""Numbers and chart for the post "Bias and variance".

Because the data is made up, we know the true curve, so bias and variance can be
measured directly: train the same model on many fresh samples and look at how its
predictions behave.
Run:  uv run examples/bias-and-variance/bias_variance.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chartstyle as cs  # noqa: E402
import market_data as md  # noqa: E402

OUT = cs.out_dir("bias-and-variance")
NOISE = 3.0             # standard deviation of the random part of next month's volatility (points)
N_TRAIN, REPEATS = 80, 300
grid = np.linspace(5, 45, 200)
truth = md.vol_curve(grid)
rng = np.random.default_rng(11)

depths = [1, 2, 3, 4, 5, 6, 8, 10, 15]
rows = []
preds_by_depth = {}
for d in depths:
    preds = np.empty((REPEATS, len(grid)))
    test_mse = []
    for k in range(REPEATS):
        x, y = md.vol_samples(N_TRAIN, seed=rng.integers(1 << 31), noise=NOISE)
        m = DecisionTreeRegressor(max_depth=d, random_state=0).fit(x.reshape(-1, 1), y)
        preds[k] = m.predict(grid.reshape(-1, 1))
        y_new = truth + rng.normal(0, NOISE, len(grid))          # fresh months, never seen
        test_mse.append(np.mean((y_new - preds[k]) ** 2))
    avg_pred = preds.mean(axis=0)
    bias2 = np.mean((avg_pred - truth) ** 2)
    var = np.mean(preds.var(axis=0))
    rows.append((d, bias2, var, NOISE ** 2, bias2 + var + NOISE ** 2, np.mean(test_mse)))
    preds_by_depth[d] = preds

print(f"noise variance sigma^2 = {NOISE}^2 = {NOISE ** 2:.4f}")
print("depth   bias^2   variance  noise   sum      measured test MSE")
for d, b2, v, n2, s, mse in rows:
    print(f"{d:5d}  {b2:7.4f}  {v:8.4f}  {n2:.4f}  {s:.4f}   {mse:.4f}")
best = min(rows, key=lambda r: r[5])
print(f"best depth by test MSE: {best[0]}")

# --- chart -------------------------------------------------------------------------------
cs.apply()
fig, axes = plt.subplots(2, 1, figsize=(8, 7.8))
ax = axes[0]
d_arr = np.array([r[0] for r in rows])
ax.plot(d_arr, [r[1] for r in rows], color=cs.BLUE, marker="o", ms=3, label="bias²")
ax.plot(d_arr, [r[2] for r in rows], color=cs.MAROON, marker="o", ms=3, label="variance")
ax.plot(d_arr, [r[5] for r in rows], color=cs.INK, marker="o", ms=3, lw=1.8, label="error on new months")
ax.axhline(NOISE ** 2, color=cs.GREY, ls="--", lw=0.8)
ax.text(10.5, NOISE ** 2 + 0.6, "noise (can't be removed)", fontsize=8)
ax.set_xlabel("Tree depth (more depth = more complex model)")
ax.set_ylabel("Squared error on new months (volatility points²)")
cs.title(ax, "Bias falls and variance rises as the tree gets deeper", 1)
ax.legend(fontsize=9)

ax = axes[1]
for d, c, lw, alpha in [(15, cs.MAROON, 0.7, 0.55), (1, cs.BLUE, 1.8, 0.9)]:
    for k in range(6):
        ax.plot(grid, preds_by_depth[d][k], color=c, lw=lw, alpha=alpha,
                label=f"depth {d}" if k == 0 else None)
ax.plot(grid, truth, color=cs.INK, lw=2, label="true average (the curve we set)")
ax.set_xlabel("This month's volatility (% a year)")
ax.set_ylabel("Next month's volatility (% a year)")
cs.title(ax, "Six training samples each: depth-1 trees agree, depth-15 trees zig-zag", 2)
ax.legend(fontsize=9, loc="lower right")
cs.save(fig, OUT / "bias-variance.svg")
