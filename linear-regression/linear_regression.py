# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "scikit-learn", "matplotlib"]
# ///
"""Numbers and chart for the post "Linear regression".

x = this month's volatility (last 20 trading days), y = next month's, both annualised %.
Run:  uv run examples/linear-regression/linear_regression.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chartstyle as cs  # noqa: E402
import market_data as md  # noqa: E402

OUT = cs.out_dir("linear-regression")

df = md.features()
tr, te = md.split(df)

# --- 1. five months of SPY, by hand ------------------------------------------------------
spy = df.xs("SPY", level="ticker")
# five SPY months at two-month gaps through the 2020 recovery, calm to stressed
dates = ["2020-06-01", "2020-08-03", "2020-10-01", "2020-12-01", "2021-02-01"]
five = spy.loc[[spy.index[spy.index.get_indexer([d], method="nearest")[0]] for d in dates]]
x = np.round(five["vol_20d"].to_numpy() * md.ANNUALISE)      # whole numbers, so the arithmetic is readable
y = np.round(five["next_vol"].to_numpy() * md.ANNUALISE)
print("dates", [d.date().isoformat() for d in five.index])
xm, ym = x.mean(), y.mean()
dx, dy = x - xm, y - ym
print("x (this month's vol %)", x, " y (next month's) ", y)
print("x mean", xm, " y mean", ym)
print("x - mean:", dx, "  y - mean:", dy)
print("products:", dx * dy, "sum", (dx * dy).sum(), "  squares:", dx ** 2, "sum", (dx ** 2).sum())
b1 = (dx * dy).sum() / (dx ** 2).sum()
b0 = ym - b1 * xm
print(f"slope b1 = {(dx * dy).sum()}/{(dx ** 2).sum()} = {b1:.4f}")
print(f"intercept b0 = {ym} - {b1:.4f}*{xm} = {b0:.4f}")
yhat = b0 + b1 * x
res = y - yhat
print("predictions", np.round(yhat, 1), " residuals", np.round(res, 1), " RSS", round((res ** 2).sum(), 1))
r2 = 1 - (res ** 2).sum() / (dy ** 2).sum()
print(f"TSS {(dy ** 2).sum():.1f};  R^2 = 1 - {(res ** 2).sum():.1f}/{(dy ** 2).sum():.1f} = {r2:.4f}")
sk = LinearRegression().fit(x.reshape(-1, 1), y)
print(f"scikit-learn agrees: intercept {sk.intercept_:.4f}, slope {sk.coef_[0]:.4f}")
print(f"forecast when this month's volatility is 20%: {b0 + b1 * 20:.1f}%")
Xd = np.column_stack([np.ones_like(x), x])
print("normal equation gives", np.linalg.inv(Xd.T @ Xd) @ Xd.T @ y)

# --- 2. three years of five tickers, one input then two --------------------------------
A = md.ANNUALISE
x1_tr, x1_te = tr[["vol_20d"]].to_numpy() * A, te[["vol_20d"]].to_numpy() * A
y_tr, y_te = tr["next_vol"].to_numpy() * A, te["next_vol"].to_numpy() * A
one = LinearRegression().fit(x1_tr, y_tr)
print(f"\none input, {len(y_tr):,} training rows (2019-2021): next vol = {one.intercept_:.2f} + {one.coef_[0]:.3f} x this month's vol")
print(f"  training R^2 {r2_score(y_tr, one.predict(x1_tr)):.3f}; test R^2 (2022-2023, {len(y_te):,} rows) {r2_score(y_te, one.predict(x1_te)):.3f}")
cols = ["vol_20d", "ret_20d"]
x2_tr = np.column_stack([tr["vol_20d"] * A, tr["ret_20d"] * 100])
x2_te = np.column_stack([te["vol_20d"] * A, te["ret_20d"] * 100])
two = LinearRegression().fit(x2_tr, y_tr)
print(f"two inputs: next vol = {two.intercept_:.2f} + {two.coef_[0]:.3f} x vol + {two.coef_[1]:.3f} x last-20-day-return(%)")
print(f"  a month that fell 10% instead of being flat -> {two.coef_[1] * -10:+.1f} points of next-month volatility, vol held the same")
print(f"  training R^2 {r2_score(y_tr, two.predict(x2_tr)):.3f}; test R^2 {r2_score(y_te, two.predict(x2_te)):.3f}")

# --- chart -------------------------------------------------------------------------------
cs.apply()
fig, axes = plt.subplots(2, 1, figsize=(8, 7.6))
ax = axes[0]
xs = np.linspace(x.min() - 3, x.max() + 3, 2)
ax.plot(xs, b0 + b1 * xs, color=cs.MAROON, lw=1.5)
for xi, yi, fi in zip(x, y, yhat):
    ax.plot([xi, xi], [yi, fi], color=cs.INK, lw=0.8, ls=":")
ax.scatter(x, y, color=cs.INK, s=30, zorder=3)
ax.set_xlabel("This month's volatility (% a year)")
ax.set_ylabel("Next month's volatility (% a year)")
cs.title(ax, "Five months of SPY: the fitted line, and each miss (dotted)", 1)
ax.text(x.min() - 2, y.max() + 1, f"next = {b0:.1f} + {b1:.2f} x this", color=cs.MAROON, fontsize=9)

ax = axes[1]
sub = tr.xs("SPY", level="ticker")
ax.scatter(sub["vol_20d"] * A, sub["next_vol"] * A, s=6, c=sub["ret_20d"] * 100, cmap="Reds_r", vmin=-15, vmax=5)
xs = np.linspace(5, 80, 2)
for ret, ls in [(0.0, "--"), (-10.0, "-")]:
    ax.plot(xs, two.intercept_ + two.coef_[0] * xs + two.coef_[1] * ret, color=cs.MAROON, ls=ls, lw=1.2,
            label=f"last month's return {ret:+.0f}%")
ax.legend(fontsize=9, loc="upper left")
ax.set_xlabel("This month's volatility (% a year)")
ax.set_ylabel("Next month's volatility (% a year)")
cs.title(ax, "SPY, 2019–2021, one dot per day (darker = the month fell more)", 2)
cs.save(fig, OUT / "linear-regression.svg")
