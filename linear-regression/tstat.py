# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "matplotlib"]
# ///
"""Standard errors, t-statistics and residual plots for the series' volatility regression, for
the "how sure is a coefficient" section of the linear regression post.
Run:  uv run examples/linear-regression/tstat.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chartstyle as cs  # noqa: E402
import market_data as md  # noqa: E402

OUT = cs.out_dir("linear-regression")
df = md.features(["SPY"])
train, test = md.split(df)
x = (train["vol_20d"] * md.ANNUALISE).to_numpy(); y = (train["next_vol"] * md.ANNUALISE).to_numpy()
X = np.column_stack([np.ones_like(x), x])
beta, *_ = np.linalg.lstsq(X, y, rcond=None)
resid = y - X @ beta
n, k = len(y), 2
s2 = (resid ** 2).sum() / (n - k)
cov = s2 * np.linalg.inv(X.T @ X)
se = np.sqrt(np.diag(cov))
print(f"rows {n}; intercept {beta[0]:.2f} (se {se[0]:.2f}, t {beta[0]/se[0]:.1f}); slope {beta[1]:.3f} (se {se[1]:.3f}, t {beta[1]/se[1]:.1f}); 95% interval for the slope {beta[1]-2*se[1]:.3f} to {beta[1]+2*se[1]:.3f}")
print(f"R^2 {1 - resid.var() / y.var():.3f}; residual sd {np.sqrt(s2):.2f}")
# a useless feature: yesterday's 1-day return
x2 = (train["ret_1d"] * 100).to_numpy(); X2 = np.column_stack([np.ones_like(x), x, x2])
b2, *_ = np.linalg.lstsq(X2, y, rcond=None); res2 = y - X2 @ b2
cov2 = ((res2 ** 2).sum() / (n - 3)) * np.linalg.inv(X2.T @ X2); se2 = np.sqrt(np.diag(cov2))
print(f"adding 1-day return: its coefficient {b2[2]:.3f} (se {se2[2]:.3f}, t {b2[2]/se2[2]:.2f}); R^2 {1 - res2.var()/y.var():.3f}")
# residual patterns
yhat = X @ beta
order = np.argsort(yhat)
print(f"residual mean in the lowest fifth of predictions {resid[order[:n//5]].mean():+.2f}, highest fifth {resid[order[-n//5:]].mean():+.2f}")
print(f"residual sd in the lowest fifth {resid[order[:n//5]].std():.2f}, highest fifth {resid[order[-n//5:]].std():.2f}  (a widening fan)")
runs = np.sign(resid); same = (runs[1:] == runs[:-1]).mean()
print(f"share of consecutive residuals with the same sign {same:.2f} (0.50 if independent): serial correlation")

cs.apply()
fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 3.8 * 2))
cs.title(a1, "Residuals against the prediction: a fan that widens to the right", 1)
a1.scatter(yhat, resid, s=6, color=cs.GREY, alpha=0.5, rasterized=True, label="one month")
a1.axhline(0, color=cs.MAROON, lw=1, label="zero: the line is right on average")
a1.set_xlabel("Predicted next-month volatility (% a year)"); a1.set_ylabel("Residual: actual minus predicted"); a1.legend()
cs.title(a2, "Residuals in time order: runs of the same sign, not a random cloud", 2)
a2.plot(train.index.get_level_values("date"), resid, color=cs.MAROON, lw=0.8, label="residual, day by day")
a2.axhline(0, color=cs.GREY, lw=0.8)
a2.set_xlabel("Date"); a2.set_ylabel("Residual"); a2.legend()
cs.save(fig, OUT / "residuals.svg")
