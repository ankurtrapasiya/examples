# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "matplotlib", "scikit-learn"]
# ///
"""Numbers and chart for the post "The data is part of the model".

Random against time-ordered splits, walk-forward folds, what filling missing values does, and
how an outlier moves a squared-loss fit against an absolute-loss fit. Blog data, scikit-learn.
Run:  uv run examples/data-is-part-of-the-model/sampling.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import HuberRegressor, LinearRegression, QuantileRegressor
from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chartstyle as cs  # noqa: E402
import market_data as md  # noqa: E402

OUT = cs.out_dir("data-is-part-of-the-model")


def rmse(y, yhat):
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def p(label, value):
    print(f"{label:<66} {value}")


df = md.features().reset_index().sort_values("date")
X = df[md.FEATURES].to_numpy(); y = (df["next_vol"] * md.ANNUALISE).to_numpy()
is_train = (df["date"] < "2024-01-01").to_numpy()
Xtr, ytr, Xte, yte = X[is_train], y[is_train], X[~is_train], y[~is_train]

print("\n== the split is part of the model ==")
rf = lambda: RandomForestRegressor(300, min_samples_leaf=5, random_state=0, n_jobs=-1)
m = rf().fit(Xtr, ytr)
p("random forest, time split (fit 2020-23, score 2024-25): test RMSE", f"{rmse(yte, m.predict(Xte)):.2f}")
kf = cross_val_score(rf(), X, y, cv=KFold(5, shuffle=True, random_state=0), scoring="neg_root_mean_squared_error")
p("same model, 5-fold shuffled cross-validation RMSE (leaks)", f"{-kf.mean():.2f} (folds {' '.join(f'{-v:.2f}' for v in kf)})")
ts = cross_val_score(rf(), X, y, cv=TimeSeriesSplit(5), scoring="neg_root_mean_squared_error")
p("same model, 5 forward folds RMSE", f"{-ts.mean():.2f} (folds {' '.join(f'{-v:.2f}' for v in ts)})")
p("baseline, predict the training mean: test RMSE", f"{rmse(yte, np.full_like(yte, ytr.mean())):.2f}")

print("\n== filling missing values ==")
rng = np.random.default_rng(0)
col = md.FEATURES.index("vol_20d")
Xm = Xtr.copy()
# values go missing *because* volatility is high (a halted stock, a missing print): not at random
mask = rng.random(len(Xm)) < np.clip((Xm[:, col] - Xm[:, col].mean()) / Xm[:, col].std(), 0, None) * 0.4
p("rows with vol_20d missing (more likely when volatility is high)", f"{mask.sum()} of {len(Xm)} ({mask.mean():.1%})")
for name, fill in [("drop the rows", None), ("fill with the column mean", Xm[~mask, col].mean()), ("fill with the mean and add a 'was missing' flag", "flag")]:
    Xf = Xm.copy()
    if fill is None:
        Xf, yf = Xm[~mask], ytr[~mask]; Xt = Xte
    elif fill == "flag":
        Xf[mask, col] = Xm[~mask, col].mean(); Xf = np.hstack([Xf, mask[:, None].astype(float)]); yf = ytr
        Xt = np.hstack([Xte, np.zeros((len(Xte), 1))])
    else:
        Xf[mask, col] = fill; yf = ytr; Xt = Xte
    m = rf().fit(Xf, yf)
    p(f"   {name}: test RMSE", f"{rmse(yte, m.predict(Xt)):.2f}")
m = rf().fit(Xtr, ytr); p("   nothing missing (for reference): test RMSE", f"{rmse(yte, m.predict(Xte)):.2f}")

print("\n== one outlier against two losses ==")
rng = np.random.default_rng(2)
xs = np.linspace(0, 10, 40); ys = 2 + 0.5 * xs + rng.normal(0, 0.5, 40)
ys_out = ys.copy(); ys_out[-1] = 30.0
fits = {}
for name, mk in [("squared loss (ordinary least squares)", LinearRegression()), ("absolute loss (median regression)", QuantileRegressor(quantile=0.5, alpha=0, solver="highs")), ("Huber loss", HuberRegressor())]:
    clean = mk.fit(xs[:, None], ys); s1, i1 = float(clean.coef_[0]), float(clean.intercept_)
    dirty = mk.fit(xs[:, None], ys_out); s2, i2 = float(dirty.coef_[0]), float(dirty.intercept_)
    fits[name] = (s1, i1, s2, i2)
    p(f"{name}: slope clean / with one outlier", f"{s1:.2f} / {s2:.2f}")
res = np.array([1, 1, 10])
p("residuals 1, 1, 10: squared total and the outlier's share", f"{(res ** 2).sum()} and {100 / 102:.0%}")
p("residuals 1, 1, 10: absolute total and the outlier's share", f"{res.sum()} and {10 / 12:.0%}")

# ------------------------------------------------------------------ chart ----
cs.apply()
fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(8, 3.8 * 3))

cs.title(a1, "Three ways to score the same model: the shuffled folds are the only ones tested on the crash year", 1)
names = ["time split\n(fit 2020-23,\nscore 2024-25)", "5 shuffled folds\n(leaks)", "5 forward folds", "baseline:\npredict the mean"]
vals = [rmse(yte, rf().fit(Xtr, ytr).predict(Xte)), -kf.mean(), -ts.mean(), rmse(yte, np.full_like(yte, ytr.mean()))]
a1.bar(names, vals, color=[cs.MAROON, cs.BLUE, cs.MAROON, cs.GREY], width=0.55, label="_x")
for i, v in enumerate(vals):
    a1.text(i, v + 0.15, f"{v:.2f}", ha="center", fontsize=9)
a1.set_xlabel("How the model was scored")
a1.set_ylabel("RMSE (volatility points)")

cs.title(a2, "Forward folds: each fold trains on everything before it and tests on what comes next", 2)
dates = df["date"].to_numpy()
for k, (tr_idx, te_idx) in enumerate(TimeSeriesSplit(5).split(X)):
    a2.plot([dates[tr_idx[0]], dates[tr_idx[-1]]], [k, k], color=cs.BLUE, lw=6, solid_capstyle="butt", label="training rows" if k == 0 else "_x")
    a2.plot([dates[te_idx[0]], dates[te_idx[-1]]], [k, k], color=cs.MAROON, lw=6, solid_capstyle="butt", label="test rows" if k == 0 else "_x")
a2.set_yticks(range(5)); a2.set_yticklabels([f"fold {k + 1}" for k in range(5)])
a2.set_xlabel("Date")
a2.set_ylabel("Fold")
a2.legend()

cs.title(a3, "One outlier: the squared-loss line tilts toward it, the absolute-loss line barely moves", 3)
a3.scatter(xs, ys_out, s=14, color=cs.GREY, label="40 points, one of them wrong by 25")
for (name, (s1, i1, s2, i2)), col in zip(fits.items(), [cs.MAROON, cs.BLUE, cs.AMBER]):
    a3.plot(xs, i2 + s2 * xs, color=col, label=f"{name.split(' (')[0]}, with the outlier: slope {s2:.2f}")
a3.plot(xs, fits["squared loss (ordinary least squares)"][1] + fits["squared loss (ordinary least squares)"][0] * xs, color=cs.GREY, lw=1, ls="--", label="the fit without the outlier: slope 0.50")
a3.set_ylim(0, 12)
a3.set_xlabel("x")
a3.set_ylabel("y")
a3.legend()
cs.save(fig, OUT / "sampling.svg")
