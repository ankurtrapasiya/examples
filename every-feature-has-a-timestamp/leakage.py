# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "matplotlib", "scikit-learn"]
# ///
"""Numbers and chart for the post "Every feature has a timestamp".

A deliberate leak and how to catch it, scaling fitted on the wrong rows, twin features and
ridge against lasso, a rolling z-score in a crash.  Run:  uv run examples/every-feature-has-a-timestamp/leakage.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chartstyle as cs  # noqa: E402
import market_data as md  # noqa: E402

OUT = cs.out_dir("every-feature-has-a-timestamp")


def rmse(y, yhat):
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def p(label, value):
    print(f"{label:<66} {value}")


df = md.features()
spy_next = (df.xs("SPY", level="ticker")["next_vol"] * md.ANNUALISE).rename("spy_same_month")
spy_last = (df.xs("SPY", level="ticker")["vol_20d"] * md.ANNUALISE).rename("spy_last_month")
d = df.join(spy_next, on="date").join(spy_last, on="date")
tr, te = md.split(d)
y_tr, y_te = tr["next_vol"] * md.ANNUALISE, te["next_vol"] * md.ANNUALISE

print("\n== the leak ==")
for name, cols in [("seven honest features", md.FEATURES), ("plus SPY's volatility over the SAME month (not knowable yet)", md.FEATURES + ["spy_same_month"]),
                   ("plus SPY's volatility over the LAST month (knowable, honest)", md.FEATURES + ["spy_last_month"])]:
    m = LinearRegression().fit(tr[cols], y_tr)
    p(f"{name}: test RMSE", f"{rmse(y_te, m.predict(te[cols])):.2f}")
# catching it: shift the suspicious feature by one month and see the score move
cols = md.FEATURES + ["spy_same_month"]
shifted = d.copy(); shifted["spy_same_month"] = shifted.groupby(level="ticker")["spy_same_month"].shift(20)
s_tr, s_te = md.split(shifted.dropna())
m = LinearRegression().fit(s_tr[cols], s_tr["next_vol"] * md.ANNUALISE)
p("the leak lagged by 20 days: test RMSE", f"{rmse(s_te['next_vol'] * md.ANNUALISE, m.predict(s_te[cols])):.2f}")

print("\n== scaling fitted on the wrong rows ==")
Xtr, Xte = tr[md.FEATURES].to_numpy(), te[md.FEATURES].to_numpy()
sc_right = StandardScaler().fit(Xtr)
sc_wrong = StandardScaler().fit(np.vstack([Xtr, Xte]))
p("vol_20d mean used to centre: training rows only / train+test pooled", f"{md.ANNUALISE * sc_right.mean_[4]:.2f} / {md.ANNUALISE * sc_wrong.mean_[4]:.2f} (% a year)")
p("   the pooled mean knows 2022 was calmer than 2020; the difference is small here, the habit is the point", "")

print("\n== twin features: ridge against lasso ==")
X2 = tr[md.FEATURES].copy(); X2["vol_20d_twin"] = X2["vol_20d"] * 1.02 + np.random.default_rng(0).normal(0, 1e-4, len(X2))
X2te = te[md.FEATURES].copy(); X2te["vol_20d_twin"] = X2te["vol_20d"] * 1.02
cols2 = list(X2.columns)
for name, mk in [("ordinary least squares", LinearRegression()), ("ridge, alpha 1", Ridge(alpha=1.0)), ("lasso, alpha 0.05", Lasso(alpha=0.05, max_iter=20000))]:
    m = make_pipeline(StandardScaler(), mk).fit(X2, y_tr)
    coef = m[-1].coef_
    i, j = cols2.index("vol_20d"), cols2.index("vol_20d_twin")
    p(f"{name}: coefficient on vol_20d / on its twin / their sum", f"{coef[i]:+.2f} / {coef[j]:+.2f} / {coef[i] + coef[j]:+.2f}; test RMSE {rmse(y_te, m.predict(X2te)):.2f}")
alphas = np.logspace(-3, 2, 40)
ridge_path = np.array([make_pipeline(StandardScaler(), Ridge(alpha=a)).fit(tr[md.FEATURES], y_tr)[-1].coef_ for a in alphas])
lasso_path = np.array([make_pipeline(StandardScaler(), Lasso(alpha=a, max_iter=20000)).fit(tr[md.FEATURES], y_tr)[-1].coef_ for a in alphas])
p("lasso: features left non-zero at alpha 0.01 / 0.1 / 1", f"{(np.abs(lasso_path[np.argmin(np.abs(alphas - 0.01))]) > 1e-6).sum()} / {(np.abs(lasso_path[np.argmin(np.abs(alphas - 0.1))]) > 1e-6).sum()} / {(np.abs(lasso_path[np.argmin(np.abs(alphas - 1))]) > 1e-6).sum()}")

print("\n== a rolling z-score in a crash ==")
px = md.prices()["SPY"]
z = (px - px.rolling(60).mean()) / px.rolling(60).std()
p("SPY on 2025-04-08 (the low): price / 60-day mean / 60-day sd / z", f"{px.loc['2025-04-08']:.0f} / {px.rolling(60).mean().loc['2025-04-08']:.0f} / {px.rolling(60).std().loc['2025-04-08']:.1f} / {z.loc['2025-04-08']:.2f}")
p("SPY on 2025-02-27 (a week into the fall): price / mean / sd / z", f"{px.loc['2025-02-27']:.0f} / {px.rolling(60).mean().loc['2025-02-27']:.0f} / {px.rolling(60).std().loc['2025-02-27']:.1f} / {z.loc['2025-02-27']:.2f}")
p("worked: window mean 100, sd 2, price 106 -> z; same gap with sd 10 -> z", f"{(106 - 100) / 2:.1f}; {(106 - 100) / 10:.1f}")

# ------------------------------------------------------------------ chart ----
cs.apply()
fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(8, 3.8 * 3))

cs.title(a1, "The leak test: lag the suspicious feature and watch the score", 1)
names = ["seven honest\nfeatures", "plus SPY's\nsame-month\nvolatility (leak)", "the leak,\nlagged 20 days", "plus SPY's\nlast-month\nvolatility (honest)"]
vals = [rmse(y_te, LinearRegression().fit(tr[md.FEATURES], y_tr).predict(te[md.FEATURES])),
        rmse(y_te, LinearRegression().fit(tr[cols], y_tr).predict(te[cols])),
        rmse(s_te['next_vol'] * md.ANNUALISE, m.predict(s_te[cols])) if False else rmse(s_te['next_vol'] * md.ANNUALISE, LinearRegression().fit(s_tr[cols], s_tr["next_vol"] * md.ANNUALISE).predict(s_te[cols])),
        rmse(y_te, LinearRegression().fit(tr[md.FEATURES + ['spy_last_month']], y_tr).predict(te[md.FEATURES + ['spy_last_month']]))]
a1.bar(names, vals, color=[cs.GREY, cs.MAROON, cs.BLUE, cs.GREY], width=0.55)
for i, v in enumerate(vals):
    a1.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=9)
a1.set_xlabel("Feature set")
a1.set_ylabel("Test RMSE (volatility points)")

cs.title(a2, "Ridge shrinks every coefficient toward zero; lasso drives some exactly to zero", 2)
for k, f in enumerate(md.FEATURES):
    a2.plot(alphas, lasso_path[:, k], color=cs.PALETTE[k % 6], label=f"{f} (lasso)")
    a2.plot(alphas, ridge_path[:, k], color=cs.PALETTE[k % 6], lw=0.9, ls=":", label="_r")
a2.plot([], [], color=cs.GREY, lw=0.9, ls=":", label="dotted: the same feature under ridge")
a2.axhline(0, color=cs.GREY, lw=0.8)
a2.set_xscale("log")
a2.set_xlabel("Penalty strength alpha (log scale)")
a2.set_ylabel("Standardised coefficient")
a2.legend()

cs.title(a3, "A rolling z-score hides a crash: the window's spread balloons with the fall", 3)
a3.plot(z.loc["2025-01":"2025-09"].index, z.loc["2025-01":"2025-09"], color=cs.MAROON, label="SPY price z-score, 60-day window")
a3.axhline(-2, color=cs.BLUE, lw=1, ls="--", label="z = −2")
a3.axhline(0, color=cs.GREY, lw=0.8)
a3.set_xlabel("Date")
a3.set_ylabel("z-score")
a3.legend()
cs.save(fig, OUT / "leakage.svg")
