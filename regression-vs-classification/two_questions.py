# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "scikit-learn", "matplotlib"]
# ///
"""Numbers and chart for the post "Regression vs classification".

Same feature table (examples/market_data.py), two questions about the coming month:
  classification: will it be a stress month (volatility above 30% a year)?   (yes / no)
  regression:     how volatile will it be?                                   (a number)
Run:  uv run examples/regression-vs-classification/two_questions.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chartstyle as cs  # noqa: E402
import market_data as md  # noqa: E402

OUT = cs.out_dir("regression-vs-classification")
STRESS = 30.0   # a month whose daily moves annualise to more than 30% is a stress month

df = md.features()
tr, te = md.split(df)
F = md.FEATURES
X_tr, X_te = tr[F].to_numpy(), te[F].to_numpy()
vol_tr, vol_te = tr["next_vol"].to_numpy() * md.ANNUALISE, te["next_vol"].to_numpy() * md.ANNUALISE
print(f"{len(df):,} ticker-days; train {len(tr):,} (2019-2021), test {len(te):,} (2022-2023)")

# --- classification: will next month be a stress month? ---------------------------------
y_tr, y_te = (vol_tr > STRESS).astype(int), (vol_te > STRESS).astype(int)
print(f"\nclassification: stress months are {y_tr.mean():.1%} of training rows, {y_te.mean():.1%} of test rows")
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(X_tr, y_tr)   # scale, then fit
pred = clf.predict(X_te)
proba = clf.predict_proba(X_te)[:, 1]
print(f"  accuracy {accuracy_score(y_te, pred):.2%}  (always saying 'calm' scores {1 - y_te.mean():.2%})")
print(f"  training accuracy {accuracy_score(y_tr, clf.predict(X_tr)):.2%}")
print(f"  test error rate {np.mean(pred != y_te):.2%}; caught {int(((pred == 1) & (y_te == 1)).sum())} of {int(y_te.sum())} real stress months")
# two example days
spy_te = te[te.index.get_level_values("ticker") == "SPY"]
calm = spy_te.iloc[[spy_te["vol_20d"].to_numpy().argmin()]]      # SPY's calmest month in the test years
wild = spy_te.iloc[[spy_te["vol_20d"].to_numpy().argmax()]]      # and its wildest
for name, row in [("calm", calm), ("wild", wild)]:
    d, t = row.index[0]
    print(f"  {name}: {t} on {d.date()}, last month's volatility {row['vol_20d'].iloc[0] * md.ANNUALISE:.1f}%, "
          f"last 20-day return {row['ret_20d'].iloc[0]:+.1%} -> P(stress) = {clf.predict_proba(row[F].to_numpy())[0, 1]:.2f}; "
          f"actual next month {row['next_vol'].iloc[0] * md.ANNUALISE:.1f}%")

# --- regression: how volatile will next month be? ---------------------------------------
reg = LinearRegression().fit(X_tr, vol_tr)
v_hat = reg.predict(X_te)
print(f"\nregression: test months average {vol_te.mean():.1f}% volatility (train {vol_tr.mean():.1f}%)")
mse = np.mean((vol_te - v_hat) ** 2)
print(f"  test MSE {mse:.1f}  RMSE {np.sqrt(mse):.1f} points  MAE {mean_absolute_error(vol_te, v_hat):.1f} points  R2 {r2_score(vol_te, v_hat):.3f}")
for name, row in [("calm", calm), ("wild", wild)]:
    print(f"  {name} day: forecast {reg.predict(row[F].to_numpy())[0]:.1f}%")

# --- chart: the two kinds of answer ----------------------------------------------------
cs.apply()
fig, axes = plt.subplots(2, 1, figsize=(8, 7.6))
ax = axes[0]
order = np.argsort(proba)
ax.scatter(np.arange(len(proba)), proba[order], s=2, c=np.where(y_te[order] == 1, cs.MAROON, cs.GREY), rasterized=True)
ax.axhline(0.5, color=cs.INK, lw=0.8, ls="--")
ax.text(len(proba) * 0.02, 0.53, "above the line: predict 'stress'", fontsize=9)
cs.title(ax, "Classification: will next month be a stress month?", 1)
ax.set_xlabel("Test ticker-days, sorted by predicted probability (red = a stress month followed)")
ax.set_ylabel("Predicted probability of a stress month")
ax.set_ylim(0, 1)

ax = axes[1]
ax.scatter(vol_te, v_hat, s=6, c=cs.MAROON, alpha=0.5, rasterized=True)
lo, hi = 0, 70
ax.plot([lo, hi], [lo, hi], color=cs.INK, lw=0.8, ls="--")
ax.text(lo + 2, hi - 8, "on the line = perfect guess", fontsize=9)
cs.title(ax, "Regression: how volatile will next month be?", 2)
ax.set_xlabel("Actual volatility of the next 20 days (% a year)")
ax.set_ylabel("Predicted volatility (% a year)")
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
cs.save(fig, OUT / "two-questions.svg")
