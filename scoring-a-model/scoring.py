# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "matplotlib", "scikit-learn"]
# ///
"""Numbers and chart for the post "Scoring a model".

Regression metrics on the next-month volatility problem, then classification metrics on the
stress-month problem: confusion matrix, thresholds, precision and recall, ROC and AUC. Blog data,
scikit-learn.  Run:  uv run examples/scoring-a-model/scoring.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chartstyle as cs  # noqa: E402
import market_data as md  # noqa: E402

OUT = cs.out_dir("scoring-a-model")
STRESS = 30.0   # a month with annualised volatility above this is a "stress month", as in the series


def p(label, value):
    print(f"{label:<66} {value}")


df = md.features()
train, test = md.split(df)
Xtr, Xte = train[md.FEATURES].to_numpy(), test[md.FEATURES].to_numpy()
ytr, yte = (train["next_vol"] * md.ANNUALISE).to_numpy(), (test["next_vol"] * md.ANNUALISE).to_numpy()

print("\n== regression metrics: next month's volatility ==")
lin = LinearRegression().fit(Xtr, ytr)
pred = lin.predict(Xte)
base = np.full_like(yte, ytr.mean())
for name, yhat in [("predict the training mean (baseline)", base), ("linear regression", pred)]:
    e = yte - yhat
    p(f"{name}: MAE / MSE / RMSE", f"{np.abs(e).mean():.2f} / {(e ** 2).mean():.1f} / {np.sqrt((e ** 2).mean()):.2f}")
p("linear: largest single miss", f"{np.abs(yte - pred).max():.1f} volatility points")
p("misses 1, 1, 1, 5: MAE / MSE / RMSE", f"{np.mean([1, 1, 1, 5]):.2f} / {np.mean([1, 1, 1, 25]):.0f} / {np.sqrt(np.mean([1, 1, 1, 25])):.2f}")

print("\n== classification metrics: stress month or not ==")
ctr, cte = (ytr > STRESS).astype(int), (yte > STRESS).astype(int)
p("share of stress months, train / test", f"{ctr.mean():.1%} / {cte.mean():.1%}")
clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=20, random_state=0, n_jobs=-1).fit(Xtr, ctr)
prob = clf.predict_proba(Xte)[:, 1]
p("AUC", f"{roc_auc_score(cte, prob):.3f}")
p("accuracy of always saying 'calm'", f"{1 - cte.mean():.1%}")
rows = []
for thr in [0.5, 0.3, 0.2]:
    yhat = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(cte, yhat).ravel()
    acc = (tp + tn) / len(cte); rec = tp / (tp + fn); spec = tn / (tn + fp); prec = tp / (tp + fp) if tp + fp else float("nan")
    rows.append((thr, tn, fp, fn, tp, acc, rec, spec, prec))
    p(f"threshold {thr}: TP / FP / FN / TN", f"{tp} / {fp} / {fn} / {tn}")
    p(f"   accuracy / recall (sensitivity) / specificity / precision", f"{acc:.1%} / {rec:.1%} / {spec:.1%} / {prec:.1%}")
fpr, tpr, thr = roc_curve(cte, prob)

# ------------------------------------------------------------------ chart ----
cs.apply()
fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(8, 3.8 * 3))

cs.title(a1, "Predicted against actual next-month volatility on the test months", 1)
a1.scatter(yte, pred, s=6, color=cs.GREY, alpha=0.5, rasterized=True, label="one ticker-month")
lim = [0, max(yte.max(), pred.max()) * 1.02]
a1.plot(lim, lim, color=cs.MAROON, lw=1, ls="--", label="perfect prediction")
a1.axhline(ytr.mean(), color=cs.BLUE, lw=1, ls=":", label=f"the baseline: always predict {ytr.mean():.1f}")
a1.set_xlabel("Actual next-month volatility (% a year)")
a1.set_ylabel("Predicted (% a year)")
a1.legend()

cs.title(a2, "The same probabilities, three thresholds: what each one catches and what it costs", 2)
labels = ["caught stress months (recall)", "false alarms among calm months", "precision of a stress call"]
x = np.arange(3); wdt = 0.25
for i, (t, tn, fp, fn, tp, acc, rec, spec, prec) in enumerate(rows):
    vals = [100 * rec, 100 * fp / (fp + tn), 100 * prec]
    a2.bar(x + (i - 1) * wdt, vals, wdt, color=[cs.MAROON, cs.BLUE, cs.AMBER][i], label=f"threshold {t}")
a2.set_xticks(x); a2.set_xticklabels(labels)
a2.set_xlabel("Measure")
a2.set_ylabel("Percent")
a2.legend()

cs.title(a3, "ROC curve: every threshold at once", 3)
a3.plot(100 * fpr, 100 * tpr, color=cs.MAROON, label=f"random forest, AUC {roc_auc_score(cte, prob):.2f}")
a3.plot([0, 100], [0, 100], color=cs.GREY, lw=1, ls="--", label="coin flip, AUC 0.5")
for t, tn, fp, fn, tp, acc, rec, spec, prec in rows:
    a3.scatter([100 * fp / (fp + tn)], [100 * rec], color=cs.BLUE, zorder=3, label=f"threshold {t}" if t == 0.5 else f"_t{t}")
a3.set_xlabel("False alarm rate (% of calm months called stress)")
a3.set_ylabel("Recall (% of stress months caught)")
a3.legend()
cs.save(fig, OUT / "scoring.svg")
