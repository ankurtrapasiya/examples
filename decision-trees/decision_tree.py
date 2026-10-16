# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "scikit-learn"]
# ///
"""Numbers for the post "Decision trees" (the tree diagram in the post is drawn from the rules printed here).

Classification: will next month be a stress month (volatility above 30% a year)?
Run:  uv run examples/decision-trees/decision_tree.py
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier, export_text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import market_data as md  # noqa: E402

STRESS = 30.0
df = md.features()
tr, te = md.split(df)
F = md.FEATURES
# features in the units the post quotes: returns in %, volatilities in % a year
scale = np.array([100, 100, 100, md.ANNUALISE, md.ANNUALISE, 100, 1])
labels = ["1-day return %", "5-day return %", "20-day return %", "5-day vol %", "20-day vol %", "price vs 20-day avg %", "Bollinger"]
X_tr, X_te = tr[F].to_numpy() * scale, te[F].to_numpy() * scale
y_tr, y_te = (tr["next_vol"].to_numpy() * md.ANNUALISE > STRESS).astype(int), (te["next_vol"].to_numpy() * md.ANNUALISE > STRESS).astype(int)


def gini(counts):
    p = np.asarray(counts) / np.sum(counts)
    return 1 - np.sum(p ** 2)


# --- 1. Gini by hand for the first question ------------------------------------------
n, stress = len(y_tr), int(y_tr.sum())
print(f"training rows {n:,}, stress months {stress} ({stress / n:.2%})")
g_root = gini([n - stress, stress])
print(f"root Gini = 1 - ({(n - stress) / n:.4f}^2 + {stress / n:.4f}^2) = {g_root:.4f}")

print("\nbest single question on each feature (a depth-1 tree per feature):")
for j, name in enumerate(labels):
    stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=20, random_state=0).fit(X_tr[:, [j]], y_tr)
    thr = stump.tree_.threshold[0]
    left = X_tr[:, j] <= thr
    nl, nr = left.sum(), (~left).sum()
    sl, sr = y_tr[left].sum(), y_tr[~left].sum()
    gl, gr = gini([nl - sl, sl]), gini([nr - sr, sr])
    weighted = nl / n * gl + nr / n * gr
    print(f"  {name:22s} <= {thr:7.2f}: {nl:5d} rows, {sl:4d} stress (Gini {gl:.4f}) | > : {nr:5d} rows, {sr:4d} stress (Gini {gr:.4f}) "
          f"| weighted {weighted:.4f}, decrease {g_root - weighted:.4f}")

# --- 2. a small, readable tree ---------------------------------------------------------
small = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, random_state=0).fit(X_tr, y_tr)
print()
print(export_text(small, feature_names=labels, decimals=1, show_weights=True))
print(f"depth-3 tree test AUC {roc_auc_score(y_te, small.predict_proba(X_te)[:, 1]):.3f}")
print("feature importance:", {k: round(float(v), 3) for k, v in zip(labels, small.feature_importances_)})
spy = te[te.index.get_level_values("ticker") == "SPY"]
for name, row in [("calm", spy.iloc[[spy["vol_20d"].to_numpy().argmin()]]), ("wild", spy.iloc[[spy["vol_20d"].to_numpy().argmax()]])]:
    xr = row[F].to_numpy() * scale
    print(f"{name} SPY day {row.index[0][0].date()}: 20-day vol {xr[0, 4]:.1f}%, 5-day vol {xr[0, 3]:.1f}%, 20-day return {xr[0, 2]:+.1f}% "
          f"-> P(stress) {small.predict_proba(xr)[0, 1]:.3f}")

# --- 3. deeper trees: training score keeps rising, test score doesn't -------------------
print()
for d in [1, 2, 3, 5, 8, 12, None]:
    m = DecisionTreeClassifier(max_depth=d, random_state=0).fit(X_tr, y_tr)
    a_tr = roc_auc_score(y_tr, m.predict_proba(X_tr)[:, 1])
    a_te = roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])
    print(f"max_depth {str(d):>4}: leaves {m.get_n_leaves():5d}  train AUC {a_tr:.3f}  test AUC {a_te:.3f}")
