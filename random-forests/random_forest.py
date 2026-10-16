# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "scikit-learn", "matplotlib"]
# ///
"""Numbers and chart for the post "Random forests".

Uses scikit-learn's RandomForestClassifier, BaggingClassifier and ExtraTreesClassifier.
Run:  uv run examples/random-forests/random_forest.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chartstyle as cs  # noqa: E402
import market_data as md  # noqa: E402

OUT = cs.out_dir("random-forests")

STRESS = 30.0
df = md.features()
tr, te = md.split(df)
names = md.FEATURES
X_tr, X_te = tr[names].to_numpy(), te[names].to_numpy()
y_tr = (tr["next_vol"].to_numpy() * md.ANNUALISE > STRESS).astype(int)
y_te = (te["next_vol"].to_numpy() * md.ANNUALISE > STRESS).astype(int)
LEAF = 20   # every tree stops at leaves of 20 rows, so only the randomness differs


def first_question(trees):
    """How often each feature is the top question across the trees."""
    counts = {}
    for t in trees:
        f = names[t.tree_.feature[0]]
        counts[f] = counts.get(f, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


models = {
    "bagging (all features at every split)": BaggingClassifier(
        DecisionTreeClassifier(min_samples_leaf=LEAF), n_estimators=300, random_state=0, n_jobs=-1),
    "random forest, 3 of 7 features per split": RandomForestClassifier(
        n_estimators=300, max_features=3, min_samples_leaf=LEAF, random_state=0, n_jobs=-1),
    "random forest, 1 of 7 features per split": RandomForestClassifier(
        n_estimators=300, max_features=1, min_samples_leaf=LEAF, random_state=0, n_jobs=-1),
    "extra trees, 3 of 7, random cut-offs": ExtraTreesClassifier(
        n_estimators=300, max_features=3, min_samples_leaf=LEAF, random_state=0, n_jobs=-1),
}
single = DecisionTreeClassifier(min_samples_leaf=LEAF, random_state=0).fit(X_tr, y_tr)
print(f"test accuracy of always saying calm: {1 - y_te.mean():.3f}")
print(f"one tree (leaves of {LEAF})  test AUC {roc_auc_score(y_te, single.predict_proba(X_te)[:, 1]):.3f}")
for name, m in models.items():
    m.fit(X_tr, y_tr)
    auc = roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])
    print(f"{name:42s} test AUC {auc:.3f}  first question: {first_question(m.estimators_)}")

rf = models["random forest, 3 of 7 features per split"]
print(f"\nout-of-bag accuracy of that forest: {RandomForestClassifier(n_estimators=300, max_features=3, min_samples_leaf=LEAF, oob_score=True, random_state=0, n_jobs=-1).fit(X_tr, y_tr).oob_score_:.3f}")

# --- are the gaps real? same models, five random seeds -------------------------------------
print("\ntest AUC over seeds 0-4 (mean, min-max):")
SEEDS = range(5)
def over_seeds(make):
    aucs = [roc_auc_score(y_te, make(seed).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]) for seed in SEEDS]
    return f"{np.mean(aucs):.3f}  ({min(aucs):.3f}-{max(aucs):.3f})"
print("  bagging                  ", over_seeds(lambda s: BaggingClassifier(DecisionTreeClassifier(min_samples_leaf=LEAF), n_estimators=300, random_state=s, n_jobs=-1)))
print("  random forest, 3 of 7    ", over_seeds(lambda s: RandomForestClassifier(n_estimators=300, max_features=3, min_samples_leaf=LEAF, random_state=s, n_jobs=-1)))
print("  random forest, 1 of 7    ", over_seeds(lambda s: RandomForestClassifier(n_estimators=300, max_features=1, min_samples_leaf=LEAF, random_state=s, n_jobs=-1)))
print("  extra trees, 3 of 7      ", over_seeds(lambda s: ExtraTreesClassifier(n_estimators=300, max_features=3, min_samples_leaf=LEAF, random_state=s, n_jobs=-1)))

# --- feature importance, two ways ------------------------------------------------------------
print("\nimpurity importance:", {n: round(float(v), 3) for n, v in zip(names, rf.feature_importances_)})
perm = permutation_importance(rf, X_te, y_te, scoring="roc_auc", n_repeats=10, random_state=0, n_jobs=-1)
print("permutation importance (drop in test AUC):", {n: round(float(v), 3) for n, v in zip(names, perm.importances_mean)})

# --- chart ------------------------------------------------------------------------------------
cs.apply()
fig, ax = plt.subplots(figsize=(8, 3.4))
order = np.argsort(perm.importances_mean)
labels = {"ret_1d": "1-day return", "ret_5d": "5-day return", "ret_20d": "20-day return", "vol_5d": "5-day volatility",
          "vol_20d": "20-day volatility", "sma_ratio": "price vs 20-day average", "bollinger": "Bollinger position"}
ax.barh([labels[names[i]] for i in order], perm.importances_mean[order], color=cs.MAROON,
        xerr=perm.importances_std[order], ecolor=cs.INK)
ax.axvline(0, color=cs.INK, lw=0.6)
ax.set_xlabel("Drop in test AUC when the feature is shuffled (bar = mean of 10 shuffles, line = spread)")
cs.title(ax, "Permutation importance: which features the forest relies on")
cs.save(fig, OUT / "importance.svg")
