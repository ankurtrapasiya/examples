# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "scikit-learn", "matplotlib"]
# ///
"""Numbers and chart for the post "Bagging".

Uses scikit-learn's BaggingRegressor / BaggingClassifier around decision trees.
Run:  uv run examples/bagging/bagging.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chartstyle as cs  # noqa: E402
import market_data as md  # noqa: E402

OUT = cs.out_dir("bagging")
rng = np.random.default_rng(13)

# --- 1. what a bootstrap sample looks like ---------------------------------------------
ids = np.arange(1, 9)
for k in range(3):
    s = np.sort(rng.choice(ids, size=len(ids), replace=True))
    print(f"bootstrap {k + 1}: {s.tolist()}  left out: {sorted(int(i) for i in set(ids) - set(s))}")
n = 10_000
left_out = np.mean([len(set(range(n)) - set(rng.integers(0, n, n))) / n for _ in range(20)])
print(f"share of rows left out of a bootstrap of size {n:,}: {left_out:.4f}  (1/e = {np.exp(-1):.4f})")

# --- 2. variance falls as we average more trees -------------------------------------------
NOISE, N_TRAIN, REPEATS = 3.0, 80, 200
grid = np.linspace(5, 45, 200)
truth = md.vol_curve(grid)
sizes = [1, 2, 5, 10, 25, 50, 100]
results = []
for B in sizes:
    preds = np.empty((REPEATS, len(grid)))
    for k in range(REPEATS):
        x, y = md.vol_samples(N_TRAIN, seed=rng.integers(1 << 31), noise=NOISE)
        if B == 1:
            m = DecisionTreeRegressor(random_state=0)                 # one fully grown tree, no bootstrap
        else:
            m = BaggingRegressor(DecisionTreeRegressor(), n_estimators=B, random_state=int(rng.integers(1 << 31)))
        preds[k] = m.fit(x.reshape(-1, 1), y).predict(grid.reshape(-1, 1))
    bias2 = np.mean((preds.mean(axis=0) - truth) ** 2)
    var = np.mean(preds.var(axis=0))
    results.append((B, bias2, var, bias2 + var + NOISE ** 2))
print("\ntrees  bias^2   variance  expected error on new months")
for B, b2, v, e in results:
    print(f"{B:5d}  {b2:.4f}   {v:.4f}    {e:.4f}")

# --- 3. classification: will next month be a stress month? -------------------------------
STRESS = 30.0
df = md.features()
tr, te = md.split(df)
X_tr, X_te = tr[md.FEATURES].to_numpy(), te[md.FEATURES].to_numpy()
y_tr = (tr["next_vol"].to_numpy() * md.ANNUALISE > STRESS).astype(int)
y_te = (te["next_vol"].to_numpy() * md.ANNUALISE > STRESS).astype(int)
single = DecisionTreeClassifier(random_state=0).fit(X_tr, y_tr)
bag = BaggingClassifier(DecisionTreeClassifier(), n_estimators=200, oob_score=True,
                        random_state=0, n_jobs=-1).fit(X_tr, y_tr)
print(f"\none full tree    test AUC {roc_auc_score(y_te, single.predict_proba(X_te)[:, 1]):.3f}")
print(f"bag of 200 trees test AUC {roc_auc_score(y_te, bag.predict_proba(X_te)[:, 1]):.3f}")
oob_auc = roc_auc_score(y_tr, bag.oob_decision_function_[:, 1])
print(f"out-of-bag AUC (from training data only) {oob_auc:.3f}")
for d in (3, 5, 8):
    one_d = DecisionTreeClassifier(max_depth=d, random_state=0).fit(X_tr, y_tr)
    bag_d = BaggingClassifier(DecisionTreeClassifier(max_depth=d), n_estimators=200, random_state=0, n_jobs=-1).fit(X_tr, y_tr)
    print(f"depth {d}: one tree test AUC {roc_auc_score(y_te, one_d.predict_proba(X_te)[:, 1]):.3f}, "
          f"bag of 200 test AUC {roc_auc_score(y_te, bag_d.predict_proba(X_te)[:, 1]):.3f}")

# --- chart ------------------------------------------------------------------------------
cs.apply()
fig, axes = plt.subplots(2, 1, figsize=(8, 7.6))
ax = axes[0]
x, y = md.vol_samples(N_TRAIN, seed=99, noise=NOISE)
one = DecisionTreeRegressor(random_state=0).fit(x.reshape(-1, 1), y)
bagged = BaggingRegressor(DecisionTreeRegressor(), n_estimators=100, random_state=0).fit(x.reshape(-1, 1), y)
ax.scatter(x, y, s=10, color=cs.GREY, label="80 months")
ax.plot(grid, one.predict(grid.reshape(-1, 1)), color=cs.MAROON_LIGHT, lw=0.8, label="one deep tree")
ax.plot(grid, bagged.predict(grid.reshape(-1, 1)), color=cs.MAROON, lw=2, label="average of 100 trees")
ax.plot(grid, truth, color=cs.INK, lw=1.2, ls="--", label="true average (the curve we set)")
ax.set_xlabel("This month's volatility (% a year)")
ax.set_ylabel("Next month's volatility (% a year)")
cs.title(ax, "Same 80 months: one deep tree against the average of 100", 1)
ax.legend(fontsize=8, loc="lower right")

ax = axes[1]
Bs = [r[0] for r in results]
ax.plot(Bs, [r[2] for r in results], color=cs.MAROON, marker="o", ms=3, label="variance")
ax.plot(Bs, [r[1] for r in results], color=cs.BLUE, marker="o", ms=3, label="bias²")
ax.set_xscale("log")
ax.set_xticks(Bs)
ax.set_xticklabels([str(b) for b in Bs])
ax.set_xlabel("Number of trees averaged (log scale)")
ax.set_ylabel("Squared error on new months (volatility points²)")
cs.title(ax, "Variance falls with more trees, then levels off; bias stays flat", 2)
ax.legend(fontsize=9)
cs.save(fig, OUT / "bagging.svg")
