# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "scikit-learn", "matplotlib"]
# ///
"""Numbers and chart for the post "Boosting".

Run:  uv run examples/boosting/boosting.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chartstyle as cs  # noqa: E402
import market_data as md  # noqa: E402

OUT = cs.out_dir("boosting")
print("scikit-learn", sklearn.__version__)

# --- 1. one AdaBoost round by hand, 10 trades, 3 mistakes -------------------------------
w = np.full(10, 0.1)
wrong = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], bool)
eps = w[wrong].sum()
alpha = 0.5 * np.log((1 - eps) / eps)
print(f"error eps = {eps:.1f}; alpha = 0.5*ln(0.7/0.3) = {alpha:.4f}")
print(f"wrong x e^alpha = {np.exp(alpha):.4f}, right x e^-alpha = {np.exp(-alpha):.4f}")
w_new = w * np.exp(np.where(wrong, alpha, -alpha))
print(f"before normalising: wrong {w_new[wrong][0]:.5f} each, right {w_new[~wrong][0]:.5f} each, total {w_new.sum():.4f}")
w_new /= w_new.sum()
print(f"after: wrong {w_new[wrong][0]:.4f} each (sum {w_new[wrong].sum():.4f}), "
      f"right {w_new[~wrong][0]:.4f} each (sum {w_new[~wrong].sum():.4f})")

# --- 2. AdaBoost on the settlement data ----------------------------------------------------
STRESS = 30.0
df = md.features()
tr_all, te_df = md.split(df)
names = md.FEATURES
X_tr, X_te = tr_all[names].to_numpy(), te_df[names].to_numpy()
y_tr = (tr_all["next_vol"].to_numpy() * md.ANNUALISE > STRESS).astype(int)
y_te = (te_df["next_vol"].to_numpy() * md.ANNUALISE > STRESS).astype(int)
print(f"stress months: {y_tr.mean():.1%} of training rows; always saying calm scores {1 - y_te.mean():.1%} on test")

stump = DecisionTreeClassifier(max_depth=1).fit(X_tr, y_tr)
print(f"\none stump test AUC {roc_auc_score(y_te, stump.predict_proba(X_te)[:, 1]):.3f}")

ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=300, learning_rate=1.0,
                         random_state=0).fit(X_tr, y_tr)
for t in range(3):
    e = ada.estimator_errors_[t]
    print(f"round {t + 1}: stump asks about {names[ada.estimators_[t].tree_.feature[0]]}, "
          f"weighted error {e:.4f}, sklearn weight {ada.estimator_weights_[t]:.4f}, "
          f"ln((1-e)/e) = {np.log((1 - e) / e):.4f}")

# --- 3. gradient boosting, with the number of trees chosen on a validation split ----------
# The best round is picked on validation data (early stopping), never on the test set.
# On a time series the validation months come AFTER the fitting months: 2019-2020 fit, 2021 validate.
dates = tr_all.index.get_level_values("date")
fit_mask = dates < "2021-01-01"
X_fit, X_val, y_fit, y_val = X_tr[fit_mask], X_tr[~fit_mask], y_tr[fit_mask], y_tr[~fit_mask]
rf = RandomForestClassifier(n_estimators=300, max_features=3, min_samples_leaf=20, random_state=0, n_jobs=-1).fit(X_tr, y_tr)
rf_auc = roc_auc_score(y_te, rf.predict_proba(X_te)[:, 1])
models = {
    "AdaBoost, stumps": AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=300, learning_rate=1.0, random_state=0),
    "gradient boosting, depth-3 trees, rate 0.1": GradientBoostingClassifier(n_estimators=300, learning_rate=0.1, max_depth=3, random_state=0),
    "gradient boosting, depth-3 trees, rate 1.0": GradientBoostingClassifier(n_estimators=300, learning_rate=1.0, max_depth=3, random_state=0),
}
curves, chosen = {}, {}
print(f"\ntrained on {len(y_fit):,} rows (2019-2020), round chosen on {len(y_val):,} validation rows (2021), scored on {len(y_te):,} test rows (2022-2023)")
for name, model in models.items():
    model.fit(X_fit, y_fit)
    val = [roc_auc_score(y_val, p[:, 1]) for p in model.staged_predict_proba(X_val)]
    test = [roc_auc_score(y_te, p[:, 1]) for p in model.staged_predict_proba(X_te)]
    curves[name] = test
    best = int(np.argmax(val))
    chosen[name] = best + 1
    print(f"{name:44s} test AUC after 10 trees {test[9]:.3f}, 50 {test[49]:.3f}, 300 {test[-1]:.3f}; "
          f"validation picks {best + 1} trees -> test {test[best]:.3f}")
print(f"random forest (from the last post)           test AUC {rf_auc:.3f}")

# --- chart ------------------------------------------------------------------------------------
cs.apply()
fig, ax = plt.subplots(figsize=(8, 3.8))
styles = {"AdaBoost, stumps": (cs.BLUE, "-"), "gradient boosting, depth-3 trees, rate 0.1": (cs.MAROON, "-"),
          "gradient boosting, depth-3 trees, rate 1.0": (cs.MAROON_LIGHT, "--")}
for name, aucs in curves.items():
    c, ls = styles[name]
    ax.plot(np.arange(1, len(aucs) + 1), aucs, color=c, ls=ls, lw=1.5, label=name)
    n = chosen[name]
    ax.plot(n, aucs[n - 1], "o", color=c, ms=5)   # the round the validation split chose
ax.axhline(rf_auc, color=cs.GREY, ls="--", lw=1.2, label=f"random forest ({rf_auc:.3f})")
ax.set_xlabel("Number of trees in the model")
ax.set_ylabel("Test AUC, 2022–2023 (0.5 = coin flip, 1 = perfect)")
ax.set_ylim(0.5, 0.9)
cs.title(ax, "Test score as trees are added (dot = round chosen on validation)")
ax.legend(fontsize=8, loc="lower right")
cs.save(fig, OUT / "boosting.svg")
