# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "scikit-learn", "matplotlib"]
# ///
"""Numbers and chart for the post "Building models with scikit-learn".

Run:  uv run examples/scikit-learn/stress_month.py

The stress-month problem from the ML series: will the next 20 trading days be volatile
(annualised volatility above 30%)? Features known at today's close, one row per ticker and day.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, KFold, TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chartstyle as cs  # noqa: E402
import market_data as md  # noqa: E402

OUT = cs.out_dir("scikit-learn")
OUT.mkdir(parents=True, exist_ok=True)
print("scikit-learn", sklearn.__version__)

# --- the data ------------------------------------------------------------------------
STRESS = 30.0
df = md.features().reset_index()                       # columns: date, ticker, features, targets
df["stress"] = (df["next_vol"] * md.ANNUALISE > STRESS).astype(int)
df = df.sort_values(["date", "ticker"]).reset_index(drop=True)   # time order, for the splits
train, test = df[df["date"] < "2022-01-01"], df[df["date"] >= "2022-01-01"]
X_tr, y_tr = train[md.FEATURES + ["ticker"]], train["stress"]
X_te, y_te = test[md.FEATURES + ["ticker"]], test["stress"]
print(f"rows: train {len(train)}, test {len(test)}; stress months: train {y_tr.mean():.1%}, test {y_te.mean():.1%}")


def auc(model, X=X_te, y=y_te):
    return roc_auc_score(y, model.predict_proba(X)[:, 1])


# --- 1. the estimator API: three models, one interface ------------------------------
def prep():
    """Scale the numeric columns, one-hot the ticker. A fresh object per pipeline: estimators are
    stateful, and Pipeline fits its steps in place, so two pipelines must not share one."""
    return ColumnTransformer([("num", StandardScaler(), md.FEATURES),
                              ("ticker", OneHotEncoder(handle_unknown="ignore"), ["ticker"])])


models = {
    "always 'calm' (DummyClassifier)": DummyClassifier(strategy="prior"),
    "logistic regression": Pipeline([("prep", prep()), ("model", LogisticRegression(max_iter=1000))]),
    "random forest": Pipeline([("prep", prep()), ("model", RandomForestClassifier(
        n_estimators=300, max_features=3, min_samples_leaf=20, random_state=0, n_jobs=-1))]),
    "gradient boosting (HistGradientBoosting)": Pipeline([("prep", prep()), ("model", HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=200, random_state=0))]),
}
print("\n1. same interface, four models (fit on 2019-2021, scored on 2022-2023):")
fitted = {}
for name, m in models.items():
    m.fit(X_tr, y_tr)
    fitted[name] = m
    acc = m.score(X_te, y_te)
    print(f"  {name:42s} accuracy {acc:.3f}   AUC {auc(m):.3f}   avg precision {average_precision_score(y_te, m.predict_proba(X_te)[:, 1]):.3f}")

# --- 2. accuracy is the wrong score here: the confusion matrix -------------------------
lr = fitted["logistic regression"]
print("\n2. logistic regression at the default 0.5 threshold:")
print(confusion_matrix(y_te, lr.predict(X_te)))
print(classification_report(y_te, lr.predict(X_te), target_names=["calm", "stress"], digits=3))
lr_bal = Pipeline([("prep", prep()), ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))]).fit(X_tr, y_tr)
print("   with class_weight='balanced':")
print(confusion_matrix(y_te, lr_bal.predict(X_te)))
print(f"   AUC unchanged? plain {auc(lr):.3f}, balanced {auc(lr_bal):.3f}")

# --- 3. cross-validation: shuffled folds lie on time series ----------------------------
print("\n3. cross-validated AUC of the gradient boosting pipeline on 2019-2021 only:")
gb = models["gradient boosting (HistGradientBoosting)"]
kf = cross_val_score(gb, X_tr, y_tr, cv=KFold(5, shuffle=True, random_state=0), scoring="roc_auc")
ts = cross_val_score(gb, X_tr, y_tr, cv=TimeSeriesSplit(5), scoring="roc_auc")
print(f"  KFold, shuffled:  folds {np.round(kf, 3)}  mean {kf.mean():.3f}")
print(f"  TimeSeriesSplit:  folds {np.round(ts, 3)}  mean {ts.mean():.3f}")
print(f"  actual 2022-2023 test AUC: {auc(fitted['gradient boosting (HistGradientBoosting)']):.3f}")

# --- 4. GridSearchCV with the right splitter ----------------------------------------
print("\n4. grid search over the boosting settings, TimeSeriesSplit(5), scored by AUC:")
grid = GridSearchCV(gb, {"model__learning_rate": [0.01, 0.03, 0.1, 0.3], "model__max_depth": [2, 3, 5]},
                    cv=TimeSeriesSplit(5), scoring="roc_auc", n_jobs=-1)
grid.fit(X_tr, y_tr)
res = pd.DataFrame(grid.cv_results_)[["param_model__learning_rate", "param_model__max_depth", "mean_test_score", "std_test_score"]]
res.columns = ["learning_rate", "max_depth", "validation AUC", "sd"]
print(res.sort_values("validation AUC", ascending=False).to_string(index=False))
print(f"  best: {grid.best_params_}  validation AUC {grid.best_score_:.3f}  test AUC {auc(grid.best_estimator_):.3f}")
lr_grid = GridSearchCV(lr, {"model__C": np.logspace(-3, 2, 11)}, cv=TimeSeriesSplit(5), scoring="roc_auc", n_jobs=-1).fit(X_tr, y_tr)
print(f"  logistic regression, best C {lr_grid.best_params_['model__C']:.3g}: validation AUC {lr_grid.best_score_:.3f}, test AUC {auc(lr_grid.best_estimator_):.3f}")


# --- 5. a custom transformer that fits the same interface ----------------------------
class Annualise(BaseEstimator, TransformerMixin):
    """Turn the daily volatility columns into % a year, so their numbers mean something."""
    def __init__(self, columns=("vol_5d", "vol_20d")):
        self.columns = columns

    def fit(self, X, y=None):
        return self                                      # nothing to learn

    def transform(self, X):
        X = X.copy()
        X[list(self.columns)] = X[list(self.columns)] * md.ANNUALISE
        return X


custom = make_pipeline(Annualise(), prep(), LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)
print(f"\n5. pipeline with a custom step: test AUC {auc(custom):.3f} (scaling makes it the same model)")

# --- 6. what the best model looks at --------------------------------------------------
best = grid.best_estimator_
perm = permutation_importance(best, X_te, y_te, scoring="roc_auc", n_repeats=10, random_state=0, n_jobs=-1)
imp = sorted(zip(md.FEATURES + ["ticker"], perm.importances_mean), key=lambda kv: -kv[1])
print("\n6. permutation importance of the tuned boosting model (drop in test AUC):")
for n, v in imp:
    print(f"  {n:10s} {v:+.3f}")

# --- chart ---------------------------------------------------------------------------
cs.apply()
fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(8, 3.8 * 3))

folds = np.arange(1, 6)
a1.plot(folds, kf, "o-", color=cs.AMBER, label="KFold, shuffled (future rows leak into training)")
a1.plot(folds, ts, "o-", color=cs.MAROON, label="TimeSeriesSplit (train on the past, validate on what follows)")
a1.axhline(auc(fitted["gradient boosting (HistGradientBoosting)"]), color=cs.BLUE, ls="--", lw=1, label="the real test, 2022-2023")
cs.title(a1, "Cross-validated AUC on 2019-2021, shuffled folds vs time-ordered folds", 1)
a1.set_xlabel("Validation fold (1 to 5)")
a1.set_ylabel("AUC on the validation fold (0.5 = coin flip)")
a1.set_xticks(folds)
a1.legend()

for depth, col in zip([2, 3, 5], [cs.GREY, cs.MAROON, cs.BLUE]):
    r = res[res["max_depth"] == depth].sort_values("learning_rate")
    a2.plot(r["learning_rate"], r["validation AUC"], "o-", color=col, label=f"trees of depth {depth}")
a2.set_xscale("log")
cs.title(a2, "Grid search: validation AUC for each learning rate and tree depth", 2)
a2.set_xlabel("Learning rate (log scale)")
a2.set_ylabel("Mean validation AUC over 5 time-ordered folds")
a2.legend()

for (name, m), col in zip(fitted.items(), [cs.GREY, cs.PURPLE, cs.BLUE, cs.MAROON]):
    if name.startswith("always"):
        continue
    fpr, tpr, _ = roc_curve(y_te, m.predict_proba(X_te)[:, 1])
    a3.plot(fpr * 100, tpr * 100, color=col, lw=1.4, label=f"{name.split(' (')[0]} (AUC {auc(m):.3f})")
a3.plot([0, 100], [0, 100], color=cs.GREY, ls=":", lw=1, label="_coin flip")
cs.title(a3, "ROC curves on the 2022-2023 test months", 3)
a3.set_xlabel("Calm months wrongly flagged as stress (% of calm months)")
a3.set_ylabel("Stress months caught (% of stress months)")
a3.legend()

cs.save(fig, OUT / "stress-month.svg")
print("\nchart written to", OUT / "stress-month.svg")
