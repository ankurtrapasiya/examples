# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "matplotlib", "scikit-learn"]
# ///
"""Numbers and chart for the post "Choosing a model".

Five models on the same problem: accuracy, training time, prediction time, tail latency per
single prediction, and how each extrapolates.  Run:  uv run examples/choosing-a-model/choose.py
"""
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chartstyle as cs  # noqa: E402
import market_data as md  # noqa: E402

OUT = cs.out_dir("choosing-a-model")


def rmse(y, yhat):
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def p(label, value):
    print(f"{label:<66} {value}")


df = md.features()
train, test = md.split(df)
Xtr, Xte = train[md.FEATURES].to_numpy(), test[md.FEATURES].to_numpy()
ytr, yte = (train["next_vol"] * md.ANNUALISE).to_numpy(), (test["next_vol"] * md.ANNUALISE).to_numpy()

models = {
    "linear regression": LinearRegression(),
    "nearest neighbours, k=100": make_pipeline(StandardScaler(), KNeighborsRegressor(100)),
    "one tree, 200 rows per leaf": DecisionTreeRegressor(min_samples_leaf=200, random_state=0),
    "random forest, 300 trees": RandomForestRegressor(300, min_samples_leaf=20, random_state=0, n_jobs=1),
    "gradient boosting, 200 trees": GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0),
}
print("\n== five models, one problem ==")
rows = {}
for name, m in models.items():
    t0 = time.perf_counter(); m.fit(Xtr, ytr); fit = time.perf_counter() - t0
    t0 = time.perf_counter(); pred = m.predict(Xte); batch = time.perf_counter() - t0
    singles = []
    for i in range(200):
        t0 = time.perf_counter(); m.predict(Xte[i:i + 1]); singles.append(time.perf_counter() - t0)
    singles = np.array(singles) * 1000
    rows[name] = dict(rmse=rmse(yte, pred), fit=fit, batch=batch, p50=np.median(singles), p99=np.percentile(singles, 99))
    p(f"{name}: test RMSE / fit / predict 2,405 rows / one row p50 / p99", f"{rows[name]['rmse']:.2f} / {1000 * fit:.0f} ms / {1000 * batch:.1f} ms / {rows[name]['p50']:.2f} ms / {rows[name]['p99']:.2f} ms")
p("baseline, predict the training mean", f"{rmse(yte, np.full_like(yte, ytr.mean())):.2f}")

print("\n== extrapolation: a month twice as volatile as anything seen in training ==")
q = Xtr[np.argmax(Xtr[:, 4])].copy()
p("the most volatile training month: vol_20d, actual next-month volatility", f"{md.ANNUALISE * q[4]:.0f}, {ytr[np.argmax(Xtr[:, 4])]:.0f}")
p("highest next-month volatility ever seen in training", f"{ytr.max():.0f}")
q2 = q.copy(); q2[4] *= 2; q2[3] *= 2
for name, m in models.items():
    p(f"{name}: prediction at double that volatility", f"{m.predict(q2[None, :])[0]:.0f}")

print("\n== interpretability ==")
lin = models["linear regression"]
p("linear coefficients (per unit of each raw feature)", " ".join(f"{f}:{c:+.0f}" for f, c in zip(md.FEATURES, lin.coef_)))
tree = models["one tree, 200 rows per leaf"]
p("one tree: number of leaves / depth", f"{tree.get_n_leaves()} / {tree.get_depth()}")
p("random forest: number of leaves across all trees", f"{sum(t.get_n_leaves() for t in models['random forest, 300 trees'].estimators_):,}")

# ------------------------------------------------------------------ chart ----
cs.apply()
fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(8, 3.8 * 3))
names = list(rows)
short = ["linear", "k-NN", "one tree", "forest", "boosting"]

cs.title(a1, "Accuracy on the unseen months: the spread between models is one volatility point", 1)
a1.bar(short, [rows[n]["rmse"] for n in names], color=[cs.MAROON, cs.BLUE, cs.GREY, cs.PURPLE, cs.AMBER], width=0.55)
a1.axhline(rmse(yte, np.full_like(yte, ytr.mean())), color=cs.GREY, lw=1, ls="--", label="baseline: predict the mean")
for i, n in enumerate(names):
    a1.text(i, rows[n]["rmse"] + 0.1, f"{rows[n]['rmse']:.2f}", ha="center", fontsize=9)
a1.set_ylim(6, 10.5)
a1.set_xlabel("Model")
a1.set_ylabel("Test RMSE (volatility points)")
a1.legend()

cs.title(a2, "Cost: time to train, and time to answer one question (median and worst 1 in 100)", 2)
x = np.arange(5); w = 0.27
a2.bar(x - w, [1000 * rows[n]["fit"] for n in names], w, color=cs.GREY, label="training time (ms)")
a2.bar(x, [rows[n]["p50"] for n in names], w, color=cs.BLUE, label="one prediction, median (ms)")
a2.bar(x + w, [rows[n]["p99"] for n in names], w, color=cs.MAROON, label="one prediction, 99th percentile (ms)")
a2.set_yscale("log")
a2.set_xticks(x); a2.set_xticklabels(short)
a2.set_xlabel("Model")
a2.set_ylabel("Milliseconds (log scale)")
a2.legend()

cs.title(a3, "Extrapolation: what each model says about a month wilder than any it has seen", 3)
vals = [models[n].predict(q2[None, :])[0] for n in names]
a3.bar(short, vals, color=[cs.MAROON, cs.BLUE, cs.GREY, cs.PURPLE, cs.AMBER], width=0.55)
a3.axhline(ytr.max(), color=cs.GREY, lw=1, ls="--", label=f"highest next-month volatility ever seen in training: {ytr.max():.0f}")
for i, v in enumerate(vals):
    a3.text(i, v + 1, f"{v:.0f}", ha="center", fontsize=9)
a3.set_xlabel("Model")
a3.set_ylabel("Predicted next-month volatility (% a year)")
a3.legend()
cs.save(fig, OUT / "choose.svg")
