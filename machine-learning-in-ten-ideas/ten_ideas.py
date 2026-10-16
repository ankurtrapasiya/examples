# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "matplotlib", "scikit-learn"]
# ///
"""Numbers and chart for the post "Machine learning for quantitative finance in ten ideas".

Everything runs on the blog's shared market data (examples/market_data.py: SPY, AAPL, JPM, XOM,
GLD, 2019-2023, seven price-derived features, target = next month's volatility) with scikit-learn,
plus two tiny made-up datasets that favour a line or a tree.
Run:  uv run examples/machine-learning-in-ten-ideas/ten_ideas.py
"""
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chartstyle as cs  # noqa: E402
import market_data as md  # noqa: E402

OUT = cs.out_dir("machine-learning-in-ten-ideas")


def rmse(y, yhat):
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def p(label, value):
    print(f"{label:<64} {value}")


df = md.features()
train, test = md.split(df)                     # 2019-2021 train, 2022-2023 test, in time order
Xtr, ytr = train[md.FEATURES].to_numpy(), (train["next_vol"] * md.ANNUALISE).to_numpy()
Xte, yte = test[md.FEATURES].to_numpy(), (test["next_vol"] * md.ANNUALISE).to_numpy()
print("\n== data ==")
p("training rows / test rows", f"{len(ytr)} / {len(yte)}")
p("target: next month's volatility, % a year; test mean / sd", f"{yte.mean():.1f} / {yte.std():.1f}")
p("baseline: predict the training mean, test RMSE", f"{rmse(yte, np.full_like(yte, ytr.mean())):.2f}")

print("\n== 9. one tree, leaf size swept ==")
leaves = list(range(1, 31)) + [40, 50, 75, 100, 150, 200]
tree_tr, tree_te, tree_t = [], [], []
for ls in leaves:
    m = DecisionTreeRegressor(min_samples_leaf=ls, random_state=0)
    t0 = time.perf_counter(); m.fit(Xtr, ytr); tree_t.append(time.perf_counter() - t0)
    tree_tr.append(rmse(ytr, m.predict(Xtr))); tree_te.append(rmse(yte, m.predict(Xte)))
for ls, a, b in zip(leaves, tree_tr, tree_te):
    if ls in (1, 5, 10, 20, 30, 50, 100, 200):
        p(f"leaf size {ls:3}: train / test RMSE", f"{a:.2f} / {b:.2f}")
best = int(np.argmin(tree_te))
p("best test RMSE", f"{tree_te[best]:.2f} at leaf size {leaves[best]}")

print("\n== 7. twenty bagged trees, same sweep ==")
bag_tr, bag_te = [], []
for ls in leaves:
    m = BaggingRegressor(DecisionTreeRegressor(min_samples_leaf=ls), n_estimators=20, random_state=0, n_jobs=-1)
    m.fit(Xtr, ytr)
    bag_tr.append(rmse(ytr, m.predict(Xtr))); bag_te.append(rmse(yte, m.predict(Xte)))
for ls, a, b in zip(leaves, bag_tr, bag_te):
    if ls in (1, 5, 10, 20, 30, 50, 100, 200):
        p(f"leaf size {ls:3}: bagged train / test RMSE", f"{a:.2f} / {b:.2f}")
p("bagged test RMSE range over all leaf sizes", f"{min(bag_te):.2f} to {max(bag_te):.2f}")
for b_ in [1, 5, 10, 20, 50, 100]:
    m = BaggingRegressor(DecisionTreeRegressor(min_samples_leaf=1), n_estimators=b_, random_state=0, n_jobs=-1).fit(Xtr, ytr)
    p(f"bags {b_:3} of fully grown trees: test RMSE", f"{rmse(yte, m.predict(Xte)):.2f}")

print("\n== 7. random splits against chosen splits ==")
rt_te, rt_t = [], []
for ls in leaves:
    tes, ts = [], []
    for seed in range(10):
        m = DecisionTreeRegressor(min_samples_leaf=ls, splitter="random", random_state=seed)
        t0 = time.perf_counter(); m.fit(Xtr, ytr); ts.append(time.perf_counter() - t0)
        tes.append(rmse(yte, m.predict(Xte)))
    rt_te.append(float(np.mean(tes))); rt_t.append(float(np.mean(ts)))
for ls in (1, 10, 30):
    i = leaves.index(ls)
    p(f"leaf size {ls:3}: chosen vs random split test RMSE", f"{tree_te[i]:.2f} vs {rt_te[i]:.2f}; fit {1000*tree_t[i]:.1f} vs {1000*rt_t[i]:.1f} ms")
m = BaggingRegressor(DecisionTreeRegressor(min_samples_leaf=1, splitter="random"), n_estimators=20, random_state=0, n_jobs=-1).fit(Xtr, ytr)
p("20 bagged random-split trees, leaf 1: test RMSE", f"{rmse(yte, m.predict(Xte)):.2f}")

print("\n== 4 and 6. a line or a tree, on data that favours each ==")
rng = np.random.default_rng(0)
X = rng.uniform(-1, 1, (1000, 3))
y_line = 50 * X[:, 0] + 30 * X[:, 1] - 20 * X[:, 2] + rng.uniform(-0.1, 0.1, 1000)
y_step = 8 * (X[:, 0] > 0) + 5 * (X[:, 1] > 0) + 3 * (X[:, 2] > 0) + 0.5 * X[:, 2]
idx = rng.permutation(1000); tr_i, te_i = idx[:600], idx[600:]
for name, y in [("linear surface", y_line), ("three steps", y_step)]:
    lin = LinearRegression().fit(X[tr_i], y[tr_i]); tr = DecisionTreeRegressor(min_samples_leaf=1, random_state=0).fit(X[tr_i], y[tr_i])
    p(f"{name}: linear / tree test RMSE", f"{rmse(y[te_i], lin.predict(X[te_i])):.2f} / {rmse(y[te_i], tr.predict(X[te_i])):.2f}")

print("\n== 3. the leak: a same-day feature ==")
# add SPY's same-day next-month volatility as a feature for the other tickers: not knowable in advance
spy = df.xs("SPY", level="ticker")["next_vol"] * md.ANNUALISE
leak = df.join(spy.rename("spy_same_month"), on="date")
ltr, lte = md.split(leak)
m = LinearRegression().fit(ltr[md.FEATURES], ltr["next_vol"] * md.ANNUALISE)
p("linear, honest features: test RMSE", f"{rmse(lte['next_vol'] * md.ANNUALISE, m.predict(lte[md.FEATURES])):.2f}")
m = LinearRegression().fit(ltr[md.FEATURES + ['spy_same_month']], ltr["next_vol"] * md.ANNUALISE)
p("linear, plus a same-month feature (leak): test RMSE", f"{rmse(lte['next_vol'] * md.ANNUALISE, m.predict(lte[md.FEATURES + ['spy_same_month']])):.2f}")

# ------------------------------------------------------------------ chart ----
cs.apply()
fig, (a1, a2, a3, a4) = plt.subplots(4, 1, figsize=(8, 3.8 * 4))
cs.title(a1, "One regression tree: error on seen and unseen months as leaves grow", 1)
a1.plot(leaves, tree_tr, color=cs.BLUE, marker="o", ms=3, label="training months (seen)")
a1.plot(leaves, tree_te, color=cs.MAROON, marker="o", ms=3, label="test months (unseen)")
a1.set_xscale("log"); a1.set_xlabel("Leaf size (minimum rows in a leaf, log scale)"); a1.set_ylabel("RMSE (volatility points)")
a1.legend()
cs.title(a2, "Twenty bagged trees: the test curve goes flat", 2)
a2.plot(leaves, bag_tr, color=cs.BLUE, marker="o", ms=3, label="training months, bagged")
a2.plot(leaves, bag_te, color=cs.MAROON, marker="o", ms=3, label="test months, bagged")
a2.plot(leaves, tree_te, color=cs.GREY, lw=1, label="test months, one tree (from panel 1)")
a2.set_xscale("log"); a2.set_xlabel("Leaf size (minimum rows in a leaf, log scale)"); a2.set_ylabel("RMSE (volatility points)")
a2.legend()
cs.title(a3, "Random splits against chosen splits: error on the test months", 3)
a3.plot(leaves, tree_te, color=cs.MAROON, marker="o", ms=3, label="tree that searches for the best split")
a3.plot(leaves, rt_te, color=cs.BLUE, marker="o", ms=3, label="tree that splits at random (mean of 10)")
a3.set_xscale("log"); a3.set_xlabel("Leaf size (minimum rows in a leaf, log scale)"); a3.set_ylabel("RMSE (volatility points)")
a3.legend()
cs.title(a4, "Time to build the tree", 4)
a4.plot(leaves, [1000 * t for t in tree_t], color=cs.MAROON, marker="o", ms=3, label="tree that searches for the best split")
a4.plot(leaves, [1000 * t for t in rt_t], color=cs.BLUE, marker="o", ms=3, label="tree that splits at random (mean of 10)")
a4.set_xscale("log"); a4.set_xlabel("Leaf size (minimum rows in a leaf, log scale)"); a4.set_ylabel("Training time (milliseconds)")
a4.legend()
cs.save(fig, OUT / "leaf-size-sweep.svg")
