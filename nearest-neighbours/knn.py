# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "matplotlib", "scikit-learn"]
# ///
"""Numbers and chart for the post "Nearest neighbours".

k-nearest neighbours on the next-month volatility problem: the k dial, scaling, prediction cost,
and the curse of dimensionality.  Run:  uv run examples/nearest-neighbours/knn.py
"""
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chartstyle as cs  # noqa: E402
import market_data as md  # noqa: E402

OUT = cs.out_dir("nearest-neighbours")


def rmse(y, yhat):
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def p(label, value):
    print(f"{label:<66} {value}")


df = md.features()
train, test = md.split(df)
Xtr, Xte = train[md.FEATURES].to_numpy(), test[md.FEATURES].to_numpy()
ytr, yte = (train["next_vol"] * md.ANNUALISE).to_numpy(), (test["next_vol"] * md.ANNUALISE).to_numpy()
p("training / test rows", f"{len(ytr)} / {len(yte)}")
p("baseline (training mean) test RMSE", f"{rmse(yte, np.full_like(yte, ytr.mean())):.2f}")
p("linear regression test RMSE", f"{rmse(yte, LinearRegression().fit(Xtr, ytr).predict(Xte)):.2f}")

print("\n== scaling ==")
p("feature spreads (std) on training rows", " ".join(f"{f}:{s:.4f}" for f, s in zip(md.FEATURES, Xtr.std(0))))
for name, mk in [("raw features", lambda k: KNeighborsRegressor(k)), ("standardised features", lambda k: make_pipeline(StandardScaler(), KNeighborsRegressor(k)))]:
    m = mk(20).fit(Xtr, ytr)
    p(f"k = 20, {name}: test RMSE", f"{rmse(yte, m.predict(Xte)):.2f}")

print("\n== the k dial ==")
ks = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, len(ytr)]
tr_e, te_e = [], []
for k in ks:
    m = make_pipeline(StandardScaler(), KNeighborsRegressor(k)).fit(Xtr, ytr)
    tr_e.append(rmse(ytr, m.predict(Xtr))); te_e.append(rmse(yte, m.predict(Xte)))
    if k in (1, 5, 20, 100, 500, len(ytr)):
        p(f"k = {k:4}: train / test RMSE", f"{tr_e[-1]:.2f} / {te_e[-1]:.2f}")
best = int(np.argmin(te_e)); p("best k on test", f"{ks[best]} (RMSE {te_e[best]:.2f})")
m = make_pipeline(StandardScaler(), KNeighborsRegressor(20, weights="distance")).fit(Xtr, ytr)
p("k = 20 weighted by 1/distance: test RMSE", f"{rmse(yte, m.predict(Xte)):.2f}")

print("\n== cost: training is free, prediction scans everything ==")
knn = make_pipeline(StandardScaler(), KNeighborsRegressor(20)); lin = LinearRegression()
for name, mdl in [("KNN", knn), ("linear", lin)]:
    t0 = time.perf_counter(); mdl.fit(Xtr, ytr); tf = time.perf_counter() - t0
    t0 = time.perf_counter(); mdl.predict(Xte); tp = time.perf_counter() - t0
    p(f"{name}: fit / predict {len(Xte):,} rows", f"{1000 * tf:.1f} ms / {1000 * tp:.1f} ms")

print("\n== the curse of dimensionality ==")
for pdim in (1, 2, 5, 10, 20, 100):
    p(f"box side to hold 10% of uniform points in {pdim} dimensions", f"{0.1 ** (1 / pdim):.2f}")
rng = np.random.default_rng(0)
noise_e = []
for extra in [0, 5, 10, 20, 50]:
    Ztr = np.hstack([Xtr, rng.normal(size=(len(Xtr), extra))]); Zte = np.hstack([Xte, rng.normal(size=(len(Xte), extra))])
    kn = make_pipeline(StandardScaler(), KNeighborsRegressor(20)).fit(Ztr, ytr)
    ln = LinearRegression().fit(Ztr, ytr)
    noise_e.append((extra, rmse(yte, kn.predict(Zte)), rmse(yte, ln.predict(Zte))))
    p(f"{extra:2} useless noise features added: KNN / linear test RMSE", f"{noise_e[-1][1]:.2f} / {noise_e[-1][2]:.2f}")

# ------------------------------------------------------------------ chart ----
cs.apply()
fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(8, 3.8 * 3))

cs.title(a1, "The k dial runs backwards: small k memorises, huge k predicts the average", 1)
a1.plot(ks, tr_e, color=cs.BLUE, marker="o", ms=3, label="training months (seen)")
a1.plot(ks, te_e, color=cs.MAROON, marker="o", ms=3, label="test months (unseen)")
a1.axhline(rmse(yte, np.full_like(yte, ytr.mean())), color=cs.GREY, lw=1, ls="--", label="baseline: always predict the training mean")
a1.set_xscale("log"); a1.set_xlabel("k, the number of neighbours averaged (log scale)"); a1.set_ylabel("RMSE (volatility points)")
a1.legend()

cs.title(a2, "Two features, one query: the twenty nearest months, before and after scaling", 2)
q = Xte[100]
i20, i5 = md.FEATURES.index("vol_20d"), md.FEATURES.index("ret_5d")
d_raw = np.sqrt(((Xtr - q) ** 2).sum(1)); near_raw = np.argsort(d_raw)[:20]
Z = (Xtr - Xtr.mean(0)) / Xtr.std(0); zq = (q - Xtr.mean(0)) / Xtr.std(0)
d_sc = np.sqrt(((Z - zq) ** 2).sum(1)); near_sc = np.argsort(d_sc)[:20]
a2.scatter(100 * Xtr[:, i5], md.ANNUALISE * Xtr[:, i20], s=4, color=cs.GREY, alpha=0.35, rasterized=True, label="training months")
a2.scatter(100 * Xtr[near_raw, i5], md.ANNUALISE * Xtr[near_raw, i20], s=18, color=cs.BLUE, label="20 nearest on raw features (volatility decides everything)")
a2.scatter(100 * Xtr[near_sc, i5], md.ANNUALISE * Xtr[near_sc, i20], s=18, color=cs.MAROON, marker="s", label="20 nearest on standardised features")
a2.scatter([100 * q[i5]], [md.ANNUALISE * q[i20]], s=60, color=cs.AMBER, marker="*", zorder=4, label="the month being predicted")
a2.set_xlabel("5-day return (%)")
a2.set_ylabel("20-day volatility (% a year)")
a2.legend()

cs.title(a3, "Useless features hurt a neighbour model far more than a linear one", 3)
a3.plot([e for e, *_ in noise_e], [k for _, k, _ in noise_e], color=cs.MAROON, marker="o", label="k = 20 neighbours, standardised")
a3.plot([e for e, *_ in noise_e], [l for *_, l in noise_e], color=cs.BLUE, marker="o", label="linear regression")
a3.set_xlabel("Number of random noise features added to the seven real ones")
a3.set_ylabel("Test RMSE (volatility points)")
a3.legend()
cs.save(fig, OUT / "knn.svg")
