# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "scikit-learn", "matplotlib"]
# ///
"""Numbers and charts for the post "Neural networks: linear regression, stacked and bent".

Target: next month's volatility (the standard deviation of the next 20 daily returns, % a year),
for the blog's five tickers. Features: the seven price features of market_data.FEATURES.
Train on 2020 to 2022, choose settings on 2023 (validation), test once on 2024 to October 2025.
The script shows (1) that two linear layers multiply out to one, by hand, (2) that a network
with no bend scores like linear regression, (3) how a bent network with a weight penalty
compares with linear regression on the same rows, over five random starts, and (4) training against validation
error, epoch by epoch, for a network with too much freedom.
Run:  uv run examples/neural-networks-linear-regression-stacked-and-bent/neural_net.py
"""
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chartstyle as cs  # noqa: E402
import market_data as md  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)
OUT = cs.out_dir("neural-networks-linear-regression-stacked-and-bent")
A = md.ANNUALISE
SEEDS = range(5)


def show(label, value):
    print(f"{label:<64} {value}")


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


# --- 1. two linear layers are one line, by hand ----------------------------------------------------
# one input x, two hidden units, one output
w1, b1 = np.array([2.0, -1.0]), np.array([1.0, 3.0])     # hidden unit k: w1[k] * x + b1[k]
w2, b2 = np.array([0.5, 0.5]), 0.0                        # output: w2 . h + b2
print("Two layers, no bend: slope", w2 @ w1, "intercept", w2 @ b1 + b2)
for x in (0.0, 2.0, 4.0, 6.0):
    lin = w2 @ (w1 * x + b1) + b2
    relu = w2 @ np.maximum(0, w1 * x + b1) + b2
    show(f"  x = {x:.0f}: hidden (no bend) {w1 * x + b1}, output", f"{lin:.2f};  with ReLU {relu:.2f}")
print()

# --- data ------------------------------------------------------------------------------------------
try:
    df = md.features()
except (OSError, KeyError) as e:
    sys.exit(f"cannot read the prices file data/prices.csv: {e}")
dates = df.index.get_level_values("date")
parts = {"train": df[dates < "2023-01-01"],
         "validation": df[(dates >= "2023-01-01") & (dates < "2024-01-01")],
         "test": df[dates >= "2024-01-01"]}
X = {k: v[md.FEATURES].to_numpy() for k, v in parts.items()}
y = {k: v["next_vol"].to_numpy() * A for k, v in parts.items()}
for k in parts:
    show(f"rows in {k}", f"{len(y[k])}  ({parts[k].index.get_level_values('date').min().date()} to "
         f"{parts[k].index.get_level_values('date').max().date()})")
print()


def fit_score(model):
    model.fit(X["train"], y["train"])
    return {k: rmse(y[k], model.predict(X[k])) for k in ("train", "validation", "test")}


def mlp(activation="relu", hidden=(32, 32), alpha=1e-4, seed=0, **kw):
    return make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=hidden, activation=activation,
                                                        alpha=alpha, max_iter=500, random_state=seed, **kw))


# --- 2. baseline, linear regression, a network with no bend ----------------------------------------
mean_guess = {k: rmse(y[k], np.full_like(y[k], y["train"].mean())) for k in ("train", "validation", "test")}
show("guess the training average: RMSE train / validation / test", " / ".join(f"{v:.2f}" for v in mean_guess.values()))
lin = fit_score(make_pipeline(StandardScaler(), LinearRegression()))
show("linear regression: RMSE train / validation / test", " / ".join(f"{v:.2f}" for v in lin.values()))
ident = [fit_score(mlp("identity", seed=s)) for s in SEEDS]
show("network 32-32, no bend (identity), 5 seeds: test RMSE", " ".join(f"{d['test']:.2f}" for d in ident))
print()

# --- 3. a bent network with room to spare, at four penalty strengths, 5 seeds ---------------------
# alpha is the weight penalty (ridge on the weights); scikit-learn divides it by the number of rows
results = {}
for alpha in (1e-4, 10.0, 100.0, 1000.0):
    runs = [fit_score(mlp("relu", (256, 256), alpha, s)) for s in SEEDS]
    results[alpha] = runs
    tr, va, te = (np.array([r[k] for r in runs]) for k in ("train", "validation", "test"))
    show(f"ReLU 256-256, alpha {alpha:g}: train / val / test (mean; test range)",
         f"{tr.mean():.2f} / {va.mean():.2f} / {te.mean():.2f}  ({te.min():.2f} to {te.max():.2f})")
best = min(results, key=lambda k: np.mean([r["validation"] for r in results[k]]))
beat = sum(r["test"] < lin["test"] for r in results[best])
show("penalty chosen on the validation year", f"alpha {best:g}; test mean "
     f"{np.mean([r['test'] for r in results[best]]):.2f}, below linear in {beat} of {len(SEEDS)} seeds")
n_w = 7 * 256 + 256 + 256 * 256 + 256 + 256 + 1
show("weights and biases in a 7-256-256-1 network / in linear regression", f"{n_w} / 8")
print()

# --- 4. epoch by epoch: training against validation error, no penalty to speak of -----------------
sc = StandardScaler().fit(X["train"])
Xs = {k: sc.transform(v) for k, v in X.items()}
net = MLPRegressor(hidden_layer_sizes=(256, 256), alpha=1e-4, random_state=0)
curve = []
for epoch in range(1, 301):
    net.partial_fit(Xs["train"], y["train"])
    curve.append([rmse(y[k], net.predict(Xs[k])) for k in ("train", "validation", "test")])
curve = np.array(curve)
stop = int(curve[:, 1].argmin()) + 1
for e in (1, 10, 50, stop, 300):
    show(f"  epoch {e}: RMSE train / validation / test", " / ".join(f"{v:.2f}" for v in curve[e - 1]))
show("validation RMSE lowest at epoch", stop)
print()

# --- charts ----------------------------------------------------------------------------------------
cs.apply()
# chart 1: one input, this month's volatility, linear regression against a small bent network
x1 = {k: v[["vol_20d"]].to_numpy() * A for k, v in parts.items()}
one_lin = LinearRegression().fit(x1["train"], y["train"])
one_net = mlp("relu", (8,), 1.0, 0).fit(x1["train"], y["train"])
grid = np.linspace(x1["train"].min(), x1["train"].max(), 300).reshape(-1, 1)
show("one input: linear test RMSE / network test RMSE",
     f"{rmse(y['test'], one_lin.predict(x1['test'])):.2f} / {rmse(y['test'], one_net.predict(x1['test'])):.2f}")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 3.8 * 2))
ax1.scatter(x1["train"], y["train"], s=4, color=cs.GREY, alpha=0.35, rasterized=True, label="_days")
ax1.plot(grid, one_lin.predict(grid), color=cs.BLUE, label="linear regression")
ax1.plot(grid, one_net.predict(grid), color=cs.MAROON, label="network, 8 ReLU units")
ax1.set_xlabel("This month's volatility (% a year)")
ax1.set_ylabel("Next month's volatility (% a year)")
ax1.legend()
cs.title(ax1, "One input, 2020 to 2022: the network bends where the line cannot", 1)
ep = np.arange(1, len(curve) + 1)
ax2.plot(ep, curve[:, 0], color=cs.BLUE, label="training years, 2020 to 2022")
ax2.plot(ep, curve[:, 1], color=cs.MAROON, label="validation year, 2023")
ax2.axvline(stop, color=cs.GREY, lw=0.8, ls="--")
ax2.set_xlabel("Epoch (passes through the training rows)")
ax2.set_ylabel("Typical miss, RMSE (points of % a year)")
ax2.legend()
cs.title(ax2, "A 256-256 network: training error keeps falling, validation error turns up", 2)
cs.save(fig, OUT / "neural_net.svg")
