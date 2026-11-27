# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "scikit-learn", "matplotlib"]
# ///
"""Numbers and chart for the post "Sequence models and responsible AI".

Input: a window of the last 20 daily moves (absolute daily return, %) for one ticker, newest
first, the "flattened" sequence of ISLP section 10.5.2. Target: next month's volatility, the
standard deviation of the next 20 daily returns (% a year). Five tickers, 2020 to October 2025.
The script shows
  (1) a shuffled split against a time-ordered split, scored on exactly the same test rows,
      for a linear model and a random forest, on two inputs (the seven summary features of
      market_data.FEATURES, and the raw 20-day window), with and without a 20-day gap;
  (2) that the order inside the window carries information: the same models with each row's
      window shuffled, so they see the same 20 numbers but not which day each came from;
  (3) the linear model's weight on each day of the window;
  (4) who the error falls on: the test miss for each ticker, and month by month.
Run:  uv run examples/sequence-models-and-responsible-ai/sequence_split.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chartstyle as cs  # noqa: E402
import market_data as md  # noqa: E402

OUT = cs.out_dir("sequence-models-and-responsible-ai")
A = md.ANNUALISE
L = 20                      # window length, trading days
H = 20                      # target horizon, trading days
TEST_FROM = "2024-01-01"
LAGS = [f"lag_{k}" for k in range(L)]   # lag_0 = today's move, lag_19 = 19 days ago


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def show(label, value):
    print(f"{label:<66} {value}")


# --- data: one row per ticker and day -------------------------------------------------------------
try:
    feats = md.features()                            # the seven summary features and the target
except (OSError, KeyError) as e:
    sys.exit(f"cannot read the prices file data/prices.csv: {e}")
px = md.prices()
frames = []
for t in px.columns:
    r = px[t].pct_change()
    d = pd.DataFrame({f"lag_{k}": r.abs().shift(k) * 100 for k in range(L)})
    d["ticker"] = t
    frames.append(d.dropna())
lags = pd.concat(frames).set_index("ticker", append=True)
df = feats.join(lags, how="inner").reset_index(level="ticker")
dates = df.index
y = df["next_vol"].to_numpy() * A                    # std of the next 20 daily returns, % a year
INPUTS = {"seven features": df[md.FEATURES].to_numpy(), "20-day window": df[LAGS].to_numpy()}
X = INPUTS["20-day window"]
show("rows (ticker-days)", f"{len(df)}  ({dates.min().date()} to {dates.max().date()})")
ac = np.mean([df.loc[df.ticker == t, "next_vol"].autocorr(1) for t in px.columns])
show("correlation of a day's target with the next day's (mean of five tickers)", f"{ac:.3f}")


def linear():
    return LinearRegression()


def forest():
    return RandomForestRegressor(n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=0)


MODELS = {"linear": linear, "forest": forest}

# --- 1. shuffled against time-ordered, on the same test rows ---------------------------------------
rng = np.random.default_rng(0)
held = rng.random(len(df)) < 0.2                     # a random 20% of all rows is held out
late = dates >= TEST_FROM
test = held & late                                   # score only the held-out rows in 2024-25
show("held-out rows in 2024 to Oct 2025 (the test rows)", int(test.sum()))
train_dates = np.sort(dates[~late].unique())
gap_start = train_dates[-H]                          # the last 20 training days' targets reach into 2024
splits = {
    "shuffled": ~held,                               # the other 80%, from every year
    "time-ordered": ~late,                           # 2020 to 2023 only
    "time-ordered, 20-day gap": dates < gap_start,
}
for name, tr in splits.items():
    show(f"  {name}: training rows", int(tr.sum()))
show("baseline: guess the 2020-2023 average", f"{rmse(y[test], np.full(test.sum(), y[~late].mean())):.2f}")
print()
for iname, Xi in INPUTS.items():
    for mname, make in MODELS.items():
        row = []
        for sname, tr in splits.items():
            row.append(rmse(y[test], make().fit(Xi[tr], y[tr]).predict(Xi[test])))
        show(f"{iname:<15} {mname:<7} shuffled, time-ordered, with gap", ", ".join(f"{v:.2f}" for v in row))
print()

# the same comparison for five different random hold-outs
for iname, Xi in INPUTS.items():
    for mname, make in MODELS.items():
        diff = []
        for seed in range(5):
            h = np.random.default_rng(seed).random(len(df)) < 0.2
            te = h & late
            a = rmse(y[te], make().fit(Xi[~h], y[~h]).predict(Xi[te]))
            b = rmse(y[te], make().fit(Xi[~late], y[~late]).predict(Xi[te]))
            diff.append(b - a)
        show(f"{iname:<15} {mname:<7} five hold-outs: time-ordered minus shuffled",
             f"{min(diff):.2f} to {max(diff):.2f}")
print()

# why the two inputs leak so differently: is yesterday's row close to today's?
for iname, Xi in INPUTS.items():
    Z = (Xi - Xi.mean(0)) / Xi.std(0)                # each column on the same scale
    nb, rd = [], []
    for t in px.columns:
        Zt = Z[(df.ticker == t).to_numpy()]
        nb.append(np.linalg.norm(Zt[1:] - Zt[:-1], axis=1))
        j = np.random.default_rng(2).permutation(len(Zt))
        rd.append(np.linalg.norm(Zt - Zt[j], axis=1))
    show(f"{iname:<15} distance to the day before, as a share of distance to a random day",
         f"{np.median(np.concatenate(nb)) / np.median(np.concatenate(rd)) * 100:.0f}%")
print()

# --- 2. does the order inside the window matter? ----------------------------------------------------
perm_rng = np.random.default_rng(1)
Xp = np.array([row[perm_rng.permutation(L)] for row in X])   # same 20 numbers, order destroyed
for mname, make in MODELS.items():
    a = rmse(y[late], make().fit(X[~late], y[~late]).predict(X[late]))
    b = rmse(y[late], make().fit(Xp[~late], y[~late]).predict(Xp[late]))
    show(f"{mname:<7} test 2024-25, window in order", f"{a:.2f}")
    show(f"{mname:<7} test 2024-25, window shuffled inside each row", f"{b:.2f}")
eq = LinearRegression().fit(X[~late].mean(axis=1, keepdims=True), y[~late])
show("linear  test 2024-25, one input: the plain average of the 20 moves",
     f"{rmse(y[late], eq.predict(X[late].mean(axis=1, keepdims=True))):.2f}")
print()

# --- 3. the linear model's weight on each day --------------------------------------------------------
lin = LinearRegression().fit(X[~late], y[~late])
w = lin.coef_
show("linear weights, today and 1, 2, 5, 10, 19 days ago",
     ", ".join(f"{w[k]:.2f}" for k in (0, 1, 2, 5, 10, 19)))
show("linear: share of total weight on the newest 5 days", f"{w[:5].sum() / w.sum() * 100:.1f}%")
show("linear intercept", f"{lin.intercept_:.2f}")
print()

# --- 4. who the error falls on ------------------------------------------------------------------------
p_lin = lin.predict(X[late])
yt, tk, dt = y[late], df["ticker"].to_numpy()[late], dates[late]
show("linear  test miss, all five tickers together", f"{rmse(yt, p_lin):.2f}")
show("linear  training miss, 2020-2023", f"{rmse(y[~late], lin.predict(X[~late])):.2f}")
for t in px.columns:
    k = tk == t
    e = rmse(yt[k], p_lin[k])
    show(f"  {t}: test miss, average actual volatility, miss as % of it",
         f"{e:.2f}, {yt[k].mean():.1f}, {e / yt[k].mean() * 100:.0f}%")
months = pd.DataFrame({"m": dt.to_period("M"), "e2": (yt - p_lin) ** 2})
monthly = np.sqrt(months.groupby("m")["e2"].mean())
worst = monthly.idxmax()
show("linear  worst test month, its miss", f"{worst}, {monthly.max():.2f}")
show("linear  months whose miss is above the training miss",
     f"{int((monthly > rmse(y[~late], lin.predict(X[~late]))).sum())} of {len(monthly)}")
show("linear  median monthly miss", f"{monthly.median():.2f}")

# --- chart ---------------------------------------------------------------------------------------------
cs.apply()
fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 7.6))
a1.bar(np.arange(L), w, color=cs.MAROON, label="weight the linear model gives each day")
a1.axhline(w.mean(), color=cs.GREY, ls="--", lw=1, label="the same total, spread evenly")
a1.set_xlabel("Days before today (0 = today)")
a1.set_ylabel("Weight (points of volatility per 1% move)")
a1.set_xticks(range(0, L, 2))
a1.legend()
cs.title(a1, "The newest days in the window get the most weight", 1)
train_miss = rmse(y[~late], lin.predict(X[~late]))
x_m = monthly.index.to_timestamp()
a2.plot(x_m, monthly.to_numpy(), color=cs.MAROON, marker="o", label="typical miss, each test month")
a2.axhline(train_miss, color=cs.BLUE, ls="--", lw=1, label="typical miss on the training years")
a2.set_xlabel("Test month")
a2.set_ylabel("Typical miss (points of % a year)")
a2.set_ylim(0, monthly.max() * 1.12)
a2.legend()
cs.title(a2, "One average hides the months the model gets badly wrong", 2)
cs.save(fig, OUT / "sequence_split.svg")
print("chart:", OUT / "sequence_split.svg")
