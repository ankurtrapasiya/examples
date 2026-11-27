# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "matplotlib"]
# ///
"""Numbers and chart for the post "AI in the investment process: a tool inside a process, not the
process".

One model, three ways to use it. The model is a linear regression that forecasts each ticker's
return over the next 21 trading days from the seven price features the ML posts share
(market_data.features). It is refitted every month on all the rows whose answer was already known
that day, so it never sees the future. The same forecasts then go through three processes:

  A. the model is the process: all the money in the ticker with the highest forecast
  B. the model is one input: the top pick gets at most 40%, the other 60% stays in SPY
  C. as B, plus a monitoring rule: if the model's picks lagged the average of the five tickers in
     4 or more of the last 6 months, hold SPY only until that stops being true

Each month's trades pay 0.1% of the value traded. The script also prints the model's
coefficients, the explanation a linear model gives for free, so a reader can do the smell test.
Run:  uv run examples/ai-in-the-investment-process/model_inside_process.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chartstyle as cs  # noqa: E402
import market_data as md  # noqa: E402

OUT = cs.out_dir("ai-in-the-investment-process")
MONTH = 21            # trading days held, and the forecast horizon
CAP = 0.40            # process B: the most one model pick may hold
LOOKBACK, BAD = 6, 4  # process C: switch the model off if its pick lagged in BAD of the last LOOKBACK months
COST = 0.001          # 0.1% of the value traded
START = "2021-01-01"  # first forecast: the model needs about a year of answers to learn from


def show(label, value):
    print(f"{label:<66} {value}")


try:
    df = md.features(horizon=MONTH)
except (OSError, KeyError) as e:
    sys.exit(f"cannot read the prices file data/prices.csv: {e}")
X_all, y_all = df[md.FEATURES], df["next_ret"]
dates = df.index.get_level_values("date").unique().sort_values()
rebal = dates[dates >= START][::MONTH]
rebal = rebal[rebal <= dates[-1 - MONTH]]          # every hold must end inside the data
known = {d: dates[max(0, dates.get_loc(d) - MONTH)] for d in rebal}   # last day whose answer is known


def fit(X, y):
    """Least squares on standardised features; returns (mean, std, coefficients with intercept)."""
    mu, sd = X.mean(), X.std()
    A = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    beta, *_ = np.linalg.lstsq(A, y.to_numpy(), rcond=None)
    return mu, sd, beta


rows, last_beta = [], None
for d in rebal:
    train_dates = df.index.get_level_values("date") < known[d]
    mu, sd, beta = fit(X_all[train_dates], y_all[train_dates])
    x = X_all.xs(d, level="date")
    f = beta[0] + ((x - mu) / sd).to_numpy() @ beta[1:]
    real = y_all.xs(d, level="date")
    rows.append({"date": d, **{f"f_{t}": v for t, v in zip(x.index, f)},
                 **{f"r_{t}": real[t] for t in x.index}})
    last_beta = beta
m = pd.DataFrame(rows).set_index("date")
T = md.TICKERS
F, R = m[[f"f_{t}" for t in T]].set_axis(T, axis=1), m[[f"r_{t}" for t in T]].set_axis(T, axis=1)
pick = F.idxmax(axis=1)
beat = np.array([R.loc[d, pick[d]] > R.loc[d].mean() for d in m.index])

# --- the model on its own --------------------------------------------------------------------------
print(f"Model: linear regression on {len(md.FEATURES)} features, refitted monthly, {len(m)} months "
      f"from {m.index[0].date()} to {m.index[-1].date()}")
show("  months the top pick beat the average of the five tickers", f"{beat.sum()} of {len(m)} ({beat.mean():.0%})")
show("  how often each ticker was the top pick", ", ".join(f"{t} {n}" for t, n in pick.value_counts().items()))
hivol = pd.Series({d: X_all.xs(d, level="date")["vol_20d"].idxmax() for d in m.index})
show("  months the top pick was also the most volatile ticker (20-day)", f"{(hivol == pick).sum()} of {len(m)}")
corr = [np.corrcoef(F.loc[d], R.loc[d])[0, 1] for d in m.index]
show("  average correlation between forecast and real return, across tickers", f"{np.mean(corr):+.3f}")
print("  coefficients of the last fit (per one standard deviation of the feature, % over 21 days):")
for name, b in zip(md.FEATURES, last_beta[1:]):
    show(f"    {name}", f"{100 * b:+.2f}")
print()

# --- three processes -------------------------------------------------------------------------------
def weights(process):
    W = pd.DataFrame(0.0, index=m.index, columns=T)
    off = []
    for i, d in enumerate(m.index):
        if process == "A":
            W.loc[d, pick[d]] = 1.0
            off.append(False); continue
        recent = beat[max(0, i - LOOKBACK):i]           # only months that have finished
        stop = process == "C" and len(recent) == LOOKBACK and (~recent).sum() >= BAD
        off.append(stop)
        W.loc[d, "SPY"] = 1.0 if stop else 1 - CAP
        if not stop:
            W.loc[d, pick[d]] += CAP
    return W, np.array(off)


def run(W):
    gross = (W * R).sum(axis=1)
    turnover = W.diff().abs().sum(axis=1)
    turnover.iloc[0] = W.iloc[0].abs().sum()
    return gross - COST * turnover


res, offs = {}, {}
spy = R["SPY"] - COST * np.r_[1, np.zeros(len(m) - 1)]
res["SPY alone"] = spy
res["NVDA alone"] = R["NVDA"] - COST * np.r_[1, np.zeros(len(m) - 1)]
for p, name in (("A", "A. All in the model's top pick"), ("B", "B. Top pick capped at 40%, rest in SPY"),
                ("C", "C. As B, switched off when the picks keep lagging")):
    W, off = weights(p)
    res[name], offs[name] = run(W), off
    show(f"  {name}: largest weight in a pick other than SPY, months off",
         f"{W.drop(columns='SPY').max().max():.0%}, {off.sum()}")
offC = offs["C. As B, switched off when the picks keep lagging"]
show("  C switched off in", ", ".join(f"{y}: {n}" for y, n in pd.Series(m.index[offC].year).value_counts().sort_index().items()))
print()

print(f"Monthly returns, {len(m)} months, after 0.1% costs")
print(f"  {'process':<52} {'average':>8} {'worst':>8} {'best':>8} {'max fall':>9} {'growth':>8}")
for name, r in res.items():
    g = (1 + r).cumprod()
    dd = (g / g.cummax().clip(lower=1) - 1).min()          # from the start value or a later peak
    print(f"  {name:<52} {100 * r.mean():+7.2f}% {100 * r.min():+7.1f}% {100 * r.max():+7.1f}% "
          f"{100 * dd:+8.1f}% {100 * (g.iloc[-1] - 1):+7.1f}%")
for name, r in res.items():
    show(f"  worst month of {name[:40]}", f"{r.idxmin().date()} {100 * r.min():+.1f}%")
wA = res["A. All in the model's top pick"]
worst = wA.idxmin()
show(f"  worst month of A starts {worst.date()}: pick, A / B / C / SPY",
     f"{pick[worst]}, " + " / ".join(f"{100 * r[worst]:+.1f}%" for r in
                                       (wA, res["B. Top pick capped at 40%, rest in SPY"],
                                        res["C. As B, switched off when the picks keep lagging"], spy)))

# --- chart -----------------------------------------------------------------------------------------
cs.apply()
fig, ax = plt.subplots(figsize=(8, 4.2))
cols = {"SPY alone": cs.GREY, "NVDA alone": cs.AMBER, "A. All in the model's top pick": cs.MAROON,
        "B. Top pick capped at 40%, rest in SPY": cs.BLUE,
        "C. As B, switched off when the picks keep lagging": cs.PURPLE}
for name, r in res.items():
    g = 10_000 * (1 + r).cumprod()
    ax.plot(g.index, g, color=cols[name], ls="--" if name.endswith("alone") else "-", label=name)
ax.set_xlabel("Month held (its first day)"); ax.set_ylabel("Value of 10,000 dollars at its end (dollars)")
ax.set_yscale("log")
ax.legend(); cs.title(ax, "The same forecasts through three processes (log scale)")
cs.save(fig, OUT / "processes.svg")
