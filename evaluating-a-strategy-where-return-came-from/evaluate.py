# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "matplotlib"]
# ///
"""Numbers and chart for the post "Evaluating a strategy: where the return came from".

One rule, evaluated the way the post describes: each of the five tickers gets a 20% slice, held
while its 50-day average close is above its 200-day average (decided at yesterday's close) and
in cash, earning nothing, otherwise. Then: three benchmarks on the same dates, return,
volatility, drawdown and turnover side by side, a split of the return into market, ticker and
timing parts, a window search on 2021-2023 checked on 2024-2025, and the rule re-run on two
periods and two sets of tickers.
Run:  uv run examples/evaluating-a-strategy-where-return-came-from/evaluate.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chartstyle as cs  # noqa: E402
import market_data as md  # noqa: E402

OUT = cs.out_dir("evaluating-a-strategy-where-return-came-from")
D = md.TRADING_DAYS
COST = 0.001                                   # 0.1% of every unit of weight bought or sold
START = "2021-01-01"                           # every window up to 250 days exists by then
try:
    px = md.prices()
except (OSError, KeyError) as e:
    sys.exit(f"cannot read the prices file data/prices.csv: {e}")
r_all = md.daily_returns(px)


def p(label, value):
    print(f"{label:<66} {value}")


def weights(tickers, fast=50, slow=200):
    """Each ticker's weight for day t, decided from closes up to day t-1."""
    sub = px[tickers]
    on = (sub.rolling(fast).mean() > sub.rolling(slow).mean()).astype(float)
    return (on / len(tickers)).shift(1).loc[r_all.index]


def run(tickers, fast=50, slow=200, start=START, end=None):
    """Daily net return of the rule and the turnover behind it, on [start, end]."""
    w = weights(tickers, fast, slow).loc[start:end]
    r = r_all[tickers].loc[start:end]
    turnover = w.diff().abs().sum(axis=1)
    turnover.iloc[0] = w.iloc[0].abs().sum()   # buying the first positions counts too
    gross = (w * r).sum(axis=1)
    return gross - COST * turnover, gross, turnover, w


def stats(ret, turnover=None):
    v = (1 + ret).cumprod()
    years = len(ret) / D
    out = {
        "total": v.iloc[-1] - 1,
        "cagr": v.iloc[-1] ** (1 / years) - 1,
        "vol": ret.std() * np.sqrt(D),
        "sharpe": ret.mean() / ret.std() * np.sqrt(D) if ret.std() > 0 else np.nan,   # risk-free taken as 0
        "maxdd": (v / v.cummax() - 1).min(),
    }
    out["turnover"] = np.nan if turnover is None else turnover.sum() / years
    return out


def fmt(s):
    t = "" if np.isnan(s["turnover"]) else f"  turnover {100 * s['turnover']:5.0f}%/yr"
    return (f"total {100 * s['total']:+6.1f}%  a year {100 * s['cagr']:+5.1f}%  vol {100 * s['vol']:4.1f}%"
            f"  Sharpe {s['sharpe']:5.2f}  max drawdown {100 * s['maxdd']:6.1f}%{t}")


T = md.TICKERS
net, gross, to, w = run(T)
r = r_all.loc[START:]
spy = r["SPY"]
ew = r[T].mean(axis=1)                         # 20% in each, reset daily: the same opportunity set
cash = pd.Series(0.0, index=r.index)
print(f"Evaluation window: {r.index[0].date()} to {r.index[-1].date()}, {len(r)} days, {len(r) / D:.2f} years")

# --- 1. the benchmarks ---------------------------------------------------------------------------
print("\n1. The rule against three benchmarks, same dates")
S = {"rule, net of 0.1% costs": stats(net, to), "rule, before costs": stats(gross, to),
     "cash (earning nothing)": stats(cash), "SPY, buy and hold": stats(spy),
     "five tickers, 20% each, buy and hold": stats(ew)}
for k, s in S.items():
    p(f"  {k}", fmt(s))
p("  rule minus cash, minus SPY, minus the five held (total return, points)",
  "  ".join(f"{100 * (S['rule, net of 0.1% costs']['total'] - S[k]['total']):+.1f}"
            for k in ("cash (earning nothing)", "SPY, buy and hold", "five tickers, 20% each, buy and hold")))
p("  average share of the money invested", f"{100 * w.sum(axis=1).mean():.0f}%")
p("  number of switches (a ticker going in or out)", f"{int((w.diff().abs() > 0).sum().sum())}")
p("  cost paid, total (points of return, simple sum)", f"{100 * COST * to.sum():.2f}")
v = (1 + net).cumprod()
dd = v / v.cummax() - 1
p("  rule: worst drawdown date", f"{dd.idxmin().date()}  peak before it {v.loc[:dd.idxmin()].idxmax().date()}")
vs = (1 + spy).cumprod()
dds = vs / vs.cummax() - 1
p("  SPY: worst drawdown date", f"{dds.idxmin().date()}  peak before it {vs.loc[:dds.idxmin()].idxmax().date()}")
ve = (1 + ew).cumprod()
dde = ve / ve.cummax() - 1
p("  five held: worst drawdown date", f"{dde.idxmin().date()}")

# --- 2. year by year ---------------------------------------------------------------------------
print("\n2. Year by year: return and Sharpe of the rule (net), and of the five held")
for y, g in net.groupby(net.index.year):
    e = ew.loc[g.index]
    p(f"  {y}", f"rule {100 * ((1 + g).prod() - 1):+6.1f}%  Sharpe {g.mean() / g.std() * np.sqrt(D):5.2f}"
      f"   five held {100 * ((1 + e).prod() - 1):+6.1f}%  Sharpe {e.mean() / e.std() * np.sqrt(D):5.2f}")

# --- 3. attribution -----------------------------------------------------------------------------
print("\n3. Where the return came from (average daily return x 252, so the parts add up)")
X = np.column_stack([np.ones(len(spy)), spy.values])
coef, res, *_ = np.linalg.lstsq(X, net.values, rcond=None)
alpha_d, beta = coef
resid = net.values - X @ coef
s2 = resid @ resid / (len(resid) - 2)
se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
p("  rule's average return a year (arithmetic)", f"{100 * D * net.mean():.1f}%")
p("  beta to SPY", f"{beta:.2f}")
p("  R squared", f"{1 - resid.var() / net.var():.2f}")
p("  market part: beta x SPY's average return a year", f"{100 * beta * D * spy.mean():.1f}%  (SPY {100 * D * spy.mean():.1f}%)")
p("  alpha a year (intercept x 252)", f"{100 * D * alpha_d:+.1f}%")
p("  alpha standard error a year, t", f"{100 * D * se[0]:.1f}%  t = {alpha_d / se[0]:.2f}")
contrib = D * (w * r).mean()                  # each slice's share of the gross average return
held = D * (r[T] / len(T)).mean()              # each slice if simply held
print("  by ticker: held all the time | what the rule earned | timing (rule minus held)")
for t in T:
    p(f"    {t}", f"{100 * held[t]:6.1f}%  {100 * contrib[t]:6.1f}%  {100 * (contrib[t] - held[t]):+6.1f}%")
p("    sum", f"{100 * held.sum():6.1f}%  {100 * contrib.sum():6.1f}%  {100 * (contrib.sum() - held.sum()):+6.1f}%")
p("    costs a year", f"{-100 * D * (COST * to).mean():.2f}%")
p("    NVDA's share of the rule's gross average return", f"{100 * contrib['NVDA'] / contrib.sum():.0f}%")
net4, *_ = run([t for t in T if t != "NVDA"])
ew4 = r[[t for t in T if t != "NVDA"]].mean(axis=1)
p("  without NVDA (4 tickers, 25% each): rule", fmt(stats(net4)))
p("  without NVDA: four held", fmt(stats(ew4)))

# --- 4. choosing the windows on one period, testing on the next -----------------------------------
print("\n4. Window search on 2021-2023 (Sharpe, net), checked on 2024-2025")
rows = []
for f in range(10, 101, 10):
    for s in range(50, 251, 25):
        if f >= s:
            continue
        ins = stats(run(T, f, s, START, "2023-12-31")[0])
        oos = stats(run(T, f, s, "2024-01-01")[0])
        rows.append((f, s, ins["sharpe"], oos["sharpe"], ins["total"], oos["total"]))
g = pd.DataFrame(rows, columns=["fast", "slow", "sh_in", "sh_out", "tot_in", "tot_out"])
p("  pairs tried", f"{len(g)}")
g = g.sort_values("sh_in", ascending=False)
print("  top five in-sample, then 50/200:")
for _, x in pd.concat([g.head(5), g[(g.fast == 50) & (g.slow == 200)]]).iterrows():
    p(f"    {int(x.fast)}/{int(x.slow)}", f"Sharpe in {x.sh_in:5.2f} out {x.sh_out:5.2f}   total in {100 * x.tot_in:+6.1f}% out {100 * x.tot_out:+6.1f}%")
p("  rank correlation of in-sample and out-of-sample Sharpe", f"{g.sh_in.rank().corr(g.sh_out.rank()):.2f}")
p("  median out-of-sample Sharpe, all pairs", f"{g.sh_out.median():.2f}")
p("  best pair's in-sample Sharpe rank among all, out-of-sample", f"{int((g.sh_out > g.iloc[0].sh_out).sum()) + 1} of {len(g)}")
e_in, e_out = ew.loc[:"2023-12-31"], ew.loc["2024-01-01":]
p("  five held, Sharpe in, out", f"{stats(e_in)['sharpe']:.2f}  {stats(e_out)['sharpe']:.2f}")

# --- 5. more than one period, more than one set of stocks -------------------------------------------
print("\n5. The same 50/200 rule on two periods and two sets of tickers (net)")
sets = {"all five": T, "the three stocks (NVDA, GS, XOM)": ["NVDA", "GS", "XOM"], "the two funds (SPY, GLD)": ["SPY", "GLD"]}
periods = {"2021-2022": (START, "2022-12-31"), "2023-2025": ("2023-01-01", None)}
for name, ts in sets.items():
    for pn, (a, b) in periods.items():
        n_, *_ = run(ts, start=a, end=b)
        e_ = r_all[ts].loc[a:b].mean(axis=1)
        sr, se_ = stats(n_), stats(e_)
        p(f"  {name}, {pn}", f"rule Sharpe {sr['sharpe']:5.2f} dd {100 * sr['maxdd']:6.1f}%   held Sharpe {se_['sharpe']:5.2f} dd {100 * se_['maxdd']:6.1f}%")

# --- chart -------------------------------------------------------------------------------------------
cs.apply()
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 3.8 * 3))
for ser, col, lab in ((ve, cs.BLUE, "five tickers held, 20% each"), (vs, cs.GREY, "SPY held"),
                      (v, cs.MAROON, "50/200 rule, net of 0.1% costs")):
    ax1.plot(ser.index, 10_000 * ser.values, color=col, label=lab)
ax1.set_ylabel("Value of 10,000 dollars (dollars)")
ax1.set_xlabel("Date")
cs.title(ax1, "What 10,000 dollars became, 2021 to 2025", 1)
ax1.legend()

for ser, col, lab in ((dde, cs.BLUE, "five tickers held, 20% each"), (dds, cs.GREY, "SPY held"),
                      (dd, cs.MAROON, "50/200 rule, net of 0.1% costs")):
    ax2.plot(ser.index, 100 * ser.values, color=col, label=lab)
ax2.set_ylabel("Fall from the highest value so far (%)")
ax2.set_xlabel("Date")
cs.title(ax2, "Drawdown: how far below its last peak", 2)
ax2.legend()

x = np.arange(len(T))
ax3.bar(x - 0.2, 100 * held.values, width=0.4, color=cs.BLUE, label="slice held all the time")
ax3.bar(x + 0.2, 100 * contrib.values, width=0.4, color=cs.MAROON, label="slice under the 50/200 rule")
ax3.axhline(0, color=cs.GREY, lw=0.8)
ax3.set_xticks(x, T)
ax3.set_xlabel("Ticker (a 20% slice each)")
ax3.set_ylabel("Contribution to the return (% a year)")
cs.title(ax3, "Where the return came from, ticker by ticker", 3)
ax3.legend()

try:
    cs.save(fig, OUT / "evaluate.svg")
except OSError as e:
    sys.exit(f"cannot write the chart: {e}")
print(f"\nchart -> {OUT / 'evaluate.svg'}")
