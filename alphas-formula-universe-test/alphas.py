# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "matplotlib", "scipy"]
# ///
"""Numbers and chart for the post "Alphas: a formula, a universe and a test of whether it worked".

Two formulas from Kakushadze, "101 Formulaic Alphas" (2015), Appendix A, computed on the blog's
five tickers from closes only: Alpha#9 (a time-series signal in price units) and Alpha#10 (the
same idea ranked across the universe each day). Then the test: the rank correlation between the
signal and the return over the next h days, for several h (how fast the signal fades), year by
year (whether it held), and the correlation between the two alphas (how much they overlap).
Run:  uv run examples/alphas-formula-universe-test/alphas.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chartstyle as cs  # noqa: E402
import market_data as md  # noqa: E402

OUT = cs.out_dir("alphas-formula-universe-test")
HORIZONS = [1, 2, 3, 5, 10, 20]
SHOW_DAY = "2022-06-17"


def p(label, value):
    print(f"{label:<70} {value}")


px = md.prices()
d1 = px.diff()                                              # delta(close, 1): today's close minus yesterday's


def alpha9_like(window):
    """Appendix A, Alpha#9 (window 5) and the inside of Alpha#10 (window 4):
    if the close has risen every day of the window, keep today's rise (momentum);
    if it has fallen every day, keep today's fall (momentum);
    otherwise flip the sign of today's move (mean reversion)."""
    lo = d1.rolling(window).min()
    hi = d1.rolling(window).max()
    return d1.where(lo > 0, d1.where(hi < 0, -d1))


a9 = alpha9_like(5)                                         # Alpha#9, in dollars per share
a10 = alpha9_like(4).rank(axis=1, pct=True)                 # Alpha#10: rank(...) across the five, 0.2 .. 1.0
a9, a10 = a9.dropna(), a10.dropna()

# --- what one day looks like ------------------------------------------------------------------
day = pd.Timestamp(SHOW_DAY)
print(f"One day, {SHOW_DAY}: the inside of Alpha#10 (window 4) and its rank")
tbl = pd.DataFrame({
    "close": px.loc[day].round(2),
    "delta(close,1)": d1.loc[day].round(2),
    "4-day min": d1.rolling(4).min().loc[day].round(2),
    "4-day max": d1.rolling(4).max().loc[day].round(2),
    "signal": alpha9_like(4).loc[day].round(2),
    "rank": a10.loc[day].round(1),
})
print(tbl.to_string())

print()
print("How often each branch fires (share of days, Alpha#9, window 5)")
lo5, hi5 = d1.rolling(5).min(), d1.rolling(5).max()
branch = pd.DataFrame({"five rises": (lo5 > 0).mean(), "five falls": (hi5 < 0).mean()}).dropna()
branch["mixed: reversal"] = 1 - branch.sum(axis=1)
print((100 * branch).round(1).to_string())

# --- the test: rank correlation with what came next -------------------------------------------
fwd = {h: px.shift(-h) / px - 1 for h in HORIZONS}          # return from today's close to h days later


def ic_timeseries(sig, h, years=None):
    """Alpha#9 is in price units, so compare each ticker with its own future: Spearman per
    ticker, then the average of the five."""
    out = []
    for t in md.TICKERS:
        s, f = sig[t], fwd[h][t]
        if years is not None:
            s = s[s.index.year.isin(years)]
        both = pd.concat([s, f], axis=1, sort=False).dropna()
        out.append(spearmanr(both.iloc[:, 0], both.iloc[:, 1]).statistic)
    return float(np.mean(out))


def ic_crosssection(sig, h, years=None):
    """Alpha#10 is a daily ranking of the five, so the test is daily too: Spearman across the
    five tickers on each day, then the average over days."""
    s = sig if years is None else sig[sig.index.year.isin(years)]
    f = fwd[h].reindex(s.index)
    vals = [spearmanr(s.loc[i], f.loc[i]).statistic for i in s.index if f.loc[i].notna().all()]
    return float(np.mean(vals))


print()
print("Information coefficient: rank correlation of the signal with the next-h-day return, 2019-2023")
rows = []
for h in HORIZONS:
    rows.append({"horizon (days)": h, "Alpha#9 (per ticker)": ic_timeseries(a9, h),
                 "Alpha#10 (across the five)": ic_crosssection(a10, h)})
ic = pd.DataFrame(rows).set_index("horizon (days)")
print(ic.round(3).to_string())

print()
print("The same, 1-day horizon, year by year")
yr_rows = []
for y in range(2019, 2024):
    yr_rows.append({"year": y, "Alpha#9": ic_timeseries(a9, 1, [y]), "Alpha#10": ic_crosssection(a10, 1, [y])})
yr = pd.DataFrame(yr_rows).set_index("year")
print(yr.round(3).to_string())

print()
p("Alpha#9 at 1 day, 2019-2021 (first three years)", f"{ic_timeseries(a9, 1, [2019, 2020, 2021]):.3f}")
p("Alpha#9 at 1 day, 2022-2023 (held-out two years)", f"{ic_timeseries(a9, 1, [2022, 2023]):.3f}")
p("Days in the sample per ticker", len(a9))

# --- overlap ----------------------------------------------------------------------------------
print()
print("Overlap between signals (average per-ticker Spearman)")
rev = -d1.dropna()                                          # plain one-day reversal: minus today's move


def overlap(x, y):
    both = [spearmanr(*pd.concat([x[t], y[t]], axis=1, sort=False).dropna().values.T).statistic for t in md.TICKERS]
    return float(np.mean(both))


p("Alpha#9 (window 5) vs the inside of Alpha#10 (window 4)", f"{overlap(a9, alpha9_like(4).dropna()):.3f}")
p("Alpha#9 vs plain reversal, minus today's move", f"{overlap(a9, rev):.3f}")

# --- chart ------------------------------------------------------------------------------------
cs.apply()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 3.8 * 2))
ax1.plot(ic.index, ic["Alpha#9 (per ticker)"], color=cs.MAROON, marker="o", ms=4,
         label="Alpha#9, each ticker against its own future")
ax1.plot(ic.index, ic["Alpha#10 (across the five)"], color=cs.BLUE, marker="o", ms=4,
         label="Alpha#10, ranked across the five each day")
ax1.axhline(0, color=cs.GREY, lw=0.8)
ax1.set_xticks(HORIZONS)
ax1.set_xlabel("Days ahead the return is measured over")
ax1.set_ylabel("Rank correlation with that return (IC)")
cs.title(ax1, "Correlation with the return h days out: small at every horizon", 1)
ax1.legend()

x = np.arange(len(yr))
w = 0.38
ax2.bar(x - w / 2, yr["Alpha#9"], width=w, color=cs.MAROON, label="Alpha#9, 1-day horizon")
ax2.bar(x + w / 2, yr["Alpha#10"], width=w, color=cs.BLUE, label="Alpha#10, 1-day horizon")
ax2.axhline(0, color=cs.GREY, lw=0.8)
ax2.set_xticks(x, [str(y) for y in yr.index])
ax2.set_xlabel("Year")
ax2.set_ylabel("Rank correlation with next day's return (IC)")
cs.title(ax2, "Did it hold? The 1-day IC, one year at a time", 2)
ax2.legend()

cs.save(fig, OUT / "alphas.svg")
print(f"\nchart -> {OUT / 'alphas.svg'}")
