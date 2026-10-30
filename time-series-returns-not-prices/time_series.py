# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "matplotlib"]
# ///
"""Numbers and chart for the post "Time series: returns are the data, prices are not".

Autocorrelation of prices, returns and absolute returns; a shuffle test for what "no
autocorrelation" looks like; volatility clustering; a trailing against a centred moving average.
All on SPY, 2020-2025, from the blog's shared prices.
Run:  uv run examples/time-series-returns-not-prices/time_series.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chartstyle as cs  # noqa: E402
import market_data as md  # noqa: E402

OUT = cs.out_dir("time-series-returns-not-prices")
LAGS = range(1, 21)


def p(label, value):
    print(f"{label:<64} {value}")


px = md.prices()["SPY"]
r = md.daily_returns(md.prices())["SPY"]
n = len(r)

print("\n== autocorrelation at lag 1 (SPY) ==")
ac_price = [px.autocorr(k) for k in LAGS]
ac_ret = [r.autocorr(k) for k in LAGS]
ac_abs = [r.abs().autocorr(k) for k in LAGS]
p("daily returns in the sample", n)
p("price, lag 1 / lag 20", f"{ac_price[0]:.3f} / {ac_price[-1]:.3f}")
p("return, lag 1 / lag 2 / lag 5", f"{ac_ret[0]:+.3f} / {ac_ret[1]:+.3f} / {ac_ret[4]:+.3f}")
p("absolute return, lag 1 / lag 5 / lag 20", f"{ac_abs[0]:+.3f} / {ac_abs[4]:+.3f} / {ac_abs[-1]:+.3f}")
no2020 = r[r.index.year != 2020]
p("return, lag 1: 2020 alone / all years except 2020", f"{r['2020'].autocorr(1):+.3f} / {no2020.autocorr(1):+.3f}")
p("absolute return, lag 1, all years except 2020", f"{no2020.abs().autocorr(1):+.3f}")

print("\n== shuffle test: how big can lag-1 autocorrelation be by luck? ==")
rng = np.random.default_rng(0)
shuffled = np.array([pd.Series(rng.permutation(r.values)).autocorr(1) for _ in range(1000)])
band = np.quantile(np.abs(shuffled), 0.95)
p("1,000 shuffles of the same returns: 95% of |lag-1 autocorr| below", f"{band:.3f}")
p("absolute returns, lag 1, in shuffles (95% below)",
  f"{np.quantile(np.abs([pd.Series(np.abs(rng.permutation(r.values))).autocorr(1) for _ in range(1000)]), 0.95):.3f}")

print("\n== volatility clustering ==")
big = r.abs() >= r.abs().quantile(0.95)
by_month = big.groupby(r.index.to_period("M")).sum().sort_values(ascending=False)
p("days in the top 5% of |return|", int(big.sum()))
p("of those, in the three busiest months", f"{int(by_month.iloc[:3].sum())}  ({', '.join(str(m) for m in by_month.index[:3])})")
n_months = r.index.to_period("M").nunique()
p(f"months out of {n_months} with none of them", n_months - int((by_month > 0).sum()))
nxt = r.abs().shift(-1)
p("mean |return| next day: after a top-5% day / after any other day",
  f"{nxt[big].mean():.2%} / {nxt[~big].mean():.2%}")

print("\n== rolling 20-day volatility, annualised ==")
vol = r.rolling(20).std() * np.sqrt(252) * 100
p("whole-sample volatility (% a year)", f"{r.std() * np.sqrt(252) * 100:.1f}")
p("20-day volatility: lowest / highest (% a year)", f"{vol.min():.1f} ({vol.idxmin():%Y-%m-%d}) / {vol.max():.1f} ({vol.idxmax():%Y-%m-%d})")

print("\n== trailing against centred 20-day moving average, 2020 ==")
trail = px.rolling(20).mean()
centred = px.rolling(20, center=True).mean()
low = px["2020-02":"2020-05"].idxmin()
t_up = trail["2020-03":].diff().gt(0).idxmax()
c_up = centred["2020-03":].diff().gt(0).idxmax()
p("SPY low of 2020", f"{low:%Y-%m-%d}")
p("centred average starts rising on / trailing average starts rising on", f"{c_up:%Y-%m-%d} / {t_up:%Y-%m-%d}")
p("days the centred average 'knew' before the trailing one", f"{len(px[c_up:t_up]) - 1}")
p("centred average on the low needs prices up to", f"{px.index[px.index.get_loc(low) + 10]:%Y-%m-%d}")

# ------------------------------------------------------------------ chart ----
cs.apply()
fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(8, 3.8 * 3))

cs.title(a1, "Correlation of SPY's daily return with the return k days earlier, with and without its sign", 1)
x = np.array(LAGS)
w = 0.38
a1.bar(x - w / 2, ac_ret, w, color=cs.MAROON, label="daily return")
a1.bar(x + w / 2, ac_abs, w, color=cs.BLUE, label="size of daily return, sign dropped")
a1.axhline(band, color=cs.INK, lw=0.8, ls="--")
a1.axhline(-band, color=cs.INK, lw=0.8, ls="--", label=f"as large as luck gets in 1,000 shuffles (±{band:.2f})")
a1.set_xticks(x[::2])
a1.set_xlabel("Lag k (trading days)")
a1.set_ylabel("Autocorrelation (−1 to 1)")
a1.legend()

cs.title(a2, "SPY daily returns: the big moves arrive in clusters", 2)
a2.plot(r.index, r * 100, color=cs.MAROON, lw=0.7, label="daily return")
a2.plot(vol.index, vol / np.sqrt(252), color=cs.BLUE, lw=1.2, label="trailing 20-day volatility, per day")
a2.plot(vol.index, -vol / np.sqrt(252), color=cs.BLUE, lw=1.2, label="_")
a2.set_xlabel("Date")
a2.set_ylabel("Daily return (%)")
a2.legend()

cs.title(a3, "Two 20-day moving averages at the 2020 low: one uses the future", 3)
seg = slice("2020-02-01", "2020-06-30")
a3.plot(px[seg].index, px[seg], color=cs.GREY, lw=1, label="SPY price")
a3.plot(trail[seg].index, trail[seg], color=cs.MAROON, label="trailing average (last 20 days)")
a3.plot(centred[seg].index, centred[seg], color=cs.BLUE, label="centred average (10 days each side)")
a3.axvline(low, color=cs.INK, lw=0.8, ls="--")
a3.set_xlabel("Date (2020)")
a3.set_ylabel("Price (dollars)")
a3.legend()

cs.save(fig, OUT / "time-series.svg")
