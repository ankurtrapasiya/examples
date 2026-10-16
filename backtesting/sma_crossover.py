# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["numpy", "pandas", "matplotlib", "backtesting", "vectorbt"]
# ///
"""Numbers and chart for the post "Backtesting in Python".

Run:  uv run examples/backtesting/sma_crossover.py

One strategy throughout: buy SPY when its fast moving average crosses above the slow one,
sell when it crosses back below. Three experiments:
  1. the strategy by hand in pandas (vectorised), with and without a one-day lag, to show
     look-ahead bias;
  2. the same strategy in backtesting.py (event-driven), with a commission;
  3. a parameter sweep in vectorbt on 2019-2021, best pair applied to 2022-2023, to show
     over-fitting.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import vectorbt as vbt
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chartstyle as cs  # noqa: E402
import market_data as md  # noqa: E402

OUT = cs.out_dir("backtesting")
OUT.mkdir(parents=True, exist_ok=True)
FAST, SLOW = 20, 100
FEE = 0.001  # 0.1% of each trade's value

spy = md.prices(["SPY"])["SPY"]
ret = spy.pct_change().fillna(0.0)


def total(r):
    return float((1 + r).prod() - 1)


def sharpe(r):
    return float(r.mean() / r.std() * np.sqrt(md.TRADING_DAYS))


def max_drawdown(r):
    eq = (1 + r).cumprod()
    return float((eq / eq.cummax() - 1).min())


def summary(name, r):
    print(f"{name:42s} return {total(r):+7.1%}   Sharpe {sharpe(r):5.2f}   max drawdown {max_drawdown(r):+6.1%}")
    return r


# --- 1. by hand, vectorised: the look-ahead trap --------------------------------------
fast, slow = spy.rolling(FAST).mean(), spy.rolling(SLOW).mean()
in_market = (fast > slow).astype(int)           # 1 = hold SPY, 0 = hold cash
print(f"SPY {spy.index[0].date()} to {spy.index[-1].date()}, {len(spy)} days; fast {FAST}, slow {SLOW}")
summary("buy and hold", ret)
cheat = summary("crossover, position set on the same close", in_market * ret)
honest = summary("crossover, position set on the next day", in_market.shift(1).fillna(0) * ret)
print(f"look-ahead inflates the return by {total(cheat) - total(honest):+.1%} points")
print("days in the market:", int(in_market.sum()), "of", len(spy))
# what backtesting.py will do below: ignore the warm-up (fast already above slow on the first
# day both averages exist, which is not a crossing), and fill one day later still
first_cross = in_market.index[(in_market.diff() != 0) & in_market.index.isin(in_market.index[SLOW:])][0]
strict = in_market.where(in_market.index >= first_cross, 0)
summary("same, no warm-up trade, filled a day later", strict.shift(2).fillna(0) * ret)


# --- 2. backtesting.py: event-driven, with commission ----------------------------------
class SmaCross(Strategy):
    fast = FAST
    slow = SLOW

    def init(self):
        sma = lambda s, n: pd.Series(s).rolling(n).mean()  # noqa: E731
        self.fast_ma = self.I(sma, self.data.Close, self.fast)
        self.slow_ma = self.I(sma, self.data.Close, self.slow)

    def next(self):
        if crossover(self.fast_ma, self.slow_ma):
            self.buy()
        elif crossover(self.slow_ma, self.fast_ma):
            self.position.close()


ohlc = pd.DataFrame({c: spy for c in ["Open", "High", "Low", "Close"]})  # closes only: fills at the close
bt = Backtest(ohlc, SmaCross, cash=10_000, commission=FEE, exclusive_orders=True, finalize_trades=True)
stats = bt.run()
print("\nbacktesting.py")
for k in ["Return [%]", "Buy & Hold Return [%]", "Sharpe Ratio", "Max. Drawdown [%]", "# Trades", "Win Rate [%]", "Exposure Time [%]"]:
    print(f"  {k:24s} {stats[k]:8.2f}")
trades = stats["_trades"]
print(f"  commission paid {(trades['Size'].abs() * (trades['EntryPrice'] + trades['ExitPrice']) * FEE).sum():.0f} on 10,000")


# --- 3. vectorbt: sweep on 2019-2021, test on 2022-2023 ------------------------------
train, test = spy[:"2021"], spy["2022":]
windows = np.arange(5, 255, 5)


def sweep(px):
    """Every fast/slow pair from `windows` (fast < slow) on one price series: one call."""
    fast_ma, slow_ma = vbt.MA.run_combs(px, window=windows, r=2, short_names=["fast", "slow"])
    pf = vbt.Portfolio.from_signals(px, fast_ma.ma_crossed_above(slow_ma), fast_ma.ma_crossed_below(slow_ma),
                                    fees=FEE, freq="1D")
    return pf.total_return()


tr, oos_all = sweep(train), sweep(test)
best = tr.idxmax()
bh_train, bh_test = train.iloc[-1] / train.iloc[0] - 1, test.iloc[-1] / test.iloc[0] - 1
print(f"\nvectorbt sweep: {len(tr)} pairs on 2019-2021; best fast={best[0]}, slow={best[1]} "
      f"return {tr.max():+.1%}; median {tr.median():+.1%}; worst {tr.min():+.1%}; buy and hold {bh_train:+.1%}")
print(f"pairs beating buy and hold in-sample: {(tr > bh_train).sum()} of {len(tr)}")
top = tr.sort_values(ascending=False).head(5)
print("\ntop 5 in-sample pairs, then the same pair on 2022-2023:")
for k, r in top.items():
    print(f"  fast {k[0]:3d} slow {k[1]:3d}   2019-21 {r:+7.1%}   2022-23 {oos_all[k]:+7.1%}")
print(f"  {'buy and hold':20s}   2019-21 {bh_train:+7.1%}   2022-23 {bh_test:+7.1%}")
print(f"  {'fast 20 slow 100':20s}   2019-21 {tr[(FAST, SLOW)]:+7.1%}   2022-23 {oos_all[(FAST, SLOW)]:+7.1%}")
print(f"all {len(tr)} pairs on 2022-23: median {oos_all.median():+.1%}, best {oos_all.max():+.1%}, worst {oos_all.min():+.1%}; "
      f"beating buy and hold {(oos_all > bh_test).sum()} of {len(tr)}; "
      f"rank correlation in-sample vs out-of-sample {tr.corr(oos_all, method='spearman'):.2f}")
print(f"top 10% in-sample pairs: median out-of-sample {oos_all[tr >= tr.quantile(0.9)].median():+.1%}; "
      f"bottom 10%: {oos_all[tr <= tr.quantile(0.1)].median():+.1%}")

# --- chart ---------------------------------------------------------------------------
cs.apply()
fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(8, 3.8 * 3))

a1.plot(spy.index, spy, color=cs.GREY, lw=1, label="SPY close")
a1.plot(spy.index, fast, color=cs.MAROON, lw=1.2, label=f"{FAST}-day average")
a1.plot(spy.index, slow, color=cs.BLUE, lw=1.2, label=f"{SLOW}-day average")
flip = in_market.diff().fillna(0)
a1.scatter(spy.index[flip > 0], spy[flip > 0], marker="^", s=48, color=cs.GREEN, zorder=3, label="buy")
a1.scatter(spy.index[flip < 0], spy[flip < 0], marker="v", s=48, color=cs.AMBER, zorder=3, label="sell")
cs.title(a1, "The signal: buy when the fast average crosses above the slow one", 1)
a1.set_ylabel("SPY price (dollars, adjusted)")
a1.set_xlabel("Date")
a1.legend()

eq_bh = (1 + ret).cumprod() * 10_000
eq_bt = stats["_equity_curve"]["Equity"]
eq_cheat = (1 + in_market * ret).cumprod() * 10_000
a2.plot(eq_cheat.index, eq_cheat, color=cs.AMBER, lw=1.2, ls="--", label="crossover with look-ahead (cheating)")
a2.plot(eq_bt.index, eq_bt, color=cs.MAROON, lw=1.4, label="crossover, next-day fills, 0.1% commission")
a2.plot(eq_bh.index, eq_bh, color=cs.BLUE, lw=1.4, label="buy and hold")
cs.title(a2, "What 10,000 dollars became under each rule, 2019-2023", 2)
a2.set_ylabel("Portfolio value (dollars)")
a2.set_xlabel("Date")
a2.legend()

a3.scatter(tr.values * 100, oos_all.values * 100, s=22, color=cs.GREY, alpha=0.8, label="one fast/slow pair", rasterized=True)
a3.scatter([tr[best] * 100], [oos_all[best] * 100], s=70, color=cs.MAROON, zorder=3, label=f"best in-sample pair (fast {best[0]}, slow {best[1]})")
a3.scatter([tr[(FAST, SLOW)] * 100], [oos_all[(FAST, SLOW)] * 100], s=70, color=cs.PURPLE, zorder=3, label=f"the pair the post uses (fast {FAST}, slow {SLOW})")
a3.axhline(100 * bh_test, color=cs.BLUE, lw=1, ls="--", label="buy and hold, 2022-2023")
cs.title(a3, f"{len(tr):,} parameter pairs: return where they were chosen vs. return afterwards", 3)
a3.set_xlabel("Return on 2019-2021, the years the pair was chosen on (%)")
a3.set_ylabel("Return on 2022-2023, unseen years (%)")
a3.legend()

cs.save(fig, OUT / "sma-crossover.svg")
print("\nchart written to", OUT / "sma-crossover.svg")
