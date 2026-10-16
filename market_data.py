"""Daily prices shared by the blog's posts: five tickers, 2019-2023, adjusted closes
downloaded once by fetch_prices.py and kept in data/prices.csv so the numbers never move."""
from pathlib import Path

import numpy as np
import pandas as pd

TICKERS = ["SPY", "AAPL", "JPM", "XOM", "GLD"]
TRADING_DAYS = 252


def prices(tickers=None, start=None, end=None):
    """Adjusted closes, one column per ticker, indexed by date."""
    px = pd.read_csv(Path(__file__).resolve().parent / "data" / "prices.csv", index_col="date", parse_dates=True)
    px = px[list(tickers or TICKERS)]
    return px.loc[start:end]


def daily_returns(px):
    """r_t = p_t / p_{t-1} - 1, first day dropped."""
    return (px / px.shift(1) - 1).iloc[1:]


def features(tickers=None, horizon=20, start=None, end=None):
    """The feature table the ML posts share: one row per ticker and trading day, from prices only.

    Features, all known at the close of day t:
      ret_1d, ret_5d, ret_20d   the last 1-, 5- and 20-day returns
      vol_5d, vol_20d           standard deviation of daily returns over the last 5 and 20 days
      sma_ratio                 price divided by its 20-day average, minus one
      bollinger                 how many 20-day standard deviations the price is from that average
    Targets: next_vol, the standard deviation of the NEXT `horizon` daily returns (20 days, about
    a month: what the market is about to do in risk terms), and next_ret, the return over those days.
    ANNUALISE turns a daily standard deviation into the yearly percentage the posts quote.
    Returns a DataFrame indexed by (date, ticker).
    """
    frames = []
    for t in (tickers or TICKERS):
        p = prices([t], start, end)[t]
        r = p.pct_change()
        sma = p.rolling(20).mean()
        sd20 = r.rolling(20).std()
        df = pd.DataFrame({
            "ret_1d": r,
            "ret_5d": p / p.shift(5) - 1,
            "ret_20d": p / p.shift(20) - 1,
            "vol_5d": r.rolling(5).std(),
            "vol_20d": sd20,
            "sma_ratio": p / sma - 1,
            "bollinger": (p - sma) / (2 * p.rolling(20).std()),
            "next_vol": r[::-1].rolling(horizon).std()[::-1].shift(-1),
            "next_ret": p.shift(-horizon) / p - 1,
        })
        df["ticker"] = t
        frames.append(df.dropna())
    out = pd.concat(frames).set_index("ticker", append=True).sort_index()
    return out


ANNUALISE = np.sqrt(TRADING_DAYS) * 100
FEATURES = ["ret_1d", "ret_5d", "ret_20d", "vol_5d", "vol_20d", "sma_ratio", "bollinger"]


def split(df, test_from="2022-01-01"):
    """Train on the years before `test_from`, test on the rest. Time order, never shuffled."""
    dates = df.index.get_level_values("date")
    return df[dates < test_from], df[dates >= test_from]


# --- a made-up version of the same problem, for the posts that need to know the truth ----------
def vol_curve(x):
    """A made-up "true" next-month volatility for a given this-month volatility x (both % a year):
    a gentle slope, plus a step up once the market is already in a stress regime (above ~25%).
    Shaped like the real SPY data, but invented so bias and variance can be measured."""
    x = np.asarray(x, dtype=float)
    return 6 + 0.45 * x + 8 / (1 + np.exp(-(x - 25) / 3))


def vol_samples(n=80, seed=0, noise=3.0):
    """n made-up months: this-month volatility between 5% and 45%, next-month = curve + noise."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(5, 45, n)
    return x, vol_curve(x) + rng.normal(0, noise, n)
