# /// script
# requires-python = ">=3.11"
# dependencies = ["yfinance", "pandas"]
# ///
"""Download daily adjusted closes once and keep them in the repo, so every post's numbers
are reproducible without a network. Run:  uv run examples/fetch_prices.py"""
from pathlib import Path
import yfinance as yf

TICKERS = ["SPY", "AAPL", "JPM", "XOM", "GLD"]
out = Path(__file__).resolve().parent / "data" / "prices.csv"
px = yf.download(TICKERS, start="2019-01-01", end="2024-01-01", auto_adjust=True, progress=False)["Close"]
px = px[TICKERS].round(4)
px.index.name = "date"
px.to_csv(out)
print(f"wrote {out}: {px.shape[0]} days x {px.shape[1]} tickers, {px.index[0].date()} to {px.index[-1].date()}")
print(px.tail(2))
