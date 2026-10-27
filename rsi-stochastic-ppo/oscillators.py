# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "matplotlib"]
# ///
"""Numbers and chart for the post "RSI, stochastic %K and PPO: three oscillators".

Each of the three computed on SPY's close with its textbook settings (RSI 14, stochastic 14 with a
3-day %D, PPO 12/26/9), one day worked by hand for each, how often each sits past its extreme
line, and how much the three agree. The data has closes only, so the stochastic's highest high
and lowest low are the highest and lowest CLOSE of the window. Concept only: no rule is run.
Run:  uv run examples/rsi-stochastic-ppo/oscillators.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chartstyle as cs  # noqa: E402
import market_data as md  # noqa: E402

OUT = cs.out_dir("rsi-stochastic-ppo")
HAND_DAY = "2020-03-16"


def p(label, value):
    print(f"{label:<60} {value}")


def rsi(close, n=14):
    """Wilder's RSI: first averages are plain means of the first n gains and losses, after that
    each average is (previous * (n-1) + today) / n."""
    change = close.diff()
    gain, loss = change.clip(lower=0), (-change).clip(lower=0)
    avg_gain, avg_loss = gain.copy() * float("nan"), loss.copy() * float("nan")
    avg_gain.iloc[n] = gain.iloc[1:n + 1].mean()
    avg_loss.iloc[n] = loss.iloc[1:n + 1].mean()
    for i in range(n + 1, len(close)):
        avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (n - 1) + gain.iloc[i]) / n
        avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (n - 1) + loss.iloc[i]) / n
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs), avg_gain, avg_loss


px = md.prices(["SPY"])["SPY"]

r, ag, al = rsi(px)
k = 100 * (px - px.rolling(14).min()) / (px.rolling(14).max() - px.rolling(14).min())
d = k.rolling(3).mean()
ema12, ema26 = px.ewm(span=12, adjust=False).mean(), px.ewm(span=26, adjust=False).mean()
ppo = 100 * (ema12 - ema26) / ema26
signal = ppo.ewm(span=9, adjust=False).mean()
hist = ppo - signal

# --- RSI by hand -----------------------------------------------------------------------------
print(f"RSI(14) on {HAND_DAY}, by hand")
i = px.index.get_loc(HAND_DAY)
ch = px.iloc[i] - px.iloc[i - 1]
p("  today's change in the close", f"{ch:+.2f}")
p("  yesterday's average gain, average loss", f"{ag.iloc[i - 1]:.3f}, {al.iloc[i - 1]:.3f}")
p("  today's average gain: (prev * 13 + gain) / 14", f"{ag.iloc[i]:.3f}")
p("  today's average loss: (prev * 13 + loss) / 14", f"{al.iloc[i]:.3f}")
p("  RS = average gain / average loss", f"{ag.iloc[i] / al.iloc[i]:.3f}")
p("  RSI = 100 - 100 / (1 + RS)", f"{r.iloc[i]:.1f}")
first14 = px.iloc[:15]
p("  (the chain starts from plain means over the first 14 days)",
  f"gain {first14.diff().clip(lower=0).iloc[1:].mean():.3f}, loss {(-first14.diff()).clip(lower=0).iloc[1:].mean():.3f}")

# --- stochastic by hand ------------------------------------------------------------------------
print()
print(f"Stochastic %K(14) on {HAND_DAY}, by hand (closes only)")
w = px.loc[:HAND_DAY].iloc[-14:]
p("  highest close of the last 14 days", f"{w.max():.2f} on {w.idxmax().date()}")
p("  lowest close of the last 14 days", f"{w.min():.2f} on {w.idxmin().date()}")
p("  today's close", f"{w.iloc[-1]:.2f}")
p("  %K = 100 * (close - lowest) / (highest - lowest)", f"{k[HAND_DAY]:.1f}")
p("  %D = mean of the last three %K", f"{d[HAND_DAY]:.1f}")

# --- PPO by hand ------------------------------------------------------------------------------
print()
print(f"PPO(12,26,9) on {HAND_DAY}, by hand")
p("  12-day EMA (multiplier 2/13)", f"{ema12[HAND_DAY]:.2f}")
p("  26-day EMA (multiplier 2/27)", f"{ema26[HAND_DAY]:.2f}")
p("  PPO = 100 * (EMA12 - EMA26) / EMA26", f"{ppo[HAND_DAY]:+.2f}%")
p("  signal = 9-day EMA of PPO", f"{signal[HAND_DAY]:+.2f}%")
p("  histogram = PPO - signal", f"{hist[HAND_DAY]:+.2f}%")
p("  MACD the same day, in dollars", f"{ema12[HAND_DAY] - ema26[HAND_DAY]:+.2f}")

# --- how often each sits past its line -----------------------------------------------------------
print()
print("Share of days past the extreme lines, 2019 to 2023")
p("  RSI above 70 / below 30", f"{100 * (r > 70).mean():.1f}% / {100 * (r < 30).mean():.1f}%")
p("  %K above 80 / below 20", f"{100 * (k > 80).mean():.1f}% / {100 * (k < 20).mean():.1f}%")
p("  PPO above 0 / below 0", f"{100 * (ppo > 0).mean():.1f}% / {100 * (ppo < 0).mean():.1f}%")
p("  RSI range seen", f"{r.min():.1f} to {r.max():.1f}")
p("  PPO range seen", f"{ppo.min():+.2f}% to {ppo.max():+.2f}%")
p("  RSI's lowest day", f"{r.idxmin().date()}  (price low: {px['2020'].idxmin().date()})")
p("  PPO's lowest day", f"{ppo.idxmin().date()}")
p("  RSI on the price low 2020-03-23", f"{r['2020-03-23']:.1f}")
p("  %K first back above 20 after the low", f"{k['2020-03-23':].gt(20).idxmax().date()}")
p("  PPO back above 0 after the low", f"{ppo['2020-03-23':].gt(0).idxmax().date()}")
p("  RSI back above 50 after the low", f"{r['2020-03-23':].gt(50).idxmax().date()}")

# --- a calm uptrend day, for contrast -------------------------------------------------------------
CALM = "2021-08-13"
print()
print(f"The same three on {CALM}, a calm uptrend")
p("  RSI, %K, PPO", f"{r[CALM]:.1f}, {k[CALM]:.1f}, {ppo[CALM]:+.2f}%")

# --- agreement --------------------------------------------------------------------------------
print()
print("Correlation between the three, 2019 to 2023")
print(pd.DataFrame({"rsi": r, "stoch_k": k, "ppo": ppo}).dropna().corr().round(2).to_string())

# --- chart ----------------------------------------------------------------------------------
cs.apply()
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 3.8 * 4))
win = slice("2020-01-01", "2020-08-31")
ax1.plot(px[win].index, px[win], color=cs.MAROON, label="SPY close")
ax1.set_xlabel("Date (2020)")
ax1.set_ylabel("Price (dollars)")
cs.title(ax1, "SPY's close, January to August 2020", 1)
ax1.legend()

ax2.plot(r[win].index, r[win], color=cs.MAROON, label="RSI, 14 days")
ax2.axhline(70, color=cs.GREY, lw=0.8, ls="--")
ax2.axhline(30, color=cs.GREY, lw=0.8, ls="--")
ax2.set_ylim(0, 100)
ax2.set_xlabel("Date (2020)")
ax2.set_ylabel("RSI (0 to 100)")
cs.title(ax2, "RSI: below 30 in the crash, above 70 in the recovery", 2)
ax2.legend()

ax3.plot(k[win].index, k[win], color=cs.MAROON, label="%K, 14 days, from closes")
ax3.plot(d[win].index, d[win], color=cs.BLUE, label="%D, 3-day average of %K")
ax3.axhline(80, color=cs.GREY, lw=0.8, ls="--")
ax3.axhline(20, color=cs.GREY, lw=0.8, ls="--")
ax3.set_ylim(0, 100)
ax3.set_xlabel("Date (2020)")
ax3.set_ylabel("Stochastic (0 to 100)")
cs.title(ax3, "Stochastic: pinned to 0 as every close is a new low", 3)
ax3.legend()

ax4.plot(ppo[win].index, ppo[win], color=cs.MAROON, label="PPO, 12/26-day")
ax4.plot(signal[win].index, signal[win], color=cs.BLUE, label="signal, 9-day average of PPO")
ax4.axhline(0, color=cs.GREY, lw=0.8)
ax4.set_xlabel("Date (2020)")
ax4.set_ylabel("PPO (% of the 26-day average)")
cs.title(ax4, "PPO: the fast average's lead over the slow one, in percent", 4)
ax4.legend()

cs.save(fig, OUT / "oscillators.svg")
print(f"\nchart -> {OUT / 'oscillators.svg'}")
