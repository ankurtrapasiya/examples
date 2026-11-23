# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "matplotlib"]
# ///
"""Numbers and chart for the post "Options: a payoff is a bent line".

Payoffs at expiry for calls, puts and four strategies built from them, on SPY. There are no option
quotes in the data, so premiums come from the Black-Scholes model (Clarke, Options and Futures: A
Tutorial, chapter 5) with SPY's own volatility over the previous 21 trading days and an interest
rate of zero. The model is checked against Clarke's Table 5.4 first. The last part rolls a
one-month protective put and a one-month covered call on SPY from 2020 to 2025 and compares their
monthly returns with holding SPY alone.
Run:  uv run examples/options-a-payoff-is-a-bent-line/options.py
"""
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chartstyle as cs  # noqa: E402
import market_data as md  # noqa: E402

OUT = cs.out_dir("options-a-payoff-is-a-bent-line")
MONTH = 21                    # trading days in one option's life
T = MONTH / md.TRADING_DAYS   # the same, in years
R = 0.0                       # interest rate, set to zero (see the check below for what 4% changes)


def show(label, value):
    print(f"{label:<64} {value}")


def N(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs(S, K, T, sigma, r=R):
    """Black-Scholes call and put prices (Clarke, chapter 5)."""
    d1 = (math.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    call = S * N(d1) - K * math.exp(-r * T) * N(d2)
    put = call - S + K * math.exp(-r * T)          # put-call parity
    return call, put


# --- 0. the model against Clarke's Table 5.4 -------------------------------------------------------
print("Check against Clarke Table 5.4 (S = 50, r = 10%, sigma = 20%), Black-Scholes calls")
for days, book in ((30, (5.39, 1.36, 0.08)), (120, (6.75, 3.15, 1.09))):
    mine = tuple(round(bs(50, k, days / 365, 0.20, 0.10)[0], 2) for k in (45, 50, 55))
    show(f"  {days} days, strikes 45 / 50 / 55: book / this code", f"{book} / {mine}")
print()

try:
    px = md.prices(["SPY"])["SPY"]
except (OSError, KeyError) as e:
    sys.exit(f"cannot read the prices file data/prices.csv: {e}")
r = md.daily_returns(px.to_frame())["SPY"]
vol = r.rolling(MONTH).std() * math.sqrt(md.TRADING_DAYS)

# --- 1. one option on the last day of data ---------------------------------------------------------
day = px.index[-1]
S0, sig = px[day], vol[day]
K = round(S0)                                      # a whole-dollar strike next to the price
C, P = bs(S0, K, T, sig)
show(f"SPY on {day.date()}, volatility over the last {MONTH} days", f"{S0:.2f}, {100 * sig:.1f}% a year")
show(f"  one-month strike {K}: call / put premium", f"{C:.2f} / {P:.2f}")
show("  same with a 4% interest rate", "{:.2f} / {:.2f}".format(*bs(S0, K, T, sig, 0.04)))
show("  call break-even K + C / put break-even K - P", f"{K + C:.2f} / {K - P:.2f}")
show("  move needed to break even, call / put", f"{100 * (K + C - S0) / S0:+.2f}% / {100 * (K - P - S0) / S0:+.2f}%")
print("  SPY at expiry: call payoff, call profit, put payoff, put profit")
for ST in (K - 40, K - 20, K, K + 20, K + 40):
    cp, pp = max(ST - K, 0), max(K - ST, 0)
    show(f"    {ST}", f"{cp:6.2f} {cp - C:+7.2f} {pp:6.2f} {pp - P:+7.2f}")
print()

# --- 2. intrinsic and time value across strikes ----------------------------------------------------
print(f"Call prices today by strike (SPY {S0:.2f}): price = intrinsic + time value")
strikes = [round(S0 * m) for m in (0.90, 0.95, 1.00, 1.05, 1.10)]
for k in strikes:
    c = bs(S0, k, T, sig)[0]
    iv = max(S0 - k, 0)
    show(f"  strike {k}", f"price {c:6.2f} = intrinsic {iv:6.2f} + time value {c - iv:5.2f}")
print()

# --- 3. leverage: the same move in SPY and in the call ---------------------------------------------
print("If SPY moves today, the call moves (same volatility, same time left):")
for move in (-0.05, -0.02, 0.02, 0.05):
    c1 = bs(S0 * (1 + move), K, T, sig)[0]
    show(f"  SPY {100 * move:+.0f}%", f"call {C:.2f} -> {c1:.2f}, {100 * (c1 / C - 1):+.0f}%")
print()

# --- 4. strategies at expiry -----------------------------------------------------------------------
Klo, Khi = round(S0 * 0.95), round(S0 * 1.05)
Pp = bs(S0, Klo, T, sig)[1]                       # protective put, 5% below
Cc = bs(S0, Khi, T, sig)[0]                       # covered call, 5% above
Cb = bs(S0, Khi, T, sig)[0]
show(f"Protective put: SPY + put at {Klo} for {Pp:.2f}", f"floor {Klo - S0 - Pp:+.2f} ({100 * (Klo - S0 - Pp) / S0:+.2f}%)")
show(f"Covered call: SPY - call at {Khi} for {Cc:.2f}", f"cap {Khi - S0 + Cc:+.2f} ({100 * (Khi - S0 + Cc) / S0:+.2f}%)")
show(f"Straddle at {K}: call + put cost {C + P:.2f}", f"break-evens {K - C - P:.2f} and {K + C + P:.2f} ({100 * (C + P) / S0:.2f}% either way)")
show(f"Bull call spread: buy {K} call, sell {Khi} call, cost {C - Cb:.2f}",
     f"max gain {Khi - K - (C - Cb):.2f}, break-even {K + C - Cb:.2f}")
print()

# --- 5. rolled every month on real SPY -------------------------------------------------------------
starts = px.index[MONTH::MONTH]
rows = []
for a, b in zip(starts[:-1], starts[1:]):
    s0, s1, sg = px[a], px[b], vol[a]
    kp, kc = s0 * 0.95, s0 * 1.05
    p = bs(s0, kp, T, sg)[1]
    c = bs(s0, kc, T, sg)[0]
    rows.append({"start": a, "spy": s1 / s0 - 1,
                 "put": (max(s1, kp) - p) / s0 - 1,       # SPY + put, per dollar of SPY at the start
                 "call": (min(s1, kc) + c) / s0 - 1,      # SPY - call
                 "p": p / s0, "c": c / s0})
m = pd.DataFrame(rows).set_index("start")
print(f"Rolled monthly, {len(m)} months from {m.index[0].date()} (5% out of the money, 21 trading days)")
show("  average premium, put / call (% of SPY)", f"{100 * m.p.mean():.2f}% / {100 * m.c.mean():.2f}%")
for col, name in (("spy", "SPY alone"), ("put", "protective put"), ("call", "covered call")):
    x = m[col]
    show(f"  {name}: mean / worst / best month / growth", f"{100 * x.mean():+.2f}% / {100 * x.min():+.1f}% / {100 * x.max():+.1f}% / {100 * ((1 + x).prod() - 1):+.1f}%")
show("  months the put paid off (SPY fell more than 5%)", f"{int((m.spy < -0.05).sum())}")
show("  months the call capped the gain (SPY rose more than 5%)", f"{int((m.spy > 0.05).sum())}")
worst = m.spy.idxmin()
show(f"  worst month for SPY, starting {worst.date()}: SPY / protected / covered",
     f"{100 * m.spy[worst]:+.1f}% / {100 * m.put[worst]:+.1f}% / {100 * m.call[worst]:+.1f}%")
print()

# --- chart -----------------------------------------------------------------------------------------
cs.apply()
fig, axs = plt.subplots(3, 1, figsize=(8, 3.8 * 3))
S = np.linspace(K - 60, K + 60, 241)
ax = axs[0]
ax.plot(S, np.maximum(S - K, 0) - C, color=cs.MAROON, label=f"Long call, strike {K}, after its premium")
ax.plot(S, np.maximum(K - S, 0) - P, color=cs.BLUE, label=f"Long put, strike {K}, after its premium")
ax.plot(S, S - S0, color=cs.GREY, ls="--", label="One SPY share bought today")
ax.axhline(0, color=cs.GREY, lw=0.8)
ax.set_xlabel("SPY at expiry (dollars)"); ax.set_ylabel("Profit at expiry (dollars a share)")
ax.legend(); cs.title(ax, "A call and a put: two lines bent at the strike", 1)
ax = axs[1]
ax.plot(S, np.maximum(S, Klo) - S0 - Pp, color=cs.BLUE, label=f"Protective put: SPY plus a {Klo} put")
ax.plot(S, np.minimum(S, Khi) - S0 + Cc, color=cs.MAROON, label=f"Covered call: SPY minus a {Khi} call")
ax.plot(S, S - S0, color=cs.GREY, ls="--", label="SPY alone")
ax.axhline(0, color=cs.GREY, lw=0.8)
ax.set_xlabel("SPY at expiry (dollars)"); ax.set_ylabel("Profit at expiry (dollars a share)")
ax.legend(); cs.title(ax, "SPY with a floor, and SPY with a ceiling", 2)
ax = axs[2]
rank = np.arange(1, len(m) + 1)
for col, c_, name in (("spy", cs.GREY, "SPY alone"), ("put", cs.BLUE, "Protective put, 5% below"), ("call", cs.MAROON, "Covered call, 5% above")):
    ax.plot(rank, 100 * np.sort(m[col].to_numpy()), marker="o", ms=3, lw=0.8, color=c_, label=name)
ax.axhline(0, color=cs.GREY, lw=0.8)
ax.set_xlabel(f"Month, sorted from worst to best (of {len(m)})"); ax.set_ylabel("Return over the month (%)")
ax.legend(); cs.title(ax, "Rolled every month on SPY, 2020 to 2025, sorted", 3)
cs.save(fig, OUT / "options.svg")
