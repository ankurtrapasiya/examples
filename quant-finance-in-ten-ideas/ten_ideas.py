# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "matplotlib", "scipy"]
# ///
"""Numbers and charts for the post "Quantitative finance in ten ideas".

A doubling-rule betting simulation on a European roulette wheel, and a Sharpe-maximising optimiser
(SciPy SLSQP) run on the blog's shared prices (SPY, NVDA, MS, OXY, GLD, 2020-2025).
Run:  uv run examples/quant-finance-in-ten-ideas/ten_ideas.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chartstyle as cs  # noqa: E402
import market_data as md  # noqa: E402

OUT = cs.out_dir("quant-finance-in-ten-ideas")
K = np.sqrt(252)


def p(label, value):
    print(f"{label:<66} {value}")


# ------------------------------------------------------------ the doubling rule ----
# European roulette: 18 of 37 pockets are red and a red bet pays even money, so the house keeps
# 2.7% of every dollar. The doubling rule: bet 1, double after every loss, back to 1 after a
# win, stop at +50.
P_WIN, TARGET, SPINS = 18 / 37, 50, 2000


def play(rng, bankroll=None):
    """One play: winnings after each round, frozen once the target is hit or the bankroll is
    gone. bankroll=None means unlimited."""
    w = np.zeros(SPINS + 1)
    bet, total = 1, 0
    for s in range(1, SPINS + 1):
        if total >= TARGET:
            w[s:] = total; break
        if bankroll is not None:
            if bankroll + total <= 0:
                w[s:] = -bankroll; break
            bet = min(bet, bankroll + total)
        if rng.random() < P_WIN:
            total += bet; bet = 1
        else:
            total -= bet; bet *= 2
        w[s] = total
    return w


rng = np.random.default_rng(42)
print("\n== the doubling rule ==")
p("P(win a spin)", f"{P_WIN:.4f}; expected value per dollar bet {P_WIN - (1 - P_WIN):+.4f}")
for k in (5, 7, 9):
    p(f"P({k} losses in a row)", f"{(1 - P_WIN) ** k:.3%}; after them: lost {2**k - 1}, next bet {2**k}")
ten = np.array([play(rng) for _ in range(10)])
bankrolls = [31, 63, 127, 255, 511, 1023, None]
sweep = []
for bk in bankrolls:
    plays = np.array([play(rng, bk) for _ in range(2000)])
    f = plays[:, -1]
    hit = (f >= TARGET).mean(); bust = (bk is not None) and (f <= -bk).mean() or 0.0
    sweep.append((bk, hit, bust, f.mean(), np.median(f), f.std()))
    p(f"bankroll {str(bk) if bk else 'unlimited':>9}: hit +50 / lose all / mean / median", f"{hit:.1%} / {bust:.1%} / {f.mean():+.1f} / {np.median(f):+.0f}")
unl = plays
p("unlimited: median spins to reach +50", f"{np.median([(row != row[-1]).sum() for row in unl]):.0f}")
p("unlimited: worst dip seen across 2,000 plays", f"{unl.min():+.0f}")

# ---------------------------------------------------------------- optimiser ----
px = md.prices()
r = md.daily_returns(px)


def sharpe_of(v):
    rr = np.diff(v) / v[:-1]
    return K * rr.mean() / rr.std(ddof=1)


def optimise(prices):
    norm = (prices / prices.iloc[0]).to_numpy()
    n = norm.shape[1]
    res = minimize(lambda w: -sharpe_of(norm @ w), np.full(n, 1 / n), method="SLSQP",
                   bounds=[(0, 1)] * n, constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
    w = np.clip(res.x, 0, 1); return w / w.sum()


def perf(prices, w):
    norm = (prices / prices.iloc[0]).to_numpy()
    v = norm @ w
    return sharpe_of(v), v[-1] / v[0] - 1


print("\n== optimiser: chosen on one period, run on the next ==")
windows = [("2020", "2021"), ("2021", "2022"), ("2022", "2023"), ("2023", "2024"), ("2024", "2025"), ("2020-2023", "2024-2025")]
rows = []
for a, b in windows:
    pa = px.loc[a.split("-")[0]:a.split("-")[-1]]
    pb = px.loc[b.split("-")[0]:b.split("-")[-1]]
    w = optimise(pa)
    eq = np.full(5, 0.2)
    s_in, _ = perf(pa, w); s_out, r_out = perf(pb, w); s_eq, r_eq = perf(pb, eq)
    singles = {t: sharpe_of((pa[t] / pa[t].iloc[0]).to_numpy()) for t in md.TICKERS}
    rows.append((a, b, w, s_in, s_out, r_out, s_eq, r_eq))
    p(f"chosen on {a}: weights", " ".join(f"{t}:{x:.0%}" for t, x in zip(md.TICKERS, w)))
    p(f"   singles' Sharpe on {a}", " ".join(f"{t}:{s:.2f}" for t, s in singles.items()))
    p(f"   Sharpe in-sample {a} / out-of-sample {b} / equal weights {b}", f"{s_in:.2f} / {s_out:.2f} / {s_eq:.2f}")
    p(f"   return on {b}: optimised / equal weights", f"{r_out:+.1%} / {r_eq:+.1%}")

print("\n== diversification: four-asset mix vs its members ==")
for yr in ["2020", "2022"]:
    sub = r.loc[yr, ["SPY", "NVDA", "MS", "OXY"]]
    mix = sub.mean(axis=1)
    p(f"{yr}: daily sigma of each", " ".join(f"{t}:{sub[t].std():.2%}" for t in sub))
    p(f"{yr}: equal-weight mix sigma, average pairwise correlation", f"{mix.std():.2%}, {sub.corr().values[np.triu_indices(4, 1)].mean():.2f}")

# ------------------------------------------------------------------- charts ----
cs.apply()
fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 3.8 * 2))
rounds = np.arange(SPINS + 1)
cs.title(a1, "Ten plays of the doubling rule with no bankroll limit", 1)
for i in range(10):
    a1.plot(rounds, ten[i], color=cs.MAROON if i == 0 else cs.GREY, lw=1.2 if i == 0 else 0.9,
            alpha=1 if i == 0 else 0.6, label="one play (winnings after each spin)" if i == 0 else "_x")
a1.axhline(TARGET, color=cs.BLUE, lw=1, ls="--", label="the +50 target")
a1.set_xlim(0, 250); a1.set_ylim(-520, 80); a1.set_xlabel("Spin number"); a1.set_ylabel("Winnings so far (dollars)"); a1.legend()
cs.title(a2, "A bigger bankroll wins more often, and the average play still loses", 2)
xs = [bk if bk else 4095 for bk, *_ in sweep]
a2.plot(xs, [100 * h for _, h, *_ in sweep], color=cs.MAROON, marker="o", ms=4, label="plays that reach +50 (%)")
a2.plot(xs, [m for *_, m, _, _ in sweep], color=cs.BLUE, marker="o", ms=4, label="mean final winnings (dollars)")
a2.axhline(0, color=cs.GREY, lw=0.8)
a2.set_xscale("log", base=2); a2.set_xticks(xs); a2.set_xticklabels([str(bk) if bk else "no limit" for bk, *_ in sweep])
a2.set_xlabel("Bankroll (dollars, log scale)"); a2.set_ylabel("Percent of plays, or dollars"); a2.legend()
cs.save(fig, OUT / "doubling-rule.svg")

fig, ax = plt.subplots(1, 1, figsize=(8, 3.8))
cs.title(ax, "Weights chosen on 2020 to 2023, then run on 2024 to 2025, against equal weights")
a, b, w, *_ = rows[-1]
pb = px.loc["2024":"2025"]
norm = (pb / pb.iloc[0]).to_numpy()
ax.plot(pb.index, norm @ w, color=cs.MAROON, label="Sharpe-maximising weights from 2020 to 2023")
ax.plot(pb.index, norm @ np.full(5, 0.2), color=cs.BLUE, label="equal weights, 20% each")
ax.plot(pb.index, (pb["SPY"] / pb["SPY"].iloc[0]).to_numpy(), color=cs.GREY, label="SPY")
ax.set_xlabel("Date"); ax.set_ylabel("Value of 1 dollar invested at the start of 2024"); ax.legend()
cs.save(fig, OUT / "optimised-vs-equal.svg")
