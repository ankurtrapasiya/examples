# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "matplotlib"]
# ///
"""Numbers and chart for the post "Why prices are hard to beat".

Random-walk checks on SPY (autocorrelation, up-after-up, streak lengths against a shuffled
series), a pair-trade z-score on MS against OXY, and a doubling-rule simulation on an American
roulette wheel with a bankroll sweep.  Run:  uv run examples/why-prices-are-hard-to-beat/efficiency.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chartstyle as cs  # noqa: E402
import market_data as md  # noqa: E402

OUT = cs.out_dir("why-prices-are-hard-to-beat")


def p(label, value):
    print(f"{label:<66} {value}")


px = md.prices()
r = md.daily_returns(px)
spy = r["SPY"]

print("\n== is SPY a random walk? ==")
p("days", f"{len(spy)}")
for lag in (1, 2, 5):
    p(f"autocorrelation of daily returns at lag {lag}", f"{spy.autocorr(lag):+.3f}")
up = spy > 0
p("share of up days", f"{up.mean():.1%}")
p("P(up | yesterday up) / P(up | yesterday down)", f"{up[up.shift(1) == True].mean():.1%} / {up[up.shift(1) == False].mean():.1%}")


def longest_run(s):
    best = cur = 0; prev = None
    for v in s:
        cur = cur + 1 if v == prev else 1; prev = v; best = max(best, cur)
    return best


rng = np.random.default_rng(0)
shuffled = [longest_run(rng.permutation(up.values)) for _ in range(1000)]
p("longest run of same-direction days, real SPY", f"{longest_run(up.values)}")
p("longest run in 1,000 shuffles of the same days: median / 95th pct", f"{np.median(shuffled):.0f} / {np.percentile(shuffled, 95):.0f}")
# streak-following rule vs reversal rule
pos_follow = up.astype(float).shift(1).fillna(0)
pos_fade = (~up).astype(float).shift(1).fillna(0)
p("hold after an up day, no costs: total / buy and hold", f"{np.prod(1 + pos_follow * spy) - 1:+.1%} / {np.prod(1 + spy) - 1:+.1%}")
p("hold after a down day, no costs: total", f"{np.prod(1 + pos_fade * spy) - 1:+.1%}")

print("\n== a pair: MS against OXY ==")
ratio = np.log(px["MS"] / px["OXY"])
z = (ratio - ratio.rolling(60).mean()) / ratio.rolling(60).std()
p("log price ratio, 60-day z-score: min / max", f"{z.min():.1f} / {z.max():.1f}")
p("days beyond |z| > 2", f"{(z.abs() > 2).sum()} of {z.notna().sum()}")
# after |z|>2, did the ratio move back toward the mean over the next 20 days?
sig = z[(z.abs() > 2)]
back = []
for d in sig.index:
    i = z.index.get_loc(d)
    if i + 20 < len(z):
        back.append(abs(z.iloc[i + 20]) < abs(z.iloc[i]))
p("after |z| > 2, share of cases closer to the mean 20 days later", f"{np.mean(back):.0%} of {len(back)}")
corr60 = r["MS"].rolling(60).corr(r["OXY"])
p("rolling 60-day correlation MS-OXY: min / max", f"{corr60.min():.2f} / {corr60.max():.2f}")

print("\n== the doubling rule on American roulette ==")
P_WIN, TARGET, SPINS = 18 / 38, 40, 2000


def play(rng, bankroll=None):
    w = np.zeros(SPINS + 1); bet, total = 1, 0
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


rng = np.random.default_rng(7)
p("P(win a spin) / expected value per dollar", f"{P_WIN:.4f} / {P_WIN - (1 - P_WIN):+.4f}")
for k in (6, 8, 10):
    p(f"P({k} losses in a row) / already lost / next bet", f"{(1 - P_WIN) ** k:.2%} / {2 ** k - 1} / {2 ** k}")
ten = np.array([play(rng) for _ in range(10)])
sweep = []
for bk in [63, 127, 255, 511, 1023, 2047, None]:
    plays = np.array([play(rng, bk) for _ in range(2000)]); f = plays[:, -1]
    hit = (f >= TARGET).mean(); bust = (bk is not None) and (f <= -bk).mean() or 0.0
    sweep.append((bk, hit, bust, f.mean(), np.median(f)))
    p(f"bankroll {str(bk) if bk else 'unlimited':>9}: reach +40 / lose all / mean / median", f"{hit:.1%} / {bust:.1%} / {f.mean():+.1f} / {np.median(f):+.0f}")
p("unlimited: median spins to reach +40", f"{np.median([(row != row[-1]).sum() for row in plays]):.0f}")

# ------------------------------------------------------------------ chart ----
cs.apply()
fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(8, 3.8 * 3))

cs.title(a1, "SPY's daily return against the previous day's: a small negative slope", 1)
a1.scatter(100 * spy.shift(1), 100 * spy, s=5, color=cs.GREY, alpha=0.5, rasterized=True, label="one trading day")
a1.axhline(0, color=cs.GREY, lw=0.8); a1.axvline(0, color=cs.GREY, lw=0.8)
b = spy.autocorr(1)
xs = np.linspace(-8, 8, 10)
a1.plot(xs, b * xs, color=cs.MAROON, label=f"fitted line, slope {b:+.3f}")
a1.set_xlabel("Yesterday's return (%)")
a1.set_ylabel("Today's return (%)")
a1.legend()

cs.title(a2, "A pair's z-score: how far MS has drifted from OXY, in 60-day standard deviations", 2)
a2.plot(z.index, z, color=cs.MAROON, lw=1, label="z-score of the MS/OXY price ratio")
for lvl, lab in [(2, "plus or minus 2: the usual entry signal"), (-2, "_")]:
    a2.axhline(lvl, color=cs.BLUE, lw=1, ls="--", label=lab)
a2.axhline(0, color=cs.GREY, lw=0.8)
a2.set_xlabel("Date")
a2.set_ylabel("z-score")
a2.legend()

cs.title(a3, "The doubling rule on American roulette: hit rate rises with the bankroll, the mean does not", 3)
xs = [bk if bk else 4095 for bk, *_ in sweep]
a3.plot(xs, [100 * h for _, h, *_ in sweep], color=cs.MAROON, marker="o", ms=4, label="plays that reach +40 (%)")
a3.plot(xs, [m for *_, m, _ in sweep], color=cs.BLUE, marker="o", ms=4, label="mean final winnings (dollars)")
a3.axhline(0, color=cs.GREY, lw=0.8)
a3.set_xscale("log", base=2); a3.set_xticks(xs); a3.set_xticklabels([str(bk) if bk else "no limit" for bk, *_ in sweep])
a3.set_xlabel("Bankroll (dollars, log scale)")
a3.set_ylabel("Percent of plays, or dollars")
a3.legend()
cs.save(fig, OUT / "efficiency.svg")
