# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "matplotlib"]
# ///
"""Numbers and chart for the post "The fundamental law: skill times the root of independent bets".

Three pieces of arithmetic, nothing learned:
  1. Romero and Balch's biased coin (chapter 9): a 51% coin bet once for $1,000 or 1,000 times
     for $1. Same expected profit, risk shrinks by the square root of the number of bets.
  2. Grinold's law, IR = IC x sqrt(breadth), on a few worked pairs of skill and breadth.
  3. How correlated the blog's five tickers are, as a reminder that five positions are fewer
     than five independent bets.
Run:  uv run examples/fundamental-law-skill-times-breadth/fundamental_law.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chartstyle as cs  # noqa: E402
import market_data as md  # noqa: E402

OUT = cs.out_dir("fundamental-law-skill-times-breadth")
P_HEADS, STAKE = 0.51, 1000.0


def p(label, value):
    print(f"{label:<62} {value}")


def coin(n_bets, p_heads=P_HEADS, stake=STAKE):
    """Expected profit and standard deviation of the total when `stake` is split over n equal
    even-money bets on a coin that pays +bet with probability p_heads, -bet otherwise."""
    bet = stake / n_bets
    mean_one = bet * (p_heads - (1 - p_heads))             # +bet * p  + (-bet) * (1 - p)
    var_one = bet**2 - mean_one**2                         # E[x^2] - E[x]^2, x^2 is always bet^2
    return n_bets * mean_one, np.sqrt(n_bets * var_one)


print("1. A 51% coin, $1,000 to bet (Romero and Balch, chapter 9)")
p("  expected profit per $1 bet, 0.51 x 1 + 0.49 x (-1)", f"${2 * P_HEADS - 1:.2f}")
for n in (1, 1000):
    m, s = coin(n)
    p(f"  {n:>4} bet(s) of ${STAKE / n:,.0f}: expected profit, std dev, ratio", f"${m:.2f}, ${s:.2f}, {m / s:.4f}")
p("  chance of losing the whole $1,000 with one bet", f"{1 - P_HEADS:.0%}")
p("  chance of losing the whole $1,000 with 1,000 bets, 0.49^1000", f"{(1 - P_HEADS) ** 1000:.1e}")
p("  ratio of the two reward-to-risk ratios", f"{coin(1000)[0] / coin(1000)[1] / (coin(1)[0] / coin(1)[1]):.2f} = sqrt(1000) = {np.sqrt(1000):.2f}")

print("\n2. The law: IR = IC x sqrt(BR)")
for ic, br in ((0.05, 100), (0.05, 400), (0.10, 25), (0.15, 10), (0.03, 1000)):
    p(f"  IC {ic:.2f}, breadth {br:>4}", f"IR = {ic * np.sqrt(br):.2f}")
p("  double the breadth, IR multiplied by sqrt(2)", f"{np.sqrt(2):.3f}")
p("  quadruple the breadth, IR multiplied by sqrt(4)", f"{np.sqrt(4):.3f}")

print("\n3. The five tickers, 2019 to 2023: correlation of daily returns")
r = md.daily_returns(md.prices())
c = r.corr()
print(c.round(2).to_string())
off = c.values[np.triu_indices(len(c), k=1)]
p("  average pairwise correlation", f"{off.mean():.2f}")
p("  highest pair", f"{c.where(np.triu(np.ones(c.shape, bool), 1)).stack().idxmax()} {off.max():.2f}")
p("  lowest pair", f"{c.where(np.triu(np.ones(c.shape, bool), 1)).stack().idxmin()} {off.min():.2f}")

# --- chart ----------------------------------------------------------------------------------
cs.apply()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 3.8 * 2))

br = np.arange(1, 1001)
for ic, colour in ((0.10, cs.MAROON), (0.05, cs.BLUE), (0.03, cs.GREY)):
    ax1.plot(br, ic * np.sqrt(br), color=colour, label=f"skill (information coefficient) {ic:.2f}")
ax1.set_xlabel("Breadth (independent bets a year)")
ax1.set_ylabel("Information ratio the law predicts")
cs.title(ax1, "Skill scales the result; breadth only its square root", 1)
ax1.legend()

ns = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])
means, stds = zip(*(coin(n) for n in ns))
ax2.plot(ns, stds, color=cs.MAROON, marker="o", label="standard deviation of the total profit")
ax2.plot(ns, means, color=cs.BLUE, marker="o", label="expected total profit")
ax2.set_xscale("log")
ax2.set_xticks(ns)
ax2.set_xticklabels([str(n) for n in ns])
ax2.set_xlabel("Number of equal bets the $1,000 is split into")
ax2.set_ylabel("Dollars")
cs.title(ax2, "A 51% coin: the same $20 expected, with risk shrinking by the square root", 2)
ax2.legend()

cs.save(fig, OUT / "fundamental_law.svg")
print(f"\nchart -> {OUT / 'fundamental_law.svg'}")
