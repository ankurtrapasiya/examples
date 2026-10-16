# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "matplotlib"]
# ///
"""Numbers and chart for the post "Value is an estimate".

Present value, the Gordon growth formula, a small discounted-cash-flow model and how its answer
swings with the discount rate and the growth guess. No market data needed.
Run:  uv run examples/value-is-an-estimate/valuation.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chartstyle as cs  # noqa: E402

OUT = cs.out_dir("value-is-an-estimate")


def p(label, value):
    print(f"{label:<66} {value}")


print("\n== present value ==")
for fv, r, n in [(121, 0.10, 2), (121, 0.12, 2), (10_000, 0.10, 1), (10_000, 0.10, 2), (1, 0.08, 50)]:
    p(f"{fv:,} due in {n} year(s) at {r:.0%}", f"{fv / (1 + r) ** n:,.2f}")
p("$1 a year for 10 years at 8%", f"{sum(1 / 1.08 ** k for k in range(1, 11)):.2f}")
p("$1 a year forever at 8% (perpetuity, 1/r)", f"{1 / 0.08:.2f}")

print("\n== Gordon growth ==")
for d1, r, g in [(3, 0.10, 0.05), (2, 0.10, 0.04), (2, 0.10, 0.05), (2, 0.10, 0.08), (2, 0.10, 0.095)]:
    p(f"D1 {d1}, r {r:.0%}, g {g:.1%}", f"{d1 / (r - g):.2f}")

print("\n== a small DCF: 5 years of cash flows then a terminal value ==")


def dcf(cf0=100, growth=0.08, r=0.10, g_term=0.03, years=5):
    cfs = [cf0 * (1 + growth) ** k for k in range(1, years + 1)]
    pv_cfs = sum(cf / (1 + r) ** k for k, cf in enumerate(cfs, 1))
    terminal = cfs[-1] * (1 + g_term) / (r - g_term)
    pv_term = terminal / (1 + r) ** years
    return pv_cfs, pv_term


a, b = dcf()
p("base case: PV of 5 years' cash / PV of terminal value / total", f"{a:.0f} / {b:.0f} / {a + b:.0f}; terminal share {b / (a + b):.0%}")
grid_r = np.arange(0.07, 0.1401, 0.01)
grid_g = np.arange(0.00, 0.0501, 0.01)
for r in (0.08, 0.10, 0.12):
    row = [sum(dcf(r=r, g_term=g)) for g in grid_g]
    p(f"total value at r {r:.0%}, long-run growth 0..5%", " ".join(f"{v:,.0f}" for v in row))
p("margin of safety: value 62.50, price 50", f"{(62.5 - 50) / 62.5:.0%} (divide by value)")
p("need 25% margin on a value of 80: buy at or below", f"{80 * 0.75:.0f}")
p("book value: assets 500, liabilities 320, 30 shares, price 9", f"equity {500 - 320}, {(500 - 320) / 30:.0f} a share, P/B {9 / ((500 - 320) / 30):.1f}")
p("scenarios 25% x 40, 50% x 60, 25% x 90", f"expected {0.25 * 40 + 0.5 * 60 + 0.25 * 90:.2f}")

# ------------------------------------------------------------------ chart ----
cs.apply()
fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(8, 3.8 * 3))

cs.title(a1, "What 100 dollars in each future year is worth today, at three discount rates", 1)
yrs = np.arange(0, 31)
for r, col in [(0.05, cs.GREEN), (0.10, cs.MAROON), (0.15, cs.BLUE)]:
    a1.plot(yrs, 100 / (1 + r) ** yrs, color=col, marker="o", ms=2.5, label=f"discount rate {r:.0%}")
a1.set_xlabel("Years from now")
a1.set_ylabel("Value today of 100 dollars (dollars)")
a1.legend()

cs.title(a2, "Gordon growth: the price of a 2 dollar dividend explodes as growth approaches the discount rate", 2)
gs = np.linspace(0, 0.095, 200)
a2.plot(100 * gs, 2 / (0.10 - gs), color=cs.MAROON, label="price = 2 / (10% − growth)")
for g, lab in [(0.04, "4%: 33.33"), (0.05, "5%: 40.00"), (0.08, "8%: 100.00")]:
    a2.scatter([100 * g], [2 / (0.10 - g)], color=cs.BLUE, zorder=3, label=f"growth {lab}" if g == 0.04 else f"_g{g}")
a2.set_ylim(0, 200)
a2.set_xlabel("Long-run dividend growth (% a year), discount rate fixed at 10%")
a2.set_ylabel("Price (dollars)")
a2.legend()

cs.title(a3, "A five-year DCF: the answer against the two guesses that drive it", 3)
for g, col in [(0.00, cs.GREY), (0.02, cs.BLUE), (0.03, cs.MAROON), (0.05, cs.AMBER)]:
    a3.plot(100 * grid_r, [sum(dcf(r=r, g_term=g)) for r in grid_r], color=col, marker="o", ms=3, label=f"long-run growth {g:.0%}")
a3.set_xlabel("Discount rate (% a year)")
a3.set_ylabel("Value today (dollars; the base case is 1,816)")
a3.legend()
cs.save(fig, OUT / "valuation.svg")
