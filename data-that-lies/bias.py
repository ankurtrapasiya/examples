# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "matplotlib"]
# ///
"""Numbers and chart for the post "Data that lies".

Data snooping (1,000 random signals on SPY), look-ahead (a missing shift), survivorship built
by construction, and the difference adjusted prices make.  Run:  uv run examples/data-that-lies/bias.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chartstyle as cs  # noqa: E402
import market_data as md  # noqa: E402

OUT = cs.out_dir("data-that-lies")
K = np.sqrt(252)


def p(label, value):
    print(f"{label:<66} {value}")


px = md.prices()
r = md.daily_returns(px)
tr, te = r.loc[:"2023"], r.loc["2024":]

print("\n== data snooping: 1,000 signals that know nothing ==")
rng = np.random.default_rng(1)
N = 1000
sig_tr = rng.choice([-1.0, 1.0], size=(N, len(tr)))
sig_te = rng.choice([-1.0, 1.0], size=(N, len(te)))
pnl_tr, pnl_te = sig_tr * tr["SPY"].to_numpy(), sig_te * te["SPY"].to_numpy()
sh_tr = K * pnl_tr.mean(1) / pnl_tr.std(1, ddof=1)
sh_te = K * pnl_te.mean(1) / pnl_te.std(1, ddof=1)
t_tr = pnl_tr.mean(1) / pnl_tr.std(1, ddof=1) * np.sqrt(len(tr))
p("signals with t > 2 on 2020-2023 (expected by luck: ~2.3%)", f"{(t_tr > 2).sum()}")
p("their mean Sharpe in-sample / out-of-sample", f"{sh_tr[t_tr > 2].mean():.2f} / {sh_te[t_tr > 2].mean():+.2f}")
best = sh_tr.argmax()
p("the best of 1,000: Sharpe in-sample / out-of-sample", f"{sh_tr[best]:.2f} / {sh_te[best]:+.2f}")
p("correlation of in-sample and out-of-sample Sharpe across the 1,000", f"{np.corrcoef(sh_tr, sh_te)[0, 1]:+.3f}")
p("Bonferroni: a single test at 5% becomes", f"{0.05 / N:.5f}; t needed about {3.9:.1f}")

print("\n== look-ahead: one missing shift ==")
fast, slow = px["SPY"].rolling(20).mean(), px["SPY"].rolling(100).mean()
in_mkt = (fast > slow).astype(float).reindex(r.index).fillna(0)
cheat = (1 + in_mkt * r["SPY"]).prod() - 1
honest = (1 + in_mkt.shift(1).fillna(0) * r["SPY"]).prod() - 1
p("20/100 crossover on SPY: position from today's close (cheat) / from yesterday's", f"{cheat:+.1%} / {honest:+.1%}")
# the leak in a model: z-score with the full-sample mean
vol = r["SPY"].rolling(20).std()
z_full = (vol - vol.mean()) / vol.std()
z_roll = (vol - vol.expanding().mean()) / vol.expanding().std()
p("a 'high volatility' flag (z > 1): using the full-sample mean / only past data, days flagged", f"{(z_full > 1).sum()} / {(z_roll > 1).sum()}")
p("   first day the two flags disagree", f"{(z_full > 1).ne(z_roll > 1).idxmax().date()}")

print("\n== survivorship, built by construction ==")
# 200 made-up stocks: random walks with a common drift and spread; the universe is formed at the END
rng = np.random.default_rng(3)
n_stocks, days = 200, 252 * 3
daily = rng.normal(0.0003, 0.02, size=(n_stocks, days)) + rng.normal(0, 0.0006, size=(n_stocks, 1))
paths = np.cumprod(1 + daily, axis=1)
final = paths[:, -1]
alive = final > 0.5                      # "delisted" if it lost more than half
p("stocks / still in the index at the end (not down more than 50%)", f"{n_stocks} / {alive.sum()}")
p("average yearly return, all 200 / survivors only", f"{(final.mean() ** (1 / 3) - 1):+.1%} / {(final[alive].mean() ** (1 / 3) - 1):+.1%}")
# a 'buy the cheapest' rule: buy the 20 worst after year 1, hold two years
y1 = paths[:, 251]
cheapest = np.argsort(y1)[:20]
ret_all = (paths[cheapest, -1] / y1[cheapest]).mean() - 1
surv_cheapest = [i for i in np.argsort(y1) if alive[i]][:20]
ret_surv = (paths[surv_cheapest, -1] / y1[surv_cheapest]).mean() - 1
p("buy the 20 cheapest after year 1, hold 2 years: full universe / survivors only", f"{ret_all:+.1%} / {ret_surv:+.1%}")

print("\n== adjusted prices ==")
p("$50 to $48 with a $2 dividend: price return / total return", f"{48 / 50 - 1:+.1%} / {(48 + 2) / 50 - 1:+.1%}")
p("2-for-1 split, unadjusted: $100 to $50 looks like", f"{50 / 100 - 1:+.0%}")
p("last print $4.00, holders receive $0.50 in liquidation", f"{0.5 / 4 - 1:+.1%}")

# ------------------------------------------------------------------ chart ----
cs.apply()
fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(8, 3.8 * 3))

cs.title(a1, "1,000 coin-flip signals: Sharpe where they were chosen against Sharpe afterwards", 1)
a1.scatter(sh_tr, sh_te, s=6, color=cs.GREY, alpha=0.5, rasterized=True, label="one random signal")
a1.scatter(sh_tr[t_tr > 2], sh_te[t_tr > 2], s=14, color=cs.MAROON, label='the ones that looked "significant" on 2020 to 2023')
a1.axhline(0, color=cs.GREY, lw=0.8); a1.axvline(0, color=cs.GREY, lw=0.8)
a1.set_xlabel("Sharpe on 2020 to 2023 (where the signal was picked)")
a1.set_ylabel("Sharpe on 2024 to 2025 (unseen)")
a1.legend()

cs.title(a2, "Survivorship by construction: 200 made-up stocks, the index formed at the end", 2)
t = np.arange(days) / 252
for i in range(n_stocks):
    a2.plot(t, paths[i], color=cs.GREY if alive[i] else cs.MAROON, lw=0.4, alpha=0.5 if alive[i] else 0.8,
            label=("survived: in today's index" if i == np.where(alive)[0][0] else ("dropped: not in today's index" if i == np.where(~alive)[0][0] else "_x")))
a2.axhline(0.5, color=cs.BLUE, lw=1, ls="--", label="the cut: dropped below half its start")
a2.set_yscale("log")
a2.set_xlabel("Years")
a2.set_ylabel("Value of 1 dollar (log scale)")
a2.legend()

cs.title(a3, "A volatility flag built with the full-sample mean fires before it could have", 3)
a3.plot(vol.index, md.ANNUALISE * vol, color=cs.GREY, lw=1, label="SPY 20-day volatility (% a year)")
a3.plot(vol.index, md.ANNUALISE * vol.where(z_full > 1), color=cs.MAROON, lw=1.4, label="flagged using the full 2020 to 2025 mean (leaks)")
a3.plot(vol.index, md.ANNUALISE * vol.where(z_roll > 1), color=cs.BLUE, lw=1.4, ls=":", label="flagged using only data known so far")
a3.set_xlabel("Date")
a3.set_ylabel("Volatility (% a year)")
a3.legend()
cs.save(fig, OUT / "bias.svg")
