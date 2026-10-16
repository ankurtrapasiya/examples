"""Made-up OTC settlement data shared by the blog's ML posts.

Nothing here is real. The patterns are generic, widely known causes of settlement
fails (instruction mismatches, cross-border trades, new counterparties, late
booking), with numbers chosen to make the examples clear.
"""
import numpy as np

FEATURES = ["notional_musd", "ssi_mismatch", "cross_border", "new_counterparty", "hours_late"]


def trades(n=20_000, seed=42):
    """One row per trade. Returns (X, y_fail, feature_names).

    y_fail is 1 when the trade failed to settle on the agreed date.
    """
    rng = np.random.default_rng(seed)
    notional = rng.lognormal(np.log(5), 1.0, n)                 # $m, median about 5
    mismatch = rng.random(n) < 0.05                               # wrong settlement instructions
    cross = rng.random(n) < 0.20                                  # different countries / markets
    new_cp = rng.random(n) < 0.10                                 # counterparty we rarely trade with
    hours_late = np.where(rng.random(n) < 0.15, rng.exponential(3, n), 0.0)  # booked after cut-off

    # log-odds of a fail: each cause adds risk; cross-border with a new counterparty is worse
    z = (-4.2 + 3.2 * mismatch + 0.9 * cross + 0.8 * new_cp + 1.1 * (cross & new_cp)
         + 0.35 * np.minimum(hours_late, 8) + 0.15 * np.log(notional))
    p = 1 / (1 + np.exp(-z))
    y = (rng.random(n) < p).astype(int)
    X = np.column_stack([notional, mismatch, cross, new_cp, hours_late]).astype(float)
    return X, y, FEATURES


def daily(days=250, seed=7):
    """One row per business day. Returns (trades_k, cross_share, fails).

    fails = 20 + 6 per thousand trades + 150 per unit of cross-border share + noise.
    """
    rng = np.random.default_rng(seed)
    trades_k = rng.normal(12, 2.5, days).clip(5, None)        # thousands of trades booked
    cross_share = rng.uniform(0.10, 0.35, days)                 # share that is cross-border
    fails = 20 + 6 * trades_k + 150 * cross_share + rng.normal(0, 8, days)
    return trades_k, cross_share, fails


def delay_curve(x):
    """True average settlement delay (days) for a trade booked x hours after cut-off."""
    return 0.1 + 1.6 * (1 - np.exp(-x / 5)) + 0.25 * np.sin(x / 2)


def delays(n=80, seed=None, noise=0.35):
    """Small noisy sample from delay_curve, for the bias/variance and bagging posts."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 24, n)
    return x, delay_curve(x) + rng.normal(0, noise, n)
