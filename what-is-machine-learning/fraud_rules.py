# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scikit-learn", "matplotlib"]
# ///
"""Charts and numbers for the post "What is machine learning?".

Synthetic card transactions (made up, no real data) with two kinds of fraud:
large purchases, and tiny "card testing" charges at night. A hand-written rule
is compared with rules a decision tree learns from the same data.

Run:  uv run examples/what-is-machine-learning/fraud_rules.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chartstyle as cs  # noqa: E402

OUT = cs.out_dir("what-is-machine-learning")
MAROON, GREY, INK = cs.MAROON, cs.GREY, cs.INK
rng = np.random.default_rng(7)

# --- synthetic data ---------------------------------------------------------
n_normal, n_big, n_test = 4000, 60, 60
normal = np.column_stack([rng.lognormal(4.0, 0.9, n_normal),           # ~$55 typical
                          rng.normal(15, 3.5, n_normal).clip(0, 23.9)])  # daytime
big = np.column_stack([rng.uniform(1500, 5000, n_big), rng.uniform(0, 23.9, n_big)])
test = np.column_stack([rng.uniform(0.5, 5, n_test), rng.uniform(0, 5, n_test)])  # small, at night
X = np.vstack([normal, big, test])
y = np.r_[np.zeros(n_normal), np.ones(n_big + n_test)]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

# --- 1. hand-written rule ---------------------------------------------------
rule_pred = (X_te[:, 0] > 1000).astype(int)

# --- 2. rules learned from data ---------------------------------------------
model = DecisionTreeClassifier(max_depth=3, random_state=0)
model.fit(X_tr, y_tr)
ml_pred = model.predict(X_te)

def report(name, pred):
    caught = int(((pred == 1) & (y_te == 1)).sum())
    frauds = int((y_te == 1).sum())
    false_alarms = int(((pred == 1) & (y_te == 0)).sum())
    print(f"{name:12s} caught {caught}/{frauds} frauds, {false_alarms} false alarms")

report("hand rule", rule_pred)
report("learned", ml_pred)
print()
print(export_text(model, feature_names=["amount", "hour"], decimals=1))

# --- chart -------------------------------------------------------------------
OUT.mkdir(parents=True, exist_ok=True)
cs.apply()
fig, axes = plt.subplots(2, 1, figsize=(8, 3.8 * 2))
for n, (ax, title, pred) in enumerate([(axes[0], "Hand-written rule: amount > $1,000", rule_pred),
                                       (axes[1], "Rules learned from data", ml_pred)], start=1):
    fraud = y_te == 1
    ax.scatter(X_te[~fraud, 0], X_te[~fraud, 1], s=6, c=GREY, alpha=0.5, label="normal purchase", rasterized=True)
    hit = fraud & (pred == 1)
    miss = fraud & (pred == 0)
    ax.scatter(X_te[hit, 0], X_te[hit, 1], s=22, c=MAROON, label="fraud, caught")
    if miss.any():   # a legend entry only for what this panel shows
        ax.scatter(X_te[miss, 0], X_te[miss, 1], s=30, facecolors="none", edgecolors=MAROON,
                   linewidths=1.4, label="fraud, missed")
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([1, 10, 100, 1000]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.set_xlim(0.3, 8000)
    ax.set_ylim(-1, 25)
    ax.set_xlabel("Purchase amount ($, log scale)")
    ax.set_ylabel("Hour of day (0–24)")
    ax.legend()
    cs.title(ax, title, n)
cs.save(fig, OUT / "rule-vs-learned.svg")
