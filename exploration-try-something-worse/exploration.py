# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "matplotlib"]
# ///
"""Numbers and chart for the post "Exploration: when to try something worse on purpose".

A ten-armed bandit in the shape of Sutton and Barto's testbed (section 2.3): 2,000 tasks, each
with ten actions whose true values are drawn from a standard normal, rewards drawn around the
true value with unit variance, 1,000 steps per task, sample-average estimates. Three ways of
choosing: greedy, epsilon-greedy with epsilon 0.1, and with epsilon 0.01. Then the arithmetic of
epsilon-greedy probabilities and of a geometric epsilon decay. Synthetic data on purpose: a bandit
has one state, so nothing here is a trading agent or a Q-learner.
Run:  uv run examples/exploration-try-something-worse/exploration.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chartstyle as cs  # noqa: E402

OUT = cs.out_dir("exploration-try-something-worse")
RUNS, STEPS, ARMS = 2000, 1000, 10
EPSILONS = (0.0, 0.01, 0.1)


def p(label, value):
    print(f"{label:<66} {value}")


def testbed(eps, runs=RUNS, steps=STEPS, arms=ARMS, seed=0):
    """Average reward and share of runs taking the best action, at each step, over `runs` tasks."""
    rng = np.random.default_rng(seed)
    q_true = rng.normal(size=(runs, arms))              # each task's true action values
    best = q_true.argmax(axis=1)
    Q = np.zeros((runs, arms))                          # estimates, start at 0
    N = np.zeros((runs, arms))                          # how often each action was taken
    reward, optimal = np.zeros(steps), np.zeros(steps)
    rows = np.arange(runs)
    for t in range(steps):
        greedy = (Q + rng.random((runs, arms)) * 1e-9).argmax(axis=1)   # ties broken at random
        explore = rng.random(runs) < eps
        a = np.where(explore, rng.integers(arms, size=runs), greedy)
        r = rng.normal(q_true[rows, a])                 # reward: true value plus unit noise
        N[rows, a] += 1
        Q[rows, a] += (r - Q[rows, a]) / N[rows, a]     # sample average, incrementally (eq. 2.3)
        reward[t], optimal[t] = r.mean(), (a == best).mean()
    return reward, optimal


print(f"Ten-armed testbed, {RUNS} tasks x {STEPS} steps, sample-average estimates")
results = {eps: testbed(eps) for eps in EPSILONS}
rng0 = np.random.default_rng(0)
p("Best possible reward per step, averaged over the tasks", f"{rng0.normal(size=(RUNS, ARMS)).max(axis=1).mean():.2f}")
for eps, (rew, opt) in results.items():
    name = "greedy" if eps == 0 else f"epsilon {eps}"
    p(f"{name}: reward per step, last 100 steps", f"{rew[-100:].mean():.2f}")
    p(f"{name}: share of tasks taking the best action at step 1000", f"{100 * opt[-1]:.0f}%")
    p(f"{name}: share taking the best action at step 100", f"{100 * opt[99]:.0f}%")

print()
print("Probability epsilon-greedy picks the greedy action, 1 - eps + eps / |A|")
for eps, n in ((0.1, 10), (0.2, 3), (0.1, 4)):
    p(f"  epsilon {eps}, {n} actions: greedy", f"{1 - eps + eps / n:.3f}   each other action {eps / n:.3f}")

print()
print("Geometric decay, epsilon_n = epsilon_0 x d^n from epsilon_0 = 1")
for d in (0.9, 0.99, 0.999):
    p(f"  d = {d}: after 10, 50, 100, 500, 1000 steps",
      "  ".join(f"{d**n:.3f}" for n in (10, 50, 100, 500, 1000)))
p("  steps for d = 0.99 to fall below 0.05", int(np.ceil(np.log(0.05) / np.log(0.99))))
p("  steps for d = 0.999 to fall below 0.05", int(np.ceil(np.log(0.05) / np.log(0.999))))

# --- chart ----------------------------------------------------------------------------------
cs.apply()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 3.8 * 2))
steps = np.arange(1, STEPS + 1)
colours = {0.0: cs.GREY, 0.01: cs.BLUE, 0.1: cs.MAROON}
labels = {0.0: "greedy, never explores", 0.01: "random action 1% of the time", 0.1: "random action 10% of the time"}
for eps, (rew, opt) in results.items():
    ax1.plot(steps, 100 * opt, color=colours[eps], label=labels[eps])
ax1.set_ylim(0, 100)
ax1.set_xlabel("Step (one action chosen per step)")
ax1.set_ylabel("Tasks where the best action was taken (% of 2,000)")
cs.title(ax1, "How often each method picks the best action", 1)
ax1.legend()

n = np.arange(0, 1001)
for d, c in ((0.9, cs.GREY), (0.99, cs.BLUE), (0.999, cs.MAROON)):
    ax2.plot(n, d**n, color=c, label=f"epsilon shrinks by a factor {d} each step")
ax2.set_xlabel("Steps taken")
ax2.set_ylabel("Epsilon (share of steps that explore)")
cs.title(ax2, "Three speeds of shrinking the random share", 2)
ax2.legend()

cs.save(fig, OUT / "exploration.svg")
print(f"\nchart -> {OUT / 'exploration.svg'}")
