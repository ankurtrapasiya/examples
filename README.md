# Blog examples

The scripts behind every number and chart on [ankurtrapasiya.com](https://ankurtrapasiya.com):
the Machine Learning, Quantitative Finance and Engineering series. Each script is one post's
working: run it and it prints the post's tables and redraws its charts.

The data is [`data/prices.csv`](data/prices.csv): daily adjusted closes for SPY, NVDA, GS, XOM
and GLD, 2020 to October 2025, downloaded once so the numbers never move. Code snippets in the posts read
it straight from this repo:

```python
import pandas as pd
URL = "https://raw.githubusercontent.com/ankurtrapasiya/examples/main/data/prices.csv"
prices = pd.read_csv(URL, index_col="date", parse_dates=True)
```

## Quick run, with Docker

Three commands, nothing else to install:

```bash
git clone https://github.com/ankurtrapasiya/examples && cd examples
docker build -t blog-examples .
docker run --rm -v "$PWD/charts:/examples/charts" blog-examples scoring-a-model
```

The first build takes a few minutes (it installs every library once); after that each run starts
in seconds. Charts land in `charts/<post>/` on your machine.

| Command | Does |
|---|---|
| `docker run --rm blog-examples` | lists every script |
| `docker run --rm -v "$PWD/charts:/examples/charts" blog-examples bagging` | runs the scripts in one folder |
| `docker run --rm -v "$PWD/charts:/examples/charts" blog-examples bagging/bagging.py` | runs one script |
| `docker run --rm -v "$PWD/charts:/examples/charts" blog-examples all` | runs everything and reports any failure |

## Without Docker

With [uv](https://docs.astral.sh/uv/), each script declares its own libraries at the top, so one
command is enough:

```bash
uv run scoring-a-model/scoring.py
```

Or with plain Python 3.12: `pip install -r requirements.txt`, then `python run.py <folder>`.

## Scripts by post

| Post | Script |
|---|---|
| [Backtesting in Python](https://ankurtrapasiya.com/posts/backtesting-in-python/) | [`backtesting/sma_crossover.py`](backtesting/sma_crossover.py) |
| [Bagging: many copies of a model, averaged](https://ankurtrapasiya.com/posts/bagging/) | [`bagging/bagging.py`](bagging/bagging.py) |
| [Beta and alpha: how much of the market you carry, and what is left](https://ankurtrapasiya.com/posts/beta-is-a-slope/) | [`beta-is-a-slope/beta.py`](beta-is-a-slope/beta.py) |
| [Bias and variance: where a model's errors come from](https://ankurtrapasiya.com/posts/bias-and-variance/) | [`bias-and-variance/bias_variance.py`](bias-and-variance/bias_variance.py) |
| [Boosting: models that learn from the last one's mistakes](https://ankurtrapasiya.com/posts/boosting/) | [`boosting/boosting.py`](boosting/boosting.py) |
| [Choosing a model is a systems decision](https://ankurtrapasiya.com/posts/choosing-a-model/) | [`choosing-a-model/choose.py`](choosing-a-model/choose.py) |
| [Covariance and correlation: why a mix is safer than its parts](https://ankurtrapasiya.com/posts/covariance-and-correlation/) | [`covariance-and-correlation/portfolio_variance.py`](covariance-and-correlation/portfolio_variance.py), [`covariance-and-correlation/rolling_corr.py`](covariance-and-correlation/rolling_corr.py) |
| [The data is part of the model: splits, gaps and outliers](https://ankurtrapasiya.com/posts/data-is-part-of-the-model/) | [`data-is-part-of-the-model/sampling.py`](data-is-part-of-the-model/sampling.py), [`data-is-part-of-the-model/two_tools.py`](data-is-part-of-the-model/two_tools.py) |
| [Data that lies: the bugs that look like alpha](https://ankurtrapasiya.com/posts/data-that-lies/) | [`data-that-lies/bias.py`](data-that-lies/bias.py) |
| [Decision trees: a model made of yes/no questions](https://ankurtrapasiya.com/posts/decision-trees/) | [`decision-trees/decision_tree.py`](decision-trees/decision_tree.py) |
| [Every feature has a timestamp: leakage, scaling and twin features](https://ankurtrapasiya.com/posts/every-feature-has-a-timestamp/) | [`every-feature-has-a-timestamp/leakage.py`](every-feature-has-a-timestamp/leakage.py) |
| [Expected value: what a risky choice is worth on average](https://ankurtrapasiya.com/posts/expected-value/) | [`expected-value/expected_value.py`](expected-value/expected_value.py) |
| [Linear regression: the first model, a straight line](https://ankurtrapasiya.com/posts/linear-regression/) | [`linear-regression/linear_regression.py`](linear-regression/linear_regression.py), [`linear-regression/tstat.py`](linear-regression/tstat.py) |
| [Machine learning for quantitative finance in ten ideas](https://ankurtrapasiya.com/posts/machine-learning-in-ten-ideas/) | [`machine-learning-in-ten-ideas/ten_ideas.py`](machine-learning-in-ten-ideas/ten_ideas.py) |
| [Mean and standard deviation: return and risk in two numbers](https://ankurtrapasiya.com/posts/mean-and-standard-deviation/) | [`mean-and-standard-deviation/mean_std.py`](mean-and-standard-deviation/mean_std.py) |
| [Nearest neighbours: predict what similar past cases did](https://ankurtrapasiya.com/posts/nearest-neighbours/) | [`nearest-neighbours/knn.py`](nearest-neighbours/knn.py) |
| [Portfolio optimisation: choosing the weights](https://ankurtrapasiya.com/posts/portfolio-optimisation/) | [`portfolio-optimisation/random_portfolios.py`](portfolio-optimisation/random_portfolios.py) |
| [Probability: how often something happens](https://ankurtrapasiya.com/posts/probability-basics/) | [`probability-basics/probability.py`](probability-basics/probability.py) |
| [Quantitative finance in ten ideas](https://ankurtrapasiya.com/posts/quant-finance-in-ten-ideas/) | [`quant-finance-in-ten-ideas/ten_ideas.py`](quant-finance-in-ten-ideas/ten_ideas.py) |
| [Random forests: bagging with one extra rule](https://ankurtrapasiya.com/posts/random-forests/) | [`random-forests/random_forest.py`](random-forests/random_forest.py) |
| [Regression vs classification: predicting a number or a label](https://ankurtrapasiya.com/posts/regression-vs-classification/) | [`regression-vs-classification/two_questions.py`](regression-vs-classification/two_questions.py) |
| [Returns compound: multiply, never add](https://ankurtrapasiya.com/posts/returns-compound/) | [`returns-compound/returns.py`](returns-compound/returns.py) |
| [Sharpe, Sortino and information ratio: return per unit of risk](https://ankurtrapasiya.com/posts/sharpe-sortino-information-ratio/) | [`risk-adjusted-ratios/drawdown.py`](risk-adjusted-ratios/drawdown.py), [`risk-adjusted-ratios/ratios.py`](risk-adjusted-ratios/ratios.py) |
| [Building models with scikit-learn](https://ankurtrapasiya.com/posts/building-models-with-scikit-learn/) | [`scikit-learn/stress_month.py`](scikit-learn/stress_month.py) |
| [Scoring a model: a number means nothing without a split, a metric and a baseline](https://ankurtrapasiya.com/posts/scoring-a-model/) | [`scoring-a-model/scoring.py`](scoring-a-model/scoring.py) |
| [Value is an estimate: discounting, growth and the margin of safety](https://ankurtrapasiya.com/posts/value-is-an-estimate/) | [`value-is-an-estimate/valuation.py`](value-is-an-estimate/valuation.py) |
| [What a trade really costs](https://ankurtrapasiya.com/posts/what-a-trade-costs/) | [`what-a-trade-costs/costs.py`](what-a-trade-costs/costs.py) |
| [What machine learning is: rules learned from examples](https://ankurtrapasiya.com/posts/what-is-machine-learning/) | [`what-is-machine-learning/fraud_rules.py`](what-is-machine-learning/fraud_rules.py) |
| [Why prices are hard to beat: efficiency, random walks and a roulette wheel](https://ankurtrapasiya.com/posts/why-prices-are-hard-to-beat/) | [`why-prices-are-hard-to-beat/efficiency.py`](why-prices-are-hard-to-beat/efficiency.py) |

Shared by all of them: [`market_data.py`](market_data.py) (loads the prices and builds the
feature table the ML posts use) and [`chartstyle.py`](chartstyle.py) (the charts' look).
[`fetch_prices.py`](fetch_prices.py) downloads fresh prices and overwrites `data/`; `run.py` skips
it, so run it only on purpose.

## Versions

[`requirements.txt`](requirements.txt) pins every library to the versions the posts' numbers were
checked with (Python 3.12). Newer versions may print slightly different decimals.

This repo is published from the blog's source; changes made here directly are overwritten by the
next sync.
