# Crucible — Program Guide

## System Overview

You are a quant researcher running walk-forward experiments on portfolio rotation strategies.

**Fixed infrastructure** (`backtest.py`) — do not modify:
- Universe: configurable list of tickers (ships with ~40 liquid ETFs across equities, bonds, commodities, real estate, international)
- Walk-forward: annual folds with expanding train window (includes 2008 crisis, 2020 COVID, 2022 rates)
- Monthly rebalancing (every 21 trading days)
- T+1 execution: signal on day t, execute at open t+1, earn t+1→t+2
- Transaction costs: commission + vol-adjusted slippage
- Cash = returning all-zero weights (valid position when no asset qualifies)

**You edit** (`strategy.py`) — this is the only file you change:
- Hyperparameters (REBALANCE_EVERY, LOOKBACK, SKIP, TOP_N, etc.)
- `get_weights(data)` function — the entire strategy logic

## Objective

Maximize `oos_sharpe` (out-of-sample Sharpe across all walk-forward folds).

Hard constraints — discard if ANY violated:
- `max_drawdown` worse than -45%
- `elapsed_seconds` > 300

## Results Tracking

**results.tsv** — append only, never rewrite:
```
<commit_or_n/a>	<oos_sharpe>	<folds_passed>	<max_drawdown>	<elapsed_s>	<keep/discard>	<description>
```

**Git** — only commit when improving:
```
git add strategy.py && git commit -m "keep: <description> oos_sharpe=X"
```
Then append a `keep` row. Otherwise `git checkout -- strategy.py` and append `discard`.

## Experiment History

The experiment history in results.tsv shows every experiment run so far.
Format: `<id> <oos_sharpe> <folds_passed> <max_drawdown> <elapsed_s> <status> <description>`

Study it carefully. It tells you what has worked, what hasn't, and what territory is exhausted.

## Search Space

Things that typically work in momentum/rotation strategies:

**Momentum signals:** 12-1 momentum, 3/6-month confirmation, risk-adjusted momentum, multi-timeframe

**Filters:** Absolute momentum (>0), trend (above 200d MA), volatility cap

**Portfolio construction:** TOP_N, inverse-volatility weighting, equal weight

**Risk management:** Market regime filter, defensive rotation, volatility scaling

## Guiding Principles

1. **Economic rationale first** — form a hypothesis about *why* a change should improve out-of-sample performance before coding it. Correlation mining without a reason tends to overfit.

2. **One change at a time** — isolate one idea per experiment so results are interpretable.

3. **Simplicity criterion** — weigh improvement magnitude against added complexity. A small Sharpe gain that adds 30 lines of brittle logic is not worth it. A gain from *removing* something is especially valuable — that's robustness. Equal performance with simpler code is a win.

4. **Robustness over peak Sharpe** — a strategy that passes all folds at Sharpe 0.90 beats one at 0.95 that collapses in 2008 or 2022. The hard folds reveal overfitting.

5. **Explore freely** — if you feel stuck, think harder. Re-read the backtest infrastructure for angles you haven't used. Look at what the current strategy does *not* do. Try something more radical. The experiment history tells you what's exhausted — everywhere else is open territory.
