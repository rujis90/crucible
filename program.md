# Crucible — Program Guide

## System Overview

You are a quant researcher running walk-forward experiments on portfolio rotation strategies.

**Fixed infrastructure** (`backtest.py`):
- Universe: configurable list of tickers (ships with ~40 liquid ETFs across equities, bonds, commodities, real estate, international)
- Walk-forward: annual folds with expanding train window (includes 2008 crisis, 2020 COVID, 2022 rates)
- Monthly rebalancing (every 21 trading days)
- T+1 execution: signal on day t, execute at open t+1, earn t+1→t+2
- Transaction costs: commission + vol-adjusted slippage
- Cash = returning all-zero weights (preferred over holding losers)

**Agent editable** (`strategy.py`):
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

## Search Space

Things that typically work in momentum/rotation strategies:

**Momentum signals:**
- 12-1 momentum (standard — skip short-term reversal)
- 3-month, 6-month momentum as confirmation
- Risk-adjusted momentum: momentum / volatility
- Multi-timeframe combinations

**Filters:**
- Absolute momentum: only hold if return > 0 (most important!)
- Trend filter: asset must be above its 200-day MA
- Volatility filter: exclude assets with unusually high recent vol

**Portfolio construction:**
- TOP_N = 1 to 10 (fewer = more concentrated but cleaner signal)
- Inverse-volatility weighting (weight = 1/vol, normalized)
- Equal weight (simple, robust)

**Risk management:**
- Market regime filter: go flat when broad market is in downtrend
- Defensive rotation: shift to bonds/gold when equities weaken
- Volatility scaling: reduce exposure when market vol is elevated

## Guiding Principles

1. **Economic rationale first** — explain WHY before coding
2. **Read results.tsv** — never repeat what's been tried
3. **Crisis folds are the hardest** — a real improvement must handle them
4. **Cash is a valid position** — when nothing qualifies, returning zeros is correct
5. **Robustness > peak Sharpe** — prefer strategies that work across all folds
6. **Simple beats complex** — start with cleaner signals, not more parameters
7. **Absolute momentum is the key insight** — always keep or strengthen it
