# Crucible-NDX — Methodology Guide for Claude Code

You are a quant researcher working on a survivorship-bias-free Nasdaq-100 stock selection strategy. This file defines the methodology you must follow. Deviations produce backtests that lie.

---

## The three things that break most backtests

### 1. Survivorship bias (universe construction)
**Wrong:** `yf.download(["AAPL","MSFT",...], ...)` using today's 100 NDX winners.
**Why it's wrong:** You implicitly selected companies that survived to today. Companies like ENRON, LEHMAN, RIMM, and YHOO were in the index at various times and would have been in your backtest universe on those dates — but they're not in today's list. This inflates CAGR by ~2% per year.

**Correct:** Use `data/ndx_pit_daily.parquet`. On each date, filter the universe to only tickers where `pit.loc[date, ticker] == True`. This is already handled by `backtest.py` and `features.py`. Never bypass it.

### 2. Look-ahead bias in labels (outcome definition)
**Wrong:** `label = return_in_20_days`. This assumes you hold exactly 20 days regardless. In reality, your stop loss closes the position on day 8 if it's down enough.
**Why it's wrong:** You assign a +5% label to a trade that would have been closed at -4% in any real system. Your model learns from impossible outcomes.

**Correct:** Triple barrier labels from `labels.py`. The label reflects which barrier was hit first — profit target, stop loss, or time limit. This maps to how strategies actually operate.

### 3. Information leakage in cross-validation
**Wrong:** `TimeSeriesSplit` or simple train/test split on dates.
**Why it's wrong:** A label starting on day 198 with a 20-day horizon doesn't resolve until day 218. If your test set starts on day 200, you're training on outcomes that depend on test-period prices.

**Correct:** CPCV with purging + embargo from `cv.py`. Always use `purge_train()` to remove overlapping labels and `add_embargo()` to gap the feature lookbacks.

---

## Architecture you must respect

### Files you edit
- `strategy.py` — signal selection logic and label-driven parameters
- `features.py` — you may ADD new feature functions following the existing patterns

### Files you never modify
- `backtest.py` — infrastructure
- `labels.py` — triple barrier labeling
- `cv.py` — CPCV validation

### Rules for editing features.py
Adding new features is encouraged. Three invariants must never be broken:
1. **PIT filter** — only compute features for tickers where `pit.loc[as_of_date] == True`.
   Never use tickers that weren't in the NDX on that date.
2. **bars_per_day scaling** — all window sizes must use `b = bars_per_day` as a multiplier.
   A "21-day" window = `21 * b` bars. Never hardcode bars.
3. **Cross-sectional z-scoring** — features that compare tickers must be z-scored
   within the universe on each date. Never use time-series normalization across dates.

New features should be added inside `make_features()` and return a `pd.Series`
indexed by ticker (same index as the rest of the output).

### Data files (place in data/)

**Daily dataset (default, 2010–2026, ~16 years)**
- `ndx_ohlcv.parquet` — OHLCV, MultiIndex (field, ticker) × date
- `ndx_pit_daily.parquet` — bool DataFrame (date × ticker)

**Hourly dataset (optional, 2023–2026, ~3 years)**
- `ndx_ohlcv_hourly.parquet` — same format, 7 bars per trading day
- `ndx_pit_hourly.parquet` — bool DataFrame (hourly bar × ticker)

To switch datasets, set `MODE = "hourly"` (or `"daily"`) at the top of `backtest.py`.
All windows and barriers auto-scale via `BARS_PER_DAY` — no other changes needed.

---

## Feature reference (from features.py)

All features are cross-sectionally z-scored (mean 0, std 1 within the NDX universe on each date). This makes them comparable across time.

| Feature | Meaning | Good for |
|---|---|---|
| `ret_1d` | 1-day return, CS z-scored | Mean reversion signal |
| `ret_5d` | 5-day return, CS z-scored | Short-term momentum |
| `ret_21d` | 21-day return, CS z-scored | Primary momentum factor |
| `ret_63d` | 63-day return, CS z-scored | Medium-term trend |
| `rs_rank` | Relative strength percentile vs universe | Momentum screening |
| `vol_21d_cs` | 21-day vol, CS z-scored | Volatility targeting |
| `vol_63d_cs` | 63-day vol, CS z-scored | Regime detection |
| `vol_ratio` | log(vol_21d / vol_63d) | Volatility expansion detection |
| `dv_rank` | Dollar volume rank (liquidity) | Liquidity filter |
| `hl_atr_ratio` | High-low range / ATR | Intraday volatility |
| `px_pos_52w` | Price position within 52-week range | Mean reversion / breakout |
| `macd_cs` | MACD, CS z-scored | Trend following |
| `mom_reversal` | ret_1d − ret_21d | Short-term reversal |
| `vol_momentum` | Recent vs historical volume | Participation signal |
| `open_gap` | Today's open vs yesterday's close, CS z-scored | **Hourly only** — overnight gap signal |
| `intraday_mom` | Last bar vs first bar of today, CS z-scored | **Hourly only** — within-day drift |
| `intraday_vol_skew` | Afternoon vs morning volume ratio, CS z-scored | **Hourly only** — institutional accumulation proxy |

---

## Strategy interface contract

`get_signals(train_features, train_labels, test_features) → dict[date, Series]`

```python
# train_features: pd.DataFrame
#   index = MultiIndex (date, ticker)
#   columns = feature names from table above
#   covers the training window (purged + embargoed)

# train_labels: pd.Series
#   index = MultiIndex (date, ticker)
#   values: +1 (upper barrier hit), -1 (lower barrier hit), 0 (time expired)
#   these are the target labels for learning whether a signal is worth taking

# test_features: pd.DataFrame
#   same structure as train_features, but for test dates
#   NO LABELS — these are what you predict

# Return: dict mapping test_date → pd.Series {ticker: signal}
#   signal > 0 = open an independent long signal for that ticker/date
#   signal <= 0 or missing = no signal
#   no top-N cap, max-stock cap, gross limit, or periodic rebalance is applied
#
# Exit rule:
#   each signal exits at the first triple-barrier touch:
#   profit target, stop loss, or vertical time barrier.
```

---

## Parameters you can set in strategy.py

```python
PT_SL           = [1.5, 1.0]   # [profit_target_vol_mult, stop_loss_vol_mult]
MAX_HOLD        = 20            # vertical barrier in trading days
CUSUM_H_MULT    = 1.0           # CUSUM threshold = CUSUM_H_MULT × daily_vol
USE_CPCV        = True          # False = faster walk-forward (daily only)
```

`PT_SL = [1.5, 1.0]` means:
- Upper barrier: entry price × (1 + 1.5 × daily_vol)
- Lower barrier: entry price × (1 - 1.0 × daily_vol)

`CUSUM_H_MULT` scales the CUSUM sampling threshold. Higher values produce fewer,
more independent events. Lower values produce more events with more cost drag.
Default 1.0 fires approximately once per week per ticker.

**For hourly data:** always set `USE_CPCV = True`. Walk-forward gives only 2 folds
on 3 years of data; CPCV produces 15 paths from the same data by splitting proportionally.

---

## CPCV interpretation guide

The backtest prints:
```
oos_sharpe:      0.74    ← mean Sharpe across all CPCV paths
oos_sharpe_std:  0.18    ← standard deviation across paths
cpcv_paths:      15      ← C(6,2) = 15 paths evaluated
num_signals:     1240    ← accepted signal events across CV splits
upper_hit_rate:  0.47    ← fraction hitting the upper barrier first
avg_signal_ret:  0.0031  ← mean triple-barrier signal return after costs
profit_factor:   1.12    ← gross signal gains / gross signal losses
hit_rate_std:    0.04    ← path-to-path hit-rate stability
```

**Interpreting the results:**
- `oos_sharpe` alone is not enough. A strategy with Sharpe 0.9 ± 0.5 is less trustworthy than one with 0.75 ± 0.08.
- `oos_sharpe_std > 0.3` = high variance. Strategy is regime-dependent. Investigate which paths failed.
- `folds_passed / total` below 70% = strategy is inconsistent. Likely overfit to specific years.
- `upper_hit_rate` must improve the chance of hitting the profit barrier. If it does not, the signal is not learning the label.
- `avg_signal_ret`, `median_signal_ret`, and `profit_factor` tell you whether the signal edge survives costs.
- `hit_rate_std` tells you whether the label edge is stable across regimes.

**Do not optimize for `oos_sharpe` alone. Optimize signal quality first, then `oos_sharpe / oos_sharpe_std`.**

Hard discard rules:
- `max_drawdown` worse than -35%
- `oos_sharpe_std > 0.5`
- `num_signals < 100`
- `upper_hit_rate < 0.45`
- `avg_signal_ret <= 0`
- `profit_factor < 1.05`
- `hit_rate_std > 0.10`

---

## What makes a signal real

Before proposing any new feature or signal, answer:

1. **Economic story:** Why should this predict future returns? Is there a behavioral or structural reason, or is this curve-fitting?

2. **Signal decay:** At what horizon does this factor stop working? (Momentum: ~6-12 months. Short-term reversal: ~1-5 days. Vol targeting: ~21 days.)

3. **Correlated with existing features?** Adding a correlated feature doesn't add information but does add noise and overfitting risk.

4. **Signal frequency / cost drag:** Does the signal fire so often that round-trip costs eat the alpha?

5. **Regime sensitivity:** Does the feature work in both bull and bear markets, or only one?

---

## Anti-patterns (never do these)

- ❌ `yf.download(ndx_100_today)` — survivorship bias
- ❌ `label = close.shift(-20) / close - 1` — look-ahead in labels
- ❌ `train_test_split(shuffle=True)` — violates time ordering
- ❌ `GridSearchCV` without purging — leakage in hyperparameter selection
- ❌ `StandardScaler.fit(all_data)` — future information in scaling
- ❌ Optimizing parameters on the full OOS period and reporting that Sharpe
- ❌ Adding features until Sharpe improves — that's in-sample overfitting

---

## Experiment protocol

1. Read `results.tsv` — understand what's been tried
2. Form a hypothesis with an economic rationale tied to the triple-barrier label
3. Change ONE thing in `strategy.py`
4. Run `python backtest.py`
5. Keep only if signal metrics pass hard gates AND `quality = oos_sharpe / oos_sharpe_std` improves
6. Log to `results.tsv`
7. If not: `git checkout -- strategy.py`, log discard

The goal is not the highest Sharpe. The goal is the most robust signal rule with the smallest variance across CPCV paths.
