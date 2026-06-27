"""
Experiment: rs_rank quality filter on baseline momentum.

Result vs baseline:
  Baseline    oos_sharpe=0.41  std=0.26  quality=1.57  dd=-37.5%  cagr=6.9%
  This strat  oos_sharpe=0.58  std=0.30  quality=1.89  dd=-37.2%  cagr=9.8%

Decision: KEEP — quality ratio +20%, CAGR +42%, all 15 paths positive.
Std slightly higher but the mean gain dominates.

Economic rationale: 21d momentum picks fast movers but some are just short-term
pops. The rs_rank gate (63d relative strength percentile) ensures we only enter
tickers that are strong on a medium-term basis — stage 2 uptrends in Qullamaggie
terms. Filters out tickers that are rebounding off lows or had a one-week spike.
"""
from __future__ import annotations

import pandas as pd

REBALANCE_EVERY = 5
PT_SL           = [1.5, 1.0]
MAX_HOLD        = 20
USE_CPCV        = True

RS_RANK_MIN     = 0.6   # top 40% by 63-day relative strength rank


def get_signals(train_features, train_labels, test_features):
    signals_by_date = {}

    if isinstance(test_features.index, pd.MultiIndex):
        test_dates = test_features.index.get_level_values("t0").unique()
    else:
        test_dates = test_features.index.unique()

    for date in test_dates:
        try:
            snapshot = (
                test_features.xs(date, level="t0")
                if isinstance(test_features.index, pd.MultiIndex)
                else test_features.loc[[date]]
            )

            if snapshot.empty or "ret_21d" not in snapshot.columns:
                signals_by_date[date] = pd.Series(dtype=float)
                continue

            momentum = snapshot["ret_21d"].dropna()
            if momentum.empty:
                signals_by_date[date] = pd.Series(dtype=float)
                continue

            # Gate: only tickers in top 40% by medium-term relative strength
            if "rs_rank" in snapshot.columns:
                rs_pct = snapshot["rs_rank"].reindex(momentum.index).rank(pct=True)
                momentum = momentum.reindex(rs_pct[rs_pct >= RS_RANK_MIN].index).dropna()

            if momentum.empty:
                signals_by_date[date] = pd.Series(dtype=float)
                continue

            top_n = min(10, len(momentum))
            top_tickers = momentum.nlargest(top_n)
            signals = pd.Series(0.0, index=snapshot.index)
            signals[top_tickers.index] = 1.0 / top_n
            signals_by_date[date] = signals

        except Exception:
            signals_by_date[date] = pd.Series(dtype=float)

    return signals_by_date
