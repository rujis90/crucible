"""
strategy.py — AGENT EDITABLE. Modify freely.

Interface contract (do not rename these):
  REBALANCE_EVERY : int   — calendar days between portfolio rebalances
  PT_SL           : list  — [profit_target_mult, stop_loss_mult] in daily vol units
  MAX_HOLD        : int   — triple barrier vertical limit in trading days
  USE_CPCV        : bool  — True = full CPCV, False = fast walk-forward
  get_signals(train_features, train_labels, test_features) → dict[date, Series]

The agent edits the signal logic inside get_signals().
backtest.py handles universe construction, label generation, and execution.

─────────────────────────────────────────────────────────────────────────────
Baseline strategy: rank tickers by 21-day momentum (ret_21d feature),
go long the top 10 within the NDX PIT universe.

No ML, no fitting — pure cross-sectional momentum signal.
Use this as a starting benchmark. It should be easy to beat.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import pandas as pd

# ── Parameters ────────────────────────────────────────────────────────────────
REBALANCE_EVERY = 5       # trading days between rebalances
PT_SL           = [1.5, 1.0]  # profit target = 1.5× daily vol, stop = 1.0×
MAX_HOLD        = 20      # vertical barrier: close position after 20 days max
USE_CPCV        = True    # set False for fast walk-forward iteration


def get_signals(
    train_features: pd.DataFrame,
    train_labels: pd.Series,
    test_features: pd.DataFrame,
) -> dict[pd.Timestamp, pd.Series]:
    """
    Generate trading signals for each test event date.

    Args:
        train_features: DataFrame indexed by (date, ticker), columns = feature names
        train_labels:   Series indexed by (date, ticker), values = {-1, 0, 1}
        test_features:  same structure as train_features, but for test dates

    Returns:
        dict mapping each test date → Series {ticker: signal_strength}
        signal_strength in [0, 1]: 0 = no position, 1 = full weight
        backtest.py normalises weights to respect MAX_POSITION and GROSS_LIMIT
    """
    signals_by_date: dict[pd.Timestamp, pd.Series] = {}

    # Get unique test dates
    if isinstance(test_features.index, pd.MultiIndex):
        test_dates = test_features.index.get_level_values("t0").unique()
    else:
        test_dates = test_features.index.unique()

    for date in test_dates:
        try:
            if isinstance(test_features.index, pd.MultiIndex):
                snapshot = test_features.xs(date, level="t0")
            else:
                snapshot = test_features.loc[[date]]

            if snapshot.empty or "ret_21d" not in snapshot.columns:
                signals_by_date[date] = pd.Series(dtype=float)
                continue

            # Rank by 21-day relative strength within the NDX PIT universe.
            # Higher ret_21d = stronger momentum = higher signal.
            momentum = snapshot["ret_21d"].dropna()
            if momentum.empty:
                signals_by_date[date] = pd.Series(dtype=float)
                continue

            # Long top 10 by momentum rank
            top_n = 10
            top_tickers = momentum.nlargest(top_n)

            # Equal weight among top-N
            signals = pd.Series(0.0, index=momentum.index)
            signals[top_tickers.index] = 1.0 / top_n

            signals_by_date[date] = signals

        except Exception:
            signals_by_date[date] = pd.Series(dtype=float)

    return signals_by_date
