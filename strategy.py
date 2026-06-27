"""
strategy.py — the only file you modify.

Interface (do not rename):
  REBALANCE_EVERY : int   — trading days between rebalances
  PT_SL           : list  — [profit_target_mult, stop_loss_mult] in daily vol units
  MAX_HOLD        : int   — vertical barrier in trading days
  USE_CPCV        : bool  — True = full CPCV (15 paths), False = fast walk-forward
  get_signals(train_features, train_labels, test_features) → dict[date, Series]

─────────────────────────────────────────────────────────────────────────────
Baseline: cross-sectional momentum top-10.

Ranks every NDX-universe ticker by 21-day return (CS z-scored), goes long the
top 10 with equal weight. No ML, no fitting.

This is the floor — every experiment must beat this on oos_sharpe / oos_sharpe_std.
See program.md for the full search space and guiding principles.
See examples/ for strategies that have already been tested.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import pandas as pd

# ── Parameters (tune these too) ───────────────────────────────────────────────
REBALANCE_EVERY = 5         # weekly rebalance
PT_SL           = [1.5, 1.0]  # profit target 1.5×vol, stop 1.0×vol
MAX_HOLD        = 20        # close after 20 trading days if neither barrier hit
USE_CPCV        = True      # set False for fast iteration, True before keeping


def get_signals(
    train_features: pd.DataFrame,
    train_labels: pd.Series,
    test_features: pd.DataFrame,
) -> dict[pd.Timestamp, pd.Series]:
    """
    Return signals for each test date.

    Args:
        train_features  — (date, ticker) MultiIndex, columns = feature names
        train_labels    — (date, ticker) MultiIndex, values ∈ {-1, 0, +1}
        test_features   — same structure, no labels

    Returns:
        dict: test_date → Series{ticker: signal_strength ∈ [0, 1]}
        0 = no position, 1 = maximum weight
        backtest.py caps each position at MAX_POSITION and normalises gross to 1.
    """
    signals_by_date: dict[pd.Timestamp, pd.Series] = {}

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

            top_n = 10
            top_tickers = momentum.nlargest(top_n)

            signals = pd.Series(0.0, index=momentum.index)
            signals[top_tickers.index] = 1.0 / top_n
            signals_by_date[date] = signals

        except Exception:
            signals_by_date[date] = pd.Series(dtype=float)

    return signals_by_date
