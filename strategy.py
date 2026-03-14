"""
strategy.py — AGENT EDITABLE. Modify freely.

Only this file changes between experiments. backtest.py is fixed.

Interface contract (do not rename these):
  - REBALANCE_EVERY : int        — trading days between rebalances
  - get_weights(data) -> pd.Series — return target portfolio weights

data dict keys (all DataFrames: DatetimeIndex rows x ticker columns):
  data['close']   adjusted close prices
  data['volume']  daily volume
  data['high']    daily high  (adjusted)
  data['low']     daily low   (adjusted)

All data sliced up to and including the signal date. No future data is present.

Weights returned:
  All values >= 0 (long-only)
  Returning all zeros = cash (flat) — valid and preferred over holding losers
  backtest.py clips each position to MAX_POSITION and gross to GROSS_LIMIT
"""

import pandas as pd

REBALANCE_EVERY = 21    # monthly rebalance (~21 trading days)


def get_weights(data: dict) -> pd.Series:
    """
    Baseline: equal-weight all assets (1/N portfolio).
    No signal, no filter — just split the portfolio evenly.
    Use this as a weak baseline to beat with a better strategy.
    """
    close = data["close"]
    tickers = close.columns
    return pd.Series(1.0 / len(tickers), index=tickers)
