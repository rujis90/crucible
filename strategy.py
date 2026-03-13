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

import numpy as np
import pandas as pd

REBALANCE_EVERY = 21    # monthly rebalance (~21 trading days)
LOOKBACK        = 252   # 12-month momentum window
SKIP            = 21    # skip most recent 1 month (short-term reversal)
TOP_N           = 5     # number of assets to hold


def get_weights(data: dict) -> pd.Series:
    """
    Top-N rotation by 12-1 momentum with absolute momentum filter.
    Equal-weight the winners. This is the classic Faber/Antonacci baseline.
    """
    close = data["close"]

    if len(close) < LOOKBACK + SKIP:
        return pd.Series(0.0, index=close.columns)

    price_now  = close.iloc[-SKIP]
    price_then = close.iloc[-LOOKBACK]
    momentum   = (price_now / price_then - 1).dropna()

    # Absolute momentum filter: only hold assets with positive 12m return
    momentum = momentum[momentum > 0]

    if momentum.empty:
        return pd.Series(0.0, index=close.columns)

    top = momentum.nlargest(TOP_N).index.tolist()

    weights = pd.Series(0.0, index=close.columns)
    weights[top] = 1.0 / len(top)
    return weights
