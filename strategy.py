"""
strategy.py — AGENT EDITABLE. Modify freely.

Only this file changes between experiments. backtest.py is fixed.

Interface contract (do not rename these):
  - REBALANCE_EVERY : int        — trading days between rebalances
  - get_weights(data) -> pd.Series — return target portfolio weights

─────────────────────────────────────────────────────────────────────────────
Example strategy: hold a single asset 100%.

Change ASSET to any ticker in the universe (see backtest.py for full list).
Common choices: "QQQ", "SPY", "GLD", "TLT"
─────────────────────────────────────────────────────────────────────────────
"""

import pandas as pd

REBALANCE_EVERY = 21       # trading days between rebalances (~1 month)
ASSET           = "QQQ"   # ticker to hold — must be in backtest.py UNIVERSE


def get_weights(data: dict) -> pd.Series:
    weights = pd.Series(0.0, index=data["close"].columns)
    if ASSET in weights.index:
        weights[ASSET] = 1.0
    return weights
