"""
strategy.py — AGENT EDITABLE. Modify freely.

Only this file changes between experiments. backtest.py is fixed.

Interface contract (do not rename these):
  - REBALANCE_EVERY : int        — trading days between rebalances
  - get_weights(data) -> pd.Series — return target portfolio weights

─────────────────────────────────────────────────────────────────────────────
Baseline: Buy and hold QQQ.
─────────────────────────────────────────────────────────────────────────────
"""

import pandas as pd

REBALANCE_EVERY = 21


def get_weights(data: dict) -> pd.Series:
    close = data["close"]
    weights = pd.Series(0.0, index=close.columns)
    if "QQQ" in weights.index:
        weights["QQQ"] = 1.0
    return weights
