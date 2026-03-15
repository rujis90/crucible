"""
strategy.py — AGENT EDITABLE. Modify freely.

Only this file changes between experiments. backtest.py is fixed.

Interface contract (do not rename these):
  - REBALANCE_EVERY : int        — trading days between rebalances
  - get_weights(data) -> pd.Series — return target portfolio weights

─────────────────────────────────────────────────────────────────────────────
R01: MA200 regime gate — QQQ below MA200 → SHY (defensive)
─────────────────────────────────────────────────────────────────────────────
"""

import pandas as pd

REBALANCE_EVERY = 21
MA_SLOW = 200


def get_weights(data: dict) -> pd.Series:
    close = data["close"]
    weights = pd.Series(0.0, index=close.columns)
    if "QQQ" not in close.columns:
        return weights
    if len(close) < MA_SLOW:
        weights["QQQ"] = 1.0
        return weights
    ma200 = close["QQQ"].iloc[-MA_SLOW:].mean()
    if close["QQQ"].iloc[-1] >= ma200:
        weights["QQQ"] = 1.0
    elif "SHY" in close.columns:
        weights["SHY"] = 1.0
    return weights
